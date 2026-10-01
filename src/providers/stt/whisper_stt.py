"""OpenAI Whisper — the accuracy-comparison speech-to-text provider.

Whisper is in the benchmark because it is the reference point everyone already
knows. It is roughly 40% more expensive per hour than the Deepgram baseline and
it cannot stream at all, so it will never be the live provider for this agent.
What it offers is a credible second opinion on accuracy: if Nova-3 and Whisper
disagree about a corpus file, the ground truth is worth re-checking.

Two differences from Deepgram are visible in the result object and are
documented rather than papered over:

* **No confidence score.** The OpenAI transcription API does not return one, so
  :attr:`~src.providers.base.TranscriptResult.confidence` is ``-1.0``.
* **Retries are this provider's job.** Whisper requests are long-running
  (the whole file is uploaded and processed before anything comes back), which
  makes them more exposed to transient 429s and 5xxs than a streaming provider.
  Retrying happens here, inside the latency measurement, so the benchmark
  reports the latency a caller actually experienced rather than a best case.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from src.providers.base import STTProvider, TranscriptResult, WordTiming
from src.providers.stt.deepgram_stt import _duration_from_wav

__all__ = ["WhisperSTT"]

#: USD per hour of audio for ``whisper-1``.
#: $0.006/min x 60 = $0.36/hr. Checked against openai.com/api/pricing on
#: 2026-10-01.
_COST_PER_HOUR = 0.36

#: Pinned model. ``whisper-1`` is the API-hosted Whisper large-v2 endpoint.
_MODEL = "whisper-1"

#: Whisper exposes no per-result confidence. ``-1.0`` is the project-wide
#: sentinel for "unknown", chosen over ``0.0`` so that a missing score can never
#: be aggregated as if the provider were certain it was wrong.
_NO_CONFIDENCE = -1.0

#: Transient-failure retry budget. Three attempts with exponential backoff:
#: enough to ride out a rate-limit blip, few enough that a genuinely broken key
#: fails the run in seconds rather than minutes.
_MAX_ATTEMPTS = 3


class WhisperSTT(STTProvider):
    """Speech-to-text backed by OpenAI's hosted Whisper API.

    Args:
        api_key: OpenAI API key. Defaults to the ``OPENAI_API_KEY``
            environment variable.
        model: Transcription model identifier. Defaults to ``"whisper-1"``.
        language: ISO-639-1 language hint. Supplying it skips Whisper's
            language-detection pass, which both speeds up the call and stops it
            occasionally deciding a short clip is not English.
        prompt: Optional biasing prompt. This is Whisper's nearest equivalent to
            Deepgram keyterm boost — proper nouns listed here are more likely to
            be spelled correctly — but it is advisory, not a hard boost, so the
            two are not strictly comparable. Left ``None`` by default to keep
            the accuracy comparison clean.

    Raises:
        RuntimeError: If no API key is available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        language: str = "en",
        prompt: str | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or pass api_key= explicitly."
            )
        self._api_key = key
        self._model = model
        self._language = language
        self._prompt = prompt

    @property
    def name(self) -> str:
        """Return ``"whisper"``."""
        return "whisper"

    @property
    def cost_per_hour(self) -> float:
        """Return USD per hour of audio: ``0.36`` for ``whisper-1``."""
        return _COST_PER_HOUR

    @property
    def model(self) -> str:
        """Return the pinned model identifier."""
        return self._model

    async def transcribe(self, audio_bytes: bytes) -> TranscriptResult:
        """Transcribe a complete audio buffer via the OpenAI API.

        Retries up to three times with exponential backoff on transient
        failures. The latency clock covers all attempts, so a result that needed
        a retry honestly reports the longer wait.

        Args:
            audio_bytes: Complete WAV file contents.

        Returns:
            A final :class:`~src.providers.base.TranscriptResult` with
            ``confidence == -1.0``.

        Raises:
            RuntimeError: If the SDKs are not installed or every attempt failed.
        """
        # Lazy imports: a contributor benchmarking only Deepgram does not need
        # the OpenAI SDK installed for this module to be importable.
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "openai is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            from tenacity import (
                AsyncRetrying,
                retry_if_exception_type,
                stop_after_attempt,
                wait_exponential,
            )
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "tenacity is not installed. Run: pip install -r requirements.txt"
            ) from exc

        audio_seconds = _duration_from_wav(audio_bytes)
        client = AsyncOpenAI(api_key=self._api_key)

        request: dict[str, Any] = {
            "model": self._model,
            "language": self._language,
            # verbose_json is required to get segment metadata back; the default
            # json response is text only.
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if self._prompt:
            request["prompt"] = self._prompt

        start = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    # A fresh file handle per attempt: the stream is consumed by
                    # the upload, so a retry on the same handle would send zero
                    # bytes and "succeed" with an empty transcript.
                    handle = io.BytesIO(audio_bytes)
                    handle.name = "audio.wav"
                    response = await client.audio.transcriptions.create(
                        file=handle, **request
                    )
        except Exception as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"Whisper transcription failed after {_MAX_ATTEMPTS} attempts: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        raw = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else {"text": getattr(response, "text", "")}
        )
        return self._parse(raw, latency_ms=latency_ms, audio_seconds=audio_seconds)

    def _parse(
        self, raw: dict[str, Any], *, latency_ms: float, audio_seconds: float
    ) -> TranscriptResult:
        """Convert an OpenAI transcription response into a ``TranscriptResult``.

        Args:
            raw: The response as a dict.
            latency_ms: Measured wall-clock latency across all attempts.
            audio_seconds: Locally computed audio duration.

        Returns:
            The normalised result.
        """
        timings = [
            WordTiming(
                word=str(word.get("word", "")),
                start=float(word.get("start", 0.0)),
                end=float(word.get("end", 0.0)),
                # Per-word confidence is not exposed either.
                confidence=_NO_CONFIDENCE,
            )
            for word in (raw.get("words") or [])
        ]

        return TranscriptResult(
            text=str(raw.get("text", "")).strip(),
            # Not a measurement — a sentinel. See _NO_CONFIDENCE.
            confidence=_NO_CONFIDENCE,
            latency_ms=latency_ms,
            provider=self.name,
            # Prefer Whisper's own duration when present; it agrees with the
            # local WAV header for well-formed files and covers formats the
            # `wave` module cannot parse.
            audio_seconds=float(raw.get("duration") or audio_seconds),
            word_timings=timings,
            is_final=True,
            raw=raw,
        )

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """Not supported by this provider.

        Unlike the Deepgram and AssemblyAI gaps, this one is permanent rather
        than scheduled: the OpenAI Whisper transcription API is file-upload only
        and has no real-time socket. That is the single strongest reason Whisper
        is a benchmark comparison point in this project and not a candidate for
        the live agent path.

        Args:
            stream: Async iterator of audio chunks.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Whisper API does not support streaming transcription - it is a "
            "file-upload endpoint with no real-time socket. Use transcribe() "
            "for batch audio, or pick a streaming provider (deepgram) for the "
            "live agent path."
        )
        yield  # pragma: no cover - marks this an async generator
