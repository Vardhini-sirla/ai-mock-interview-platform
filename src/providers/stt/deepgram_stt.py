"""Deepgram Nova-3 — the Sprint 1 speech-to-text baseline.

Nova-3 is the baseline the other two providers are measured against, chosen
before any benchmark ran for two reasons that the benchmark cannot change:

* It is the cheapest of the three per hour of audio.
* It supports **keyterm boost**, which matters disproportionately for this
  project. A mock interview is full of proper nouns a general model has never
  seen — the candidate's own name, the companies on their resume, the
  frameworks in the job description. Without boosting, "Jobnova" becomes "job
  nova" and every downstream mention is scored as two errors.

Being the baseline is a measurement decision, not a verdict. If ``benchmarks/``
shows Whisper or AssemblyAI winning on this corpus, ``STT_PROVIDER`` changes
and nothing else does.
"""

from __future__ import annotations

import io
import os
import time
import wave
from collections.abc import AsyncIterator
from typing import Any

from src.providers.base import STTProvider, TranscriptResult, WordTiming

__all__ = ["DeepgramSTT", "DEFAULT_KEYTERMS"]

#: Proper nouns boosted on every request.
#:
#: These are the terms that recur across the capstone's own corpus — the
#: author, the product, and the vendors in the stack. A general-purpose
#: acoustic model mis-hears all of them, and each miss costs two WER errors
#: (one deletion, one insertion) rather than one.
DEFAULT_KEYTERMS: tuple[str, ...] = (
    "Vardhini",
    "Shia",
    "Jobnova",
    "LiveKit",
    "Tavus",
    "Deepgram",
    "Cartesia",
    "Clark",
    "Capstone",
)

#: USD per hour of pre-recorded audio, Nova-3 pay-as-you-go.
#: $0.0043/min x 60 = $0.258/hr. Checked against deepgram.com/pricing on
#: 2026-10-01. Update this literal in the same commit as any model change.
_COST_PER_HOUR = 0.258

#: Deepgram's current flagship model. Pinned, not "latest", so a benchmark run
#: from six months ago stays reproducible.
_MODEL = "nova-3"


def _duration_from_wav(audio_bytes: bytes) -> float:
    """Return the duration in seconds of a WAV buffer.

    Duration is read locally rather than taken from the provider response for
    two reasons: the benchmark needs it to price the call even when the request
    fails, and a locally computed figure is identical across all three
    providers, which keeps the cost column comparable.

    Args:
        audio_bytes: Complete WAV file contents, header included.

    Returns:
        Duration in seconds, or ``0.0`` if the buffer is not parseable WAV.
        Returning zero rather than raising keeps one malformed corpus file from
        aborting a whole benchmark run — the row simply reports no duration.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (wave.Error, EOFError, OSError):
        return 0.0


class DeepgramSTT(STTProvider):
    """Speech-to-text backed by Deepgram's Nova-3 model.

    Args:
        api_key: Deepgram API key. Defaults to the ``DEEPGRAM_API_KEY``
            environment variable.
        model: Deepgram model identifier. Defaults to ``"nova-3"``.
        keyterms: Proper nouns to boost. Defaults to :data:`DEFAULT_KEYTERMS`.
            Pass an empty sequence to measure the unboosted baseline, which is
            worth doing once to quantify what boosting buys.
        language: BCP-47 language tag passed to Deepgram.

    Raises:
        RuntimeError: If no API key is available. Raised in ``__init__`` so a
            misconfigured environment fails before any audio is read.

    Example:
        >>> stt = DeepgramSTT()                      # doctest: +SKIP
        >>> result = await stt.transcribe(wav_bytes)  # doctest: +SKIP
        >>> result.provider                           # doctest: +SKIP
        'deepgram'
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        keyterms: list[str] | tuple[str, ...] | None = None,
        language: str = "en-US",
    ) -> None:
        key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or pass api_key= explicitly."
            )
        self._api_key = key
        self._model = model
        self._keyterms = list(DEFAULT_KEYTERMS if keyterms is None else keyterms)
        self._language = language

    @property
    def name(self) -> str:
        """Return ``"deepgram"``."""
        return "deepgram"

    @property
    def cost_per_hour(self) -> float:
        """Return USD per hour of audio: ``0.258`` for Nova-3 pay-as-you-go."""
        return _COST_PER_HOUR

    @property
    def model(self) -> str:
        """Return the pinned Deepgram model identifier."""
        return self._model

    @property
    def keyterms(self) -> list[str]:
        """Return a copy of the boosted keyterm list."""
        return list(self._keyterms)

    async def transcribe(self, audio_bytes: bytes) -> TranscriptResult:
        """Transcribe a complete audio buffer via Deepgram's REST API.

        Args:
            audio_bytes: Complete WAV file contents.

        Returns:
            A final :class:`~src.providers.base.TranscriptResult`.

        Raises:
            RuntimeError: If the SDK is not installed, the request fails, or
                the response carries no alternatives.
        """
        # Lazy import: keeps `import src.providers` working for someone who has
        # only installed the Whisper or AssemblyAI SDK.
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "deepgram-sdk is not installed. Run: pip install -r requirements.txt"
            ) from exc

        audio_seconds = _duration_from_wav(audio_bytes)

        options_kwargs: dict[str, Any] = {
            "model": self._model,
            "language": self._language,
            "smart_format": True,
            "punctuate": True,
            # Word timings are requested even though Sprint 1 only scores text:
            # the barge-in work in Sprint 5 needs them, and asking now means the
            # benchmark corpus doubles as a fixture for that.
            "utterances": True,
        }
        if self._keyterms:
            # Nova-3 spells this `keyterm`; pre-Nova-3 models used `keywords`.
            options_kwargs["keyterm"] = self._keyterms

        client = DeepgramClient(self._api_key)
        options = PrerecordedOptions(**options_kwargs)
        payload = {"buffer": audio_bytes}

        start = time.perf_counter()
        try:
            response = await client.listen.asyncrest.v("1").transcribe_file(
                payload, options
            )
        except Exception as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Deepgram transcription failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        raw = response.to_dict() if hasattr(response, "to_dict") else dict(response)
        return self._parse(raw, latency_ms=latency_ms, audio_seconds=audio_seconds)

    def _parse(
        self, raw: dict[str, Any], *, latency_ms: float, audio_seconds: float
    ) -> TranscriptResult:
        """Convert a Deepgram response dict into a :class:`TranscriptResult`.

        Args:
            raw: Deepgram's JSON response as a dict.
            latency_ms: Measured wall-clock latency of the call.
            audio_seconds: Locally computed audio duration.

        Returns:
            The normalised result.

        Raises:
            RuntimeError: If the response contains no transcript alternative.
        """
        try:
            alternative = raw["results"]["channels"][0]["alternatives"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Deepgram returned no transcript alternatives: {raw!r}"
            ) from exc

        timings = [
            WordTiming(
                word=word.get("punctuated_word") or word.get("word", ""),
                start=float(word.get("start", 0.0)),
                end=float(word.get("end", 0.0)),
                confidence=float(word.get("confidence", -1.0)),
            )
            for word in alternative.get("words", [])
        ]

        return TranscriptResult(
            text=alternative.get("transcript", ""),
            confidence=float(alternative.get("confidence", -1.0)),
            latency_ms=latency_ms,
            provider=self.name,
            audio_seconds=audio_seconds,
            word_timings=timings,
            is_final=True,
            raw=raw,
        )

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """Not implemented in Sprint 1.

        Deepgram does support real-time streaming over WebSocket, and this is
        the provider whose streaming path the agent will actually use. It is
        deliberately left out of Sprint 1 because streaming is only testable
        against a live audio track, and Sprint 1's deliverable is the offline
        WER comparison.

        Args:
            stream: Async iterator of audio chunks.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Deepgram streaming transcription lands with agent wire-up "
            "(Sprint 4). Use transcribe() for batch audio."
        )
        yield  # pragma: no cover - marks this an async generator
