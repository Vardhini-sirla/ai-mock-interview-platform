"""AssemblyAI — the third speech-to-text comparison provider.

AssemblyAI is the most expensive of the three per hour, so it earns its place in
the benchmark on capability rather than price. Two things make it a genuine
contender for an interview agent:

* **``word_boost`` is a direct analogue of Deepgram's keyterm boost.** It takes
  the same list of proper nouns with the same intent, which makes the
  Deepgram-vs-AssemblyAI comparison the only apples-to-apples boosted pair in
  the benchmark. (Whisper's ``prompt`` is advisory and not equivalent.)
* **Its speaker-diarisation and disfluency handling are strong**, which matters
  for scoring a candidate's filler words later in the project.

The SDK is synchronous. Rather than reimplement its upload-and-poll flow over
``aiohttp``, the blocking call is pushed to a worker thread with
:func:`asyncio.to_thread`. That keeps the event loop free — the only property
the rest of the system actually requires of an async provider — without
maintaining a parallel HTTP client whose retry and error semantics would drift
from the vendor's own.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from src.providers.base import STTProvider, TranscriptResult, WordTiming
from src.providers.stt.deepgram_stt import DEFAULT_KEYTERMS, _duration_from_wav

__all__ = ["AssemblyAISTT"]

#: USD per hour of audio, Best tier async transcription.
#: $0.0062/min x 60 ~= $0.37/hr. Checked against assemblyai.com/pricing on
#: 2026-10-01.
_COST_PER_HOUR = 0.37

#: AssemblyAI reports word confidence in milliseconds-based word objects and
#: offsets in milliseconds, not seconds. Converted on parse so that
#: :class:`~src.providers.base.WordTiming` is in seconds for every provider.
_MS_PER_SECOND = 1000.0


class AssemblyAISTT(STTProvider):
    """Speech-to-text backed by AssemblyAI's async transcription API.

    Args:
        api_key: AssemblyAI API key. Defaults to the ``ASSEMBLYAI_API_KEY``
            environment variable.
        word_boost: Proper nouns to boost. Defaults to
            :data:`~src.providers.stt.deepgram_stt.DEFAULT_KEYTERMS`, shared
            deliberately with Deepgram so neither provider is handed a different
            vocabulary in the benchmark.
        language_code: BCP-47 language tag.
        speech_model: AssemblyAI model tier, ``"best"`` or ``"nano"``.
            ``"best"`` is the default and is what :attr:`cost_per_hour` prices;
            switching to ``"nano"`` requires updating that figure too.

    Raises:
        RuntimeError: If no API key is available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        word_boost: list[str] | tuple[str, ...] | None = None,
        language_code: str = "en_us",
        speech_model: str = "best",
    ) -> None:
        key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY is not set. Copy .env.example to .env and "
                "add your key, or pass api_key= explicitly."
            )
        self._api_key = key
        self._word_boost = list(DEFAULT_KEYTERMS if word_boost is None else word_boost)
        self._language_code = language_code
        self._speech_model = speech_model

    @property
    def name(self) -> str:
        """Return ``"assemblyai"``."""
        return "assemblyai"

    @property
    def cost_per_hour(self) -> float:
        """Return USD per hour of audio: ``0.37`` for the Best tier."""
        return _COST_PER_HOUR

    @property
    def word_boost(self) -> list[str]:
        """Return a copy of the boosted proper-noun list."""
        return list(self._word_boost)

    async def transcribe(self, audio_bytes: bytes) -> TranscriptResult:
        """Transcribe a complete audio buffer via AssemblyAI.

        The vendor SDK is synchronous, so the upload-and-poll call runs in a
        worker thread. Latency therefore includes AssemblyAI's queue wait, which
        is the honest figure for a batch API but is not comparable to a
        streaming provider's time-to-first-word.

        Args:
            audio_bytes: Complete WAV file contents.

        Returns:
            A final :class:`~src.providers.base.TranscriptResult`.

        Raises:
            RuntimeError: If the SDK is not installed or transcription failed.
        """
        # Lazy import: see the module docstring in src/providers/__init__.py.
        try:
            import assemblyai as aai
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "assemblyai is not installed. Run: pip install -r requirements.txt"
            ) from exc

        audio_seconds = _duration_from_wav(audio_bytes)

        aai.settings.api_key = self._api_key
        config = aai.TranscriptionConfig(
            language_code=self._language_code,
            speech_model=self._speech_model,
            punctuate=True,
            format_text=True,
            word_boost=self._word_boost or None,
            # "high" is the strongest boost weighting; anything less makes the
            # comparison against Deepgram keyterms unfair to AssemblyAI.
            boost_param="high" if self._word_boost else None,
        )

        def _blocking_transcribe() -> Any:
            """Run the synchronous SDK call. Executed on a worker thread."""
            return aai.Transcriber(config=config).transcribe(audio_bytes)

        start = time.perf_counter()
        try:
            transcript = await asyncio.to_thread(_blocking_transcribe)
        except Exception as exc:  # pragma: no cover - network path
            raise RuntimeError(f"AssemblyAI transcription failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        status = getattr(transcript, "status", None)
        if str(status).lower().endswith("error"):
            raise RuntimeError(
                f"AssemblyAI transcription failed: {getattr(transcript, 'error', status)}"
            )

        return self._parse(
            transcript, latency_ms=latency_ms, audio_seconds=audio_seconds
        )

    def _parse(
        self, transcript: Any, *, latency_ms: float, audio_seconds: float
    ) -> TranscriptResult:
        """Convert an AssemblyAI ``Transcript`` into a ``TranscriptResult``.

        Args:
            transcript: The SDK's transcript object.
            latency_ms: Measured wall-clock latency of the call.
            audio_seconds: Locally computed audio duration.

        Returns:
            The normalised result.
        """
        timings = [
            WordTiming(
                word=str(getattr(word, "text", "")),
                # AssemblyAI reports offsets in milliseconds.
                start=float(getattr(word, "start", 0) or 0) / _MS_PER_SECOND,
                end=float(getattr(word, "end", 0) or 0) / _MS_PER_SECOND,
                confidence=float(getattr(word, "confidence", -1.0) or -1.0),
            )
            for word in (getattr(transcript, "words", None) or [])
        ]

        duration = getattr(transcript, "audio_duration", None)
        return TranscriptResult(
            text=(getattr(transcript, "text", None) or "").strip(),
            confidence=float(getattr(transcript, "confidence", -1.0) or -1.0),
            latency_ms=latency_ms,
            provider=self.name,
            audio_seconds=float(duration) if duration else audio_seconds,
            word_timings=timings,
            is_final=True,
            raw={"id": getattr(transcript, "id", None), "status": str(
                getattr(transcript, "status", "")
            )},
        )

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """Not implemented in Sprint 1.

        AssemblyAI does offer a real-time WebSocket endpoint, but it is only
        wired up if AssemblyAI wins the Sprint 1 benchmark. Building streaming
        for a provider that is about to be ruled out on cost would be work spent
        before the measurement that justifies it.

        Args:
            stream: Async iterator of audio chunks.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "AssemblyAI streaming transcription is not implemented - it lands "
            "with agent wire-up only if AssemblyAI wins the Sprint 1 WER "
            "benchmark. Use transcribe() for batch audio."
        )
        yield  # pragma: no cover - marks this an async generator
