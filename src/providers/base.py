"""Provider contracts for the voice pipeline.

Every external service the interview agent talks to — speech-to-text,
text-to-speech, the reasoning LLM, voice-activity detection — sits behind one
of the abstract base classes in this module.

Why bother with an abstraction layer at all? Three reasons, and they are the
whole justification for Sprint 1:

1. **Benchmarking needs a uniform shape.** The WER harness in
   ``benchmarks/wer.py`` cannot compare Deepgram against Whisper against
   AssemblyAI unless all three return the same object with the same fields.
   :class:`TranscriptResult` is that object.
2. **Vendor swaps must be a config change, not a refactor.** Pricing moves,
   models get deprecated, and a provider that wins on accuracy may lose on
   latency. Keeping the call sites dependent on ``STTProvider`` rather than on
   ``deepgram.DeepgramClient`` means swapping is one env var.
3. **Cost has to be measurable at the source.** Each provider declares its own
   :attr:`STTProvider.cost_per_hour`, so the benchmark can price a run without
   a hand-maintained lookup table drifting out of date.

Sprint ownership of each contract:

==================  ======  ====================================================
Contract            Sprint  Status
==================  ======  ====================================================
:class:`STTProvider`     1  Implemented (Deepgram, Whisper, AssemblyAI)
:class:`TTSProvider`     2  Placeholder — contract only
:class:`LLMProvider`     3  Placeholder — contract only
:class:`VADProvider`     5  Placeholder — contract only
==================  ======  ====================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TranscriptResult",
    "SynthesisResult",
    "LLMResponse",
    "WordTiming",
    "STTProvider",
    "TTSProvider",
    "LLMProvider",
    "VADProvider",
]


@dataclass(slots=True)
class WordTiming:
    """Start/end offsets for a single recognised word.

    Word-level timings are what make barge-in and interruption handling
    possible later in the project: to know *where* the candidate was cut off we
    need more than a flat string. Providers that do not expose timings leave
    :attr:`TranscriptResult.word_timings` empty rather than faking values.

    Attributes:
        word: The recognised token, as the provider spelled it.
        start: Offset in seconds from the beginning of the audio.
        end: Offset in seconds from the beginning of the audio.
        confidence: Per-word confidence in ``0.0..1.0``, or ``-1.0`` when the
            provider does not report one.
    """

    word: str
    start: float
    end: float
    confidence: float = -1.0


@dataclass(slots=True)
class TranscriptResult:
    """A single transcription outcome, normalised across providers.

    This is the unit of comparison for the WER benchmark. Every field exists
    because the benchmark or the agent runtime needs it:

    Attributes:
        text: The transcript itself. Raw provider casing and punctuation are
            preserved — normalisation belongs to the scoring layer, not here,
            so that one corpus can be scored under different normalisations.
        confidence: Provider-reported confidence in ``0.0..1.0``. Set to
            ``-1.0`` when the provider does not expose a score (notably the
            OpenAI Whisper API), so that "unknown" is never silently read as
            "zero confidence".
        latency_ms: Wall-clock milliseconds for the provider call, measured by
            the provider implementation itself. Includes network time, because
            that is what the candidate actually experiences.
        provider: The provider's :attr:`STTProvider.name`, carried on the
            result so benchmark rows stay attributable after aggregation.
        audio_seconds: Duration of the submitted audio. Needed to turn
            :attr:`STTProvider.cost_per_hour` into a per-file cost and to
            compute a real-time factor.
        word_timings: Per-word offsets when the provider supplies them.
        is_final: ``False`` for interim streaming hypotheses, ``True`` for
            batch results and finalised streaming segments.
        raw: The provider's unmodified response, kept for debugging a
            surprising benchmark row without re-running the call.
    """

    text: str
    confidence: float
    latency_ms: float
    provider: str
    audio_seconds: float
    word_timings: list[WordTiming] = field(default_factory=list)
    is_final: bool = True
    raw: dict[str, Any] | None = None

    @property
    def word_count(self) -> int:
        """Return the number of whitespace-delimited tokens in the transcript."""
        return len(self.text.split())

    @property
    def real_time_factor(self) -> float:
        """Return latency divided by audio duration.

        A value below ``1.0`` means the provider transcribed faster than
        real time. Returns ``0.0`` when the audio duration is unknown, rather
        than raising, so a single bad corpus file cannot abort a benchmark run.
        """
        if self.audio_seconds <= 0:
            return 0.0
        return (self.latency_ms / 1000.0) / self.audio_seconds

    def estimated_cost_usd(self, cost_per_hour: float) -> float:
        """Return the dollar cost of this transcription at the given rate.

        Args:
            cost_per_hour: The provider's USD price per hour of audio.

        Returns:
            Cost in USD for :attr:`audio_seconds` of audio.
        """
        return (self.audio_seconds / 3600.0) * cost_per_hour


@dataclass(slots=True)
class SynthesisResult:
    """A single text-to-speech outcome, normalised across providers.

    Consumed from Sprint 2 onward. Defined here alongside
    :class:`TranscriptResult` so that both halves of the voice loop share one
    vocabulary for latency and cost.

    Attributes:
        audio: Encoded audio bytes in :attr:`encoding` at :attr:`sample_rate`.
        encoding: Container/codec label, e.g. ``"linear16"`` or ``"mp3"``.
        sample_rate: Samples per second of :attr:`audio`.
        latency_ms: Wall-clock milliseconds for the synthesis call.
        provider: The provider's :attr:`TTSProvider.name`.
        characters: Billable character count — TTS vendors price per character,
            not per second, so cost cannot be derived from duration.
        audio_seconds: Duration of the synthesised audio, when known.
    """

    audio: bytes
    encoding: str
    sample_rate: int
    latency_ms: float
    provider: str
    characters: int
    audio_seconds: float = 0.0


@dataclass(slots=True)
class LLMResponse:
    """A single LLM completion, normalised across providers.

    Consumed from Sprint 3 onward.

    Attributes:
        text: The assistant's reply.
        provider: The provider's :attr:`LLMProvider.name`.
        model: Concrete model identifier that served the request. Recorded
            separately from the provider because a provider exposes many
            models at different prices.
        latency_ms: Wall-clock milliseconds for the completion call.
        time_to_first_token_ms: Milliseconds until the first streamed token.
            This, not total latency, is what makes a spoken reply feel prompt.
        prompt_tokens: Billable input tokens.
        completion_tokens: Billable output tokens.
        finish_reason: Why generation stopped, e.g. ``"stop"`` or ``"length"``.
        raw: The provider's unmodified response.
    """

    text: str
    provider: str
    model: str
    latency_ms: float
    time_to_first_token_ms: float = -1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Return prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class STTProvider(ABC):
    """Contract for every speech-to-text backend.

    Implementations live in ``src/providers/stt/`` and are constructed through
    :func:`src.providers.get_stt`. Nothing outside that factory should import a
    concrete provider class by name.

    Implementation rules, each of which the benchmark depends on:

    * **Credentials are read in ``__init__``.** A missing API key must raise
      ``RuntimeError`` at construction, not halfway through a benchmark run.
    * **SDK imports are lazy.** Import the vendor SDK *inside* the method that
      needs it, so a contributor with only one provider installed can still
      import the package and run the rest of the suite.
    * **Latency is measured by the provider.** Use
      :func:`time.perf_counter` around the network call only, so the number
      reflects the vendor rather than local file I/O.
    * **Unknown confidence is ``-1.0``.** Never substitute ``0.0``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the short, stable provider identifier.

        This string is the factory key, the value of
        :attr:`TranscriptResult.provider`, and part of the benchmark CSV
        filename, so it must be lowercase, filename-safe, and never renamed
        once results have been committed.
        """

    @property
    @abstractmethod
    def cost_per_hour(self) -> float:
        """Return the USD list price per hour of transcribed audio.

        Declared per provider — "cost at source" — so pricing lives next to
        the integration it describes and is updated in the same commit as any
        model change. Implementations should cite the pricing page and the date
        the figure was checked in a comment.
        """

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> TranscriptResult:
        """Transcribe one complete audio buffer.

        This is the batch path used by the WER benchmark: a whole file in, one
        scored transcript out.

        Args:
            audio_bytes: Complete audio file contents, including the container
                header. The benchmark corpus is WAV, and implementations should
                not assume raw PCM.

        Returns:
            A :class:`TranscriptResult` with ``is_final=True``.

        Raises:
            RuntimeError: If the provider rejects the request or returns no
                transcript.
        """

    @abstractmethod
    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """Transcribe a live audio stream, yielding interim and final results.

        This is the path the real-time agent will use. Sprint 1 ships the batch
        path only; concrete providers raise ``NotImplementedError`` here with a
        message naming the sprint that lands it, so the gap is explicit rather
        than a silent empty iterator.

        Args:
            stream: Async iterator of audio chunks, typically 20ms frames of
                16-bit PCM from the WebRTC track.

        Yields:
            :class:`TranscriptResult` values. Interim hypotheses carry
            ``is_final=False`` and may revise earlier text; finalised segments
            carry ``is_final=True`` and are append-only.

        Raises:
            NotImplementedError: Until streaming lands for the provider.
        """

    def __repr__(self) -> str:
        """Return a debug representation naming the provider and its price."""
        return f"<{type(self).__name__} name={self.name!r} usd_per_hour={self.cost_per_hour}>"


class TTSProvider(ABC):
    """Contract for every text-to-speech backend.

    **Placeholder — Sprint 2.** The contract is defined now so that Sprint 1's
    factory, docs, and architecture diagram describe the whole pipeline rather
    than only the part that is built. No implementations exist yet, and
    :func:`src.providers.get_tts` will raise ``ValueError`` for any name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the short, stable provider identifier."""

    @property
    @abstractmethod
    def cost_per_million_chars(self) -> float:
        """Return the USD list price per million synthesised characters.

        Priced per character rather than per hour because that is how TTS
        vendors bill; the asymmetry with :attr:`STTProvider.cost_per_hour` is
        deliberate, not an oversight.
        """

    @abstractmethod
    async def synthesize(self, text: str) -> SynthesisResult:
        """Synthesise speech for a complete utterance.

        Args:
            text: The text to speak.

        Returns:
            A :class:`SynthesisResult` holding the encoded audio.
        """

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesise speech incrementally, yielding audio chunks.

        Streaming matters more for TTS than for STT: the candidate hears the
        first chunk while the rest is still being generated, which is the
        difference between a conversation and a wait.

        Args:
            text: The text to speak.

        Yields:
            Encoded audio chunks in playback order.
        """


class LLMProvider(ABC):
    """Contract for every reasoning backend that drives the interview.

    **Placeholder — Sprint 3.** Defined now for the same reason as
    :class:`TTSProvider`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the short, stable provider identifier."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the concrete model identifier this instance calls."""

    @abstractmethod
    async def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Generate a single completion for a message history.

        Args:
            messages: Chat history as ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Provider-specific overrides such as ``temperature`` or
                ``max_tokens``.

        Returns:
            An :class:`LLMResponse`.
        """

    @abstractmethod
    async def complete_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate a completion incrementally, yielding text deltas.

        Args:
            messages: Chat history as ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Provider-specific overrides.

        Yields:
            Text fragments in order. Callers forward these to the TTS layer so
            speech starts before generation finishes.
        """


class VADProvider(ABC):
    """Contract for every voice-activity / turn-detection backend.

    **Placeholder — Sprint 5.** Turn detection is what decides when the
    candidate has finished speaking. It is isolated behind its own contract
    because the choice — energy threshold, Silero, or a semantic end-of-turn
    model — changes interruption behaviour dramatically and will need its own
    benchmark, mirroring what Sprint 1 does for STT.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the short, stable provider identifier."""

    @abstractmethod
    async def is_speech(self, audio_chunk: bytes) -> bool:
        """Report whether a single audio frame contains speech.

        Args:
            audio_chunk: One frame of audio, typically 10-30ms of 16-bit PCM.

        Returns:
            ``True`` if the frame is judged to contain speech.
        """

    @abstractmethod
    async def detect_turn_end(self, stream: AsyncIterator[bytes]) -> AsyncIterator[bool]:
        """Emit a signal each time the speaker appears to have finished a turn.

        Args:
            stream: Async iterator of audio frames.

        Yields:
            ``True`` at each detected end-of-turn boundary.
        """
