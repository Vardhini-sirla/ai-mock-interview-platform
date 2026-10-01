"""Provider factories — the only place concrete vendor classes are named.

Call sites ask for a capability, not a vendor::

    from src.providers import get_stt

    stt = get_stt()                 # whatever STT_PROVIDER says
    stt = get_stt("whisper")        # or an explicit override

Two properties of this module are load-bearing:

* **Lazy imports.** A provider's module is imported only when that provider is
  requested, so a contributor who has installed ``deepgram-sdk`` but not
  ``assemblyai`` can still import this package, run the test suite, and
  benchmark Deepgram. Importing every SDK eagerly would make the whole project
  un-runnable without every API key and every dependency.
* **A registry, not an if-chain.** ``STT_PROVIDERS`` and friends are public
  constants, so ``--help`` text, error messages, and tests can all enumerate
  what is actually supported instead of hardcoding a list that drifts.

No logic beyond resolution lives here. Provider behaviour belongs in
``src/providers/stt/``; the contracts belong in :mod:`src.providers.base`.
"""

from __future__ import annotations

import os

from src.providers.base import (
    LLMProvider,
    LLMResponse,
    STTProvider,
    SynthesisResult,
    TranscriptResult,
    TTSProvider,
    VADProvider,
    WordTiming,
)

__all__ = [
    "get_stt",
    "get_tts",
    "get_llm",
    "get_vad",
    "STT_PROVIDERS",
    "TTS_PROVIDERS",
    "LLM_PROVIDERS",
    "VAD_PROVIDERS",
    "DEFAULT_STT_PROVIDER",
    "DEFAULT_TTS_PROVIDER",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_VAD_PROVIDER",
    "STTProvider",
    "TTSProvider",
    "LLMProvider",
    "VADProvider",
    "TranscriptResult",
    "SynthesisResult",
    "LLMResponse",
    "WordTiming",
]

#: Supported speech-to-text provider names. Sprint 1 deliverable.
STT_PROVIDERS: tuple[str, ...] = ("deepgram", "whisper", "assemblyai")

#: Supported text-to-speech provider names. Populated in Sprint 2.
TTS_PROVIDERS: tuple[str, ...] = ()

#: Supported LLM provider names. Populated in Sprint 3.
LLM_PROVIDERS: tuple[str, ...] = ()

#: Supported voice-activity-detection provider names. Populated in Sprint 5.
VAD_PROVIDERS: tuple[str, ...] = ()

#: Default used when ``STT_PROVIDER`` is unset. Deepgram Nova-3 is the Sprint 1
#: baseline that the other two providers are measured against.
DEFAULT_STT_PROVIDER = "deepgram"
DEFAULT_TTS_PROVIDER = "cartesia"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_VAD_PROVIDER = "silero"


def _resolve(
    requested: str | None,
    *,
    env_var: str,
    default: str,
    supported: tuple[str, ...],
    kind: str,
) -> str:
    """Normalise a provider name and verify it is supported.

    Args:
        requested: Explicit name from the caller, or ``None`` to consult the
            environment.
        env_var: Environment variable to read when ``requested`` is ``None``.
        default: Fallback when neither the caller nor the environment names one.
        supported: Provider names this factory can build.
        kind: Human-readable capability name, used in the error message.

    Returns:
        The lowercase, whitespace-stripped provider name.

    Raises:
        ValueError: If the resolved name is not in ``supported``. The message
            lists every supported name so the caller does not have to go
            reading source to find out what is available.
    """
    name = (requested or os.getenv(env_var) or default).strip().lower()
    if name not in supported:
        options = ", ".join(supported) if supported else "none yet"
        raise ValueError(
            f"Unknown {kind} provider {name!r}. Supported: {options}. "
            f"Set {env_var} or pass an explicit name."
        )
    return name


def get_stt(provider: str | None = None) -> STTProvider:
    """Build a speech-to-text provider.

    Args:
        provider: Provider name, case-insensitive. Falls back to the
            ``STT_PROVIDER`` environment variable, then to
            :data:`DEFAULT_STT_PROVIDER`.

    Returns:
        A ready-to-use :class:`~src.providers.base.STTProvider`.

    Raises:
        ValueError: If the name is not one of :data:`STT_PROVIDERS`.
        RuntimeError: If the selected provider's API key is missing. The
            provider raises this from its own constructor, so configuration
            errors surface here rather than mid-benchmark.

    Example:
        >>> stt = get_stt("deepgram")  # doctest: +SKIP
        >>> stt.name                   # doctest: +SKIP
        'deepgram'
    """
    name = _resolve(
        provider,
        env_var="STT_PROVIDER",
        default=DEFAULT_STT_PROVIDER,
        supported=STT_PROVIDERS,
        kind="STT",
    )

    if name == "deepgram":
        from src.providers.stt.deepgram_stt import DeepgramSTT

        return DeepgramSTT()
    if name == "whisper":
        from src.providers.stt.whisper_stt import WhisperSTT

        return WhisperSTT()
    if name == "assemblyai":
        from src.providers.stt.assemblyai_stt import AssemblyAISTT

        return AssemblyAISTT()

    # Unreachable: _resolve already rejected anything not in STT_PROVIDERS.
    raise ValueError(f"STT provider {name!r} is listed but not wired up.")


def get_tts(provider: str | None = None) -> TTSProvider:
    """Build a text-to-speech provider.

    **Sprint 2.** :data:`TTS_PROVIDERS` is empty, so every call currently
    raises. The factory exists now so Sprint 2 adds a branch rather than a new
    module and a new import convention.

    Args:
        provider: Provider name, case-insensitive. Falls back to the
            ``TTS_PROVIDER`` environment variable, then to
            :data:`DEFAULT_TTS_PROVIDER`.

    Returns:
        A ready-to-use :class:`~src.providers.base.TTSProvider`.

    Raises:
        ValueError: Always, until Sprint 2 registers an implementation.
    """
    name = _resolve(
        provider,
        env_var="TTS_PROVIDER",
        default=DEFAULT_TTS_PROVIDER,
        supported=TTS_PROVIDERS,
        kind="TTS",
    )
    raise ValueError(f"TTS provider {name!r} is listed but not wired up.")


def get_llm(provider: str | None = None) -> LLMProvider:
    """Build an LLM provider.

    **Sprint 3.** :data:`LLM_PROVIDERS` is empty, so every call currently
    raises.

    Args:
        provider: Provider name, case-insensitive. Falls back to the
            ``LLM_PROVIDER`` environment variable, then to
            :data:`DEFAULT_LLM_PROVIDER`.

    Returns:
        A ready-to-use :class:`~src.providers.base.LLMProvider`.

    Raises:
        ValueError: Always, until Sprint 3 registers an implementation.
    """
    name = _resolve(
        provider,
        env_var="LLM_PROVIDER",
        default=DEFAULT_LLM_PROVIDER,
        supported=LLM_PROVIDERS,
        kind="LLM",
    )
    raise ValueError(f"LLM provider {name!r} is listed but not wired up.")


def get_vad(provider: str | None = None) -> VADProvider:
    """Build a voice-activity-detection provider.

    **Sprint 5.** :data:`VAD_PROVIDERS` is empty, so every call currently
    raises.

    Args:
        provider: Provider name, case-insensitive. Falls back to the
            ``VAD_PROVIDER`` environment variable, then to
            :data:`DEFAULT_VAD_PROVIDER`.

    Returns:
        A ready-to-use :class:`~src.providers.base.VADProvider`.

    Raises:
        ValueError: Always, until Sprint 5 registers an implementation.
    """
    name = _resolve(
        provider,
        env_var="VAD_PROVIDER",
        default=DEFAULT_VAD_PROVIDER,
        supported=VAD_PROVIDERS,
        kind="VAD",
    )
    raise ValueError(f"VAD provider {name!r} is listed but not wired up.")
