"""Contract tests for the provider factories.

These tests guard the properties the rest of the system relies on, and nothing
more. They deliberately make no network calls: constructing a provider must be
free, and a test suite that needs live API credits is a test suite nobody runs.

Tests that need a real key skip rather than fail, so the suite stays green for a
contributor who has set up only one provider.
"""

from __future__ import annotations

import os

import pytest

from src.providers import (
    LLM_PROVIDERS,
    STT_PROVIDERS,
    TTS_PROVIDERS,
    VAD_PROVIDERS,
    STTProvider,
    get_llm,
    get_stt,
    get_tts,
    get_vad,
)


def test_get_stt_deepgram_returns_deepgram_instance() -> None:
    """An explicit "deepgram" request builds a DeepgramSTT with its own name/price."""
    if not os.getenv("DEEPGRAM_API_KEY"):
        pytest.skip("DEEPGRAM_API_KEY not set - cannot construct DeepgramSTT")

    from src.providers.stt.deepgram_stt import DeepgramSTT

    provider = get_stt("deepgram")

    assert isinstance(provider, DeepgramSTT)
    assert isinstance(provider, STTProvider)
    # name is the factory key, the CSV filename fragment, and the value stamped
    # onto every TranscriptResult - renaming it invalidates committed results.
    assert provider.name == "deepgram"
    assert provider.cost_per_hour == pytest.approx(0.258)


def test_get_stt_unknown_name_raises_value_error() -> None:
    """An unsupported name fails fast and names the supported alternatives."""
    with pytest.raises(ValueError) as excinfo:
        get_stt("not-a-real-provider")

    message = str(excinfo.value)
    assert "not-a-real-provider" in message
    # The error must enumerate the options; a bare "unknown provider" sends the
    # reader to the source.
    for supported in STT_PROVIDERS:
        assert supported in message


def test_get_stt_is_case_insensitive() -> None:
    """Provider names resolve regardless of case or surrounding whitespace.

    Names arrive from ``.env`` files and shell history, where casing is not
    reliable. ``STT_PROVIDER=Deepgram`` must not be a configuration error.
    """
    if not os.getenv("DEEPGRAM_API_KEY"):
        pytest.skip("DEEPGRAM_API_KEY not set - cannot construct DeepgramSTT")

    assert get_stt("DEEPGRAM").name == "deepgram"
    assert get_stt("DeepGram").name == "deepgram"
    assert get_stt("  deepgram  ").name == "deepgram"


def test_get_stt_uppercase_unknown_name_still_raises() -> None:
    """Case-insensitivity does not weaken validation.

    Runs without an API key, so it covers the uppercase path even on a machine
    with no credentials configured.
    """
    with pytest.raises(ValueError):
        get_stt("DEEPGRAMM")


def test_get_stt_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit argument, the factory honours $STT_PROVIDER.

    This is the mechanism the whole swap pattern rests on: if the env var were
    ignored, changing providers would require a code change.
    """
    monkeypatch.setenv("STT_PROVIDER", "definitely-not-a-provider")

    with pytest.raises(ValueError, match="definitely-not-a-provider"):
        get_stt()


def test_stt_provider_abc_cannot_be_instantiated() -> None:
    """STTProvider is abstract; only concrete subclasses may be constructed."""
    with pytest.raises(TypeError):
        STTProvider()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    """A subclass that skips a required member is still abstract.

    Guards against a future provider shipping without ``cost_per_hour``, which
    would leave the benchmark unable to price its results.
    """

    class HalfBuiltSTT(STTProvider):
        """Declares a name and nothing else."""

        @property
        def name(self) -> str:
            return "half-built"

    with pytest.raises(TypeError):
        HalfBuiltSTT()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("factory", "registry"),
    [
        (get_tts, TTS_PROVIDERS),
        (get_llm, LLM_PROVIDERS),
        (get_vad, VAD_PROVIDERS),
    ],
)
def test_unimplemented_factories_raise_value_error(factory, registry) -> None:
    """TTS, LLM, and VAD factories exist but have no registered providers yet.

    They must raise rather than return ``None``, so a call site written against
    a Sprint 2/3/5 capability fails loudly in Sprint 1.
    """
    assert registry == ()
    with pytest.raises(ValueError):
        factory()


def test_stt_registry_contents() -> None:
    """The Sprint 1 registry lists exactly the three benchmarked providers."""
    assert STT_PROVIDERS == ("deepgram", "whisper", "assemblyai")


@pytest.mark.parametrize("name", ["deepgram", "whisper", "assemblyai"])
def test_every_registered_provider_is_importable(name: str) -> None:
    """Each registered name maps to a real class implementing the contract.

    Imports the module without constructing the provider, so it passes with no
    API keys set. This catches a typo in a lazy-import branch, which would
    otherwise only surface mid-benchmark.
    """
    expected = {
        "deepgram": ("src.providers.stt.deepgram_stt", "DeepgramSTT"),
        "whisper": ("src.providers.stt.whisper_stt", "WhisperSTT"),
        "assemblyai": ("src.providers.stt.assemblyai_stt", "AssemblyAISTT"),
    }
    module_path, class_name = expected[name]
    module = __import__(module_path, fromlist=[class_name])
    provider_cls = getattr(module, class_name)

    assert issubclass(provider_cls, STTProvider)


@pytest.mark.parametrize("name", ["deepgram", "whisper", "assemblyai"])
def test_missing_api_key_raises_runtime_error(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider with no credentials fails at construction, not mid-run.

    Credentials are validated in ``__init__`` so a misconfigured environment
    costs nothing; discovering it on file 40 of a benchmark would have already
    spent the API budget.
    """
    for env_var in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "ASSEMBLYAI_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(RuntimeError, match="API_KEY"):
        get_stt(name)
