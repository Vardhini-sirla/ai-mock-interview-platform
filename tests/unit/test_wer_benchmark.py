"""Smoke tests for the WER benchmark harness.

The harness is a CLI, and a CLI that crashes on ``--help`` is broken for anyone
who has not already read its source. These tests make no network calls and need
no API keys: they verify the module imports, the parser builds, and the help path
exits cleanly.

Actual WER numbers are validated against the corpus in the Sprint 1 wrap-up, not
here — asserting on them would require committing audio fixtures and spending API
credits on every test run.
"""

from __future__ import annotations

import argparse

import pytest

from benchmarks import wer


def test_module_exposes_cli_entrypoints() -> None:
    """The module imports and exposes the pieces the CLI and tests rely on."""
    assert callable(wer.main)
    assert callable(wer.build_parser)
    assert callable(wer.run_benchmark)


def test_build_parser_returns_argument_parser() -> None:
    """build_parser() produces a usable parser without touching the filesystem."""
    parser = wer.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_help_exits_zero_and_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--help`` prints usage and exits 0 rather than raising.

    argparse raises SystemExit(0) for --help; anything else - a traceback, a
    nonzero code - means the parser is misconfigured.
    """
    with pytest.raises(SystemExit) as excinfo:
        wer.main(["--help"])

    assert excinfo.value.code == 0

    out = capsys.readouterr().out
    assert "usage" in out.lower()
    # Every documented flag must actually appear in the help text.
    for flag in ("--provider", "--corpus", "--truth", "--results", "--limit"):
        assert flag in out


def test_help_mentions_supported_providers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The help text names the providers, so --help is enough to run a benchmark."""
    with pytest.raises(SystemExit):
        wer.main(["--help"])

    out = capsys.readouterr().out
    for provider in ("deepgram", "whisper", "assemblyai"):
        assert provider in out


def test_parser_defaults_point_at_benchmark_directories() -> None:
    """Default paths resolve to the repo's benchmark directories.

    These are derived from the module's own location so the command works from
    any working directory; a wrong default would send results somewhere
    untracked.
    """
    args = wer.build_parser().parse_args([])

    assert args.provider is None  # falls back to $STT_PROVIDER, then deepgram
    assert args.corpus.name == "corpus"
    assert args.truth.name == "ground_truth"
    assert args.results.name == "results"
    assert args.limit is None
    assert args.no_csv is False


def test_csv_columns_are_the_documented_schema() -> None:
    """The CSV schema matches what benchmarks/README.md documents.

    Committed result files are parsed by column name; renaming or reordering one
    breaks every historical CSV, so the schema is pinned by test.
    """
    assert wer.CSV_COLUMNS == (
        "file",
        "audio_seconds",
        "provider_latency_ms",
        "wer",
        "ground_truth_words",
        "hypothesis_words",
        "estimated_cost_usd",
    )


def test_missing_corpus_directory_exits_one(
    capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    """A nonexistent corpus path is a clean error, not a traceback."""
    code = wer.main(
        [
            "--provider",
            "deepgram",
            "--corpus",
            str(tmp_path / "no-such-corpus"),
            "--truth",
            str(tmp_path / "no-such-truth"),
            "--no-csv",
        ]
    )

    assert code == 1
    assert "error:" in capsys.readouterr().err
