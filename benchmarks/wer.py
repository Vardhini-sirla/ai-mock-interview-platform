"""Word Error Rate benchmark harness for the STT provider layer.

Run it as a module so relative imports resolve from the repository root::

    python -m benchmarks.wer --provider deepgram
    python -m benchmarks.wer --provider whisper --corpus benchmarks/corpus
    python -m benchmarks.wer --provider assemblyai --limit 5

Every run writes a timestamped CSV to ``benchmarks/results/`` and prints an
aggregate summary. The CSV is the artifact — committing it is what makes a
provider choice auditable three months later, when the only question anyone
remembers is "why did we pick this one?".

Methodology, and the reasoning behind it, is in ``benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["BenchmarkRow", "build_parser", "run_benchmark", "main"]

#: Repository root, derived from this file's location so the defaults work no
#: matter which directory the command is run from.
_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS = _REPO_ROOT / "benchmarks" / "corpus"
DEFAULT_TRUTH = _REPO_ROOT / "benchmarks" / "ground_truth"
DEFAULT_RESULTS = _REPO_ROOT / "benchmarks" / "results"

#: CSV schema. Fixed and append-only: adding a column is fine, renaming or
#: reordering one silently breaks every committed result file.
CSV_COLUMNS = (
    "file",
    "audio_seconds",
    "provider_latency_ms",
    "wer",
    "ground_truth_words",
    "hypothesis_words",
    "estimated_cost_usd",
)


@dataclass(slots=True)
class BenchmarkRow:
    """One scored corpus file — exactly one row of the output CSV.

    Attributes:
        file: Corpus filename, without directory.
        audio_seconds: Duration of the audio.
        provider_latency_ms: Wall-clock latency the provider reported.
        wer: Word error rate in ``0.0..n``. Can exceed ``1.0`` when the
            hypothesis inserts more words than the reference contains, which is
            a real and informative failure mode, not a bug to clamp away.
        ground_truth_words: Reference word count after normalisation.
        hypothesis_words: Hypothesis word count after normalisation.
        estimated_cost_usd: ``audio_seconds`` priced at the provider's own
            ``cost_per_hour``.
        error: Populated instead of a score when the file could not be
            transcribed. Failures are reported as rows rather than dropped, so a
            provider cannot post a good mean WER by quietly skipping hard files.
    """

    file: str
    audio_seconds: float
    provider_latency_ms: float
    wer: float
    ground_truth_words: int
    hypothesis_words: int
    estimated_cost_usd: float
    error: str = ""

    def as_csv_dict(self) -> dict[str, object]:
        """Return this row keyed by :data:`CSV_COLUMNS`, rounded for readability."""
        return {
            "file": self.file,
            "audio_seconds": round(self.audio_seconds, 3),
            "provider_latency_ms": round(self.provider_latency_ms, 1),
            "wer": round(self.wer, 4),
            "ground_truth_words": self.ground_truth_words,
            "hypothesis_words": self.hypothesis_words,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


def _build_transform():
    """Return the jiwer normalisation applied to both reference and hypothesis.

    WER is meaningless without a stated normalisation. One provider punctuates
    and capitalises by default while another does not; scoring raw strings would
    measure formatting preferences, not recognition accuracy. The same
    transformation is applied to both sides of every comparison:

    * ``ToLowerCase`` — "Jobnova" vs "jobnova" is not a recognition error.
    * ``RemovePunctuation`` — smart-formatting differences are not errors.
    * ``RemoveMultipleSpaces`` and ``Strip`` — tokenisation hygiene.
    * ``ReduceToListOfListOfWords`` — required terminator: ``jiwer`` scores
      token lists, and raises if a transform chain hands it a bare string.

    Deliberately *not* normalised: numbers and contractions. "twenty twenty
    four" vs "2024" and "do not" vs "don't" stay scored as errors, because in a
    transcript the interviewer LLM reads, they genuinely are differences.

    Returns:
        A ``jiwer.Compose`` pipeline.

    Raises:
        RuntimeError: If jiwer is not installed.
    """
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "jiwer is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


def _discover_pairs(corpus: Path, truth: Path) -> list[tuple[Path, Path]]:
    """Pair each corpus WAV with its ground-truth transcript.

    Pairing is by stem: ``corpus/intro_01.wav`` is scored against
    ``ground_truth/intro_01.txt``. A WAV with no matching transcript is skipped
    with a warning rather than scored against an empty reference, which would
    report a perfect-looking 0.0 WER for an unverifiable file.

    Args:
        corpus: Directory of ``.wav`` files.
        truth: Directory of ``.txt`` transcripts.

    Returns:
        ``(wav_path, txt_path)`` pairs, sorted by filename so run order — and
        therefore the CSV — is deterministic.

    Raises:
        FileNotFoundError: If either directory does not exist.
    """
    if not corpus.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus}")
    if not truth.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {truth}")

    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(corpus.glob("*.wav")):
        transcript = truth / f"{wav.stem}.txt"
        if not transcript.is_file():
            print(
                f"  [skip] {wav.name}: no ground truth at {transcript.name}",
                file=sys.stderr,
            )
            continue
        pairs.append((wav, transcript))
    return pairs


async def run_benchmark(
    provider_name: str,
    corpus: Path,
    truth: Path,
    results_dir: Path,
    limit: int | None = None,
    write_csv: bool = True,
) -> tuple[list[BenchmarkRow], Path | None]:
    """Score one provider against the corpus and write a results CSV.

    Args:
        provider_name: Provider to benchmark, as named in
            :data:`src.providers.STT_PROVIDERS`.
        corpus: Directory of ``.wav`` files.
        truth: Directory of matching ``.txt`` transcripts.
        results_dir: Where the CSV is written. Created if absent.
        limit: Score at most this many files. Useful for a smoke run that does
            not spend the whole corpus's API budget.
        write_csv: Set ``False`` to score without writing a file.

    Returns:
        The scored rows and the CSV path (``None`` when ``write_csv`` is
        ``False`` or no rows were produced).

    Raises:
        FileNotFoundError: If the corpus or ground-truth directory is missing.
        ValueError: If ``provider_name`` is not supported.
        RuntimeError: If the provider's API key or SDK is missing.
    """
    from src.providers import get_stt

    transform = _build_transform()
    import jiwer  # Safe: _build_transform already verified the import.

    pairs = _discover_pairs(corpus, truth)
    if limit is not None:
        pairs = pairs[:limit]

    if not pairs:
        print(
            f"No scorable files found. Put .wav files in {corpus} and matching "
            f".txt transcripts in {truth}.",
            file=sys.stderr,
        )
        return [], None

    # Constructed once, outside the loop: a missing API key should fail before
    # the first file rather than on every one.
    provider = get_stt(provider_name)
    print(
        f"Benchmarking {provider.name} on {len(pairs)} file(s) "
        f"@ ${provider.cost_per_hour:.3f}/hr"
    )

    rows: list[BenchmarkRow] = []
    for index, (wav, transcript_path) in enumerate(pairs, start=1):
        reference = transcript_path.read_text(encoding="utf-8").strip()
        audio_bytes = wav.read_bytes()

        try:
            result = await provider.transcribe(audio_bytes)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            print(f"  [{index}/{len(pairs)}] {wav.name}: FAILED - {exc}", file=sys.stderr)
            rows.append(
                BenchmarkRow(
                    file=wav.name,
                    audio_seconds=0.0,
                    provider_latency_ms=0.0,
                    wer=float("nan"),
                    ground_truth_words=len(reference.split()),
                    hypothesis_words=0,
                    estimated_cost_usd=0.0,
                    error=str(exc),
                )
            )
            continue

        measures = jiwer.process_words(
            reference,
            result.text,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        normalised_reference = transform(reference)
        normalised_hypothesis = transform(result.text)
        cost = result.estimated_cost_usd(provider.cost_per_hour)

        rows.append(
            BenchmarkRow(
                file=wav.name,
                audio_seconds=result.audio_seconds,
                provider_latency_ms=result.latency_ms,
                wer=measures.wer,
                ground_truth_words=_count_words(normalised_reference),
                hypothesis_words=_count_words(normalised_hypothesis),
                estimated_cost_usd=cost,
            )
        )
        print(
            f"  [{index}/{len(pairs)}] {wav.name}: WER {measures.wer:.3f}  "
            f"{result.latency_ms:.0f}ms  {result.audio_seconds:.1f}s  ${cost:.5f}"
        )

    csv_path = None
    if write_csv:
        csv_path = _write_csv(rows, provider.name, results_dir)
    _print_summary(rows, provider.name, csv_path)
    return rows, csv_path


def _count_words(normalised: object) -> int:
    """Count words in a jiwer-normalised value.

    jiwer transformations return either a string or a list of token lists
    depending on which transforms are composed, so both shapes are handled.

    Args:
        normalised: The output of a ``jiwer.Compose`` pipeline.

    Returns:
        The number of words.
    """
    if isinstance(normalised, str):
        return len(normalised.split())
    if isinstance(normalised, list):
        total = 0
        for item in normalised:
            total += len(item.split()) if isinstance(item, str) else len(item)
        return total
    return 0


def _write_csv(rows: list[BenchmarkRow], provider_name: str, results_dir: Path) -> Path:
    """Write scored rows to a timestamped CSV.

    The filename carries the provider and a UTC timestamp, so repeated runs
    accumulate rather than overwrite. A benchmark you can only run once is not a
    benchmark — being able to re-run after a model update and diff the two files
    is the point.

    Args:
        rows: Scored rows, in corpus order.
        provider_name: Used in the filename.
        results_dir: Destination directory, created if absent.

    Returns:
        Path to the written CSV.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"wer_{provider_name}_{stamp}.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_dict())
    return path


def _print_summary(
    rows: list[BenchmarkRow], provider_name: str, csv_path: Path | None
) -> None:
    """Print the aggregate summary for a completed run.

    Reports mean *and* median WER because they answer different questions: the
    mean is dragged by a single catastrophic file, the median says what a typical
    utterance looks like. A large gap between them is itself the finding. Latency
    is reported at p90 rather than mean for the same reason — a candidate
    remembers the worst pause, not the average one.

    Args:
        rows: Scored rows.
        provider_name: Provider label for the header.
        csv_path: Where results were written, if they were.
    """
    scored = [row for row in rows if not row.error]
    failed = len(rows) - len(scored)

    print()
    print(f"=== WER summary: {provider_name} ===")
    print(f"  files scored      : {len(scored)}" + (f"  ({failed} failed)" if failed else ""))

    if not scored:
        print("  no successful transcriptions - nothing to aggregate")
        if csv_path:
            print(f"  results           : {csv_path}")
        return

    wers = [row.wer for row in scored]
    latencies = [row.provider_latency_ms for row in scored]
    audio_seconds = sum(row.audio_seconds for row in scored)
    total_cost = sum(row.estimated_cost_usd for row in scored)

    print(f"  audio total       : {audio_seconds:.1f}s")
    print(f"  mean WER          : {statistics.fmean(wers):.4f}")
    print(f"  median WER        : {statistics.median(wers):.4f}")
    print(f"  p90 latency       : {_percentile(latencies, 0.90):.0f}ms")
    print(f"  median latency    : {statistics.median(latencies):.0f}ms")
    print(f"  total cost        : ${total_cost:.5f}")
    if csv_path:
        print(f"  results           : {csv_path}")


def _percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile of ``values``.

    Nearest-rank rather than interpolated: at the corpus sizes this harness runs
    on (tens of files), an interpolated p90 reports a latency that no request
    actually took.

    Args:
        values: Sample values; need not be sorted.
        fraction: Percentile as a fraction, e.g. ``0.90``.

    Returns:
        The percentile value, or ``0.0`` for an empty sample.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Split out from :func:`main` so tests can assert on ``--help`` without
    running a benchmark.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.wer",
        description=(
            "Measure Word Error Rate, latency, and cost for one STT provider "
            "against the shared benchmark corpus. Writes a timestamped CSV to "
            "benchmarks/results/ and prints an aggregate summary."
        ),
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.wer --provider deepgram\n"
            "  python -m benchmarks.wer --provider whisper --limit 5\n"
            "  python -m benchmarks.wer --provider assemblyai --no-csv\n"
            "\n"
            "Methodology: benchmarks/README.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "STT provider to benchmark (deepgram, whisper, assemblyai). "
            "Defaults to $STT_PROVIDER, then to deepgram."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Directory of .wav files to transcribe (default: %(default)s)",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=DEFAULT_TRUTH,
        help=(
            "Directory of ground-truth .txt transcripts, matched to corpus "
            "files by filename stem (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory for the output CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Score at most N files - useful for a cheap smoke run",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Print the summary without writing a results CSV",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark from the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a configuration or
        credential error, ``2`` if no file could be transcribed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Loaded here rather than at import time so that `--help` works without a
    # .env file present.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - optional convenience
        pass

    started = time.perf_counter()
    try:
        rows, _ = asyncio.run(
            run_benchmark(
                provider_name=args.provider,
                corpus=args.corpus,
                truth=args.truth,
                results_dir=args.results,
                limit=args.limit,
                write_csv=not args.no_csv,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nwall clock: {time.perf_counter() - started:.1f}s")
    if rows and all(row.error for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
