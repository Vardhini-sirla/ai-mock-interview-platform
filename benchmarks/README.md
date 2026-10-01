# Benchmarks

Provider choices in this project are measured, not asserted. Every claim in
`docs/` about which provider is faster, cheaper, or more accurate traces back to
a CSV in `results/` that anyone can regenerate.

Sprint 1 covers speech-to-text only. TTS (Sprint 2) and turn detection
(Sprint 5) get their own harnesses alongside this one, following the same three
principles.

## Three principles

**1. Same corpus, every provider.** All providers are scored against the
identical set of WAV files in `corpus/` with the identical ground truth in
`ground_truth/`. Files are paired by filename stem — `corpus/intro_01.wav` is
scored against `ground_truth/intro_01.txt`. A WAV with no matching transcript is
skipped with a warning rather than scored against an empty reference, because an
unverifiable file would otherwise post a flattering 0.0 WER.

The corpus should reflect what a mock interview actually sounds like: filler
words, mid-sentence restarts, technical proper nouns, varied accents, and at
least a few recordings made on a laptop microphone in a room with background
noise. A corpus of clean studio audio will rank all three providers as excellent
and tell you nothing.

**2. Fixed configs, pinned models.** Each provider pins a concrete model
(`nova-3`, `whisper-1`, AssemblyAI `best`) rather than a "latest" alias, so a
result from six months ago still means something. Deepgram and AssemblyAI are
both given the *same* proper-noun boost list — `DEFAULT_KEYTERMS` in
`src/providers/stt/deepgram_stt.py` — so neither is handed a vocabulary
advantage. Whisper's `prompt` parameter is left unset: it is advisory biasing
rather than a hard boost, so enabling it would make the comparison less clean,
not more.

Scoring normalisation is fixed too, and applied to both reference and
hypothesis: lowercase, strip punctuation, collapse whitespace. Numbers and
contractions are deliberately **not** normalised — "2024" vs "twenty twenty
four" and "don't" vs "do not" stay scored as errors, because the interviewer LLM
downstream reads the transcript as written.

**3. Cost at source.** Each provider declares its own `cost_per_hour` next to
the integration it describes, with the pricing page and check date in a comment.
The harness never holds its own price table, so a rate can't drift away from the
code that incurs it. Changing a model and forgetting its price is then a
one-file review problem, not a cross-file one.

## Running a benchmark

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your API keys
```

Drop WAV files in `benchmarks/corpus/` and matching transcripts in
`benchmarks/ground_truth/`, then:

```bash
# WER, latency, and cost for one provider
python -m benchmarks.wer --provider deepgram
python -m benchmarks.wer --provider whisper
python -m benchmarks.wer --provider assemblyai

# cheap smoke run before spending the whole corpus's API budget
python -m benchmarks.wer --provider deepgram --limit 3

# score without writing a CSV
python -m benchmarks.wer --provider whisper --no-csv

# full option list
python -m benchmarks.wer --help
```

Omitting `--provider` falls back to `$STT_PROVIDER`, then to `deepgram`. Only
the provider whose key you have set needs to be installed — SDK imports are lazy
for exactly this reason.

## How results are tracked

Each run writes `results/wer_<provider>_<UTC-timestamp>.csv`. Runs accumulate
rather than overwrite, so re-running after a model update and diffing the two
files is the normal workflow.

| Column | Meaning |
| --- | --- |
| `file` | Corpus filename |
| `audio_seconds` | Duration of the audio |
| `provider_latency_ms` | Wall-clock latency for the provider call, including network |
| `wer` | Word error rate after normalisation; may exceed 1.0 on heavy insertion |
| `ground_truth_words` | Reference word count, post-normalisation |
| `hypothesis_words` | Hypothesis word count, post-normalisation |
| `estimated_cost_usd` | `audio_seconds` priced at the provider's own rate |

The run also prints an aggregate: mean WER, median WER, p90 latency, median
latency, and total cost.

Mean and median WER are both reported because they answer different questions.
The mean is dragged by one catastrophic file; the median says what a typical
utterance looks like. A wide gap between them is itself a finding — usually one
corpus file with a proper noun nobody boosted. Latency is reported at p90 rather
than mean because a candidate remembers the worst pause in an interview, not the
average one.

Files that fail to transcribe are written as rows with an empty score rather
than dropped, so a provider can't earn a good mean WER by quietly skipping the
hard files.

Commit the CSV for any run that informs a decision. The sprint write-up should
cite the filename.

## Interpreting the result

A provider wins on the combination, not on WER alone. Deepgram Nova-3 is the
Sprint 1 baseline because it is the cheapest of the three and the only one with
both keyterm boost and a streaming path — Whisper's API has no real-time socket
at all, which rules it out of the live agent regardless of how it scores here.
If the numbers contradict that reasoning, `STT_PROVIDER` changes and nothing
else does. That is what the abstraction layer is for.

## See also

- `docs/architecture.md` — where the provider layer sits in the voice pipeline
- `src/providers/base.py` — the contracts every provider implements
- Agile Scrum Plan in `docs/` — sprint scope and acceptance criteria
