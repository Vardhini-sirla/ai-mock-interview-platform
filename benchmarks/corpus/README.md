# Benchmark corpus

This directory holds the audio every STT provider is scored against. Reference
transcripts live one level up in `../ground_truth/`.

**The audio itself is not committed.** `.gitignore` excludes `corpus/*.wav` and
`corpus/*.mp3` — the files are large and contain real voices. Ground-truth
transcripts *are* committed, because they are small, reviewable, and the thing
that actually needs version control. Keep the audio in shared storage and
reference the recording by filename; anyone re-running a benchmark pulls the
audio separately.

## What we need

5–10 recordings, 2–15 minutes each, from different speakers.

| Property | Target |
| --- | --- |
| Count | 5–10 recordings |
| Length | 2–15 minutes each |
| Speakers | A different speaker per recording; no speaker twice |
| Format | WAV, 16-bit PCM, mono, 16 kHz or higher |
| Content | Real mock-interview answers — not read passages |

Format matters more than it looks: `_duration_from_wav()` in
`src/providers/stt/deepgram_stt.py` reads duration from the WAV header with
Python's `wave` module, and it returns `0.0` for anything it cannot parse. A
file that is secretly MP3-in-a-`.wav`-wrapper will still transcribe but will
report zero duration, which silently zeroes its cost column. Convert to real
PCM WAV before adding a file.

Read passages are excluded deliberately. A speaker reading prepared text has
none of the restarts, mid-sentence corrections, or trailing-off that a nervous
candidate produces, and those are precisely what separates a provider that works
in an interview from one that works in a demo.

### A note on recording length

The 2–15 minute range is wide on purpose, but be aware of what it does to the
statistics. The harness reports one WER number per file, so ten recordings give
a sample of ten. With files at the long end, the mean-vs-median comparison in
`../README.md` loses most of its diagnostic power — one bad patch inside a
15-minute file gets averaged away inside that file's own score rather than
showing up as an outlier.

Prefer several shorter recordings over one long one when you have the choice. If
a long recording is what you have, consider also splitting it into 2–3 minute
segments and adding those as separate corpus entries, each with its own ground
truth.

## File naming

```
YYYYMMDD_speakerID_topic.wav
```

| Part | Rule | Example |
| --- | --- | --- |
| `YYYYMMDD` | Recording date, zero-padded | `20261015` |
| `speakerID` | Stable pseudonym, no real names | `spk03` |
| `topic` | Lowercase, underscore-separated | `system_design` |

Examples:

```
20261015_spk01_behavioral_intro.wav
20261015_spk02_system_design.wav
20261018_spk03_resume_walkthrough.wav
20261018_spk03_resume_walkthrough.txt    <- in ../ground_truth/
```

Two rules are load-bearing:

**The stem is the join key.** `_discover_pairs()` in `../wer.py` pairs
`corpus/<stem>.wav` with `ground_truth/<stem>.txt` by exact filename stem. A
mismatch of even one character means the WAV is skipped with a warning rather
than scored — it will not fail loudly, so check the run output for `[skip]`
lines after adding files.

**`speakerID` must be a pseudonym and must be stable.** Never a real name — the
filename is committed to a public repository even though the audio is not.
Stability matters because reusing `spk03` for a second recording is how we can
later check whether a provider struggles with one particular voice rather than
with one particular recording. Keep the pseudonym-to-person mapping out of the
repo entirely.

## Ground truth format

For every `corpus/<stem>.wav`, add `../ground_truth/<stem>.txt`:

- UTF-8, plain text, no BOM.
- The verbatim transcript as one continuous block. Line breaks are fine; the
  scorer collapses whitespace.
- **No** timestamps, speaker labels, or annotation markup. The harness compares
  word sequences, so `[00:14] Interviewer:` is four extra tokens that score as
  insertion errors against every provider.

Four transcription rules follow from how the harness normalises text before
scoring. The scorer lowercases, strips punctuation, and collapses whitespace —
so casing and punctuation in ground truth do not matter. Everything it does
*not* normalise does matter:

**Write numbers as they are spoken.** "twenty twenty four", not "2024". "three
years", not "3 years". Digits and words are not normalised to each other, so
`2024` against a provider's `twenty twenty four` scores as three errors for a
difference nobody actually misheard.

**Do not expand contractions.** Write "don't" if the speaker said "don't".
"do not" against "don't" scores as an insertion plus a substitution.

**Exclude filler words.** Leave out "um", "uh", "er", "mm". This is the one rule
that looks like it contradicts "verbatim", and it is deliberate: neither pinned
config requests fillers (Deepgram's `filler_words` and AssemblyAI's
`disfluencies` are both left unset, so the API default of stripping them
applies), but Whisper has no such switch and emits some fillers anyway. Ground
truth that includes fillers would therefore charge Deepgram and AssemblyAI a
deletion for every "um" while charging Whisper nothing — handing Whisper a WER
advantage that has nothing to do with recognition quality. Keep them out and the
three providers stay comparable.

*Do* keep real disfluencies that are made of real words: false starts
("I worked — I led the team"), repetitions ("the the deployment"), and
self-corrections. Those are recognition problems, not transcription noise, and
they are a large part of what the benchmark is for. Write them without the
em-dash; punctuation is stripped anyway.

**Spell proper nouns correctly, then boost them.** Product names, company names,
and framework names must be spelled as intended: `LiveKit`, not `live kit`;
`Jobnova`, not `job nova`. Then add any new proper noun to `DEFAULT_KEYTERMS` in
`src/providers/stt/deepgram_stt.py`, which is shared by both the Deepgram
keyterm boost and the AssemblyAI `word_boost`. A proper noun in ground truth but
absent from the boost list costs two errors every time it occurs — one deletion
and one insertion — and that single omission can dominate a file's score.

## Inclusion criteria

The corpus exists to find out where providers break, so it has to contain
conditions that break them. A corpus of clean audio from fluent speakers in
quiet rooms will rank all three providers as excellent and tell you nothing.

Aim for spread across all four axes rather than a balanced grid — with 5–10
recordings you cannot cover every combination, so prioritise getting at least
one clearly hard recording on each axis.

**Accents.** At least three distinguishable accents or first-language
backgrounds, including non-native English speakers. This is the axis where
provider differences are usually largest and where a US-centric corpus will
most badly mislead you.

**Speech rates.** Include one noticeably fast speaker and one slow, halting
speaker with long pauses. Slow speech with long gaps is the harder case: it
interacts with turn detection, and the recordings double as fixtures for the
Sprint 5 VAD work.

**Background noise.** Span the range deliberately:

| Level | Condition |
| --- | --- |
| Clean | Headset mic, quiet room |
| Typical | Laptop built-in mic, room reverb |
| Noisy | Café, fan, keyboard, traffic, or an audible second voice |

At least one recording should be genuinely difficult. Most real interviews are
conducted on a laptop microphone, so "typical" should be the most common
condition in the corpus, not "clean".

**Devices.** Vary the capture hardware — headset, laptop built-in, phone. Device
colouration is a separate variable from room noise, and providers differ in how
well they handle each.

Content-wise, the recordings should collectively cover behavioural answers,
technical explanations, and resume walkthroughs, and should contain the proper
nouns from `DEFAULT_KEYTERMS` as they come up naturally.

## Consent and privacy

These are recordings of real people answering questions about their careers.

- Get explicit consent to record and to use the audio for benchmarking, before
  recording.
- Never commit the audio. It is gitignored; do not override that with `git add
  -f`.
- Use pseudonymous `speakerID`s in filenames and keep the mapping outside the
  repo.
- Ground-truth transcripts *are* committed and *are* public. Review each one for
  personal details — employer names, locations, anything identifying — before
  committing, and redact as needed. A redaction changes the reference text, so
  redact the audio to match or drop the recording.

## Before you commit

1. WAV is real 16-bit PCM — `python -c "import wave; print(wave.open('FILE').getparams())"`.
2. Filename matches `YYYYMMDD_speakerID_topic.wav` with a pseudonymous ID.
3. `../ground_truth/<same stem>.txt` exists, UTF-8.
4. Numbers spelled out, contractions intact, fillers removed, proper nouns spelled correctly.
5. Any new proper nouns added to `DEFAULT_KEYTERMS`.
6. Transcript reviewed for personal details.
7. `python -m benchmarks.wer --provider deepgram --limit 3 --no-csv` runs with no
   `[skip]` lines for your file.

## See also

- `../README.md` — benchmark methodology, scoring normalisation, how results are tracked
- `../wer.py` — the harness; `_discover_pairs()` is the pairing logic
- `docs/architecture.md` — where the provider layer sits in the pipeline
