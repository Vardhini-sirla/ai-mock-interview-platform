# Architecture

An AI mock-interview agent is a real-time voice loop. The candidate speaks, the
agent hears, reasons, answers, and an avatar lip-syncs the reply. Every stage is
a separate vendor, and each one adds latency that the candidate feels directly.

## The voice pipeline

```
                      ┌───────────────── one conversational turn ─────────────────┐

  ┌──────────┐    ┌───────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌────────┐    ┌──────────┐
  │ Candidate│    │ Audio │    │ VAD │    │ STT │    │ LLM │    │  TTS   │    │  Avatar  │
  │   mic    │───▶│ in    │───▶│turn │───▶│speech│──▶│inter│───▶│ speech │───▶│ lip-sync │
  │          │    │ WebRTC│    │ end │    │→text │   │viewer│   │ ← text │    │  video   │
  └──────────┘    └───────┘    └─────┘    └─────┘    └─────┘    └────────┘    └────┬─────┘
       ▲                                                                            │
       └────────────────────────── candidate sees + hears ◀────────────────────────┘

  Transport : LiveKit (WebRTC rooms, audio + video tracks)
  VAD       : Silero / semantic turn detection     ── Sprint 5
  STT       : Deepgram Nova-3 | Whisper | AssemblyAI ── Sprint 1  ◀── YOU ARE HERE
  LLM       : interviewer persona + scoring rubric  ── Sprint 3
  TTS       : Cartesia                              ── Sprint 2
  Avatar    : Tavus                                 ── Sprint 4

  Latency budget — the number that governs every provider choice:

      VAD  ~100ms │ STT ~200ms │ LLM ~500ms │ TTS ~150ms │ Avatar ~250ms
      ───────────────────────────────────────────────────────────────────
      target end-to-end: under 1,200ms from end-of-speech to first audio

  Past roughly 1.5s of silence the candidate assumes the agent didn't hear them
  and starts repeating themselves. Every stage is on the critical path, so a
  provider that is 300ms slower spends a quarter of the whole budget.
```

## Sprint 1 focus: the STT abstraction layer

```
                        ┌───────────────────────────────────┐
                        │  src/providers/__init__.py        │
   call sites ─────────▶│  get_stt()  get_tts()             │   reads STT_PROVIDER
   (agent, benchmark)   │  get_llm()  get_vad()             │   lazy-imports one module
                        └─────────────────┬─────────────────┘
                                          │  returns
                                          ▼
                        ┌───────────────────────────────────┐
                        │  src/providers/base.py            │
                        │  STTProvider (ABC)                │
                        │    .name                          │
                        │    .cost_per_hour                 │
                        │    async transcribe()             │
                        │    async transcribe_stream()      │
                        └─────────────────┬─────────────────┘
                                          │  implemented by
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                            ▼
   ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
   │   DeepgramSTT     │       │    WhisperSTT     │       │   AssemblyAISTT   │
   │   nova-3          │       │    whisper-1      │       │   best tier       │
   │   $0.258/hr       │       │    $0.360/hr      │       │   $0.370/hr       │
   │   keyterm boost ✓ │       │    no confidence  │       │   word_boost ✓    │
   │   streaming: S4   │       │    streaming: ✗   │       │   streaming: TBD  │
   └─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │  all return
                                         ▼
                              ┌────────────────────────┐
                              │   TranscriptResult     │
                              │   text, confidence,    │
                              │   latency_ms, provider,│
                              │   audio_seconds,       │
                              │   word_timings         │
                              └───────────┬────────────┘
                                          │  scored by
                                          ▼
                              ┌────────────────────────┐
                              │  benchmarks/wer.py     │
                              │  WER · latency · cost  │
                              │  → results/*.csv       │
                              └────────────────────────┘
```

Sprint 1 ships three STT providers behind one contract, plus the harness that
measures them. The TTS, LLM, and VAD contracts are defined in `base.py` but have
no implementations yet — the shape is fixed now so later sprints add a factory
branch rather than a new convention.

Three design decisions do the work:

**One result type.** Deepgram, Whisper, and AssemblyAI return three different
JSON shapes with different field names, units, and conventions — AssemblyAI
reports word offsets in milliseconds, Deepgram in seconds, Whisper exposes no
confidence at all. Each provider normalises into `TranscriptResult` at its own
boundary. The benchmark compares providers because they are already comparable,
not because it special-cases three vendors.

**Unknown is `-1.0`, never `0.0`.** The OpenAI Whisper API does not return a
confidence score. Defaulting it to `0.0` would make Whisper look maximally
uncertain in every aggregate; `-1.0` is a sentinel that filters cleanly.

**Cost is declared next to the integration.** Each provider owns its
`cost_per_hour`, with the pricing page and check date in a comment. The harness
has no price table of its own, so a rate cannot drift away from the code that
incurs it.

## The component swap pattern

Call sites depend on the capability, never on the vendor. Selecting a provider is
an env var; `src/providers/__init__.py` is the only module that names a concrete
class.

```python
# .env
STT_PROVIDER=deepgram
```

```python
from src.providers import get_stt

# The agent asks for "speech-to-text", not for Deepgram.
stt = get_stt()                      # honours $STT_PROVIDER
result = await stt.transcribe(wav_bytes)

print(result.text)                   # same field for all three providers
print(result.latency_ms)             # same measurement
print(result.estimated_cost_usd(stt.cost_per_hour))
```

Swapping the provider is one line of config — no call site changes:

```bash
STT_PROVIDER=whisper python -m benchmarks.wer
```

And an explicit name overrides the environment, which is how the benchmark
scores all three in one session:

```python
for name in ("deepgram", "whisper", "assemblyai"):
    stt = get_stt(name)              # same contract, three vendors
    result = await stt.transcribe(wav_bytes)
    print(f"{stt.name:12} {result.text[:60]!r} {result.latency_ms:.0f}ms")
```

Adding a fourth provider is three steps and touches no existing call site:

1. Write `src/providers/stt/<vendor>_stt.py` subclassing `STTProvider`.
2. Add the name to `STT_PROVIDERS` and a lazy-import branch in `get_stt()`.
3. Run `python -m benchmarks.wer --provider <vendor>` and commit the CSV.

The same pattern carries into later sprints: Sprint 2's `get_tts()` and Sprint
3's `get_llm()` differ only in the contract they return, so swapping Cartesia for
another TTS vendor will be the same one-line config change.

## Layout

```
src/
  providers/
    __init__.py          factories: get_stt / get_tts / get_llm / get_vad
    base.py              contracts + TranscriptResult / SynthesisResult / LLMResponse
    stt/
      deepgram_stt.py    nova-3, keyterm boost, Sprint 1 baseline
      whisper_stt.py     whisper-1, tenacity retries, no confidence score
      assemblyai_stt.py  best tier, word_boost, sync SDK on a worker thread
benchmarks/
  wer.py                 WER + latency + cost harness
  README.md              methodology
  corpus/                input .wav files
  ground_truth/          reference .txt transcripts, paired by filename stem
  results/               timestamped CSVs, committed
tests/unit/              factory contract + harness smoke tests
docs/                    this file, Agile Scrum Plan
```

## See also

- `benchmarks/README.md` — benchmark methodology and the three principles
- `src/providers/base.py` — the contracts, with the reasoning in the docstrings
- Agile Scrum Plan in `docs/` — sprint scope and acceptance criteria
