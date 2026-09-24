# Local CPU Real-Time Voice Agent — IT Staffing

Fully offline, CPU-only conversational voice agent. No GPU, no API keys, no
network calls after the one-time model download.

```
Mic → Silero VAD → Nemotron streaming ASR → Llama-3.2-3B → Kokoro/Piper → Speaker
        ↑              (decodes as you talk)                          ↓
        └────────────────── barge-in interrupt ─────────────────────--┘
```

| Stage | Model | Runtime | Disk |
|---|---|---|---|
| VAD | Silero VAD v5 | onnxruntime | 2 MB |
| **STT** | **Nemotron streaming ASR 0.6B, int8** | onnx-asr / transformers | **630 MB** |
| **TTT** | **Llama-3.2-3B-Instruct Q4_K_M** | Ollama (loopback HTTP) | **2020 MB** |
| **TTS** | **Kokoro-82M int8** (≥4 cores) | kokoro-onnx | **115 MB** |
| TTS fallback | Piper `en_US-amy-medium` | piper-tts | 63 MB |
| | | **total** | **2830 MB = 2.76 GiB** |

**The STT is cache-aware streaming, and that is the point.** Parakeet and Whisper
cannot begin until the caller stops talking, so their entire cost sits on the
critical path — 250-475 ms in live runs. Nemotron consumes audio as it arrives and
reuses cached encoder state instead of recomputing an overlapping window, so at
end-of-speech the decode is essentially done. It also brings punctuation and
capitalisation, and reports 6.93% average WER at 1.12 s chunks / 7.07% at 0.56 s.
`stt.nemotron.chunk_ms` moves along that latency/accuracy curve at runtime with no
reload.

Alternate engines still ship and are never loaded unless selected: Parakeet TDT
0.6B (671 MB, batch), Vosk `small-en-us` (68 MB, streaming), `faster-whisper-base.en`
(145 MB).

**No compiler required.** The LLM runs through **Ollama** on `127.0.0.1:11434`
(`llm.backend: ollama`). That sidesteps `llama-cpp-python`'s wheel/toolchain
problem entirely and works on Python 3.13. A `llama-server` backend also ships
(`llm.backend: server`) for the prebuilt llama.cpp binary. Either way it is 100%
offline — the socket never leaves loopback.

---

## Quick start (Windows)

```powershell
cd C:\Users\PC\Desktop\AI_calling_agent
.\setup_windows.ps1
```

Creates `.venv`, installs everything, downloads the models **and** the llama.cpp
binary, runs preflight. Then:

```powershell
.\.venv\Scripts\Activate.ps1
.\run_vosk.ps1                   # live call, Vosk streaming STT (fast)
.\run_whisper.ps1                # live call, Whisper STT (accurate)
python run_agent.py --text       # typed input, spoken output (no mic)
python diagnose.py               # per-stage check if something is wrong
python benchmark.py              # latency against the budget
python -m pytest tests\ -q       # 273 tests, no models needed
```

**Wear headphones.** On open speakers the microphone hears Piper's output and
barge-in gets twitchy. `barge_in.threshold` compensates, but there is no acoustic
echo cancellation in this build.

### Manual install

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_models.py --warm-stt
python run_agent.py --check
```

---

## Something's wrong? Run the diagnostic

```powershell
python diagnose.py
```

Tests each stage separately and tells you which one is broken or slow — including
**whether the microphone is capturing anything at all**, which `run_agent.py`
cannot distinguish from "you didn't say anything." Live input-level meter, then
Silero's speech probabilities, the transcript of what you actually said, the LLM's
tokens/sec, and Piper's per-chunk synthesis time, each against its target.

```powershell
python diagnose.py --mic-only       # just the microphone/VAD check
python diagnose.py --device 2       # force an input device (indexes are listed)
python diagnose.py --skip-mic       # stage timings only, canned text
python diagnose.py --stt-bench      # sweep STT model / threads / chunk_length
```

---

## Layout

```
config.yaml                  every tunable
run_agent.py                 entry point (--text, --check, --list-devices, --stt, --threads)
diagnose.py                  per-stage troubleshooting
benchmark.py                 latency harness
run_vosk.ps1                 launch with the streaming Vosk recognizer
run_whisper.ps1              launch with faster-whisper
cleanup.ps1                  delete models/packages this pipeline no longer uses
setup_windows.ps1            one-shot Windows setup
scripts/download_models.py   models + llama.cpp binary fetch, footprint report
voice_agent/
  config.py                  config load, hardware profile, adaptive thread plan
  prompts.py                 system prompt (full + compact) and context injection
  audio_io.py                mic capture, interruptible speaker
  vad.py                     Silero wrapper + turn/barge-in detection
  stt.py                     Nemotron (streaming), Parakeet, Vosk, Whisper
  llm.py                     llama-server client + context trimming
  tts.py                     Kokoro + Piper, with a measured fallback
  text_chunker.py            sentence chunking + spoken-form normalization
  metrics.py                 per-turn latency accounting
  orchestrator.py            the async pipeline and barge-in supervisor
tests/                       273 tests
```

---

## How the pipeline overlaps

Nothing runs strictly end-to-end.

**Synthesis overlaps generation.** `SentenceChunker` watches the token stream and
emits a chunk the moment one is complete. The **first** chunk is allowed to be
short (18 chars) because latency beats prosody when nothing is playing yet; later
chunks use a 55-char minimum since playback is already running. If no sentence
terminator arrives within 200 chars it splits at a comma or conjunction rather
than stalling.

**Playback overlaps everything.** Chunk 1 is in the speaker queue while the LLM is
still generating chunk 2. Backpressure kicks in above ~2.5 s of queued audio so
Piper stops competing with llama.cpp for cores.

**The system prompt is pre-ingested.** `warmup()` sends the real system prompt at
startup with `cache_prompt`, so llama-server has that KV prefix cached before the
call begins. Without this the ~800-token persona is ingested on the caller's first
question — which measured **22.8 seconds** on a 2-core box. The `compact` prompt
(~200 tokens, the default) cuts it further.

## Two STT engines

```powershell
.\run_vosk.ps1                      # streaming, low latency  (config default)
.\run_whisper.ps1                   # more accurate, higher latency
.\run_whisper.ps1 -Model small.en   # more accurate again, ~3x slower
python run_agent.py --stt-engine vosk      # same thing without the wrapper
```

Or set `stt.engine: vosk | whisper` in `config.yaml`.

| | Vosk | Whisper |
|---|---|---|
| Decoding | **streams while you speak** | can only start once you stop |
| Finalize cost | **~90 ms** | ~500–1200 ms on 2 cores |
| Accuracy | good on clean close-mic speech | better, especially on accents and noise |
| Casing / punctuation | none (lowercase, no commas) | yes |
| Size | 68 MB | 145 MB (base.en) |

**Why Vosk is the default.** Whisper's cost lands entirely on the critical path:
it cannot begin until the caller stops talking, and it pads every clip to a fixed
mel window, so a one-second utterance costs about the same as a ten-second one.
Vosk consumes 32 ms frames as the VAD produces them, so by end-of-speech the
decode is essentially done.

Measured on a single core, 5.2 s of speech:

| | |
|---|---|
| Vosk, streamed — finalize | **94 ms** |
| Vosk, batch (no streaming) | 3136 ms |
| Whisper distil-small.en | 2838 ms |

Feeding costs ~6 ms per 32 ms frame. That's 20% of the frame budget, so it runs
in the STT executor pool rather than on the event loop — the pool has one worker,
which keeps frames in order and makes the later `finalize()` queue behind them
naturally.

Compare them on your own hardware:

```powershell
python diagnose.py --stt-bench
```

Sweeps Vosk plus Whisper × threads × `chunk_length`, printing latency with the
transcript beside it so you can see where accuracy breaks.

### Whisper-specific notes

- **The model must be a CTranslate2 build.** `distil-whisper/distil-small.en` on
  the Hub is the transformers format and fails with `Unable to open file
  'model.bin'`. Use the `Systran/faster-*` conversions.
- `chunk_length: 15` shrinks the encoder's padded window from 30 s. Lower is
  faster; below ~8 accuracy slips.
- `beam_size: 1` (greedy) roughly halves the cost.
- **Hallucination filtering is on.** Whisper reliably invents text from silence —
  "Thank you.", "Thanks for watching!", "Bye." A real session produced three
  phantom turns in a row from the agent's own audio leaking into the mic. Segments
  above `no_speech_threshold` *and* below `min_avg_logprob` are dropped, plus a
  blocklist of the canonical phrases. Rejected turns are logged, not silently
  swallowed.

### Threads

Within a turn the stages are *sequential* — VAD ends → STT alone → LLM → TTS
overlapping the LLM's later tokens — so STT gets every physical core. It
previously got `cores // 2`, which on a 2-core box meant **one thread** for the
stage that already dominated.

---

## Barge-in

When Silero sees sustained speech during playback:

1. `SpeakerStream.interrupt()` clears the output deque — the next PortAudio
   callback (≤ 20 ms away) emits silence.
2. The turn's `threading.Event` is set and the HTTP response is closed, which
   makes llama-server abort the slot instead of burning cores.
3. The asyncio turn task is cancelled, dropping queued chunks. **`_barge_in()`
   never awaits model work**, so it cannot be blocked by an in-flight Whisper or
   Piper call.
4. What the agent actually said is kept in history, and `INTERRUPTION_NOTE` is
   injected next turn so it picks up cleanly.

Each turn carries its own `Event` rather than reading `self._stop_generation`, so
a new turn can never un-cancel the previous one. There's a regression test for it.

While the agent speaks, barge-in requires probability ≥ 0.70 sustained for 180 ms
(vs. 0.50 / 200 ms when idle), with a 150 ms cooldown after the cut.

---

## Threading

`n_threads` defaults to **physical** cores via `psutil.cpu_count(logical=False)` —
oversubscribing hyperthreads hurts Q4_K_M throughput. llama.cpp's threads live in
the subprocess; the Python-side LLM thread just reads an SSE socket.

| Thread | Work |
|---|---|
| PortAudio input | 32 ms frames → asyncio queue |
| Event loop | Silero (~0.26 ms/frame, 1 thread), turn segmentation, barge-in supervision |
| `stt` pool (1) | Vosk frame feeding + finalize, or Whisper; all physical cores |
| LLM thread (per turn) | token stream → chunker → queue |
| `tts` pool (1) | Piper, `cores // 2` — it overlaps the LLM so it must not take the machine |
| PortAudio output | drains the speaker deque |

---

## Domain configuration

**The call is outbound.** Riya dials the candidate; they did not call her. This
matters more than it sounds — an earlier inbound-shaped prompt had the agent
asking people it had just phoned why they were calling, and a test caller
eventually replied *"no no, you called me."*

The requisition lives in `config.yaml` under `call:`, and is interpolated into
the system prompt, so the agent always knows why it rang:

```yaml
call:
  candidate_name: ""                  # blank if unknown
  role: "Senior Data Engineer"
  client: "a healthcare analytics client"
  location: "Bengaluru, hybrid three days onsite"
  engagement: "12-month contract-to-hire"
  must_haves: "Python, Spark, AWS, and production pipeline ownership"
  source: "your profile came up in our database"
```

Change `role` and the whole conversation retargets. There is deliberately no
`rate` field: Riya must never quote a number on a first call.

`voice_agent/prompts.py` builds the persona around that brief. `compact` is the
default; `full` adds objection-handling depth. Both are static prefixes pinned in
llama-server's KV cache at warmup, so their length costs startup time rather than
per-turn latency — which is why per-turn notes are appended to the last **user**
message and never to the system prompt.

Two design choices there are load-bearing on a 1.7B model:

- **`AGENDA` is an ordered checklist** (bad time? → open to the role? → hands-on
  experience → notice period → rate → work authorization → location → book the
  follow-up). The model answers the first item it doesn't have. Question loops
  happen when a small model has nowhere to go and falls back on its last
  sentence; an agenda guarantees a next question.
- **No example contains a concrete fact.** Few-shot examples get copied verbatim
  at this size. An earlier prompt's example line — *"Not since Friday, I'm
  chasing the hiring manager today"* — came back word for word at a caller who
  had just said "I'm a data engineer". Examples now teach shape, not content.

### Turn-taking

Sitting between the VAD and the model, all in `orchestrator.py`:

| Guard | What it does |
|---|---|
| Dangling-connector hold | A transcript ending on `so`, `and`, `because`, `um`… is a thinking pause, not a finished turn. Held for `vad.continuation_grace_ms` and stitched to the next segment, or released if the caller never resumes. |
| Backchannel filter | "Mm-hmm", "yeah", "right" are not turns. Replying to them produced four identical questions in a row. |
| Talk-over absorption | If the caller speaks while the agent is still *thinking* (nothing played yet), that is one utterance split by a pause, not an interruption. The turn is folded into the next one instead of being binned — nine turns vanished in one call before this, each leaving an unanswered user message in history. |

### Output guards

A 1.7B does not reliably obey a system prompt for a whole call. Every guard below
exists because the exact failure was observed on a live call; each rejects the
reply **before Piper runs** and regenerates once, naming the specific failure.
Rejections are counted in the run summary — a high count means the model is
fighting the prompt, not that the hardware is slow.

| Guard | Live failure it catches |
|---|---|
| Repetition | The same question asked four turns running |
| Assistant voice | *"Hello there! How can I assist you today? 🌟"* |
| Artifact | *"You: I called about a role at [Client Name]"*, stray `</think>`, list numbering, uninterpolated `{role}` |
| Agenda echo | *"Are they open to hearing about a Senior Data Engineer role at all?"* — a checklist note read aloud, in the third person |
| Parrot | *"Working with the big team."* — the caller's own words handed back |

Two prompt-design rules follow from the same failures, and are enforced by tests:

- **No speaker labels anywhere in the prompt.** When the style examples were
  written `Them: … / You: …`, the model copied the *format* and began replies
  with a literal `You:`. Removing concrete facts from examples was not enough —
  the labels themselves had to go. Examples are label-free prose now.
- **Checklist items are fragments, not sentences.** `open to the role?` cannot be
  spoken as-is; `are they open to hearing about the role at all` was read out
  verbatim, third person included.

### Prompt cache

`llm.turn_reminder` is attached to **every** user message in the rendered prompt,
never just the last one. This looks wasteful and is not: llama-server reuses the
KV cache for the longest common prefix, so attaching it only to the newest turn
means the previous turn is re-sent *without* it, the prefix diverges at the first
user message, and the whole conversation is re-ingested every turn. That mistake
pinned time-to-first-token flat at ~1940 ms regardless of how little had changed.
`test_every_turn_reuses_the_previous_turns_prefix` guards it.

### STT fixups

Phone-band audio eats leading consonants, so `_STT_FIXUPS` repairs the terms that
matter in this domain — `WS`→`AWS`, `dev ops`→`DevOps`, `c2c`→`C2C`, `h1b`→`H1B`,
`air flow`→`Airflow`. Extend the list there; all entries are covered by tests.

### Live-data injection

```python
agent.context_provider = lambda transcript: (
    "Req 4521 — Senior AWS DevOps Engineer, client Acme Corp, "
    "bill rate $85-95/hr, remote US, fill within 3 weeks"
    if "acme" in transcript.lower() else None
)
```

Keep it under ~150 tokens. Wire it to your ATS/CRM lookup.

### Spoken-form normalization

Piper reads text literally, so `normalize_for_tts()` rewrites shorthand first:

| Written | Spoken |
|---|---|
| `C2H` / `C2C` / `W2` / `1099` | contract to hire / corp to corp / W two / ten ninety nine |
| `$85-95/hr` | eighty five to ninety five dollars an hour |
| `$140k` | one hundred forty thousand dollars |
| `Req #4521` | requisition number four five two one |
| `H1B` / `GC` / `RTR` | H one B / green card / right to represent |

Extend `_ACRONYMS` in `text_chunker.py`. All covered by tests.

---

## Tuning

| Symptom | Fix |
|---|---|
| `Unable to open file 'model.bin'` | you pointed `stt.model_name` at a transformers repo — use a `Systran/faster-*` one |
| `dropped N mic frame(s)` | the machine can't keep up. `Systran/faster-whisper-tiny.en`, `--llm-size 0.5b`, or a `*-low` Piper voice |
| `llama-server exited immediately` | check `models/llama-server.log`; if a flag was rejected, delete it from `llm.server.extra_args` |
| Port 8080 in use | change `llm.server.port`, or let it reuse the existing server |
| Agent cuts you off mid-sentence | raise `vad.continuation_grace_ms` first — it only costs time on sentences that actually trail off. `vad.min_silence_ms` 700 is the blunt version and taxes every turn |
| Long pause before it answers | lower `vad.min_silence_ms` to 400 |
| First word of your sentence is missing | raise `vad.speech_pad_ms`. The pre-roll buffer is sized to hold this *plus* the frames the trigger consumes, so the value is real padding, not a budget the trigger eats |
| Agent asks the same question repeatedly | check `llm.repeat_last_n` (≥256) and that the `AGENDA` in `prompts.py` covers your call — a loop means the model ran out of things to ask |
| Agent thinks you called it | you're on an old `agent.greeting` / prompt; the shipped ones are outbound |
| One turn takes ~10 s after an interruption | you're on a build without the abort path; barge-in must call `llm.abort()`, not just set the stop event |
| Agent interrupts itself | raise `barge_in.threshold` to 0.8, or use headphones |
| Silero sees no speech | check the mic in `diagnose.py`; lower `vad.threshold` to 0.35 |
| STT too slow | use `.\run_vosk.ps1`; run `diagnose.py --stt-bench` |
| STT misheard you | `.\run_whisper.ps1 -Model base.en`, or a bigger Vosk model (`--vosk-model en-us-0.22`) |
| Agent replies to things you didn't say | Whisper hallucinating on silence — lower `stt.whisper.no_speech_threshold` to 0.4, or switch to Vosk |
| Agent barges in on itself | already at 0.85/300 ms; use headphones, that's the real fix |
| Agent sounds like a generic chatbot | `llm.system_prompt: full` (costs ~330 more prompt tokens, cached after warmup) |
| TTFT too slow | `--llm-size 0.6b`, or lower `llm.n_ctx` |
| Agent gives long rambling answers | `llm.max_sentences` (default 2) is the hard cap; `max_tokens` alone does not hold it |
| Agent repeats your words back at you | the parrot guard catches short echoes; if it slips through, lower `llm.temperature` |
| Agent drifts into chatbot voice late in a call | expected on a 1.7B as history grows — `prompts.TURN_REMINDER` is restated every turn and the guard catches the rest. A smaller history (`llm.history_token_budget`) helps more than a sterner prompt |

---

## Adapting to the machine

Nothing in `config.yaml` is pinned to a core count. At startup `plan_threads()`
in `voice_agent/config.py` splits the machine between the stages that actually
run at the same time, and the engines adapt with it:

| Physical cores | Tier | LLM gen / batch | TTS | STT | Default TTS engine |
|---|---|---|---|---|---|
| 2 | minimal | 2 / 2 | 1 | 1 | Piper |
| 4 | low | 3 / 4 | 1 | 2 | Piper |
| 8 | mid | 6 / 8 | 2 | 2 | Kokoro |
| 16 | high | 8 / 16 | 3 | 4 | Kokoro |

Two things worth knowing about that table:

**Only one window is contended.** STT runs while the caller speaks; the LLM runs
after end-of-speech; TTS overlaps the tail of generation. So the budget that
matters is `llm_gen + tts <= cores`. The previous policy gave each stage its own
maximum — llama.cpp `cores`, STT `cores`, TTS `cores//2` — promising 5 threads on
a 2-core laptop. That is why Piper was reporting 800-1400 ms for a first chunk
that should take ~120 ms: it was fighting llama.cpp for the same two cores.

**Prompt ingest gets the whole machine.** It runs before any audio exists, so it
is never contended — hence separate `-t` and `-tb` values. Generation threads cap
at 8 because llama.cpp stops scaling on small models beyond that.

`tts.engine: auto` picks Kokoro from 4 cores up, but the decision is then
*measured*: `build_tts` synthesises a line at warmup and falls back to Piper if
the realtime factor exceeds `tts.max_realtime_factor`. Above 1.0 synthesis is
slower than speech and the call stalls mid-sentence, so this is checked rather
than assumed — Kokoro's RTF swings by an order of magnitude across laptops.

---

## Hardware reality check

The <1300 ms target assumes roughly 4 physical cores. On **2 cores with the 1.7B**
it is not reachable — a measured session averaged 2518 ms:

| Stage | 2 cores, Qwen3-1.7B | Why |
|---|---|---|
| Parakeet | ~335 ms | fine |
| LLM first token | ~1136 ms, rising with history | prompt ingest is linear in context, and 2 threads is the floor |
| Piper first audio | ~1048 ms, p90 2436 ms | 1 TTS thread competing with llama.cpp for the same 2 cores |

Both LLM and TTS stages are core-starved, and they contend with each other. The
fix is a smaller model, not more tuning:

```powershell
python scripts/download_models.py --llm-size 0.6b
# then set llm.model_file to Qwen_Qwen3-0.6B-Q4_K_M.gguf
```

Qwen3-0.6B roughly halves time-to-first-token and frees a core for Piper — but
it follows instructions *less* reliably, so expect more guard rejections, and
each rejection costs a full regeneration. Watch the "Turns regenerated" line in
the summary: if it climbs, the smaller model is costing more than it saves.

### The honest limit

A free-form outbound sales conversation is at the edge of what a 1.7B does
reliably, and the guards in this repo are containment, not a cure. If output
quality is still the blocker after all of the above, the options in increasing
order of effort are:

1. **More cores.** Qwen3-4B on 4-8 physical cores is a different conversation
   partner entirely, and the latency budget becomes reachable again.
2. **Constrain the dialogue.** Drive the call from an explicit state machine —
   one agenda item per state, the LLM only phrasing the question and classifying
   the answer. A 1.7B is reliable at "is this a yes, a no, or neither?" and
   unreliable at "hold a persona across twenty turns". This is the standard
   production shape for small-model voice agents and it is what the AGENDA list
   is already half-way towards.

---

## How the questions are written (read this before changing the prompt)

The single most consequential decision in this project was to **stop letting the
model write the questions.** Riya's turn is now assembled from two parts:

1. **A one-line reaction** — the only thing the model still composes.
2. **A fixed question**, looked up in `config.yaml → agent.questions` by agenda
   topic and appended by Python (`_append_question`).

Almost every content defect the project ever had — agenda notes read aloud, two
questions in one turn, re-asking, off-agenda drift, invented commitments, third
person — traced to that one freedom. Templating removed the whole class at once,
and it *improved* latency rather than defending it, because the model now
generates fewer tokens per turn.

Two rules that follow, and both were learned the hard way:

- **The templated question is never written to the model's history.** Only the
  model's own reaction line goes in. Storing the question re-teaches the model
  that its job is to write questions, and it is the thing the model then echoes.
  There is a deliberate asymmetry: history lines no longer end in `"?"`, so
  `_is_backchannel` reads `_last_spoken` instead. Do not "simplify" that back.
- **An empty reaction is allowed.** Once the question is synthesised separately,
  the caller still hears a turn even if the reaction is cut to nothing. Cutting
  a bad line is strictly better than leaving it, and it costs no regeneration.

### When a guard misses a new wording: key on the invariant, not the bound

The repair rules in `orchestrator.py` are deterministic regexes. Each time one
has missed a fresh 3B phrasing, the tempting fix is to widen its length bound
and the correct fix is to find what the wordings share:

- A rule keyed on a **verb list** missed `"our clients often match candidates"`.
  Keyed on the **subject** instead, one rule covers the family.
- `_CALL_NARRATION`'s gap `[a-z\s]{0,24}` missed `"information on SQL skills"`
  by **two characters**, and looked correct for three runs. It is now keyed on
  (discourse subject) × (discourse verb) with **no length window at all**.

Raising a bound catches the run in front of you and misses the next one.

`_MAY_EMPTY` is the companion mechanism: a label in that set may empty the whole
reply, which turns "detect the defect" into "the defect costs nothing". Prefer
adding a label there over adding a pattern that merely reports.

---

## Measured performance

Driven through the real turn path (`_handle_turn`) against the real 3B, on the
reference machine, over a 10-turn scripted screening call:

| metric | value | target |
|---|---|---|
| TTFA average | **696 ms** | < 1300 ms |
| TTFA p90 | **967 ms** | — |
| regenerations | **0** | 0 |

Representative per-stage split on an ordinary turn: STT ~0 ms (streaming, done
by end-of-speech), LLM TTFT ~130–160 ms, TTS 205–660 ms. **TTS now dominates the
budget** — the read-back line is a template and a long one, so Piper costs
800–920 ms on that turn alone. If p90 ever needs to come down, that is where to
look: `tts.engine: kokoro` on a 4-core box, or a shorter `confirm` template.

---

## Known limitations

- **Two physical cores is below what this stack wants.** The tuning above should
  bring a 2-core box from ~4.3 s to roughly 1–1.5 s to first audio, but the
  brief's 1.2 s with the *accurate* models needs more cores, not more config.
- **No acoustic echo cancellation.** Headphones, or accept twitchy barge-in.
- **English only** as configured.
- **No telephony transport.** The audio boundary is `MicStream` / `SpeakerStream`;
  swapping in SIP or WebRTC means replacing those two classes, nothing else.
- **`llm.ollama.num_ctx: 3072` caps call length.** Long calls drop their oldest
  turns rather than summarizing them.
- **The hangup arms at two farewells, but the caller can still outlast it.**
  Once the checklist is finished the agent signs off; if the caller keeps
  talking, the loop keeps giving them turns. Observed on a 10-turn replay,
  where an off-agenda turn after sign-off produced a promise the agent has no
  mandate to make. A production deployment wants a hard turn ceiling.

---

## Cleanup

```powershell
.\cleanup.ps1            # dry run, shows what would go
.\cleanup.ps1 -Force     # ~2.4 GB freed
```

Removes Kokoro, Moonshine, the Qwen2.5-3B GGUF, `config.lowspec.yaml`,
`MIGRATION.md`, stray Linux llama.cpp binaries and `__pycache__`, then prints the
`pip uninstall` line for the packages no longer imported.
