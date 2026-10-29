# AI Voice Calling Agent — Project Description

## 1. Overview

**AI Calling Agent ("Riya")** is a fully offline, CPU-only conversational voice  
agent that automates outbound recruiting phone calls for an IT staffing firm.  
The agent introduces itself, screens the candidate against a job requisition,  
conducts a structured Q\&A, reads back the captured details, and closes the call  
— all in real time, with no human operator on the line.

The system is designed to run on commodity hardware with **no GPU, no cloud  
APIs, and no network dependency** after a one-time model download. Every  
component — speech recognition, language model, and text-to-speech — runs  
locally on the machine's CPU.

---

## 2. Business Problem Solved

Recruiters spend a significant portion of their day on initial screening calls  
that follow a predictable pattern: introduce the role, confirm the candidate's  
qualifications, verify their tech stack, discuss notice period, and schedule a  
follow-up. These calls are repetitive, time-consuming, and difficult to scale.

Riya automates this first-touch screening call. A recruiter configures the job  
requisition (role, client, must-have skills, location, engagement type) in a  
simple YAML file, and the agent places the call, conducts the screening, and  
produces a structured summary of the candidate's responses. The recruiter  
reviews only the candidates who pass the initial filter.

---

## 3. Technical Architecture

### Pipeline

```
Microphone -> Silero VAD -> Speech-to-Text -> LLM (response generation) -> Text-to-Speech -> Speaker
     ^                    (decodes as the caller speaks)                                |
     |                                                                                   |
     +---------------------- Barge-in interrupt (caller can cut the agent off) ---------+
```

The pipeline is **not sequential end-to-end**. Stages overlap:

- **STT streams while the caller speaks.** By the time the caller finishes,  
  transcription is nearly complete — the system does not wait for silence to  
  begin decoding.
- **TTS synthesis overlaps LLM generation.** As soon as the first complete  
  sentence is generated, it enters the speaker queue while the model continues  
  producing the rest of the reply.
- **Playback overlaps everything.** Audio for sentence 1 is playing while the  
  LLM generates sentence 2 and TTS synthesizes it.

This overlap is what makes sub-1.3-second response latency achievable on CPU.

### Component Stack

| Stage                          | Model / Engine                            | Role                                                              | Disk Footprint |
| ------------------------------ | ----------------------------------------- | ----------------------------------------------------------------- | -------------- |
| Voice Activity Detection (VAD) | Silero VAD v5 (ONNX)                      | Detects speech boundaries, triggers turn start/end                | 2 MB           |
| Speech-to-Text (STT)           | NVIDIA Parakeet TDT 0.6B (int8 ONNX)      | Transcribes caller speech to text; batch mode at end-of-utterance | 671 MB         |
| STT (alternate)                | Vosk small-en-us                          | Streaming Kaldi recognizer — lowest latency, least accurate       | 68 MB          |
| STT (alternate)                | faster-whisper base.en (int8)             | CTranslate2 build — most accurate, highest latency                | 145 MB         |
| Language Model (LLM)           | Llama-3.2-3B-Instruct (Q4_K_M GGUF)       | Generates the agent's conversational responses                    | 2,020 MB       |
| LLM Runtime                    | Ollama (loopback HTTP on 127.0.0.1:11434) | Serves the model locally; no compiler or GPU required             | —              |
| Text-to-Speech (TTS)           | Piper en_US-amy-medium                    | Synthesizes the agent's voice; extremely fast (RTF ~0.15)         | 63 MB          |
| TTS (alternate)                | Kokoro-82M (int8 ONNX)                    | Higher-quality voice; auto-selected on 4+ core machines           | 115 MB         |
| **Total model footprint**      |                                           |                                                                   | **~2.8 GB**    |

### Key Files

| File                          | Purpose                                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `config.yaml`                 | All tunable parameters — model selection, VAD thresholds, thread allocation, call brief, question templates |
| `run_agent.py`                | Entry point — live mic mode, text mode, device listing, preflight check                                     |
| `voice_agent/orchestrator.py` | Async pipeline coordinator — manages turn-taking, barge-in, guard logic, question templating                |
| `voice_agent/vad.py`          | Silero VAD wrapper + turn/barge-in detection                                                                |
| `voice_agent/stt.py`          | Multi-engine STT abstraction (Parakeet, Vosk, Whisper)                                                      |
| `voice_agent/llm.py`          | LLM client (Ollama / llama-server) + context trimming                                                       |
| `voice_agent/tts.py`          | TTS engine abstraction (Piper / Kokoro) with measured fallback                                              |
| `voice_agent/prompts.py`      | System prompt construction, agenda management, persona definition                                           |
| `voice_agent/text_chunker.py` | Sentence chunking + spoken-form normalization (e.g., "$85/hr" -> "eighty five dollars an hour")             |
| `voice_agent/config.py`       | Config loading, hardware detection, adaptive thread planning                                                |
| `voice_agent/audio_io.py`     | Microphone capture + interruptible speaker output                                                           |
| `voice_agent/metrics.py`      | Per-turn latency measurement and reporting                                                                  |
| `diagnose.py`                 | Per-stage diagnostic tool — isolates which stage is broken or slow                                          |
| `benchmark.py`                | Latency harness — measures TTFA against the target budget                                                   |
| `tests/`                      | 408 tests covering all pipeline logic, guards, and normalization                                            |

---

## 4. How the Conversation Works

### The Call is Outbound

Riya dials the candidate. The system prompt is explicitly shaped for an  
outbound call — the agent knows why it called and never asks the candidate  
why they're on the line.

### Configurable Job Requisition

The requisition is defined in `config.yaml` and injected into the system  
prompt at startup:

```yaml
call:
  role: "Senior Data Engineer"
  client: "a healthcare analytics client"
  location: "Bengaluru, hybrid three days onsite"
  engagement: "12-month contract-to-hire"
  must_haves: "Python, Spark, AWS, and production pipeline ownership"
```

Changing the `role` field retargets the entire conversation. The agent always  
knows the role, the client, and the must-have skills before the call begins.

### Structured Agenda

The agent follows an ordered checklist during the call:

1. **Qualification and field** — What is your highest qualification?
2. **Years of experience** — How many years of total work experience?
3. **Tech stack** — Which technologies do you work with day to day?
4. **Read back and confirm** — Repeat captured details, ask for confirmation.
5. **Close the call** — Thank the candidate and sign off.

The model asks the first checklist item it doesn't yet have an answer for.  
This prevents the question-loop failure common in small models (repeating the  
last question when they have nowhere to go).

### Question Templating (Key Design Decision)

The agent's questions are **not generated by the LLM**. They are fixed  
templates stored in `config.yaml`, selected by agenda topic and appended by  
Python. The model's only job is to compose a **one-line reaction** to what the  
candidate just said (e.g., "That's a solid background." or "Got it, four years  
in the field."). Python then appends the next scripted question.

This decision was driven by empirical evidence: when the 3B model was allowed  
to compose its own questions, it produced a new failure mode on every run —  
reading checklist notes aloud in the third person, asking two questions in one  
turn, re-asking answered questions, inventing commitments the agent has no  
mandate to make. Templating eliminated the entire defect family at once and  
also reduced per-turn token generation, improving latency.

### Turn-Taking Intelligence

Between the VAD and the model, the orchestrator applies several guards:

- **Dangling-connector hold**: A transcript ending on "so", "and", "because"  
  is a thinking pause, not a finished turn. The system holds briefly and  
  stitches the next segment.
- **Backchannel filter**: "Mm-hmm", "yeah", "right" are not turns. The agent  
  does not reply to them.
- **Talk-over absorption**: If the candidate speaks while the agent is still  
  thinking (nothing played yet), it's treated as one utterance split by a  
  pause, not an interruption.

### Barge-In (Interruption Handling)

When the candidate starts speaking while the agent is talking:

1. Playback stops within ~20 ms (next audio callback emits silence).
2. The LLM generation is aborted (the HTTP response is closed, freeing CPU).
3. Queued audio chunks are dropped.
4. What the agent actually said is preserved in history, and an interruption  
   note is injected next turn so the conversation resumes cleanly.

Each turn carries its own cancellation event, so a new turn can never  
accidentally un-cancel the previous one.

---

## 5. Output Quality Guards

A 3B-parameter model does not reliably obey a system prompt for an entire call.  
The orchestrator applies deterministic guards that inspect the model's output  
**before** it reaches TTS. Each guard exists because the specific failure was  
observed on a live call.

| Guard               | Failure Caught                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Repetition**      | Same question asked multiple turns in a row                                                               |
| **Assistant voice** | Generic chatbot language ("How can I assist you today?")                                                  |
| **Artifact**        | Stray formatting, uninterpolated template variables, list numbering                                       |
| **Agenda echo**     | Internal checklist notes read aloud verbatim in third person                                              |
| **Parrot**          | Echoing the candidate's own words back at them                                                            |
| **Reaction gate**   | Agent talking about itself, the call, a technology, or the candidate's manner instead of to the candidate |

Guard outcomes are three-tiered:

- **Cut**: The offending text is dropped, the templated question still goes out.  
  No regeneration. Costs zero latency.
- **Warn**: The text is spoken but logged for review.
- **Retry (disabled)**: A second generation was found to be consistently worse  
  than the first (a 3B asked to stop inventing simply invents something  
  different). Retries are disabled by configuration.

---

## 6. Spoken-Form Normalization

TTS reads text literally, so the system normalizes domain-specific shorthand  
before synthesis:

| Written               | Spoken                                                    |
| --------------------- | --------------------------------------------------------- |
| C2H / C2C / W2 / 1099 | contract to hire / corp to corp / W two / ten ninety nine |
| $85-95/hr             | eighty five to ninety five dollars an hour                |
| $140k                 | one hundred forty thousand dollars                        |
| H1B / GC / RTR        | H one B / green card / right to represent                 |

STT fixups also repair common mishearings from phone-band audio: "WS" becomes  
"AWS", "dev ops" becomes "DevOps", "air flow" becomes "Airflow".


## 8. Performance

### Latency Target: Time-to-First-Audio (TTFA) < 1,300 ms average

Measured on live microphone calls over 6-7 turns each:

| Run | Avg TTFA | p90      | Regenerations | Result          |
| --- | -------- | -------- | ------------- | --------------- |
| 43  | 1,211 ms | 1,474 ms | 0             | PASS            |
| 44  | 1,097 ms | 1,274 ms | 0             | PASS            |
| 45  | 1,230 ms | 1,253 ms | 0             | PASS            |
| 46  | 1,235 ms | 1,335 ms | 0             | PASS            |
| 47  | 1,197 ms | 1,464 ms | 0             | PASS            |
| 48  | 1,135 ms | —        | 0             | PASS            |
| 49  | 965 ms   | 1,131 ms | 0             | PASS (best run) |

The latency target has been met and held across multiple consecutive runs.

### Test Suite

**408 tests passed, 1 skipped** — covering pipeline logic, output guards,  
spoken-form normalization, STT fixups, LLM backend abstraction, and  
orchestrator flow.

---

## 9. Deployment

### One-Command Setup (Windows)

```powershell
.\setup_windows.ps1
```

Creates a virtual environment, installs all dependencies, downloads the models  
and LLM binary, and runs a preflight check.

### Running a Call

```powershell
# Live voice call (microphone + speaker)
python run_agent.py

# Text input mode (no microphone needed — for testing)
python run_agent.py --text

# Diagnose issues
python diagnose.py

# Benchmark latency
python benchmark.py
```

### Requirements

- **OS**: Windows (Linux/macOS compatible with minor path adjustments)
- **Python**: 3.10 – 3.13
- **CPU**: 4+ physical cores recommended (2 cores functional but slower)
- **RAM**: ~4 GB free (models total ~2.8 GB on disk)
- **Disk**: ~3 GB for models + dependencies
- **GPU**: Not required
- **Network**: Only for initial model download; none at runtime

---

## 10. Known Limitations

1. **No telephony transport**: The audio boundary is the local  
   microphone/speaker. Integrating SIP or WebRTC requires replacing two  
   classes (`MicStream` / `SpeakerStream`) — the rest of the pipeline is  
   transport-agnostic.
2. **No acoustic echo cancellation**: Headphones are recommended to prevent  
   the microphone from hearing the agent's output.
3. **English only** as currently configured.
4. **Context window caps call length**: The 3,072-token context window holds  
   roughly 6-10 turns; older turns are dropped beyond that.
5. **Adversarial callers**: A 3B model cannot reliably handle hostile or  
   deceptive input (fabricated qualifications, obscene questions). This is a  
   model capability limit, not a software bug.

---

## 11. Extension Points

| Extension                          | How                                                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **New job role**                   | Edit `call:` section in `config.yaml` — the conversation retargets automatically                         |
| **Live data injection (ATS/CRM)**  | Set `agent.context_provider` to a function that returns requisition details based on transcript keywords |
| **New STT engine**                 | Add to `voice_agent/stt.py`; all engines share the same interface                                        |
| **New TTS engine**                 | Add to `voice_agent/tts.py`; auto-fallback logic is engine-agnostic                                      |
| **Telephony integration**          | Replace `MicStream` / `SpeakerStream` in `voice_agent/audio_io.py` with SIP/WebRTC handlers              |
| **Additional screening questions** | Add to `agent.questions` in `config.yaml` and extend `AGENDA_TEMPLATES` in `prompts.py`                  |
| **Multi-language**                 | Swap STT model (Whisper supports 90+ languages), TTS voice, and system prompt                            |

---

## 12. Summary

This project delivers a production-architected, fully offline voice agent that  
conducts structured recruiting screening calls with sub-1.3-second response  
latency on commodity CPU hardware. The architecture prioritizes reliability  
over flexibility: questions are templated to guarantee consistent screening,  
output guards catch the failure modes of small models before they reach the  
caller, and every stage of the pipeline overlaps to minimize end-to-end  
latency. The system is configurable via a single YAML file and extensible at  
well-defined integration points.
