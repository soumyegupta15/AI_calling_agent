# Migration: distil-Whisper small.en + Qwen2.5-1.5B + Piper

Status: **planned, not implemented.** Written at a stopping point so this can be
picked up without re-deriving anything. Current code still ships Moonshine +
Kokoro + a 3B/1.5B split across two config files.

Target stack:

| Stage | Model | Runtime |
|---|---|---|
| VAD | Silero VAD v5 (unchanged) | onnxruntime |
| STT | `distil-whisper/distil-small.en`, int8 | faster-whisper (CTranslate2) |
| TTT | Qwen2.5-1.5B-Instruct Q4_K_M | llama-server subprocess |
| TTS | Piper `en_US-amy-medium` | piper-tts |

---

## 1. STT: replace Moonshine with distil-Whisper

**Runtime choice: `faster-whisper`** (CTranslate2). It is the route that actually
gives int8 on CPU — `pip install faster-whisper`, then CTranslate2 quantizes on
load, no manual conversion step. The `transformers` route needs ONNX export work
and is slower on CPU.

```powershell
pip install faster-whisper
```

Rewrite `voice_agent/stt.py`. Keep the existing class names and the
`Transcript` dataclass so `orchestrator.py` needs no changes:

```python
from faster_whisper import WhisperModel

class DistilWhisperSTT:
    def __init__(self, model_name="distil-whisper/distil-small.en",
                 compute_type="int8", num_threads=2):
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,     # int8 | int8_float32 | float32
            cpu_threads=num_threads,
            num_workers=1,
        )
        self._lock = threading.Lock()      # keep: partials race the final pass

    def transcribe(self, audio_int16, sample_rate=16000) -> Transcript:
        audio = audio_int16.astype(np.float32) / 32768.0
        started = time.perf_counter()
        with self._lock:
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                beam_size=1,               # greedy: ~2x faster, fine for short turns
                vad_filter=False,          # Silero already segmented the turn
                condition_on_previous_text=False,   # avoids drift across turns
                without_timestamps=True,
            )
            text = " ".join(s.text for s in segments).strip()
        return Transcript(text, (time.perf_counter()-started)*1000.0, False,
                          1000.0*len(audio_int16)/sample_rate)
```

`StreamingTranscriber` is unchanged — it only calls `.transcribe()`.

**Caveats to expect.** distil-small.en is ~166M params vs Moonshine base's ~60M,
so per-pass cost is higher, but unlike Moonshine it is a Whisper derivative and
**pads to 30 s internally**. That means cost is roughly constant regardless of
utterance length — bad for short turns, good for long ones. Benchmark before
trusting it: on 2 cores expect ~400–900 ms per pass. If that is too slow,
`distil-whisper/distil-small.en` with `compute_type="int8"` and `beam_size=1` is
already the fast configuration; the next lever is going back to a smaller model.

Because of the 30 s padding, **set `stt.partial_interval_ms: 0`** — repeated
partial passes no longer get cheaper on short buffers, so they just burn CPU.

---

## 2. TTT and TTS

Already done in `config.lowspec.yaml`: Qwen2.5-1.5B-Q4_K_M, `system_prompt:
compact`, `n_ctx: 1536`, `max_tokens: 90`, Piper `en_US-amy-medium`.

**Action:** fold `config.lowspec.yaml` into `config.yaml` and delete the
lowspec file. There is now only one supported stack, so two configs is dead
weight.

---

## 3. Delete (unused after the swap)

| Path | Why |
|---|---|
| `voice_agent/tts.py` → `KokoroTTS` class | Piper only. Keep `PiperTTS`, `Speech`, `build_tts` |
| `voice_agent/llm.py` → `QwenLLM` class | in-process backend unused; keep `BaseLLM` + `LlamaServerLLM` |
| `config.lowspec.yaml` | merged into `config.yaml` |
| `models/kokoro-v1.0.int8.onnx`, `models/voices-v1.0.bin` | ~115 MB |
| `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | ~1.9 GB |
| `models/llama.cpp/` Linux binaries (`llama-server`, `*.so`) | left over from my Linux testing; the Windows build is the `.exe` |
| Moonshine HF cache: `%USERPROFILE%\.cache\huggingface\hub\models--UsefulSensors--moonshine*` | ~250 MB |
| `requirements.txt`: `useful-moonshine-onnx`, `kokoro-onnx`, `tokenizers` | replaced by `faster-whisper` |

Add to `requirements.txt`: `faster-whisper>=1.0`, `piper-tts>=1.6`.

Also strip from the code: the `--tts-precision` flag, `KOKORO_*` constants and
`fetch_tts()` in `scripts/download_models.py`; the `inprocess` branch in
`run_agent.py --backend` and in `preflight()`; the Kokoro/Moonshine sections of
`README.md`.

`scripts/download_models.py` should end up fetching: Silero VAD, the llama.cpp
Windows binary, Qwen2.5-1.5B, and the Piper voice. distil-Whisper downloads
itself from HF on first use (add a `--warm-stt` that instantiates
`WhisperModel` once).

---

## 4. Tests to update

- `tests/test_llm_backends.py`: drop
  `test_inprocess_backend_gives_a_useful_message_when_the_wheel_is_missing`.
- Keep all four Silero-context tests in `tests/test_pipeline_logic.py` — those
  guard the bug that broke speech detection entirely.
- Add: a `DistilWhisperSTT` test that asserts `beam_size=1`,
  `condition_on_previous_text=False` and `vad_filter=False` are passed (a fake
  `WhisperModel` recording kwargs is enough — no model download in CI).
- `diagnose.py`: rename the STT section, and drop the "try moonshine/base"
  hint.

---

## 5. Order of work

1. `pip install faster-whisper piper-tts`
2. Rewrite `stt.py`, run `python diagnose.py --skip-mic` to get real numbers
3. Merge configs, delete Kokoro + `QwenLLM`
4. Update downloader, preflight, tests, README
5. `python -m pytest tests\ -q` then `python benchmark.py`

Do step 2 before step 3 — if distil-small.en is too slow on 2 cores you want to
know that before deleting the alternative.

---

## Already fixed and working (do not regress)

- **Silero v5 needs 64 samples of context prepended to each 512-sample frame.**
  Without it the VAD returns ~0.0 for all speech and nothing is ever captured.
  Four regression tests cover this.
- **`warmup()` primes the prefix cache with the real system prompt**, so the
  ~800-token ingest is paid at startup, not on the caller's first question.
- **Mic queue drops the oldest frame** instead of raising `QueueFull`.
- **VAD loop starts before the greeting**, so frames are consumed during it and
  the caller can barge in on the greeting.
