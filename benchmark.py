"""Latency harness for the definition of done (brief section 8).

No microphone required. Test utterances are spoken by Piper, resampled to
16 kHz, and fed back through Silero + distil-Whisper -- so the STT stage is
measured on real audio rather than a synthetic tone.

Measures, per utterance:
    distil-Whisper transcript target 50-400 ms
    Qwen2.5 time-to-first-tok target 150-400 ms
    Piper time-to-first-audio  target 150-300 ms
    TIME TO FIRST AUDIO       target      <1300 ms

Plus a standalone barge-in test (target < 200 ms) and a footprint check.

    python benchmark.py
    python benchmark.py --turns 10
    python benchmark.py --no-audio-roundtrip   # skip Piper-generated test audio
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.config import load_config, model_path  # noqa: E402
from voice_agent.llm import GenerationStats, build_llm  # noqa: E402
from voice_agent.stt import build_stt  # noqa: E402
from voice_agent.text_chunker import SentenceChunker  # noqa: E402
from voice_agent.tts import build_tts  # noqa: E402
from voice_agent.vad import SileroVAD  # noqa: E402

# Ten utterances from the CANDIDATE's side of an outbound call, because that is
# the call the agent actually places. They cover the opening confusion, the
# screening questions, the standard objections, and -- deliberately -- the two
# input shapes that broke the pipeline in live testing: a sentence that trails
# off on a connector, and a bare backchannel.
TEST_UTTERANCES = [
    "Sorry, who is this and where did you get my number?",
    "Yeah I'm a data engineer, mostly Python and Spark on AWS.",
    "I'm working right now, so I'd have a sixty day notice period.",
    "What's the rate on this one?",
    "Honestly that's lower than what I'm on at the moment.",
    "Is it fully remote or do I have to come into the office?",
    "I'm on an H1B, does that work for this client?",
    "I've already got two other interviews lined up this week.",
    # Trails off mid-thought: must be held, not answered as a complete turn.
    "Uh so I have about two years of experience and I have worked with, so",
    # Bare backchannel: must not be treated as a turn at all.
    "Mm-hmm.",
]

BARGE_IN_TARGET_MS = 200.0
TTFA_TARGET_MS = 1300.0


def resample_to_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Linear resample to 16 kHz int16. Good enough for a latency harness."""
    if src_rate == 16000:
        return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    duration = len(samples) / float(src_rate)
    n_out = int(duration * 16000)
    x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    out = np.interp(x_new, x_old, samples)
    return np.clip(out * 32767.0, -32768, 32767).astype(np.int16)


def fmt(label: str, values: list[float], lo: float, hi: float) -> str:
    if not values:
        return f"  {label:<26} (no samples)"
    avg = statistics.mean(values)
    p90 = sorted(values)[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    mark = "PASS" if avg <= hi else "OVER"
    return f"  {label:<26} avg {avg:6.0f} ms   p90 {p90:6.0f} ms   target {lo:.0f}-{hi:.0f}  [{mark}]"


def bench_barge_in(cfg) -> float:
    """Measure how long after interrupt() the output stream stops emitting audio."""
    from voice_agent.audio_io import SpeakerStream

    sr = 24000
    speaker = SpeakerStream(sample_rate=sr, blocksize=int(cfg.audio["output_blocksize"]))
    try:
        speaker.start()
    except Exception as exc:
        print(f"  (no output device available: {exc})")
        return float("nan")

    # Queue 5 s of quiet tone, let it start, then cut it.
    tone = 0.05 * np.sin(2 * np.pi * 220 * np.arange(5 * sr) / sr).astype(np.float32)
    speaker.play(tone)
    time.sleep(0.25)

    started = time.perf_counter()
    speaker.interrupt()
    while speaker.is_speaking and (time.perf_counter() - started) < 1.0:
        time.sleep(0.001)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    speaker.stop()
    return elapsed_ms


def footprint(cfg) -> tuple[float, bool]:
    models_dir = Path(cfg.paths.models_dir)
    total = sum(p.stat().st_size for p in models_dir.glob("*") if p.is_file())
    gb = total / 1024**3
    return gb, gb <= 2.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--turns", type=int, default=len(TEST_UTTERANCES))
    parser.add_argument("--stt", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-audio-roundtrip", action="store_true",
                        help="skip Piper-generated test audio; STT stage is not measured")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stt:
        cfg["stt"]["model_name"] = args.stt
    if args.threads:
        cfg["llm"]["n_threads"] = args.threads

    cores = cfg.runtime["physical_cores"]
    print("=" * 74)
    print("  Local CPU voice agent - latency benchmark")
    print(f"  physical cores {cores}   llama.cpp n_threads {cfg.llm['n_threads']}"
          f"   stt threads {cfg.runtime['stt_threads']}")
    print("=" * 74)

    print("\nLoading models...")
    t0 = time.perf_counter()
    vad = SileroVAD(model_path(cfg, cfg.vad["model_file"]), num_threads=1)
    stt = build_stt(cfg.stt, Path(cfg.paths.models_dir),
                    num_threads=cfg.runtime["stt_threads"])
    llm = build_llm(cfg.llm, model_path(cfg, cfg.llm["model_file"]), call=dict(cfg.get("call", {}) or {}))
    tts = build_tts(cfg.tts, Path(cfg.paths.models_dir), num_threads=cfg.runtime["tts_threads"])
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print("Warming up...")
    llm.warmup()
    tts.warmup()
    stt.transcribe(np.zeros(16000, dtype=np.int16))

    utterances = (TEST_UTTERANCES * 3)[: args.turns]

    stt_ms: list[float] = []
    ttft_ms: list[float] = []
    tts_ms: list[float] = []
    ttfa_ms: list[float] = []
    vad_frame_ms: list[float] = []
    response_lengths: list[int] = []

    print(f"\nRunning {len(utterances)} turn(s)...\n")
    for i, text in enumerate(utterances, 1):
        llm.reset()

        # ---- generate test audio and measure the STT stage -----------------
        transcript_text = text
        if not args.no_audio_roundtrip:
            speech = tts.synthesize(text)
            audio16 = resample_to_16k(speech.samples, speech.sample_rate)

            # VAD cost per frame (should be well under a millisecond).
            vad.reset()
            frames = len(audio16) // 512
            v0 = time.perf_counter()
            for f in range(frames):
                vad(audio16[f * 512 : (f + 1) * 512])
            if frames:
                vad_frame_ms.append((time.perf_counter() - v0) * 1000.0 / frames)

            result = stt.transcribe(audio16)
            stt_ms.append(result.latency_ms)
            transcript_text = result.text or text

        # ---- LLM + TTS, measured the way the orchestrator overlaps them ----
        llm.add_user(transcript_text)
        chunker = SentenceChunker(
            first_chunk_min_chars=int(cfg.tts["first_chunk_min_chars"]),
            chunk_min_chars=int(cfg.tts["chunk_min_chars"]),
            chunk_max_chars=int(cfg.tts["chunk_max_chars"]),
        )
        stats = GenerationStats()
        stop = threading.Event()

        turn_start = time.perf_counter()
        first_chunk: str | None = None
        full = []
        for delta in llm.stream(stop_event=stop, stats=stats):
            full.append(delta)
            chunks = chunker.push(delta)
            if chunks and first_chunk is None:
                first_chunk = chunks[0]
                stop.set()  # we only need time-to-first-audio; stop generating
                break
        if first_chunk is None:
            first_chunk = chunker.flush() or "".join(full).strip() or "Okay."

        first_chunk_ready = time.perf_counter()
        speech = tts.synthesize(first_chunk)
        first_audio = time.perf_counter()

        ttft_ms.append(stats.ttft_ms)
        tts_ms.append((first_audio - first_chunk_ready) * 1000.0)
        ttfa_ms.append((first_audio - turn_start) * 1000.0)
        response_lengths.append(len(first_chunk))

        print(f"  {i:2d}. \"{text[:52]}{'...' if len(text) > 52 else ''}\"")
        print(f"      -> \"{first_chunk[:62]}{'...' if len(first_chunk) > 62 else ''}\"")
        print(f"      stt {stt_ms[-1] if stt_ms else 0:5.0f}  ttft {stats.ttft_ms:5.0f}"
              f"  tts {tts_ms[-1]:5.0f}  |  first audio {ttfa_ms[-1]:6.0f} ms"
              f"  ({stats.tokens_per_second:.1f} tok/s)")

    # ---- barge-in ------------------------------------------------------
    print("\nMeasuring barge-in stop latency...")
    barge_ms = bench_barge_in(cfg)

    gb, under_limit = footprint(cfg)

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 74)
    print("  RESULTS")
    print("=" * 74)
    if vad_frame_ms:
        print(f"  {'Silero per 32ms frame':<26} avg {statistics.mean(vad_frame_ms):6.2f} ms")
    print(f"  {'VAD end-of-speech hold':<26} {float(cfg.vad['min_silence_ms']):6.0f} ms"
          f"   (fixed by config)      target 300-500")
    if stt_ms:
        print(fmt("distil-Whisper", stt_ms, 50, 400))
    else:
        print("  distil-Whisper             (skipped: --no-audio-roundtrip)")
    print(fmt("Qwen2.5 first token", ttft_ms, 150, 400))
    print(fmt("Piper first audio", tts_ms, 150, 300))
    print("  " + "-" * 70)
    print(fmt("TIME TO FIRST AUDIO", ttfa_ms, 0, TTFA_TARGET_MS))
    print("  " + "-" * 70)

    print("\n  Definition of done")
    checks = [
        (f"avg time-to-first-audio < {TTFA_TARGET_MS:.0f} ms",
         statistics.mean(ttfa_ms) <= TTFA_TARGET_MS),
        (f"barge-in stops playback < {BARGE_IN_TARGET_MS:.0f} ms  ({barge_ms:.0f} ms)",
         barge_ms == barge_ms and barge_ms < BARGE_IN_TARGET_MS),
        (f"model footprint <= 2.5 GB  ({gb:.2f} GB)", under_limit),
        (f"first chunk stays short  (avg {statistics.mean(response_lengths):.0f} chars)",
         statistics.mean(response_lengths) < 220),
    ]
    for label, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}]  {label}")

    print("\n  Note: the two definition-of-done items this harness cannot check are")
    print("  response quality (1-3 sentences, objection handling) and live barge-in")
    print("  against a real microphone. Run `python run_agent.py` for those.")

    try:
        llm.close()
    except Exception:
        pass
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
