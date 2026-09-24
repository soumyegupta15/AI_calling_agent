"""Isolate where Piper's first-audio latency actually goes.

The agent reports "Piper first audio" of ~770 ms per turn, but Piper's measured
realtime factor is ~0.15 -- so raw synthesis should not be the cost. This
measures the real path (voice_agent.tts.PiperTTS.synthesize) to find out
whether the 770 ms is synthesis, or time spent waiting for the chunker to
accumulate enough text before it synthesizes at all.

Run:  python scripts/bench_tts.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent.tts import PiperTTS  # noqa: E402

# Realistic first chunks, post-normalization.
CHUNKS = [
    "Got it, thanks.",
    "Sure, that makes sense.",
    "Great, and how many years?",
    "Okay. Twelve months contract to hire.",
]


def median_ms(fn, n: int) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> int:
    model = ROOT / "models" / "en_US-amy-medium.onnx"
    print(f"model: {model.name}\n")

    t0 = time.perf_counter()
    tts = PiperTTS(model, {"model_file": "en_US-amy-medium.onnx"}, num_threads=2)
    print(f"PiperTTS.__init__ (voice load)  {(time.perf_counter() - t0) * 1000:8.1f} ms")

    w = tts.warmup()
    print(f"warmup()                       {w:8.1f} ms\n")

    print("per-chunk synthesis (this is the number that matters):")
    for text in CHUNKS:
        cold = median_ms(lambda: tts.synthesize(text), 1)
        warm = median_ms(lambda: tts.synthesize(text), 7)
        speech = tts.synthesize(text)
        dur = len(speech.samples) / speech.sample_rate if speech else 0.0
        rtf = (warm / 1000.0) / dur if dur > 0 else 0.0
        print(f"  {text!r:42s} cold {cold:7.1f} ms   warm {warm:7.1f} ms   "
              f"audio {dur:5.2f}s   rtf {rtf:.3f}")

    long_text = " ".join(CHUNKS * 12)
    t0 = time.perf_counter()
    speech = tts.synthesize(long_text)
    bulk_ms = (time.perf_counter() - t0) * 1000.0
    audio_s = len(speech.samples) / speech.sample_rate
    print(f"\nbulk: {audio_s:.2f}s audio in {bulk_ms:.1f} ms -> rtf {bulk_ms / 1000 / audio_s:.3f}")

    print("\nInterpretation: if warm per-chunk is well under ~150 ms, the ~770 ms")
    print("'Piper first audio' is NOT synthesis -- it is time spent waiting for")
    print("the sentence chunker to accumulate first_chunk_min_chars of LLM tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
