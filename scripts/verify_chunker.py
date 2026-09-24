"""Prove the first-chunk fix against real replies from the agent's own logs.

Streams each reply token-by-token at a realistic CPU decode rate and reports
when the first chunk became speakable, plus how many chunks the reply split
into. Runs the same text through the OLD chunker behaviour (spaced dash is not
a break, no first-chunk deadline) side by side.

Run:  python scripts/verify_chunker.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent import text_chunker as tc  # noqa: E402

# Verbatim from the run logs -- note the " - " joins and the missing periods.
REPLIES = [
    "That sounds like a challenging requirement - 12 months contract-to-hire "
    "typically includes some training period - Years of hands-on experience "
    "with Python Spark AWS and production pipeline development?",
    "Sounds good I'm glad I could help Thanks again for taking some time today "
    "have a wonderful evening",
    "I am physically present here in Bengaluru - Do you require relocation "
    "assistance for hiring?",
    "Four years is quite impressive - How many years of hands-on data "
    "engineering experience do you think would be ideal for such a role?",
    "Good, that fits. So you have owned pipelines end to end?",
]

TOKENS_PER_SEC = 25.0
CHARS_PER_TOKEN = 4
DEADLINE_MS = 280
NEVER = re.compile(r"(?!x)x")   # matches nothing


def tokens(text: str) -> list[str]:
    n = CHARS_PER_TOKEN
    return [text[i : i + n] for i in range(0, len(text), n)]


def run(text: str, old: bool) -> tuple[float, int, list[str]]:
    saved = tc._DASH_BREAK
    if old:
        tc._DASH_BREAK = NEVER          # pre-fix: spaced dash is not a break
    try:
        c = tc.SentenceChunker(
            first_chunk_min_chars=12,
            chunk_min_chars=55,
            chunk_max_chars=140,
            max_sentences=2,
            first_chunk_max_ms=0 if old else DEADLINE_MS,
        )
        delay = 1.0 / TOKENS_PER_SEC
        t0 = time.perf_counter()
        first_ms = None
        chunks: list[str] = []

        for i, tok in enumerate(tokens(text)):
            for g in c.push(tok):
                chunks.append(g)
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000.0
            if c.done:
                break
            target = t0 + (i + 1) * delay
            while time.perf_counter() < target:
                time.sleep(0.0002)

        if not c.done:
            tail = c.flush()
            if tail:
                chunks.append(tail)
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000.0
        return (first_ms or 0.0), len(chunks), chunks
    finally:
        tc._DASH_BREAK = saved


def main() -> int:
    print(f"decode {TOKENS_PER_SEC:.0f} tok/s, first-chunk deadline {DEADLINE_MS} ms\n")
    print(f"{'reply':40s} {'OLD':>9s} {'NEW':>9s} {'OLD n':>6s} {'NEW n':>6s}")
    print("-" * 76)

    tot_old = tot_new = 0.0
    for text in REPLIES:
        o_ms, o_n, _ = run(text, old=True)
        n_ms, n_n, _ = run(text, old=False)
        tot_old += o_ms
        tot_new += n_ms
        label = (text[:38] + "..") if len(text) > 40 else text
        print(f"{label:40s} {o_ms:7.0f}ms {n_ms:7.0f}ms {o_n:6d} {n_n:6d}")

    n = len(REPLIES)
    print("-" * 76)
    print(f"{'average':40s} {tot_old / n:7.0f}ms {tot_new / n:7.0f}ms")

    print("\nsample split, new behaviour:")
    for i, ch in enumerate(run(REPLIES[0], old=False)[2], 1):
        print(f"  {i}. {ch}")

    print("\n'OLD n' of 1 = the whole reply arrived as one chunk: streaming TTS")
    print("was effectively off and first audio waited for full generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
