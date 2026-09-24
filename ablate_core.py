"""TEMPORARY A/B harness: run replay.py with a truncated _VOICE_RULES.

Tests the hypothesis that rules added across runs 16-19 (which each chased one
defect seen in one live call) overload a 3B model and made quality worse, not
better.

    python ablate_core.py --keep-through 8 --reps 6 --quiet
    python ablate_core.py --keep-through 9 --reps 6 --quiet

--keep-through N keeps voice rules 1..N and drops the rest. Omit it to run the
full rule set (the control arm).

Delete once the question is settled.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import voice_agent.prompts as P
import replay

_orig = P._VOICE_RULES

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--keep-through", type=int, default=None,
                help="keep rules 1..N, drop the rest (default: keep all)")
known, rest = ap.parse_known_args()
# replay.main() runs argparse over sys.argv too, and would choke on
# --keep-through. Hand it only the args it knows.
sys.argv = [sys.argv[0], *rest]

if known.keep_through is None:
    print(f"[ablate] CONTROL: full _VOICE_RULES ({len(_orig)} chars)")
else:
    n = known.keep_through
    marker = f"\n{n + 1}. "
    assert marker in _orig, f"marker {marker!r} not found -- rule text changed"
    core = _orig.split(marker)[0].rstrip()
    assert len(core) < len(_orig)
    print(f"[ablate] KEEP 1-{n}: _VOICE_RULES {len(_orig)} -> {len(core)} chars")
    P._VOICE_RULES = core

sys.exit(asyncio.run(replay.main()))
