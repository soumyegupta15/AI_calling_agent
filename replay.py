"""Score conversation quality OFFLINE, from a scripted call.

Why this exists: a live call is one sample. A 3B model at temperature 0.4
fails differently every time, so judging a prompt change from a single call
reads noise as signal -- which is how several "fixes" here made things worse
while appearing to work. This harness drives the REAL turn path
(VoiceAgent.scripted_turn: same prompt, same agenda note, guard on, no mic and
no speaker) over a fixed script, N times, and counts rule violations.

Run it after any prompt or guard change:

    python replay.py                 # 3 reps of the standard script
    python replay.py --reps 5
    python replay.py --quiet         # summary table only

Exit code is 1 when the violation rate exceeds --fail-over, so it can gate a
change instead of requiring a human to place a call and read the log.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

from voice_agent.config import load_config
from voice_agent.orchestrator import (
    VoiceAgent,
    _echoes_agenda,
    _invents_employer,
    _invents_specifics,
    _parrots,
    _raises_deflected_topic,
    _sounds_like_a_chatbot,
    _unanswered_question,
    _wrong_person,
)
from voice_agent.prompts import AgendaTracker

# A scripted screening call. Deliberately includes the inputs that broke real
# runs: a short reply ("Yes."), an out-of-scope ask (salary), a pure
# off-topic question, an inappropriate remark, and a closing.
SCRIPT = [
    "Oh yes, I'm free.",
    "I have done bachelor's in computer science.",
    "I have five years of work experience.",
    "Mostly Python, Java, SQL and JavaScript.",
    "Yes.",
    "Do you have any openings in AI engineer?",
    "What's the capital of France?",
    "Can you tell me about the salary?",
    "Do you think I look good?",
    "No thank you.",
]

# Things the prompt forbids that no structural detector covers.
_CAPABILITY_SPEAK = (
    "i cannot", "i can't", "i'm not able", "i am not able", "i'm unable",
    "i am unable", "no further assistance", "i don't have any knowledge",
)
_SYSTEM_LEAK = (
    "our system", "system flagged", "flagged an unclear", "unclear response",
    "the system says",
)


class _NullSpeaker:
    """Swallows audio. pending_seconds stays 0 so backpressure never sleeps."""

    sample_rate = 22050
    is_speaking = False
    pending_seconds = 0.0

    def play(self, samples) -> None:  # noqa: ANN001
        return None

    def interrupt(self) -> int:
        return 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def score(reply: str, caller_said: str, corpus: str, agenda: list[str]) -> list[str]:
    """Name every rule this reply breaks. Empty list == clean."""
    if not reply.strip():
        return ["empty"]
    hits: list[str] = []
    low = reply.lower()

    if _parrots(reply, caller_said):
        hits.append("parrot")
    if _invents_employer(reply):
        hits.append("fabrication")
    if _invents_specifics(reply, corpus):
        hits.append("fabrication")
    if _raises_deflected_topic(reply, corpus):
        hits.append("off_topic")
    if _unanswered_question(reply, caller_said):
        hits.append("unanswered")
    if _echoes_agenda(reply, agenda):
        hits.append("agenda_echo")
    if _wrong_person(reply):
        hits.append("wrong_person")
    if _sounds_like_a_chatbot(reply):
        hits.append("assistant_speak")
    if any(p in low for p in _CAPABILITY_SPEAK):
        hits.append("capability_speak")
    if any(p in low for p in _SYSTEM_LEAK):
        hits.append("system_leak")
    if "?." in reply or ".?" in reply:
        hits.append("double_terminator")
    # A bare yes/no is a dead end on a call: rule 13.
    if low.strip(" .?!") in ("yes", "no", "yeah", "okay", "ok"):
        hits.append("bare_answer")
    return sorted(set(hits))


async def run_once(agent: VoiceAgent, verbose: bool) -> tuple[list[list[str]], list[float], list[tuple[str, str, list[str]]]]:
    agent.llm.reset()
    agent.agenda = AgendaTracker(agent._agenda)

    per_turn: list[list[str]] = []
    latencies: list[float] = []
    offenders: list[tuple[str, str, list[str]]] = []
    corpus = ""
    for line in SCRIPT:
        started = time.perf_counter()
        m = await agent.scripted_turn(line)
        latencies.append((time.perf_counter() - started) * 1000.0)
        reply = (m.response or "").strip()
        hits = score(reply, line, corpus, agent._agenda)
        per_turn.append(hits)
        if hits:
            offenders.append((line, reply, hits))
        corpus = f"{corpus} {line}".strip()
        if verbose:
            flag = f"  <== {', '.join(hits)}" if hits else ""
            print(f"\n  You : {line}")
            print(f"  Riya: {reply}{flag}")
    return per_turn, latencies, offenders


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3, help="repetitions (default 3)")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument("--show-bad", action="store_true",
                    help="print every failing turn with its transcript")
    ap.add_argument("--fail-over", type=float, default=1.0,
                    help="exit 1 if violations per turn exceed this")
    args = ap.parse_args()

    print(f"Replaying a scripted screening call x{args.reps} "
          f"({len(SCRIPT)} turns each)...")
    agent = VoiceAgent(load_config(), on_event=lambda k, t: None)
    # __init__ only wires config; load() is what builds VAD/STT/LLM/TTS. It
    # also builds the real SpeakerStream, so the null speaker replaces it
    # AFTER -- never call speaker.start(), or this opens an audio device.
    agent.load()
    agent.speaker = _NullSpeaker()

    all_turns: list[list[str]] = []
    all_lat: list[float] = []
    all_offenders: list[tuple[str, str, list[str]]] = []
    for rep in range(1, args.reps + 1):
        if not args.quiet:
            print(f"\n===== rep {rep}/{args.reps} =====")
        turns, lat, offenders = await run_once(agent, verbose=not args.quiet)
        all_turns.extend(turns)
        all_lat.extend(lat)
        all_offenders.extend(offenders)

    total_turns = len(all_turns)
    dirty = [t for t in all_turns if t]
    counts: dict[str, int] = {}
    for hits in all_turns:
        for h in hits:
            counts[h] = counts.get(h, 0) + 1

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"Turns scored:      {total_turns}")
    print(f"Clean turns:       {total_turns - len(dirty)}/{total_turns} "
          f"({100 * (total_turns - len(dirty)) / max(1, total_turns):.0f}%)")
    print(f"Violations/turn:   {sum(len(t) for t in all_turns) / max(1, total_turns):.2f}")
    if counts:
        print("\nBy kind (turns affected):")
        for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {kind:<20} {n:>3}  ({100 * n / max(1, total_turns):.0f}%)")
    else:
        print("\nNo violations detected.")

    # Where in the script the damage is. Aggregated over reps, because a single
    # rep says which line failed once, not which INPUT is unreliable.
    print("\nBy script line (violations / reps):")
    for i, line in enumerate(SCRIPT):
        n = sum(1 for j, hits in enumerate(all_turns) if j % len(SCRIPT) == i and hits)
        bar = "#" * n
        print(f"  {n:>2}/{args.reps:<3} {bar:<8} {line[:52]}")

    if args.show_bad and all_offenders:
        print("\n" + "-" * 62)
        print("FAILING TURNS")
        print("-" * 62)
        for caller, reply, hits in all_offenders:
            print(f"\n  [{', '.join(hits)}]")
            print(f"  You : {caller}")
            print(f"  Riya: {reply}")

    print(f"\nTurn wall time:    avg {statistics.mean(all_lat):.0f} ms  "
          f"p90 {sorted(all_lat)[max(0, int(0.9 * len(all_lat)) - 1)]:.0f} ms")

    rate = sum(len(t) for t in all_turns) / max(1, total_turns)
    print(f"\nGate (--fail-over {args.fail_over}): "
          f"{'FAIL' if rate > args.fail_over else 'PASS'}")
    return 1 if rate > args.fail_over else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
