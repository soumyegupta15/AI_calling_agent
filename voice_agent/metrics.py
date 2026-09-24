"""Per-turn latency accounting against the brief's section 5 budget."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# (label, attribute, target_low_ms, target_high_ms) -- straight from section 5.
# Labels were hardcoded to distil-Whisper/Qwen2.5 and kept saying so long after
# the engines had been swapped for Parakeet and Qwen3, which made the benchmark
# output actively misleading. They are set from config at startup now.
BUDGET = [
    ("VAD end-of-speech", "vad_ms", 300, 500),
    ("STT transcript", "stt_ms", 50, 400),
    ("LLM first token", "llm_ttft_ms", 150, 400),
    ("TTS first audio", "tts_ms", 150, 550),
]

_LABELS = {"stt": "STT transcript", "llm": "LLM first token", "tts": "TTS first audio"}


def set_engine_labels(stt: str = "", llm: str = "", tts: str = "") -> None:
    """Name the stages after whatever is actually loaded."""
    if stt:
        _LABELS["stt"] = f"{stt} transcript"
    if llm:
        _LABELS["llm"] = f"{llm} first token"
    if tts:
        _LABELS["tts"] = f"{tts} first audio"
    BUDGET[1] = (_LABELS["stt"], "stt_ms", 50, 400)
    BUDGET[2] = (_LABELS["llm"], "llm_ttft_ms", 150, 400)
    # 550, not 300. Measured floor for Piper here is ~450 ms (token arrival +
    # ~130 ms synthesis + executor/queue handoff); a 300 ms ceiling flagged
    # healthy turns red on every single call.
    BUDGET[3] = (_LABELS["tts"], "tts_ms", 150, 550)


@dataclass
class TurnMetrics:
    turn_id: int = 0
    speech_start_t: float = 0.0
    speech_end_t: float = 0.0
    stt_done_t: float = 0.0
    llm_first_token_t: float = 0.0
    llm_done_t: float = 0.0
    first_audio_ready_t: float = 0.0
    playback_start_t: float = 0.0

    user_audio_ms: float = 0.0
    stt_cached: bool = False
    llm_tokens: int = 0
    llm_prompt_tokens: int = 0
    interrupted: bool = False
    # True when the first reply was rejected by an output guard and regenerated.
    # Worth surfacing: a retry roughly doubles the turn's LLM cost, so a run with
    # many retries is slow for a reason that is not the hardware.
    retried: bool = False
    chunks: int = 0
    transcript: str = ""
    response: str = ""
    # Silence the VAD waited through before declaring end-of-turn. Fixed by
    # config (min_silence_ms); reported because it is part of what the caller
    # perceives, even though it is not "work" the pipeline is doing.
    _vad_hold_ms: float = 0.0

    # ------------------------------------------------------------ derived view
    @property
    def vad_ms(self) -> float:
        return self._vad_hold_ms

    @property
    def stt_ms(self) -> float:
        return self._ms(self.speech_end_t, self.stt_done_t)

    @property
    def llm_ttft_ms(self) -> float:
        return self._ms(self.stt_done_t, self.llm_first_token_t)

    @property
    def tts_ms(self) -> float:
        return self._ms(self.llm_first_token_t, self.first_audio_ready_t)

    @property
    def time_to_first_audio_ms(self) -> float:
        """The number in the definition of done: end-of-speech -> audio out."""
        return self._ms(self.speech_end_t, self.playback_start_t or self.first_audio_ready_t)

    @property
    def total_perceived_ms(self) -> float:
        """Including the VAD silence hold, i.e. what the caller actually waits."""
        return self._vad_hold_ms + self.time_to_first_audio_ms

    @staticmethod
    def _ms(a: float, b: float) -> float:
        return (b - a) * 1000.0 if (a and b and b >= a) else 0.0

    # ----------------------------------------------------------------- display
    def render(self) -> str:
        rows = [
            f"  turn {self.turn_id}  |  {self.user_audio_ms/1000:.1f}s of speech"
            f"  |  {self.llm_prompt_tokens} prompt tok  |  {self.llm_tokens} gen tok"
            f"  |  {self.chunks} audio chunk(s)"
        ]
        rows.append("  " + "-" * 62)
        for label, attr, lo, hi in BUDGET:
            value = getattr(self, attr)
            if attr == "vad_ms":
                value = self._vad_hold_ms
            mark = "ok " if value <= hi else "OVER"
            note = " (cached partial)" if attr == "stt_ms" and self.stt_cached else ""
            rows.append(f"  {label:<24} {value:7.0f} ms   target {lo}-{hi:<5} [{mark}]{note}")
        rows.append("  " + "-" * 62)
        ttfa = self.time_to_first_audio_ms
        mark = "ok " if ttfa <= 1300 else "OVER"
        rows.append(f"  {'TIME TO FIRST AUDIO':<24} {ttfa:7.0f} ms   target <1300  [{mark}]")
        rows.append(f"  {'incl. VAD hold':<24} {self.total_perceived_ms:7.0f} ms")
        if self.interrupted:
            rows.append("  (turn was interrupted by the caller)")
        if self.retried:
            rows.append("  (first reply was rejected by a guard and regenerated)")
        return "\n".join(rows)


class MetricsLog:
    """Aggregates turns so `benchmark.py` and the CLI can report averages."""

    def __init__(self) -> None:
        self.turns: list[TurnMetrics] = []

    def add(self, m: TurnMetrics) -> None:
        self.turns.append(m)

    def summary(self) -> str:
        completed = [t for t in self.turns if t.time_to_first_audio_ms > 0]
        if not completed:
            return "No completed turns recorded."

        def avg(attr: str) -> float:
            return sum(getattr(t, attr) for t in completed) / len(completed)

        def pctl(attr: str, p: float) -> float:
            values = sorted(getattr(t, attr) for t in completed)
            idx = min(len(values) - 1, int(round(p * (len(values) - 1))))
            return values[idx]

        lines = [
            "",
            f"=== Summary over {len(completed)} turn(s) ===",
            f"  {_LABELS['stt']:<22} avg {avg('stt_ms'):6.0f} ms   p90 {pctl('stt_ms', 0.9):6.0f} ms",
            f"  {_LABELS['llm']:<22} avg {avg('llm_ttft_ms'):6.0f} ms   p90 {pctl('llm_ttft_ms', 0.9):6.0f} ms",
            f"  {_LABELS['tts']:<22} avg {avg('tts_ms'):6.0f} ms   p90 {pctl('tts_ms', 0.9):6.0f} ms",
            f"  TIME TO FIRST AUDIO    avg {avg('time_to_first_audio_ms'):6.0f} ms"
            f"   p90 {pctl('time_to_first_audio_ms', 0.9):6.0f} ms",
        ]
        passing = avg("time_to_first_audio_ms") <= 1300
        lines.append(f"  Definition of done (<1300 ms avg): {'PASS' if passing else 'FAIL'}")
        cached = sum(1 for t in completed if t.stt_cached)
        lines.append(f"  Turns served from a stable partial transcript: {cached}/{len(completed)}")
        retried = sum(1 for t in completed if t.retried)
        if retried:
            lines.append(
                f"  Turns regenerated after a guard rejection: {retried}/{len(completed)}"
                "  (each roughly doubles that turn's LLM time)"
            )
        return "\n".join(lines)


def now() -> float:
    return time.perf_counter()
