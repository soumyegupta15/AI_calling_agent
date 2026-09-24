"""Tests for orchestration logic that does not need the models loaded:
config resolution, context trimming, VAD turn segmentation, and the
interruptible speaker buffer.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.config import (  # noqa: E402
    hardware_tier,
    load_config,
    physical_core_count,
    plan_threads,
)
from voice_agent.llm import Turn  # noqa: E402
from voice_agent.metrics import MetricsLog, TurnMetrics  # noqa: E402
from voice_agent.prompts import SYSTEM_PROMPT, build_user_message  # noqa: E402
from voice_agent.vad import TurnDetector, VADEvent  # noqa: E402


# --------------------------------------------------------------------------
#  Config
# --------------------------------------------------------------------------

def test_config_loads_and_resolves_threads():
    cfg = load_config()
    assert cfg.llm["n_threads"] >= 1
    assert cfg.llm["n_threads_batch"] >= cfg.llm["n_threads"]
    assert cfg.llm["n_threads"] <= max(2, cfg.runtime["physical_cores"])
    assert Path(cfg.paths.models_dir).is_absolute()


def test_config_matches_the_shipped_stack():
    cfg = load_config()
    # Not the Qwen3 preset this test used to check. Two things moved: the
    # model is Llama-3.2-3B (the very next assertion), and temperature was
    # then lowered 0.6 -> 0.4 because a 3B inventing screening questions was
    # the dominant defect of runs 27-33. At 0.4 it still varies its reactions
    # and fabricates markedly less.
    assert 0.3 <= cfg.llm["temperature"] <= 0.5
    assert 0.9 <= cfg.llm["top_p"] <= 0.95
    assert cfg.llm["repeat_penalty"] == 1.15
    # llama.cpp's 64-token default repeat window is shorter than one of Riya's
    # own replies, so it could not see that she had already asked this exact
    # question two turns earlier.
    assert cfg.llm["repeat_last_n"] >= 256
    # Ollama, not llama-server: same local gguf, but it owns model load/unload
    # and keep_alive, which is what keeps TTFA flat across a call.
    assert cfg.llm["backend"] == "ollama"
    assert "Llama-3.2-3B" in cfg.llm["model_file"]
    assert cfg.tts["engine"] in ("kokoro", "piper")   # "auto" resolves at load
    assert cfg.stt["engine"] in ("nemotron", "parakeet", "vosk", "whisper")
    assert cfg.stt["parakeet"]["quantization"] == "int8"
    assert cfg.stt["whisper"]["model_name"].startswith("Systran/faster-")
    assert cfg.stt["whisper"]["compute_type"] == "int8"
    assert cfg.stt["whisper"]["beam_size"] == 1
    # Encoder window; 30 is stock, lower is faster. Below 6 accuracy collapses.
    assert 6 <= cfg.stt["whisper"]["chunk_length"] <= 30
    assert cfg.stt["vosk"]["model_dir"].startswith("vosk-model")
    # Partials are a Whisper workaround and are off for both engines.
    assert cfg.stt["partial_interval_ms"] == 0
    assert 300 <= cfg.vad["min_silence_ms"] <= 700
    # Raised after the agent was barging in on its own speaker output.
    assert cfg.barge_in["threshold"] >= 0.8


def test_no_reasoning_suppression_is_needed_for_llama():
    """Llama 3.2 has no hybrid-reasoning mode.

    The Qwen3-era flags (--chat-template-kwargs enable_thinking, and
    --reasoning-budget) existed only to stop the agent reading its own scratchpad
    down the phone. They are dead weight here -- and a flag an older llama.cpp
    build rejects will refuse to start the server at all.
    """
    cfg = load_config()
    args = " ".join(str(a) for a in cfg.llm["server"]["extra_args"])
    assert "enable_thinking" not in args
    assert "--reasoning-budget" not in args
    # SentenceChunker still strips reasoning tags if you swap a thinking model in.
    from voice_agent.text_chunker import SentenceChunker

    c = SentenceChunker(18, 55, 200)
    assert "</think>" not in " ".join(c.push("scratch</think>Right, what rate? "))


def test_model_footprint_is_declared_and_bounded():
    """Nemotron 630 + Llama-3.2-3B Q4_K_M 2020 + Kokoro int8 115 + Piper 63
    + Silero 2 = 2830 MB = 2.76 GiB.

    This is over the original 2 GiB target on purpose -- the budget was raised
    to fit a 3B model. Keep the number honest rather than quietly drifting.
    """
    budget_mb = {
        "nemotron_int8": 630,
        "llama_3.2_3b_q4km": 2020,
        "kokoro_int8": 115,
        "piper_fallback": 63,
        "silero": 2,
    }
    total_mb = sum(budget_mb.values())
    assert total_mb / 1024 <= 3.0, f"{total_mb} MB is over the 3 GiB ceiling"


@pytest.mark.parametrize("cores", [1, 2, 3, 4, 6, 8, 12, 16, 32, 64])
def test_thread_plan_never_oversubscribes_the_contended_window(cores):
    """Only ONE window has two stages running at once: LLM generation + TTS.

    The old policy handed each stage its own maximum -- llama.cpp `cores`, STT
    `cores`, TTS `cores//2` -- which promised 5 threads on a 2-core laptop.
    Oversubscription does not fail loudly, it just makes everything slower and
    jittery, and it is why Piper reported 800-1400 ms for a chunk that should
    take ~120 ms.

    Below 3 cores there is nothing to divide, so a small overlap is accepted
    deliberately; above that the budget must hold.
    """
    plan = plan_threads(cores)
    if cores >= 3:
        assert plan["llm_threads"] + plan["tts_threads"] <= cores, plan
    assert all(v >= 1 for v in plan.values()), plan


@pytest.mark.parametrize("cores", [1, 2, 4, 8, 16, 64])
def test_thread_plan_scales_with_the_machine(cores):
    """The point of the profile: same config, any laptop."""
    plan = plan_threads(cores)
    # Prompt ingest is never contended, so it may use the whole machine.
    assert plan["llm_batch_threads"] >= plan["llm_threads"]
    assert plan["llm_batch_threads"] <= max(cores, 1)
    # llama.cpp stops scaling on small models past ~8 threads.
    assert plan["llm_threads"] <= 8
    assert plan["vad_threads"] == 1


def test_more_cores_never_means_fewer_threads():
    previous = plan_threads(1)
    for cores in range(2, 33):
        plan = plan_threads(cores)
        assert plan["llm_threads"] >= previous["llm_threads"] or plan["llm_threads"] == 8
        assert plan["llm_batch_threads"] >= previous["llm_batch_threads"]
        previous = plan


def test_hardware_tiers_are_ordered():
    assert hardware_tier(2, 8) == "minimal"
    assert hardware_tier(4, 16) == "low"
    assert hardware_tier(8, 16) == "mid"
    assert hardware_tier(16, 32) == "high"
    # Thin RAM demotes regardless of core count.
    assert hardware_tier(16, 4) == "minimal"


def test_kokoro_is_only_auto_selected_on_machines_that_can_stream_it():
    """Kokoro is ~20x Piper's compute per second of audio.

    Measured in this project: Piper realtime factor ~0.15, Kokoro ~2.69 on one
    core. Above 1.0 synthesis is slower than speech and the call stalls, so the
    auto default must not hand a 2-core laptop an engine it cannot feed.
    """
    import yaml

    raw = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml",
                              encoding="utf-8"))
    # "auto" is the adaptive default and what a fresh install should get. The
    # shipped file pins piper, which is what this machine actually measured as
    # streamable, so it is not a regression -- only pinning Kokoro would be.
    assert raw["tts"]["engine"] in ("auto", "piper"), (
        "shipped config must not pin an engine this machine cannot feed"
    )
    # And the runtime decision is still measured, not just guessed from cores.
    assert 0 < float(raw["tts"]["max_realtime_factor"]) < 1.0


def test_physical_core_count_is_sane():
    assert 1 <= physical_core_count() <= 512


# --------------------------------------------------------------------------
#  Prompt assembly
# --------------------------------------------------------------------------

def test_system_prompt_has_the_voice_rules():
    assert "Riya" in SYSTEM_PROMPT
    assert "One or two short sentences per turn" in SYSTEM_PROMPT
    assert "One question per turn" in SYSTEM_PROMPT
    assert "No emoji, markdown, bullets" in SYSTEM_PROMPT


def test_context_injection_wraps_the_user_turn():
    msg = build_user_message(
        "Hey, any update on the DevOps role for Acme?",
        "Req 4521 - Senior AWS DevOps Engineer, client Acme Corp, remote US",
    )
    assert msg.startswith("[Relevant context:")
    assert msg.endswith("Hey, any update on the DevOps role for Acme?")


def test_no_context_leaves_the_turn_untouched():
    assert build_user_message("  hello  ") == "hello"


# --------------------------------------------------------------------------
#  Context trimming (exercises _trimmed_history without loading llama.cpp)
# --------------------------------------------------------------------------

class _FakeLLM:
    """Exercises BaseLLM._trimmed_history without constructing a real model."""

    def __init__(self, history, budget=1800, min_kept=4):
        from voice_agent.llm import BaseLLM

        self.history = history
        self.cfg = {"history_token_budget": budget, "min_kept_turns": min_kept}
        self._trimmed_history = BaseLLM._trimmed_history.__get__(self)


def test_history_is_trimmed_to_budget():
    history = [Turn("user" if i % 2 == 0 else "assistant", f"turn {i}", 200) for i in range(20)]
    kept = _FakeLLM(history, budget=1000, min_kept=4)._trimmed_history()
    assert sum(t.tokens for t in kept) <= 1000
    # Most recent turns survive.
    assert kept[-1].content == "turn 19"


def test_history_keeps_a_minimum_even_when_over_budget():
    history = [Turn("user" if i % 2 == 0 else "assistant", f"turn {i}", 5000) for i in range(10)]
    kept = _FakeLLM(history, budget=100, min_kept=4)._trimmed_history()
    assert len(kept) == 4


def test_short_history_is_untouched():
    history = [Turn("user", "hi", 10), Turn("assistant", "hello", 10)]
    assert _FakeLLM(history)._trimmed_history() == history


# --------------------------------------------------------------------------
#  VAD turn segmentation (with a scripted fake probability source)
# --------------------------------------------------------------------------

class _ScriptedVAD:
    """Stands in for SileroVAD; returns probabilities from a list."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def __call__(self, frame):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self):
        self.i = 0


VAD_CFG = {
    "threshold": 0.5,
    "neg_threshold": 0.35,
    "min_silence_ms": 400,
    "min_speech_ms": 200,
    "speech_pad_ms": 200,
    "max_speech_ms": 30000,
}
BARGE_CFG = {"enabled": True, "threshold": 0.7, "min_speech_ms": 180, "cooldown_ms": 120}


def _detector(probs):
    return TurnDetector(_ScriptedVAD(probs), dict(VAD_CFG), dict(BARGE_CFG))


def _run(detector, n):
    frame = np.zeros(512, dtype=np.int16)
    return [detector.process(frame) for _ in range(n)]


def test_preroll_actually_preserves_audio_before_the_trigger():
    """Regression: the leading consonant was being deleted, not padded.

    The pre-roll deque was sized at speech_pad_ms alone. But it takes
    min_speech_ms of *consecutive speech* to fire the trigger, and those frames
    land in the pre-roll first -- so at trigger time the only thing left in a
    200 ms buffer was the 200 ms of speech that had just filled it. Net padding:
    zero. Under the stricter barge-in profile (min_speech_ms 300 > pad 200) the
    buffer was too small to even hold the trigger frames, so the first ~100 ms of
    the word was evicted outright. That is why "AWS" reached the model as "WS"
    and "so" as "C".

    Marker frames make the loss visible: every frame before the trigger carries a
    distinct value, so we can count how many survived into the captured turn.
    """
    lead_silence = 8            # frames of quiet before the caller starts
    detector = _detector([0.0] * lead_silence + [0.9] * 30 + [0.0] * 20)

    captured = None
    for i in range(60):
        # Value-tag each frame so we can identify it later.
        frame = np.full(512, i + 1, dtype=np.int16)
        result = detector.process(frame)
        if result.event is VADEvent.SPEECH_END:
            captured = result.audio
            break

    assert captured is not None
    first_value = int(captured[0])
    onset_value = lead_silence + 1          # frame value where speech begins

    # Nothing after the onset may be the first thing captured.
    assert first_value <= onset_value, (
        f"capture started at frame value {first_value}, after the onset at "
        f"{onset_value}: the start of the word was deleted"
    )

    # And there must be REAL padding in front of the onset. Silero's threshold
    # crossing lags the true acoustic onset, so capturing exactly from the
    # crossing still loses the attack of the first phoneme.
    pad_frames_kept = onset_value - first_value
    expected = detector.pad_frames
    assert pad_frames_kept >= expected - 1, (
        f"only {pad_frames_kept} frame(s) of lead-in survived; "
        f"speech_pad_ms asks for {expected}"
    )


def test_preroll_survives_the_stricter_barge_in_profile():
    """The barge-in trigger needs MORE frames, so the buffer must be bigger."""
    barge = dict(BARGE_CFG, min_speech_ms=300)
    detector = TurnDetector(_ScriptedVAD([0.99] * 60), dict(VAD_CFG), barge)
    barge_frames = detector.barge_min_frames
    assert detector._preroll.maxlen > barge_frames, (
        "pre-roll cannot even hold the frames the barge-in trigger consumes, "
        "so the start of the interrupting word is discarded"
    )


def test_speech_start_then_end_is_detected():
    # 32 ms/frame: 20 frames of speech (~640 ms), then 20 of silence (~640 ms).
    probs = [0.9] * 20 + [0.0] * 20
    events = [r.event for r in _run(_detector(probs), 40)]
    assert VADEvent.SPEECH_START in events
    assert VADEvent.SPEECH_END in events
    assert events.index(VADEvent.SPEECH_START) < events.index(VADEvent.SPEECH_END)


def test_short_blip_does_not_start_a_turn():
    # 200 ms minimum = ~7 frames; 3 frames of noise should be ignored.
    probs = [0.9] * 3 + [0.0] * 20
    events = [r.event for r in _run(_detector(probs), 23)]
    assert VADEvent.SPEECH_START not in events


def test_brief_pause_mid_sentence_does_not_end_the_turn():
    # 400 ms of silence is needed to end a turn = ~13 frames at 32 ms/frame.
    # The 5-frame (160 ms) pause in the middle must NOT end it; the 25-frame
    # run of silence afterwards must.
    probs = [0.9] * 15 + [0.1] * 5 + [0.9] * 15 + [0.0] * 25
    detector = _detector(probs)
    results = _run(detector, 60)
    ends = [i for i, r in enumerate(results) if r.event is VADEvent.SPEECH_END]
    assert ends, "the turn never ended"
    assert ends[0] > 35, "the turn ended during the mid-sentence pause"


def test_barge_in_needs_a_higher_probability_while_the_agent_speaks():
    # 0.6 clears the idle threshold (0.5) but not the barge-in threshold (0.7).
    detector = _detector([0.6] * 30)
    detector.agent_speaking = True
    events = [r.event for r in _run(detector, 30)]
    assert VADEvent.SPEECH_START not in events

    detector2 = _detector([0.85] * 30)
    detector2.agent_speaking = True
    events2 = [r.event for r in _run(detector2, 30)]
    assert VADEvent.SPEECH_START in events2


def test_cooldown_suppresses_the_speaker_ring_out():
    detector = _detector([0.95] * 40)
    detector.start_cooldown()
    events = [r.event for r in _run(detector, 3)]
    assert all(e is VADEvent.NONE for e in events)


def test_max_speech_forces_a_timeout():
    cfg = dict(VAD_CFG, max_speech_ms=640)  # ~20 frames
    detector = TurnDetector(_ScriptedVAD([0.95] * 200), cfg, dict(BARGE_CFG))
    events = [r.event for r in _run(detector, 60)]
    assert VADEvent.SPEECH_TIMEOUT in events


# --------------------------------------------------------------------------
#  Silero v5 input contract
#
#  Regression: the ONNX graph declares input as [None, None], so feeding it a
#  bare 512-sample frame runs without error and returns ~0.0 for even the
#  clearest speech. v5 requires 64 samples of context prepended (576 total).
#  This silently broke ALL speech detection; nothing else in the suite caught it
#  because every other VAD test uses scripted probabilities.
# --------------------------------------------------------------------------

class _RecordingSession:
    """Stands in for onnxruntime.InferenceSession, capturing input shapes."""

    class _IO:
        def __init__(self, name):
            self.name = name

    def __init__(self, v5=True):
        self._v5 = v5
        self.seen_lengths = []
        self.seen_inputs = []

    def get_inputs(self):
        names = ["input", "state", "sr"] if self._v5 else ["input", "h", "c", "sr"]
        return [self._IO(n) for n in names]

    def run(self, _outputs, feed):
        x = feed["input"]
        self.seen_lengths.append(x.shape[1])
        self.seen_inputs.append(x.copy())
        out = np.array([[0.9]], dtype=np.float32)
        if self._v5:
            return out, np.zeros((2, 1, 128), dtype=np.float32)
        return out, np.zeros((2, 1, 64), np.float32), np.zeros((2, 1, 64), np.float32)


def _vad_with_session(session):
    from voice_agent.vad import SileroVAD

    vad = SileroVAD.__new__(SileroVAD)          # bypass file loading
    vad.session = session
    vad.sample_rate = 16000
    vad._input_names = {i.name for i in session.get_inputs()}
    vad._v5 = "state" in vad._input_names
    vad._context_samples = 64 if vad._v5 else 0
    vad._sr_tensor = np.array(16000, dtype=np.int64)
    vad.reset()
    return vad


def test_silero_v5_receives_512_plus_64_context_samples():
    session = _RecordingSession(v5=True)
    vad = _vad_with_session(session)
    for _ in range(5):
        vad(np.full(512, 1000, dtype=np.int16))

    assert session.seen_lengths == [576] * 5, (
        "Silero v5 must be fed 64 context samples + a 512-sample frame; "
        "a bare 512 frame returns ~0.0 for all speech"
    )


def test_silero_v5_context_carries_the_previous_frames_tail():
    session = _RecordingSession(v5=True)
    vad = _vad_with_session(session)

    vad(np.full(512, 1000, dtype=np.int16))
    vad(np.full(512, 2000, dtype=np.int16))

    first, second = session.seen_inputs
    # Frame 1 starts from a zero context.
    assert np.allclose(first[0, :64], 0.0)
    # Frame 2's context is the tail of frame 1's combined input.
    assert np.allclose(second[0, :64], first[0, -64:])
    assert np.allclose(second[0, 64:], 2000 / 32768.0)


def test_silero_reset_clears_the_context():
    session = _RecordingSession(v5=True)
    vad = _vad_with_session(session)
    vad(np.full(512, 3000, dtype=np.int16))
    vad.reset()
    vad(np.full(512, 3000, dtype=np.int16))
    assert np.allclose(session.seen_inputs[-1][0, :64], 0.0)


def test_silero_v4_models_get_no_context():
    """Older v4 exports take a bare frame; only v5 needs the context."""
    session = _RecordingSession(v5=False)
    vad = _vad_with_session(session)
    vad(np.full(512, 1000, dtype=np.int16))
    assert session.seen_lengths == [512]


def test_preroll_keeps_audio_before_the_first_speech_frame():
    detector = _detector([0.0] * 5 + [0.9] * 20 + [0.0] * 20)
    results = _run(detector, 45)
    end = next(r for r in results if r.event is VADEvent.SPEECH_END)
    # Utterance includes the pre-roll frames, so it is longer than the speech run.
    assert end.audio is not None
    assert len(end.audio) > 20 * 512


# --------------------------------------------------------------------------
#  Speaker buffer (no audio device needed -- callback is driven manually)
# --------------------------------------------------------------------------

def _speaker_stream_class():
    """Import SpeakerStream, stubbing PortAudio if the host has no audio stack.

    The queue/callback/interrupt logic is pure Python -- it is the highest-risk
    part of the barge-in path, so it should be tested on CI boxes too, not
    skipped just because there is no sound card.
    """
    try:
        import sounddevice  # noqa: F401
    except Exception:
        import types

        stub = types.ModuleType("sounddevice")
        stub.query_devices = lambda *a, **k: []
        stub.InputStream = object
        stub.OutputStream = object
        sys.modules["sounddevice"] = stub

    from voice_agent.audio_io import SpeakerStream

    return SpeakerStream


def test_speaker_buffer_drains_and_interrupts():
    SpeakerStream = _speaker_stream_class()

    speaker = SpeakerStream(sample_rate=24000, blocksize=480)
    speaker.play(np.ones(2400, dtype=np.float32) * 0.1)
    assert speaker.is_speaking
    assert speaker.pending_seconds == pytest.approx(0.1, abs=1e-6)

    out = np.zeros((480, 1), dtype=np.float32)
    speaker._callback(out, 480, None, None)
    assert out[0, 0] == pytest.approx(0.1)
    assert speaker.pending_seconds == pytest.approx(0.08, abs=1e-6)

    dropped = speaker.interrupt()
    assert dropped == 1920
    assert not speaker.is_speaking

    out2 = np.zeros((480, 1), dtype=np.float32)
    speaker._callback(out2, 480, None, None)
    assert np.all(out2 == 0.0)  # silence within one block of the interrupt


def test_speaker_clips_out_of_range_samples():
    SpeakerStream = _speaker_stream_class()

    speaker = SpeakerStream()
    speaker.play(np.array([2.0, -3.0, 0.5], dtype=np.float32))
    out = np.zeros((3, 1), dtype=np.float32)
    speaker._callback(out, 3, None, None)
    assert out[0, 0] == 1.0 and out[1, 0] == -1.0 and out[2, 0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
#  Metrics
# --------------------------------------------------------------------------

def test_turn_metrics_computes_time_to_first_audio():
    m = TurnMetrics(turn_id=1)
    m.speech_end_t = 100.0
    m.stt_done_t = 100.10
    m.llm_first_token_t = 100.35
    m.first_audio_ready_t = m.playback_start_t = 100.60
    m._vad_hold_ms = 400.0

    assert m.stt_ms == pytest.approx(100, abs=1)
    assert m.llm_ttft_ms == pytest.approx(250, abs=1)
    assert m.tts_ms == pytest.approx(250, abs=1)
    assert m.time_to_first_audio_ms == pytest.approx(600, abs=1)
    assert m.total_perceived_ms == pytest.approx(1000, abs=1)
    assert "TIME TO FIRST AUDIO" in m.render()


def test_metrics_summary_reports_pass_under_budget():
    # Timestamps mimic perf_counter(); 0.0 legitimately means "never happened".
    log = MetricsLog()
    for i in range(3):
        m = TurnMetrics(turn_id=i)
        m.speech_end_t = 1000.0
        m.stt_done_t = 1000.08
        m.llm_first_token_t = 1000.30
        m.first_audio_ready_t = m.playback_start_t = 1000.55
        log.add(m)
    summary = log.summary()
    assert "PASS" in summary
    assert "3 turn(s)" in summary


def test_metrics_ignores_unset_timestamps():
    m = TurnMetrics(turn_id=1)
    m.speech_end_t = 1000.0  # nothing else ran (e.g. cancelled turn)
    assert m.stt_ms == 0.0
    assert m.time_to_first_audio_ms == 0.0


def test_metrics_summary_handles_no_turns():
    assert "No completed turns" in MetricsLog().summary()


# --------------------------------------------------------------------------
#  Voice rules added after run 16.
#
# That run PASSED on latency but shipped six conversation defects that no
# output guard caught. Each rule below exists to stop one specific thing Riya
# actually said on the call, so if a later edit drops one, these fail instead
# of letting it ship silently:
#    9  -> first audio waits on the first COMPLETE sentence, so keep it short
#   10 -> "Your degree background was mentioned" -- nothing had been
#   11 -> "You're asking why we stopped discussing that..." -- they said "?"
#   12 -> "I cannot provide guidance on how to respond to..." -- assistant-speak
#   13 -> a bare "No." delivered as a whole answer
#   14 -> "Five years... Five years" and "Amazon... Amazon"
#   15 -> "Senior Data Engineer positions?."
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rule",
    [
        "9. Make your FIRST sentence short",
        "10. Never say the caller mentioned",
        "11. Never explain what the caller meant",
        "12. Never describe what you can or cannot do",
        '13. Never reply with only "Yes."',
        "14. Inside one turn, name each thing once",
        "15. End every sentence with exactly one mark",
    ],
)
def test_run_16_voice_rules_survive_edits(rule):
    assert rule in SYSTEM_PROMPT


# Run 17: latency fixed (Piper 642 -> 446 ms, TTFA 1249 -> 1073 ms, both from
# rule 9) but Riya invented machinery and facts about her own side.
#   16 -> "Our system flagged an unclear response"
#   17 -> "That doesn't fit our current schedule" /
#         "You can't start working yet because of age limits" /
#         "We're actually looking at other candidates"
@pytest.mark.parametrize(
    "rule",
    [
        "16. Never mention systems, flags, or processing",
        "17. Never invent a reason, a rule, or a fact",
    ],
)
def test_run_17_voice_rules_survive_edits(rule):
    assert rule in SYSTEM_PROMPT


def test_the_first_chunk_may_be_cut_below_a_full_sentence():
    """Run 16 logged a flat 642 ms 'Piper first audio' on all 12 turns.

    Real Piper is 110-155 ms, so that time was spent WAITING for text, not
    synthesising it. Once the deadline passes, _force_split re-arms on every
    token push, so the value that matters is how little text we accept as
    speakable. 10 held audio back; 6 is about one short word pair.
    """
    cfg = load_config()
    assert int(cfg.tts["emergency_min_chars"]) <= 6
