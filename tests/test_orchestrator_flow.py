"""End-to-end orchestration test with stub models.

Verifies the wiring that is easy to get wrong and impossible to unit-test against
the real models: that chunk 1 is played before the LLM has finished generating,
that barge-in cancels a turn without deadlocking, and that a cancelled turn's
stop_event cannot be un-set by the turn that replaces it.

No audio device, no model files, no network.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub PortAudio before voice_agent.audio_io is imported anywhere.
try:  # pragma: no cover
    import sounddevice  # noqa: F401
except Exception:  # pragma: no cover
    _stub = types.ModuleType("sounddevice")
    _stub.query_devices = lambda *a, **k: []
    _stub.InputStream = object
    _stub.OutputStream = object
    sys.modules["sounddevice"] = _stub

from voice_agent.config import load_config  # noqa: E402
from voice_agent.metrics import TurnMetrics  # noqa: E402
from voice_agent.orchestrator import VoiceAgent, _CUT_KINDS  # noqa: E402


# --------------------------------------------------------------------------
#  Stubs
# --------------------------------------------------------------------------

class FakeLLM:
    """Emits a fixed reply one word at a time, slowly enough to observe overlap."""

    def __init__(self, reply: str, delay_s: float = 0.01):
        self.reply = reply
        self.delay_s = delay_s
        self.history: list = []
        self.tokens_emitted = 0
        self.stopped_early = False
        self.aborts = 0

    def add_user(self, text):
        self.history.append(("user", text))

    def add_assistant(self, text):
        self.history.append(("assistant", text))

    def recent_assistant_lines(self, limit: int = 6):
        return [t for role, t in self.history if role == "assistant"][-limit:]

    def drop_last_user(self, expected=None):
        if not self.history or self.history[-1][0] != "user":
            return False
        if expected is not None and expected.strip() not in self.history[-1][1]:
            return False
        self.history.pop()
        return True

    def abort(self):
        self.aborts += 1

    @property
    def context_tokens(self):
        return 100

    def stream(self, stop_event=None, extra_system=None, stats=None):
        self.extra_system_seen = extra_system
        for word in self.reply.split(" "):
            if stop_event is not None and stop_event.is_set():
                self.stopped_early = True
                if stats:
                    stats.interrupted = True
                return
            time.sleep(self.delay_s)
            self.tokens_emitted += 1
            if stats:
                stats.tokens = self.tokens_emitted
            yield word + " "


class FakeTTS:
    sample_rate = 24000

    def __init__(self, delay_s: float = 0.01):
        self.delay_s = delay_s
        self.calls: list[str] = []

    def synthesize(self, text):
        time.sleep(self.delay_s)
        self.calls.append(text)
        speech = types.SimpleNamespace()
        # 0.2 s of audio per chunk.
        speech.samples = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
        speech.sample_rate = self.sample_rate
        return speech


class FakeSpeaker:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.played: list[int] = []
        self.play_times: list[float] = []
        self.interrupts = 0
        self._pending = 0.0

    def play(self, samples):
        self.played.append(len(samples))
        self.play_times.append(time.perf_counter())

    def interrupt(self):
        self.interrupts += 1
        dropped = int(self._pending * self.sample_rate)
        self._pending = 0.0
        return dropped

    @property
    def is_speaking(self):
        return False

    @property
    def pending_seconds(self):
        return self._pending

    def start(self):
        pass

    def stop(self):
        pass


class FakeTurnDetector:
    agent_speaking = False
    is_speaking = False

    def __init__(self):
        self.cooldowns = 0

    def start_cooldown(self):
        self.cooldowns += 1


def make_agent(reply: str, llm_delay=0.01, tts_delay=0.01, max_sentences=None) -> VoiceAgent:
    cfg = load_config()
    if max_sentences is not None:
        # Tests about streaming mechanics need long replies; the two-sentence
        # ceiling would end generation before they can observe anything.
        cfg["llm"]["max_sentences"] = max_sentences
    agent = VoiceAgent(cfg, on_event=lambda k, t: None)
    agent.llm = FakeLLM(reply, llm_delay)
    agent.tts = FakeTTS(tts_delay)
    agent.speaker = FakeSpeaker()
    agent.turn_detector = FakeTurnDetector()
    return agent


# --------------------------------------------------------------------------
#  Tests
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_chunk_plays_before_generation_finishes():
    """The whole point of the pipeline: audio starts mid-generation."""
    reply = "Got it. So you need a senior DevOps person, contract to hire, remote is fine. Let me pull up the requisition and check where we are."
    agent = make_agent(reply, llm_delay=0.01, max_sentences=0)

    start = time.perf_counter()
    m = await agent.say_text("Any update on the DevOps role?")
    end = time.perf_counter()

    assert agent.speaker.played, "nothing was ever played"
    first_play_at = agent.speaker.play_times[0] - start
    total = end - start
    # Chunk 1 must be out well before the turn is over.
    assert first_play_at < total * 0.7, (
        f"first audio at {first_play_at:.3f}s of a {total:.3f}s turn -- not overlapping"
    )
    assert m.chunks >= 2, "response should have been split into multiple chunks"
    assert m.response.strip()
    assert agent.llm.history[-1][0] == "assistant"


@pytest.mark.asyncio
async def test_chunks_are_played_in_order_and_nothing_is_lost():
    # Not the old "Sure. I'll check the submission status." -- promising to go
    # and check is a commitment Riya cannot make, and it is cut on sight now
    # (run 36 turn 6). This test is about chunk ordering, so any two-sentence
    # reply will do.
    reply = "Sure. The submission is still pending."
    agent = make_agent(reply)
    m = await agent.say_text("Where are we?")
    joined = " ".join(agent.tts.calls)
    assert joined.split() == reply.split()
    assert m.response.split() == reply.split()


@pytest.mark.asyncio
async def test_barge_in_stops_generation_and_cancels_the_turn():
    reply = " ".join(f"word{i}." for i in range(60))
    agent = make_agent(reply, llm_delay=0.02, tts_delay=0.005, max_sentences=0)
    agent._loop = asyncio.get_running_loop()

    # Kick off a turn the way _start_turn does.
    m = TurnMetrics(turn_id=1)
    m.speech_end_t = time.perf_counter()
    stop_event = threading.Event()
    agent._stop_generation = stop_event
    agent._active_metrics = m
    agent._active_turn = asyncio.create_task(
        agent._handle_turn_text_stub(m, stop_event)
    )

    await asyncio.sleep(0.15)          # let it get going
    assert agent.speaker.played, "expected some audio before the interruption"
    played_before = len(agent.speaker.played)

    t0 = time.perf_counter()
    await agent._barge_in()
    barge_ms = (time.perf_counter() - t0) * 1000.0

    assert barge_ms < 200, f"barge-in took {barge_ms:.0f} ms"
    assert agent.speaker.interrupts == 1
    assert stop_event.is_set()
    assert m.interrupted
    assert agent.turn_detector.cooldowns == 1
    # Setting the event is not enough: the generation thread can be parked in a
    # blocking socket read that never returns on its own. Closing the connection
    # is what actually frees it -- and llama-server's slot with it. Without this
    # the dead turn held the client lock and the NEXT turn's first token came
    # back 10.5 s later.
    assert agent.llm.aborts == 1, "barge-in did not abort the in-flight request"

    await asyncio.sleep(0.15)
    assert agent.llm.stopped_early, "llama.cpp stream was not stopped"
    # No further audio should have been queued after the cut.
    assert len(agent.speaker.played) <= played_before + 1
    assert agent._pending_interruption is True


@pytest.mark.asyncio
async def test_a_new_turn_cannot_unset_the_previous_turns_stop_event():
    """Regression guard: workers must capture their own Event, not read self."""
    agent = make_agent("one. two. three. four. five.", llm_delay=0.02)
    agent._loop = asyncio.get_running_loop()

    old_event = threading.Event()
    agent._stop_generation = old_event
    old_event.set()

    # Simulate _start_turn replacing the attribute with a fresh Event.
    new_event = threading.Event()
    agent._stop_generation = new_event

    assert old_event.is_set()
    assert not new_event.is_set()

    # A worker handed old_event must still see it as set.
    m = TurnMetrics(turn_id=9)
    queue: asyncio.Queue = asyncio.Queue()
    from voice_agent.llm import GenerationStats

    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, old_event),
        daemon=True,
    )
    thread.start()
    text, _repeated = await agent._consume_chunks(queue, m, old_event)
    thread.join(timeout=2)
    assert text == "", "generation should have been suppressed by the old stop event"
    assert not agent.speaker.played


@pytest.mark.asyncio
async def test_interruption_note_is_injected_on_the_next_turn():
    agent = make_agent("Okay sure.")
    agent._pending_interruption = True
    await agent.say_text("Sorry, go on.")
    from voice_agent.prompts import INTERRUPTION_NOTE

    # say_text does not inject; the mic path does. Confirm the flag survives.
    assert agent._pending_interruption is True
    assert INTERRUPTION_NOTE.startswith("[")
    assert "cut you off" in INTERRUPTION_NOTE
    assert "Do not repeat" in INTERRUPTION_NOTE


@pytest.mark.asyncio
async def test_empty_reply_does_not_pollute_history():
    agent = make_agent("   ")
    m = await agent.say_text("hello?")
    assert m.response == ""
    assert all(role != "assistant" for role, _ in agent.llm.history)


# --------------------------------------------------------------------------
#  Mic backpressure
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_mic_queue_drops_oldest_instead_of_raising():
    """Regression: an overflowing mic queue used to raise QueueFull out of a
    call_soon_threadsafe callback, spamming the console once per 32 ms frame."""
    agent = make_agent("hello.")
    agent._loop = asyncio.get_running_loop()
    agent._frame_queue_size = 4
    agent._frames = asyncio.Queue(maxsize=4)
    agent._running = True

    for i in range(20):
        agent._enqueue_frame(np.full(512, i, dtype=np.int16))   # must not raise

    assert agent._frames.qsize() == 4
    assert agent._dropped_frames == 16

    # The four survivors are the four most recent frames, in order.
    kept = [int(agent._frames.get_nowait()[0]) for _ in range(4)]
    assert kept == [16, 17, 18, 19]


@pytest.mark.asyncio
async def test_mic_callback_never_raises_before_the_loop_is_up():
    agent = make_agent("hello.")
    agent._loop = None
    agent._on_mic_frame(np.zeros(512, dtype=np.int16))          # must not raise

    agent._loop = asyncio.get_running_loop()
    agent._running = False
    agent._on_mic_frame(np.zeros(512, dtype=np.int16))          # must not raise


@pytest.mark.asyncio
async def test_greeting_does_not_block_the_vad_loop():
    """Regression: the greeting used to be awaited BEFORE _vad_loop started, so
    mic frames accumulated unconsumed for the whole length of the greeting."""
    agent = make_agent("hello.", tts_delay=0.15)
    agent._loop = asyncio.get_running_loop()
    agent._frames = asyncio.Queue(maxsize=200)
    agent._running = True
    agent.turn_detector.is_speaking = False

    consumed = []

    async def fake_vad_loop():
        while agent._running:
            consumed.append(await agent._frames.get())

    vad_task = asyncio.create_task(fake_vad_loop())

    async def feed():
        for _ in range(12):
            agent._enqueue_frame(np.zeros(512, dtype=np.int16))
            await asyncio.sleep(0.01)

    feeder = asyncio.create_task(feed())
    await agent._speak_standalone("hello.")
    await feeder
    await asyncio.sleep(0.05)

    agent._running = False
    vad_task.cancel()

    assert agent._dropped_frames == 0
    assert len(consumed) >= 10, "frames were not consumed during the greeting"


# --------------------------------------------------------------------------
#  Helper injected onto VoiceAgent for the barge-in test
# --------------------------------------------------------------------------

async def _handle_turn_text_stub(self, m: TurnMetrics, stop_event: threading.Event) -> None:
    """Mimics _handle_turn from the point where the transcript is already known."""
    from voice_agent.llm import GenerationStats

    try:
        self.llm.add_user("stubbed transcript")
        queue: asyncio.Queue = asyncio.Queue()
        stats = GenerationStats()
        thread = threading.Thread(
            target=self._generate_worker,
            args=(queue, stats, None, m, stop_event),
            daemon=True,
        )
        thread.start()
        spoken = await self._consume_chunks(queue, m, stop_event)
        m.response = spoken
    except asyncio.CancelledError:
        stop_event.set()
        m.interrupted = True
        raise


VoiceAgent._handle_turn_text_stub = _handle_turn_text_stub


# --------------------------------------------------------------------------
#  Turn-taking: the conversational failures from the live call log
# --------------------------------------------------------------------------

from voice_agent.orchestrator import (  # noqa: E402
    _apply_stt_fixups,
    _ends_dangling,
    _is_backchannel,
    _too_similar,
)


@pytest.mark.parametrize("text", [
    "Uh so I am a data engineer and I have two years of experience, so",
    "I have worked with Python and",
    "my current notice period is um",
    "I was thinking maybe because",
])
def test_trailing_connectors_mean_the_caller_is_not_finished(text):
    """A 400 ms silence timer cannot tell a thinking pause from a finished thought.

    In the live log the caller said "...and I have two years of experience, so"
    and the agent answered the half-sentence, which then got barged in on when
    they carried on. The dangling word is the cheap signal to wait a beat.
    """
    assert _ends_dangling(text)


@pytest.mark.parametrize("text", [
    "Yes I am interested in the role.",
    "What is the rate for this position?",
    "I have two years of experience",
    "I am an AWS engineer",
])
def test_finished_sentences_are_not_held(text):
    assert not _ends_dangling(text)


@pytest.mark.parametrize("text", ["Mm-hmm.", "uh huh", "hmm", "aha"])
def test_pure_continuers_are_not_turns_when_nothing_was_asked(text):
    """Only when the agent's last line was NOT a question."""
    assert _is_backchannel(text, "I'll pull the requisition up now.")


@pytest.mark.parametrize("answer", [
    "Yes.", "Yeah", "okay", "Sure", "No", "Right.", "Correct", "Got it",
    "I see", "Alright", "Mm-hmm.", "uh huh",
])
def test_nothing_is_a_backchannel_after_a_question(answer):
    """This bug killed the call on turn two.

    Riya asks a question almost every turn by design, so the caller's reply is
    usually a one-word acknowledgement. "Have you got two minutes?" / "Yes." was
    classified as a backchannel and silently discarded -- the agent said nothing
    back, and from the caller's side it looked like a hang. Even "mm-hmm" after
    a yes/no question plainly means yes.
    """
    assert not _is_backchannel(answer, "Have you got two minutes?")


@pytest.mark.parametrize("answer", ["Yes.", "Yeah", "okay", "Sure", "Right.", "Correct"])
def test_acknowledgements_are_answers_not_noise(answer):
    """Even with no question mark, a real word is a turn, not a continuer."""
    assert not _is_backchannel(answer, "Let me know what suits.")


@pytest.mark.parametrize("answer", ["Yes", "Yeah", "okay", "ok", "sure", "no", "you"])
def test_the_stt_layer_does_not_discard_answers(answer):
    """The other half of the same bug, one layer lower.

    Parakeet inherited Whisper's hallucination blocklist, which lists "yeah",
    "okay" and "you". Those were dropped before the orchestrator ever saw them.
    Parakeet does not invent speech from silence, so it must not use that list.
    """
    from voice_agent.stt import _is_noise_only

    assert not _is_noise_only(answer)


@pytest.mark.parametrize("noise", ["uh", "um", ".", "", "  ", "mm"])
def test_the_stt_layer_still_drops_contentless_results(noise):
    from voice_agent.stt import _is_noise_only

    assert _is_noise_only(noise)


@pytest.mark.parametrize("heard,expected", [
    ("I am a WS engineer", "AWS"),
    ("we use dev ops daily", "DevOps"),
    ("is this c2c or w 2", "C2C"),
    ("I am on h1b", "H1B"),
    ("air flow and py spark", "Airflow"),
])
def test_domain_terms_are_repaired_after_stt(heard, expected):
    """Phone-band audio eats leading consonants; these are the words that matter."""
    assert expected in _apply_stt_fixups(heard)


def test_repeated_question_is_recognised_even_when_reworded():
    a = "Are you looking to fill a role, or are you on the candidate side?"
    b = "I'm here. Are you looking to fill a role, or are you on the candidate side?"
    assert _too_similar(a, b)


def test_different_questions_are_not_flagged_as_repeats():
    a = "Are you looking to fill a role, or are you on the candidate side?"
    b = "What rate are you targeting for this contract?"
    assert not _too_similar(a, b)


@pytest.mark.asyncio
async def test_a_repeated_reply_is_suppressed_before_it_reaches_the_speaker():
    """The guard must fire before Piper runs, or the caller hears the repeat start."""
    # A STATEMENT, not the question this used to use. Questions the model
    # writes are dropped at chunk level now (run 35: they were being spoken
    # and then stripped from the transcript, so the caller heard two), so a
    # question would never reach this veto path at all.
    line = "You are on the candidate side rather than the hiring side."
    agent = make_agent(line)
    agent._loop = asyncio.get_running_loop()
    agent._regenerate = True               # opt in: this tests the veto path
    agent.llm.add_assistant(line)          # she already said this

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    from voice_agent.llm import GenerationStats

    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, repeated = await agent._consume_chunks(queue, m, stop_event, guard=True)
    thread.join(timeout=2)

    assert repeated == ("repeat", line)
    assert spoken == ""
    assert not agent.speaker.played, "the caller heard the start of a repeated line"
    assert not agent.tts.calls, "Piper ran on a line we had already decided to bin"


@pytest.mark.asyncio
async def test_a_repeat_is_cut_rather_than_retried():
    """A repeat must not reach the speaker -- and must not cost a retry.

    This test used to assert the opposite: that `regenerate_kinds` overrode
    the global off switch and a repeat DID trigger a second generation, on the
    reasoning that correction_note() names the duplicated text so a re-roll has
    something concrete to avoid.

    Run 42 reversed that, with the measurement printed by the tool itself:
    "2/6 turns regenerated after a guard rejection (each roughly doubles that
    turn's LLM time)". Those turns logged TTFT 745 and 584 ms against 360-441
    ms elsewhere, and every other stage had improved on run 41. The retry
    policy was the whole regression.

    The intent is unchanged -- a repeat must never be spoken -- but the
    mechanism is now a cut instead of a retry, which is free because the
    templated question still goes out. Same outcome, one generation instead of
    two. There is nothing to salvage in a repeat; asking the model to produce
    a better version of a sentence it should not say at all was the mistake.
    """
    line = "You are on the candidate side rather than the hiring side."
    agent = make_agent(line)
    agent._loop = asyncio.get_running_loop()
    # Deliberately NOT setting agent._regenerate: that is the shipped default.
    assert agent._regenerate is False, "this test is about the shipped setting"
    # ...and the shipped config no longer buys retries for any kind.
    assert "repeat" not in agent._regen_kinds
    assert "repeat" in _CUT_KINDS
    agent.llm.add_assistant(line)  # she already said this

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    from voice_agent.llm import GenerationStats

    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, repeated = await agent._consume_chunks(queue, m, stop_event, guard=True)
    thread.join(timeout=2)

    # No veto is returned, so _handle_turn will not schedule a retry...
    assert repeated is None, f"a cut must not return a veto, got {repeated}"
    # ...and the text itself never reached the speaker.
    assert spoken == "", "a cut repeat must not reach the speaker"
    assert not agent.speaker.played


@pytest.mark.asyncio
async def test_a_fresh_reply_passes_the_guard_untouched():
    # A statement that mentions a forbidden topic the CALLER raised. The
    # caller's words have to reach the repair path (they are passed in
    # explicitly here) or the topic rules below treat her own quotation of
    # them as something she volunteered.
    agent = make_agent("The rate for this one is still open.")
    agent._loop = asyncio.get_running_loop()
    agent.llm.add_assistant("You are on the candidate side rather than the hiring side.")

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    from voice_agent.llm import GenerationStats

    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, repeated = await agent._consume_chunks(
        queue, m, stop_event, guard=True,
        caller_said="what rate are you targeting for this one",
    )
    thread.join(timeout=2)

    assert repeated is None
    assert "rate" in spoken
    assert agent.speaker.played


class _FakeTranscriber:
    """Returns a scripted transcript per finalize() call."""

    supports_streaming = False

    def __init__(self, texts):
        self.texts = list(texts)

    def finalize(self, audio):
        from voice_agent.stt import Transcript
        return Transcript(self.texts.pop(0) if self.texts else "")

    def reset(self):
        pass


async def _drive_turn(agent, text_queue, audio=b"x"):
    """Run one _handle_turn the way the VAD loop would."""
    agent._turn_id += 1
    m = TurnMetrics(turn_id=agent._turn_id)
    m.speech_end_t = time.perf_counter()
    stop_event = threading.Event()
    agent._stop_generation = stop_event
    agent._active_metrics = m
    frame = np.zeros(512, dtype=np.int16) if audio is not None else None
    await agent._handle_turn(frame, m, agent._turn_id, stop_event)
    return m


@pytest.mark.asyncio
async def test_an_unfinished_sentence_is_held_and_then_stitched():
    """The exact sequence from the live log, which lost the caller's answer.

    They said "...I have two years of experience, so", the agent answered the
    fragment, they carried on, and the barge-in threw the rest away. Holding the
    fragment means the model finally sees the whole sentence.
    """
    agent = make_agent("Got it, that's relevant.")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber([
        "I am an AWS engineer and I have two years of experience, so",
        "can you tell me more about the role",
    ])

    await _drive_turn(agent, None)
    # Nothing said yet: the sentence was not finished.
    assert agent._carry_text.endswith("so")
    assert not agent.llm.history, "answered half a sentence"
    agent._cancel_carry_flush()

    await _drive_turn(agent, None)
    user_turns = [t for role, t in agent.llm.history if role == "user"]
    assert len(user_turns) == 1
    assert "two years of experience" in user_turns[0]
    assert "tell me more about the role" in user_turns[0], "the caller's follow-up was lost"
    assert agent._carry_text == ""


@pytest.mark.asyncio
async def test_a_held_fragment_is_answered_if_the_caller_never_resumes():
    """Holding must not mean hanging. The grace timer flushes it."""
    agent = make_agent("Got it.")
    agent._loop = asyncio.get_running_loop()
    agent._carry_grace_s = 0.05
    agent.transcriber = _FakeTranscriber(["I work with Spark and"])

    await _drive_turn(agent, None)
    assert agent._carry_text

    await asyncio.sleep(0.3)
    user_turns = [t for role, t in agent.llm.history if role == "user"]
    assert user_turns == ["I work with Spark and"], "held fragment was never released"


@pytest.mark.asyncio
async def test_a_backchannel_does_not_trigger_a_reply():
    agent = make_agent("Are you open to hearing about the role?")
    agent._loop = asyncio.get_running_loop()
    agent.llm.add_assistant("Are you open to hearing about the role?")
    agent.transcriber = _FakeTranscriber(["Mm-hmm."])

    before = len(agent.llm.history)
    await _drive_turn(agent, None)
    assert len(agent.llm.history) == before, "replied to a backchannel"
    assert not agent.speaker.played


@pytest.mark.asyncio
async def test_stt_fixups_reach_the_model():
    agent = make_agent("Got it.")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["I am a WS engineer with dev ops exposure."])

    await _drive_turn(agent, None)
    user_turns = [t for role, t in agent.llm.history if role == "user"]
    assert "AWS" in user_turns[0]
    assert "DevOps" in user_turns[0]


# --------------------------------------------------------------------------
#  Persona collapse: the failures from the second live call
# --------------------------------------------------------------------------

from voice_agent.orchestrator import (  # noqa: E402
    _has_emoji,
    _parrots,
    _sounds_like_a_chatbot,
)


@pytest.mark.parametrize("reply", [
    "Hello there! How can I assist you today?",
    "Let me clarify—what exactly do you mean by skills?",
    "Could you please state that clearly so we can proceed accurately?",
    "I'm here to help with your questions.",
    "Sure thing 🌟",
])
def test_assistant_voice_is_recognised(reply):
    """By turn 14 of a live call the model had drifted back to its instruct prior.

    Every one of these is banned in the system prompt in plain words. The
    instruction was still in context, just 1000 tokens away and outvoted by the
    recent conversation, so catching the output is more reliable than asking a
    1.7B more firmly.
    """
    assert _sounds_like_a_chatbot(reply)


@pytest.mark.parametrize("reply", [
    "Got it, that fits. How many years on the pipeline side?",
    "Fair enough. Mind if I keep you on file for something closer?",
    "No problem - later today, or tomorrow morning?",
])
def test_in_character_replies_pass(reply):
    assert not _sounds_like_a_chatbot(reply)


def test_emoji_are_detected_and_stripped_before_piper():
    from voice_agent.text_chunker import normalize_for_tts

    assert _has_emoji("Sure thing 🌟")
    # The prompt bans them; this is the backstop for when the model ignores it.
    assert "🌟" not in normalize_for_tts("Sure thing 🌟")
    assert normalize_for_tts("Sure thing 🌟").strip() == "Sure thing"


def test_curly_punctuation_is_flattened_for_piper():
    from voice_agent.text_chunker import normalize_for_tts

    out = normalize_for_tts("That’s good — how many years…")
    assert "’" not in out and "…" not in out
    assert "that's" in out.lower()


def test_doubled_terminators_collapse_to_one_before_piper():
    """Run 17 ended four separate turns on "?." ("...do you have?." / "SQL?.").

    Piper reads the trailing "." as a second, shorter sentence, so the question
    never settles -- it lands as a stumble. Rule 15 already forbids it and the
    3B emitted it anyway, so enforce it here rather than asking again.
    """
    from voice_agent.text_chunker import normalize_for_tts

    assert normalize_for_tts("How many years of experience do you have?.").endswith("?")
    assert normalize_for_tts("what databases have you worked with using SQL?.").endswith("?")
    assert "?." not in normalize_for_tts("Is there anything else?.")
    # An ellipsis is not a doubled terminator -- it must survive untouched.
    assert normalize_for_tts("Hold on...") == "Hold on..."
    # A statement followed by a real question must survive too.
    assert normalize_for_tts("Right. Which languages?").endswith("?")


def test_a_checklist_note_bolted_on_the_end_is_cut_out():
    """Run 18 spoke the live item verbatim, fused to the clause before it.

    "...when you're available again qualification and field?" and "...job
    openings at our clients years of experience?" both went out loud.
    _echoes_agenda cannot catch these: its clause test compares whole clauses
    and the note is fused, and its length test needs a short reply.
    """
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = [
        "qualification and field?",
        "years of experience?",
        "tech stack -- which ones?",
        "read back and confirm?",
        "close the call",
    ]
    first = "Your time was cut short thanks for letting us know when you're available again qualification and field?"
    assert _strip_agenda_suffix(first, agenda) == (
        "Your time was cut short thanks for letting us know when you're available again."
    )
    second = "We were discussing collecting information for potential job openings at our clients years of experience?"
    assert _strip_agenda_suffix(second, agenda) == (
        "We were discussing collecting information for potential job openings at our clients."
    )
    assert "qualification and field" not in _strip_agenda_suffix(first, agenda)
    assert "years of experience" not in _strip_agenda_suffix(second, agenda)


@pytest.mark.parametrize(
    "text,caller,lost,wanted",
    [
        # 1. Unsolicited redirect: nobody raised salary, yet it opened turn 1
        #    of runs 20 and 23.
        (
            "Our recruitment team will discuss salary details separately can you "
            "confirm how many years of total work experience you currently hold?.",
            "Oh here's I am free.",
            "salary",
            "Can you confirm how many years",
        ),
        # 3. Commenting on the caller.
        (
            "You're having trouble answering That helps clarify things",
            "Um guess",
            "having trouble",
            "That helps clarify things",
        ),
        (
            "Your response was unclear Could you tell me specifically which "
            "programming languages you have been working with lately?.",
            "I said five years of total full time.",
            "response was unclear",
            "Could you tell me specifically",
        ),
        (
            "You hung up on me Sorry that conversation didn't go through Can I "
            "help match you with some relevant job openings anyway?.",
            "Opening.",
            "hung up",
            "Can I help match you",
        ),
        # 4. Claiming to know the openings book.
        (
            "No I don't currently have an opening for an A Engineer You're looking "
            "at our Senior Data Engineer role with a healthcare analytics client",
            "Do you have any A engineer openings?",
            "opening for an A Engineer",
            "You're looking at our Senior Data Engineer role",
        ),
        # 5. Apologising for the machinery.
        (
            "Sorry I'm Riya from IT Staffing Solutions Hi sorry I forgot my name "
            "earlier Thank you for confirming your qualification",
            "Oh yeah, Sean.",
            "forgot my name",
            "Thank you for confirming your qualification",
        ),
    ],
)
def test_run_23_mechanical_failures_are_cut_in_place(text, caller, lost, wanted):
    """Four failure shapes from run 23, each cut without a second generation.

    Nine prompt rules did not stop any of them; these are identifiable spans,
    so they are removed in code and the rest of the reply survives.
    """
    from voice_agent.orchestrator import _repair_chunk

    out, cut = _repair_chunk(text, caller)
    assert cut, f"nothing was cut from {text!r}"
    assert lost.lower() not in out.lower(), out
    assert wanted in out, out


@pytest.mark.parametrize(
    "text,caller",
    [
        # The redirect is correct when the caller DID raise the topic.
        (
            "Our recruitment team will discuss salary details separately. "
            "What range were you expecting?",
            "What is the salary for this role?",
        ),
        ("Sorry, could you repeat that?", "Um guess"),
        ("Can you tell me about your educational background?", "hello"),
        # "We will follow up regarding suitable openings" used to be listed
        # here as legitimate. It is not: the config bans commitments outright
        # (no callback, no "we'll be in touch"), and run 36 turn 6 went out
        # with "I'll check on any relevant job openings at our client
        # companies soon." Moved to test_a_promise_to_follow_up_is_cut.
        ("That is helpful, thank you. Have a great day.", "thanks"),
    ],
)
def test_legitimate_replies_survive_the_repairs(text, caller):
    """Cutting must not eat replies that are already right."""
    from voice_agent.orchestrator import _repair_chunk

    out, cut = _repair_chunk(text, caller)
    assert not cut, f"cut {cut} from a correct reply: {out!r}"
    assert out == text


def test_run_24_openings_claim_without_a_verb_is_cut():
    """Run 24 turn 11: "No immediate openings but our team will follow up...".

    The run-23 pattern needed "(we|I) don't have", so this grammar -- no verb
    at all -- sailed past it.
    """
    from voice_agent.orchestrator import _repair_chunk

    text = (
        "No immediate openings but our team will follow up regarding suitable roles."
    )
    out, cut = _repair_chunk(text, "Do you have openings?")
    assert "openings claim" in cut, cut
    assert "immediate openings" not in out.lower(), out
    # Run 46 turn 5 widened _COMMITMENT to a team subject: "Our team will
    # follow up regarding suitable openings at our health care companies."
    # went out uncut because the subject was I|we|let me. Same defect here, so
    # the promise is cut as well as the openings claim, and the stranded "But."
    # that is left behind is what _only_a_fragment drops in the pipeline.
    assert "commitment she cannot make" in cut, cut
    assert "follow up" not in out.lower(), out


def test_a_promise_to_follow_up_is_cut():
    """Run 36 turn 6: "I'll check on any relevant job openings at our client
    companies soon."

    Asked about openings, Riya promised to go and look. She has no client
    list and no mandate, and the config bans commitments outright -- no
    callback, no date, no "we'll be in touch". The caller still hears the
    templated line, so this costs nothing.
    """
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        "I'll check on any relevant job openings at our client companies soon.",
        "We will follow up regarding suitable openings. Have a great day.",
    ):
        out, cut = _repair_chunk(text, "thanks")
        assert "commitment she cannot make" in cut, f"not cut from {text!r}"
        assert "openings" not in out.lower(), out


def test_the_correct_about_variant_of_the_meta_comment_is_cut():
    """Run 36 turn 5.

    "You're correct about our notes so far You've got an Undergrad Computer
    Science from a four-year program followed by five years of work
    experience..." -- _META_COMMENT knew "correct that" and "correct we/I" but
    not "correct about", so the dead agreement went out with the real content
    trailing after it, uncut and rambling (38 gen tokens).
    """
    from voice_agent.orchestrator import _repair_chunk

    text = ("You're correct about our notes so far You've got an Undergrad "
            "Computer Science from a four-year program followed by five years "
            "of work experience.")
    out, cut = _repair_chunk(text, "Oh yes you are right.")
    assert "commentary about the call" in cut, cut
    assert "correct about" not in out.lower(), out
    assert out.startswith("You've got an Undergrad"), out

    # Run 37 turn 5: "You're correct on both points." Same agreement, "on"
    # instead of "about". The preposition list is open rather than fixed --
    # one entry per preposition is the loop that made runs 27-33 endless.
    out, cut = _repair_chunk("You're correct on both points.", "Oh yes you are right.")
    assert "commentary about the call" in cut, cut
    assert not out.strip(), out


def test_narrating_the_machinery_is_cut():
    """Run 37 turn 7, straight after a barge-in.

    "It seems we lost connection mid-sentence." The caller had only repeated
    themselves. Nobody on a phone call says this, and the caller cannot act on
    it. The cut has to take the whole clause: stopping at the verb left
    "Connection.", because _clause_end only walks forward from the match and
    "connection" is lowercase with no clause opener after it.
    """
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        "It seems we lost connection mid-sentence.",
        "We lost connection mid-sentence.",
        "Sorry, you cut out for a moment.",
    ):
        out, cut = _repair_chunk(text, "I said I said AI engineer rule.")
        assert "narrated the machinery" in cut, (text, cut)
        assert not out.strip(), f"left {out!r} behind"

    # A real sentence after it survives.
    out, cut = _repair_chunk(
        "It seems we lost connection mid-sentence. What is your notice period?",
        "")
    assert "narrated the machinery" in cut, cut
    assert "notice period" in out, out


def test_a_claim_about_no_positions_that_match_is_left_alone():
    """The narrow half of the new pattern: "no positions" alone is not a claim.

    "There are no positions that match your profile" would be cut to "that
    match your profile", which is worse than leaving it. Requiring a qualifier
    for position/role keeps that safe.
    """
    from voice_agent.orchestrator import _repair_chunk

    text = "There are no positions that match your profile at the moment."
    out, cut = _repair_chunk(text, "anything for me?")
    assert not cut, f"cut {cut} from a reply that is not claiming anything: {out!r}"


@pytest.mark.parametrize(
    "text,wanted",
    [
        # Run 24 turns 1, 2 and 9 each asked twice in one breath.
        (
            "Got it, thanks. How many years have you worked with Python? "
            "And what is your current tech stack?",
            "Got it, thanks. How many years have you worked with Python?",
        ),
        (
            "Hi, this is Riya from IT Staffing Solutions. Do you have a minute "
            "to talk? Are you currently working?",
            "Hi, this is Riya from IT Staffing Solutions. Do you have a minute to talk?",
        ),
    ],
)
def test_a_second_question_in_one_chunk_is_dropped(text, wanted):
    """Voice rule 1 says one question per turn; the 3B model does not obey it."""
    from voice_agent.orchestrator import _repair_chunk

    out, cut = _repair_chunk(text, "hello")
    assert "second question" in cut, cut
    assert out == wanted, out


def test_a_single_question_followed_by_a_statement_survives():
    """One question is not the defect -- only a second one is.

    Cutting at the first "?" unconditionally would turn "Are you free to talk?
    I'll be quick." into "Are you free to talk?" and lose the reassurance.
    """
    from voice_agent.orchestrator import _repair_chunk

    text = "Are you free to talk? I'll be quick."
    out, cut = _repair_chunk(text, "hello")
    assert not cut, f"cut {cut} from a single-question reply: {out!r}"
    assert out == text


def test_a_second_question_streaming_in_a_later_chunk_is_also_dropped():
    """The two questions do not always arrive in the same chunk.

    `asked` is the per-turn flag that carries "already asked one" forward.
    """
    from voice_agent.orchestrator import _repair_chunk

    asked = [False]
    first, cut1 = _repair_chunk("Thanks for that. How many years?", "", asked)
    assert not cut1, cut1
    assert asked[0] is True

    second, cut2 = _repair_chunk("Great. And what is your tech stack?", "", asked)
    assert "second question" in cut2, cut2
    assert "tech stack" not in second.lower(), second


@pytest.mark.asyncio
async def test_a_garbled_answer_does_not_end_the_call():
    """Run 25 turn 6: she tried to hang up over a transcription error.

    The caller had just clarified their answer ("I said My SQL and post Gray
    SQL") and Riya replied "Sorry it seems we didn't quite match thanks
    anyway." That is not a goodbye, so _FAREWELL never saw it and the farewell
    counter never moved -- but the caller hears it as being written off. Only
    the caller ends the call.
    """
    from voice_agent.llm import GenerationStats

    agent = make_agent("Sorry it seems we didn't quite match thanks anyway.")
    agent._loop = asyncio.get_running_loop()

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, _ = await agent._consume_chunks(
        queue, m, stop_event, guard=True, caller_said="I said MySQL and PostgreSQL"
    )
    thread.join(timeout=2)

    assert "match" not in spoken.lower(), spoken
    assert "again" in spoken.lower(), spoken
    assert agent._blocked_bail is True


@pytest.mark.asyncio
async def test_the_caller_gets_to_end_the_call():
    """The mirror of the test above -- their call, their goodbye.

    When the caller is the one backing out, "not a match" is the honest answer
    and replacing it with a request to repeat would be absurd.
    """
    from voice_agent.llm import GenerationStats

    agent = make_agent("Sounds like it's not a match thanks anyway.")
    agent._loop = asyncio.get_running_loop()

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, _ = await agent._consume_chunks(
        queue, m, stop_event, guard=True, caller_said="Actually I'm not interested."
    )
    thread.join(timeout=2)

    assert "match" in spoken.lower(), spoken
    assert agent._blocked_bail is False


@pytest.mark.parametrize(
    "text",
    [
        "Sorry it seems we didn't quite match thanks anyway.",
        "We don't seem to be a match.",
        "That doesn't really work out. Thanks anyway.",
        "I'll leave it there.",
    ],
)
def test_giving_up_mid_call_is_detected(text):
    from voice_agent.orchestrator import _BAILOUT

    assert _BAILOUT.search(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Thanks for your time, have a great day.",
        "Great, five years with Python. What about Spark?",
        "We will follow up regarding suitable roles.",
        "Can you confirm your database skills were MySQL?",
        "That doesn't seem relevant to our conversation.",
    ],
)
def test_a_working_reply_is_not_read_as_giving_up(text):
    from voice_agent.orchestrator import _BAILOUT

    assert not _BAILOUT.search(text), text


def test_a_bare_no_is_an_answer_not_a_refusal():
    r"""A bare "no" is an answer, not a refusal -- `no\s*you?` once matched
    the word "no" on its own.

    A caller answering a yes/no question says it constantly; reading that as
    "I want off this call" would have let every bail-out stand.
    """
    from voice_agent.orchestrator import _DECLINE

    for refusal in (
        "Actually I'm not interested.",
        "No thanks.",
        "I'm not looking right now.",
        "Please don't call me again.",
    ):
        assert _DECLINE.search(refusal), refusal
    for answer in ("No.", "No, I said MySQL and PostgreSQL.", "Yes.", "Uh huh."):
        assert not _DECLINE.search(answer), answer


@pytest.mark.parametrize(
    "text,caller,lost,wanted",
    [
        # The redirect that came back with a different last word.
        (
            "Our recruitment team will discuss salary details later can you "
            "confirm your highest level of education completed?",
            "Hello, I'm free.",
            "salary",
            "Can you confirm your highest level of education",
        ),
        # Meta-commentary: narrating the call instead of making it.
        (
            "I don't know how long ago your education ended can you tell me "
            "approximately how many years of total experience you have?",
            "Uh soward now",
            "I don't know",
            "Can you tell me approximately how many years",
        ),
        (
            "You are correct we should confirm your previous answers instead of "
            "repeating them let's move forward with your tech stack which "
            "specific frameworks besides Python and Java did you use during "
            "your projects?",
            "How many times do I have to tell you",
            "previous answers",
            "which specific frameworks besides Python and Java",
        ),
        # Defining jargon back at the person who asked about it.
        (
            "Data warehouse management systems refer to software tools that help "
            "manage large datasets for analysis.\n\nCan I get an idea of some "
            "databases specifically designed for storing structured information "
            "such as relational databases?",
            "What is data warehouse management system",
            "refer to software tools",
            "Can I get an idea of some databases",
        ),
        # Commenting on the caller ending the call.
        (
            "You're ending this conversation early thanks for your time goodbye.",
            "And uh Yeah.",
            "ending this conversation",
            "Thanks for your time goodbye",
        ),
    ],
)
def test_run_26_failures_are_cut_in_place(text, caller, lost, wanted):
    """Five shapes from run 26, each cut without paying for a regeneration.

    Run 26 was the first FAIL in nine runs. The model rambled (27-40 tokens
    against 13-22) and every one of these went out loud.
    """
    from voice_agent.orchestrator import _repair_chunk

    out, cut = _repair_chunk(text, caller)
    assert cut, f"nothing was cut from {text!r}"
    assert lost.lower() not in out.lower(), out
    assert wanted in out, out


def test_a_reply_that_is_only_the_redirect_is_cut_away():
    """Run 35 reversed this test, on purpose -- read before "fixing" it.

    It used to assert the opposite: "cutting it would leave silence; the
    off-topic guard rejects it instead." Both halves of that are now false.

      - Regeneration is OFF (regenerate_on_reject false, measured on runs
        11-14: +850 ms to +1800 ms per rejection and the retry was worse).
        So the guard does not REPLACE the line, it only reports it -- the
        line goes out anyway.
      - The questions are templated. Python appends one, so an empty
        reaction does not mean the caller is met with silence.

    Silence is therefore no longer what a cut costs; one wrong sentence is.
    """
    from voice_agent.orchestrator import _repair_chunk

    out, cut = _repair_chunk(
        "Our recruitment team will discuss salary details later.", "hello"
    )
    assert cut, "the redirect is the whole reply and it still went out"
    assert not out.strip(), out


@pytest.mark.asyncio
async def test_a_reply_that_is_only_a_forbidden_topic_still_leaves_a_question():
    """The guarantee the test above depends on: the caller hears a question.

    Run 35 turn 1 spoke "No salary discussion today When needed." -- no
    recruitment team, no "will discuss", nothing any redirect pattern
    recognises. Cutting it empties the reaction, which is only acceptable
    because the templated question is synthesised separately.
    """
    agent = make_agent("No salary discussion today When needed.")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(
        ["I have a masters in computer science"])

    m = await _drive_turn(agent, None)

    spoken = " ".join(agent.tts.calls)
    assert "salary" not in spoken.lower(), spoken
    assert spoken.strip(), "the reaction was cut and nothing replaced it"
    # ...and nothing of the cut reply survives. The chunker splits on clause
    # boundaries, so the tail ("When needed.") arrives as its own chunk with
    # no topic word left for any rule to match, and it used to be spoken on
    # its own in front of the question.
    assert "when needed" not in spoken.lower(), spoken

    # And the turn is MEASURED. Run 37 turn 1 reported "TIME TO FIRST AUDIO
    # 0 ms" for exactly this shape: the reaction was cut to nothing, so no
    # streamed chunk played, so _consume_chunks never stamped first audio and
    # the metric fell back to 0. A 0 ms turn drags the average down and makes
    # the definition of done meaningless.
    assert m.time_to_first_audio_ms > 0, (
        "a turn whose audio came from the templated question was not timed")


def test_a_bare_topic_word_note_is_stripped():
    """Run 26 turn 3 spoke the item as one word: "...have? Qualification"."""
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = ["qualification and field?", "years of experience?"]
    assert _strip_agenda_suffix(
        "Can you give me approximately how many years of total experience you "
        "have? Qualification",
        agenda,
    ) == "Can you give me approximately how many years of total experience you have?"
    # "years" is an agenda item's first word too, but here it is the question.
    assert _strip_agenda_suffix("How many years?", agenda) == "How many years?"


def test_a_note_in_the_middle_of_a_sentence_is_stripped():
    """Run 26 turn 12 read the item aloud mid-sentence, not as a suffix."""
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = ["qualification and field?", "read back and confirm?"]
    out = _strip_agenda_suffix(
        "No we strayed from our original purpose now let me clarify something "
        "read back and confirm what technology languages frameworks you've "
        "worked on recently.",
        agenda,
    )
    assert "read back and confirm" not in out.lower(), out
    assert "what technology languages" in out, out
    # The binder gate still protects a real question ending on the item.
    assert _strip_agenda_suffix(
        "Can you tell me about your qualification and field?", agenda
    ) == "Can you tell me about your qualification and field?"


def test_a_read_back_is_not_credited_until_it_asks_for_confirmation():
    """Run 25 turn 4 echoed the stack and moved on; nothing was confirmed.

    "You've mentioned Python, Java and SQL so far Can you tell me specifically
    which databases those SQL skills apply to?" ends in a question, so the
    checklist marked "read back and confirm?" done -- and the agenda ran out
    two turns early, which is the frame turn 6's bail-out was generated in.
    """
    from voice_agent.orchestrator import _READBACK_CONFIRM

    assert not _READBACK_CONFIRM.search(
        "You've mentioned Python, Java and SQL so far Can you tell me "
        "specifically which databases those SQL skills apply to?"
    )
    for good in (
        "You have five years, Python, Java and SQL -- does that sound right?",
        "So that's five years with Python and Spark. Is that correct?",
        "Let me read that back: five years, Python, Java. Anything to add?",
        "Did I get that right -- five years, and mostly Python?",
        "Any corrections to what I have so far?",
    ):
        assert _READBACK_CONFIRM.search(good), good


def test_a_question_shaped_like_a_thank_you_does_not_end_the_call():
    """Run 27 hung up mid-sentence on two lines that were not goodbyes.

    "can I thank you for taking time today?" and "Can I thank you for chatting
    with me today though?" both matched `thank you for (taking|chatting)`, hit
    max_farewells, and dropped the call while the caller was still speaking.
    A reply that ends in a question mark is still asking something.
    """
    from voice_agent.orchestrator import _is_farewell

    for question in (
        "I'm glad I could clarify my role briefly now can I thank you for taking time today?",
        "No there isn't an opening listed yet. Can I thank you for chatting with me today though?",
        "Thanks for your time -- anything else you'd like to add?",
    ):
        assert not _is_farewell(question), question

    for goodbye in (
        "Thanks for your time, have a great day.",
        "Have a great day thank you.",
        "Goodbye.",
        "No thank you for speaking with me today it was good talking to you.",
    ):
        assert _is_farewell(goodbye), goodbye


def test_the_call_waits_for_the_caller_after_the_last_goodbye():
    """The goodbye is Riya's; stopping is the caller's decision.

    Run 27 and run 30 both dropped the line while the caller was mid-sentence.
    """
    import voice_agent.orchestrator as orch

    src = open(orch.__file__, encoding="utf-8").read()
    # Arm rather than stop: the listen loop cancels this if they speak.
    assert "self._hangup_at = time.monotonic()" in src
    assert "(caller spoke; staying on the call)" in src
    # And it must not fire while her own goodbye is still playing.
    assert "self.speaker.pending_seconds <= 0" in src


def test_acknowledging_the_caller_is_not_reported_as_guessing():
    """Run 22 flagged "That sounds like a solid foundation..." as fabrication.

    That reply was correct -- an acknowledgement plus the next question. The
    detector was matching any "sounds like a/an/the". What actually matters is
    the subject: run 8's real failure was "Deliver sounds like an interesting
    project management tool", where a garbled word is inflated into a product.
    A generic subject, or none, means vague praise, not an invented fact.
    """
    from voice_agent.orchestrator import _GENERIC_SOUND_SUBJECTS, _GUESSING

    def fires(text: str) -> bool:
        match = _GUESSING.search(text)
        if not match:
            return False
        subject = (match.group(1) or "").strip(".,!?;:").lower()
        return bool(subject) and subject not in _GENERIC_SOUND_SUBJECTS

    # Correct acknowledgements -- must stay silent.
    assert not fires("That sounds like a solid foundation. How many years?")
    assert not fires("It sounds like a good fit for the role.")
    assert not fires("Sounds like a great background.")
    # The failure this guard exists for -- must still fire.
    assert fires("Deliver sounds like an interesting project management tool.")


def test_a_real_question_ending_on_the_item_is_left_alone():
    """Amputating a correct rewrite would be worse than the leak.

    "Can you tell me about your qualification and field?" ends on the item but
    is a genuine question -- the word before it binds to it.
    """
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = ["qualification and field?", "years of experience?"]
    for reply in (
        "Can you tell me about your qualification and field?",
        "Which years of experience?",
        "What about the years of experience?",
    ):
        assert _strip_agenda_suffix(reply, agenda) == reply


def test_a_chunk_that_is_only_the_note_leaves_nothing_to_speak():
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = ["qualification and field?", "years of experience?"]
    assert _strip_agenda_suffix("Years of experience?", agenda) == ""
    assert _strip_agenda_suffix("years of experience", agenda) == ""


def test_a_short_leftover_survives_the_agenda_strip():
    """Run 24: the strip deleted real speech and Riya stopped mid-sentence.

    The sentence chunker split one sentence across two chunks -- "...how you
    use" and then "it?\\nRead back and confirm?" -- so what was left after
    removing the note was "it?". Three characters, under the old four-character
    floor, so the whole chunk was dropped and the caller heard "...how you use"
    with no ending. A leftover that still contains letters is speech.
    """
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = ["read back and confirm?", "close the call"]
    assert _strip_agenda_suffix("it?\nRead back and confirm?", agenda) == "it?"
    # Not a second terminator bolted onto one that is already there.
    assert not _strip_agenda_suffix("it?\nRead back and confirm?", agenda).endswith("?.")


def test_an_ordinary_reply_is_untouched():
    from voice_agent.orchestrator import _strip_agenda_suffix

    agenda = ["qualification and field?", "years of experience?"]
    reply = "How many years have you been working with Python and Spark?"
    assert _strip_agenda_suffix(reply, agenda) == reply


def test_parroting_the_caller_is_caught():
    """A live turn answered "working with the big team" with the same words back.

    It repeats nothing Riya said, so the repetition guard never saw it.
    """
    assert _parrots("Working with the big team.", "Uh working with the big team")
    assert _parrots("Minimum two.", "Uh minimum two")


def test_a_real_reply_is_not_mistaken_for_parroting():
    assert not _parrots(
        "Good, that fits. How many years on Spark?",
        "I have three years of experience in data engineering",
    )
    # A question that reuses their words is still progress, not an echo.
    assert not _parrots(
        "Big team - how many engineers?",
        "working with the big team",
    )


@pytest.mark.asyncio
async def test_assistant_voice_is_suppressed_before_it_is_spoken():
    agent = make_agent("Hello there! How can I assist you today?")
    agent._loop = asyncio.get_running_loop()
    agent._regenerate = True               # opt in: this tests the veto path

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    from voice_agent.llm import GenerationStats

    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, rejected = await agent._consume_chunks(queue, m, stop_event, guard=True)
    thread.join(timeout=2)

    assert rejected is not None and rejected[0] == "assistant"
    assert spoken == ""


@pytest.mark.asyncio
async def test_only_two_sentences_are_ever_generated():
    """max_tokens alone did not hold the line on a live call.

    A 46-token reply asked three questions in a row, well inside the 90-token
    budget of the time. Breaking the stream is the structural fix.
    """
    ramble = ("Let me see. Do you mean Python? Or do you mean Spark? "
              "Or perhaps you mean AWS? There is more.")
    agent = make_agent(ramble, llm_delay=0.001)
    agent._loop = asyncio.get_running_loop()

    m = await agent.say_text("which skills?")
    sentences = sum(m.response.count(t) for t in ".!?")
    assert sentences <= 2, f"spoke {sentences} sentences: {m.response!r}"
    assert "There is more" not in m.response


# --------------------------------------------------------------------------
#  Template artifacts and the "caller kept talking" path
# --------------------------------------------------------------------------

from voice_agent.orchestrator import _has_artifact  # noqa: E402


@pytest.mark.parametrize("reply,expected", [
    ("You: I called about a Senior Data Engineer role.", "You:"),
    ("I called about a role at [Client Name].", "[Client Name]"),
    ("Is there a date that works?</think>", "</think>"),
    ("2. open to the role?", "2."),
    ("I rang about the {role} opening.", "{role}"),
])
def test_template_artifacts_are_caught(reply, expected):
    """All of these reached the speaker in one live call."""
    assert _has_artifact(reply) == expected


@pytest.mark.parametrize("reply", [
    "Got it, that fits. How many years on Spark?",
    "No problem - later today, or tomorrow morning?",
    "I rang about a data engineering role. Got two minutes?",
])
def test_ordinary_replies_have_no_artifacts(reply):
    assert not _has_artifact(reply)


@pytest.mark.asyncio
async def test_talking_over_a_thinking_agent_keeps_the_words():
    """Nothing was played, so this is one utterance, not an interruption.

    The old path logged "BARGE | dropped 0.00s" and binned the turn while
    leaving its transcript in history unanswered -- nine turns disappeared that
    way in a single call, and the prompt ended up with runs of consecutive user
    messages that the model handled badly.
    """
    agent = make_agent("Got it.")
    agent._loop = asyncio.get_running_loop()

    m = TurnMetrics(turn_id=1)
    m.transcript = "I am looking for a long term role"
    agent._active_metrics = m
    agent.llm.add_user("I am looking for a long term role")
    agent._active_turn = asyncio.create_task(asyncio.sleep(5))

    await agent._absorb_unspoken_turn()

    assert "I am looking for a long term role" in agent._carry_text
    # And it is no longer sitting in history as an unanswered user turn.
    assert not [t for role, t in agent.llm.history if role == "user"]
    assert agent.speaker.interrupts == 0, "nothing was playing; nothing to cut"


@pytest.mark.asyncio
async def test_a_real_barge_in_still_cuts_playback():
    """The quiet path must not swallow genuine interruptions."""
    agent = make_agent("word. " * 40, llm_delay=0.02, max_sentences=0)
    agent._loop = asyncio.get_running_loop()
    agent.speaker._pending = 2.0          # audio is queued, i.e. they heard us

    m = TurnMetrics(turn_id=1)
    agent._active_metrics = m
    agent._active_turn = asyncio.create_task(asyncio.sleep(5))

    await agent._barge_in()
    assert agent.speaker.interrupts == 1
    assert agent._pending_interruption is True


def test_a_checklist_note_read_aloud_is_caught():
    """Item two, spoken at the candidate about the candidate.

    A recruiter on a call says "you", never "they" -- so overlap with an agenda
    item plus third-person framing is a reliable tell.
    """
    from voice_agent.orchestrator import _echoes_agenda
    from voice_agent.prompts import DEFAULT_CALL, build_agenda

    agenda = build_agenda(DEFAULT_CALL)
    # Derived from the live checklist so this survives a change of agenda: the
    # tell is the third person, not the particular wording of an item.
    for item in agenda[:3]:
        assert _echoes_agenda(
            f"What about their {item.rstrip('?').strip()}?", agenda
        ), item


def test_the_note_read_aloud_in_the_first_person_is_caught_too():
    """Run 9 spoke the checklist item back to the caller verbatim.

    "Years of experience?" and "Read back and confirm?" were not reworded at
    all -- the directive text went straight out of the speaker. The echo check
    only looked for third-person framing, which caught the note quoted ABOUT
    the caller but not the note quoted AT them. A reply no longer than the
    item itself is the note, whatever the pronoun.
    """
    from voice_agent.orchestrator import _echoes_agenda
    from voice_agent.prompts import DEFAULT_CALL, build_agenda

    agenda = build_agenda(DEFAULT_CALL)
    for item in agenda:
        assert _echoes_agenda(item, agenda), item
        # Same words, capitalised as a sentence -- still just the note.
        assert _echoes_agenda(item.capitalize(), agenda), item

    # A real question says more than the note does.
    assert not _echoes_agenda(
        "How many years of experience do you have?", agenda)
    assert not _echoes_agenda(
        "Python Java SQL are common tools in data engineering. Which of these "
        "have you worked with?", agenda)


def test_the_same_question_asked_properly_is_allowed():
    from voice_agent.orchestrator import _echoes_agenda
    from voice_agent.prompts import DEFAULT_CALL, build_agenda

    agenda = build_agenda(DEFAULT_CALL)
    for good in (
        "Are you open to hearing about it?",
        "Do you need sponsorship, or are you good to work here?",
        "What rate are you after?",
        "Are they flexible on the rate at their end?",   # about the client
    ):
        assert not _echoes_agenda(good, agenda), good


# --------------------------------------------------------------------------
#  Reworded repeats (the "production pipelines" loop)
# --------------------------------------------------------------------------

_PIPELINE_LOOP = [
    "Do you have any experience with production pipeline ownership?",
    "Did you work on production pipelines before?",
    "Have you worked on production pipelines before?",
    "Have you ever managed production pipelines in your career?",
]


@pytest.mark.parametrize("i", [1, 2, 3])
def test_the_same_question_reworded_is_caught(i):
    """Four askings of one question, all missed by raw word overlap.

    do/did/have/any/ever/before/your dominate the token count, so two identical
    questions scored 0.43 similar. Comparing stemmed content words reduces both
    to {work, production, pipeline} and the repeat becomes obvious.
    """
    assert any(_too_similar(_PIPELINE_LOOP[i], _PIPELINE_LOOP[j]) for j in range(i)), (
        f"{_PIPELINE_LOOP[i]!r} was not recognised as a repeat"
    )


def test_plurals_and_verb_forms_compare_equal():
    from voice_agent.orchestrator import _stem

    assert _stem("pipelines") == _stem("pipeline")
    assert _stem("worked") == _stem("working") == _stem("works") == "work"
    assert _stem("years") == _stem("year")
    # ...without mangling words that merely end in s.
    assert _stem("business") == "business"


@pytest.mark.parametrize("a,b", [
    ("How many years on Spark?", "What rate are you after?"),
    ("Are you open to the role?", "Do you need sponsorship?"),
    ("What's your notice period?", "Does Bengaluru work for you?"),
    ("Got two minutes?", "What rate are you targeting?"),
])
def test_different_questions_are_still_allowed(a, b):
    assert not _too_similar(a, b)


def test_a_correctly_phrased_question_is_not_rejected_as_an_agenda_echo():
    """The tracker names the live item, so a good question resembles it.

    Rejecting on resemblance alone would reject exactly the rewrite we asked
    for. Only third-person framing -- talking about the caller as "they" while
    talking to them -- is a genuine tell.
    """
    from voice_agent.orchestrator import _echoes_agenda
    from voice_agent.prompts import DEFAULT_CALL, build_agenda

    agenda = build_agenda(DEFAULT_CALL)
    for good in (
        "How many years have you been hands-on with Python, Spark and AWS?",
        "Does Bengaluru work for you, three days onsite?",
        "What rate are you targeting?",
    ):
        assert not _echoes_agenda(good, agenda), good


# --------------------------------------------------------------------------
#  Grammatical person (talking TO the caller, not ABOUT them)
# --------------------------------------------------------------------------

from voice_agent.orchestrator import _wrong_person  # noqa: E402


@pytest.mark.parametrize("reply", [
    # "Yes, we're available." is the case this guard was BUILT for, and it is
    # no longer blocked -- see test_speaking_for_her_own_side_is_not_the_wrong_person.
    "They've confirmed interest. What's your experience with Python?",
    "Their notice period is what matters here.",
])
def test_talking_about_the_caller_is_rejected(reply):
    """All observed live, and invisible to every other guard.

    These echo nothing and invent nothing -- they are simply in the wrong
    grammatical person. Riya is on the phone with one person; that person is
    "you".
    """
    assert _wrong_person(reply)


@pytest.mark.parametrize("reply", [
    "Got it, that fits. How many years on Spark?",
    "Good. What's your notice period?",
    "Understood. Are you open to three days onsite?",
    # "They" about the CLIENT is fine, and the sentence addresses the caller.
    "They're flexible on rate - does that work for you?",
    "Fair enough. Mind if I keep you on file?",
])
def test_second_person_replies_are_allowed(reply):
    assert not _wrong_person(reply)


def test_the_turn_directive_never_says_they_or_we():
    """The root cause, not the symptom.

    The directive used to read "acknowledge what THEY just said", and the model
    took its pronouns straight from it -- answering "They've confirmed interest."
    An instruction written about the caller in the third person produces replies
    about the caller in the third person.
    """
    from voice_agent.prompts import AgendaTracker

    t = AgendaTracker(["target rate?", "work authorization?"])
    for directive in (t.directive(), (t.advance(), t.directive())[1]):
        body = directive.replace('"they"', "").replace('"we"', "")
        for pronoun in (" they ", " them ", " their ", " we "):
            assert pronoun not in f" {body.lower()} ", f"{pronoun!r} in {directive!r}"


def test_the_greeting_only_consumes_an_item_it_actually_asks():
    """greeting_covers must match what the greeting says, not what it used to.

    It was 1 on the reasoning that "have you got a couple of minutes?" was
    checklist item one -- true of an older agenda whose first item was
    "bad time?". The current checklist opens with "qualification and field?",
    which the greeting never mentions, so 1 silently discarded the first real
    screening question. Run 11 turn 1 was then rejected off_agenda for asking
    about qualifications: Riya was asking the item the tracker had thrown away.

    Permission to talk is not a screening question. Asserted as an invariant
    rather than a constant, so rewriting the greeting cannot desync it again.
    """
    from voice_agent.config import load_config
    from voice_agent.orchestrator import _content_words
    from voice_agent.prompts import AgendaTracker, build_agenda

    cfg = load_config()
    items = build_agenda(dict(cfg.get("call", {}) or {}))
    covers = int(cfg.agent.get("greeting_covers", 1))
    greeting = str(cfg.agent.get("greeting", "")).lower()

    asks_first = bool(_content_words(items[0]) & set(
        w.strip(",.!?") for w in greeting.split()))
    assert covers == (1 if asks_first else 0), (
        f"greeting_covers={covers} but the greeting "
        f"{'does' if asks_first else 'does not'} ask item one {items[0]!r}"
    )

    tracker = AgendaTracker(items)
    for _ in range(covers):
        tracker.advance()
    assert tracker.current is not None
    assert "bad time" not in tracker.current


# ---------------------------------------------------------------------------
# Guard precision: a rejection costs a whole extra generation, and the
# regenerated reply is the one that has been fabricating.
# ---------------------------------------------------------------------------

def test_a_good_tech_stack_question_is_not_rejected():
    """Regression: shortening AGENDA_TEMPLATES silently broke this guard.

    The live item "tech stack -- which ones?" has the content words
    {tech, stack, one}. Scoring a question against those gave
    "Which programming languages do you work with?" an overlap of 0.00, so it
    was rejected as off-agenda. Measured on a real call, 6 of 9 good questions
    scored under the threshold -- and each rejection discarded a correct reply
    in favour of a regenerated one that invented details ("Five plus years",
    "You currently work with Java and Python").
    """
    from voice_agent.orchestrator import _invents_question

    good = [
        ("tech stack -- which ones?",
         "Which programming languages specifically are part of your daily coding routine?"),
        ("tech stack -- which ones?",
         "What technologies specifically do you actually work with day to day?"),
        ("years of experience?",
         "Can you please confirm how much work experience you've accumulated over time?"),
        ("qualification and field?",
         "What did you study, and what degree do you hold?"),
    ]
    for item, question in good:
        assert not _invents_question(question, item), (item, question)


def test_a_question_about_another_topic_is_still_rejected():
    """The fix must not have turned the guard off."""
    from voice_agent.orchestrator import _invents_question

    bad = [
        ("years of experience?", "Are you open to a contract-to-hire engagement?"),
        ("tech stack -- which ones?", "What is your current notice period?"),
        ("qualification and field?", "What is your expected hourly rate?"),
    ]
    for item, question in bad:
        assert _invents_question(question, item), (item, question)


def test_a_closing_turn_may_not_open_a_new_data_question():
    """Run 6 turn 7 asked for the qualification again after wrap-up.

    The caller heard the same question twice and the call then hung up without
    ever confirming anything.
    """
    from voice_agent.orchestrator import _invents_question

    assert _invents_question(
        "What was your highest level of academic qualification?", "wrap-up")
    assert _invents_question(
        "How many years of experience do you have?", "close the call")
    # A closing that closes is fine.
    assert not _invents_question(
        "Thanks for your time, our team will follow up. Goodbye!", "close the call")


def test_a_read_back_may_repeat_the_callers_own_figures():
    """Confirmation means repeating what the caller said.

    "So you have five years of experience and work with Python -- is that
    right?" shares no words with the terse item "read back and confirm?", and
    it states a number. Without the caller's own words being on-topic and
    exempt from the fabrication check, every confirmation would be rejected.
    """
    from voice_agent.orchestrator import _invents_question, _invents_specifics

    caller = ("I have five years of experience. Python is my daily coding "
              "routine. Mostly machine learning.")
    assert not _invents_question(
        "So you have five years of experience and work with Python -- is that right?",
        "read back and confirm?", caller)
    assert not _invents_specifics(
        "So that's five years of experience and Python -- correct?", caller)


def test_a_spelled_out_number_is_caught_like_a_digit():
    """Run 6: "Five plus years typically qualifies us for certain roles".

    The caller had never mentioned five years -- they said it two turns later.
    The pattern only matched digits, so the claim sailed through.
    """
    from voice_agent.orchestrator import _invents_specifics

    assert _invents_specifics(
        "Five plus years typically qualifies us for certain roles.")
    assert _invents_specifics("The notice period is three months.")
    # No number, no claim.
    assert not _invents_specifics("Thanks, that's helpful.")


def test_an_off_agenda_question_in_the_second_sentence_is_still_caught():
    """Run 7: the question arrived after the guard window had already shut.

    "Years of relevant professional experience are important here. Are there
    5+ years of relevant professional experience included in your current job
    title?" -- sentence one is 61 characters and GUARD_CHARS was 60, so the
    behavioural checks closed before the question streamed in and it reached
    the caller while the live item was "read back and confirm?".
    """
    from voice_agent.orchestrator import _invents_question

    assert _invents_question(
        "Years of relevant professional experience are important here. "
        "Are there 5+ years of relevant professional experience included in "
        "your current job title?",
        "read back and confirm?",
    )


def test_a_single_topic_word_is_enough_to_reject_a_closing_turn():
    """Run 7 asked for the qualification again after the call had closed.

    "What was your highest level of education completed?" shares exactly one
    word ("education") with the qualification vocabulary, so a threshold of
    two let it through. A closing turn should carry no data question at all.
    """
    from voice_agent.orchestrator import _invents_question

    assert _invents_question(
        "What was your highest level of education completed?", "close the call")
    assert _invents_question(
        "What was your highest qualification when you graduated?", "wrap-up")
    assert not _invents_question(
        "Thanks very much for chatting with me.", "close the call")


def test_speaking_for_her_own_side_is_not_the_wrong_person():
    """Run 7 rejected a reply for containing the word "our".

    The screening brief REQUIRES "our recruitment team will discuss that with
    you separately", and the opening names "our database" -- so the guard was
    rejecting phrasing the prompt itself mandates. That was one of three
    rejections that call, each costing a full extra generation.
    """
    from voice_agent.orchestrator import _wrong_person

    for ok in ("Our recruitment team will follow up regarding suitable openings.",
               "This conversation is so our recruitment team will discuss "
               "suitable openings with you separately.",
               "Your profile came up in our database.",
               "Our client is a healthcare analytics firm."):
        assert not _wrong_person(ok), ok

    # Talking ABOUT the caller is still wrong, singular as well as plural --
    # "he"/"she" were named in the function's own docstring but were missing
    # from the pattern, so it could never have caught them.
    for bad in ("They have confirmed interest.",
                "Has he worked with Spark before?",
                "Does she have five years?"):
        assert _wrong_person(bad), bad


def test_first_person_plural_never_costs_a_regeneration():
    """Run 13 rejected three replies for "we"/"our"/"We" and made them worse.

    Turn 13's retry opened "I shouldn't say 'we'." -- the model narrating the
    rule it had just been handed, instead of answering the caller. Turn 9's
    retry still said "our" anyway. Three full generations spent, and two of
    the three replacements were worse than what they replaced.

    "Yes, we're available" is genuinely wrong and is now NOT caught; that is
    the accepted cost. Third person about the caller still blocks, because
    "they have confirmed interest" confuses the caller in a way "we will
    follow up" does not.
    """
    from voice_agent.orchestrator import _wrong_person

    for ok in ("Yes, we're available. How long have you been considering roles?",
               "We're glad to hear that. Are you open to a Senior Data Engineer role?",
               "Yes, we are available.",
               "Our recruitment team will follow up."):
        assert not _wrong_person(ok), ok

    # Third person about the caller still blocks.
    assert _wrong_person("They have confirmed interest.")


def test_a_placeholder_employer_is_caught_but_an_employer_question_is_not():
    """Run 7: "You currently work at XYZ company ...".

    The call brief names no employer, so Riya has no basis for one. Checked
    regardless of the trailing "?", because the run-on reply put a fabricated
    employer and a question in the same unpunctuated sentence -- which made
    the numeric-fact check skip it as a question.
    """
    from voice_agent.orchestrator import _invents_employer

    assert _invents_employer(
        "You currently work at XYZ company can you tell me some technologies "
        "or tools like SQL Server Reporting Services?")
    assert _invents_employer("You currently work for Infosys right?")
    # Legitimate questions about the employer, not claims about it.
    assert not _invents_employer("Do you work at Acme Corp?")
    assert not _invents_employer(
        "Can you tell me which company you currently work for?")


def test_one_decisive_topic_word_outranks_the_overlap_ratio():
    """Run 8 rejected a correct question for being long, not for being wrong.

    "Four years sounds good Can you confirm how many years specifically?"
    shares only "year" with the experience vocabulary -- 0.17 against a 0.25
    threshold -- because the polite padding diluted the denominator. One
    unmistakable topic word is stronger evidence than a ratio, so anchors
    short-circuit the ratio test.
    """
    from voice_agent.orchestrator import _invents_question

    assert not _invents_question(
        "Four years sounds good Can you confirm how many years specifically?",
        "years of experience?")


def test_adjectival_forms_still_match_their_topic():
    """_stem does not strip "-al", so "educational" != "education".

    Run 8's opening asked about the caller's "educational background" and was
    rejected as off-agenda purely on that missing suffix.
    """
    from voice_agent.orchestrator import _invents_question

    assert not _invents_question(
        "So can you give me some details about your educational background?",
        "qualification and field?")


def test_a_closing_turn_may_only_ask_about_the_follow_up():
    """Run 8 asked the candidate to opine on AI/ML after the call had closed.

    "Can AI/ML be applied generally across different types of projects besides
    health analytics?" shares no vocabulary with the checklist at all, so a
    topic blacklist could never catch it -- that was true in runs 6, 7 and 8,
    three different backwards questions. At closing, only arranging the
    follow-up is still a legitimate question.
    """
    from voice_agent.orchestrator import _invents_question

    assert _invents_question(
        "Can AI/ML be applied generally across different types of projects "
        "besides health analytics?", "close the call")
    # Arranging the follow-up is fine.
    assert not _invents_question(
        "Shall I have our recruitment team follow up with you?", "close the call")
    assert not _invents_question("Thanks very much for chatting with me.",
                                 "close the call")


def test_guessing_at_a_misheard_word_is_caught():
    """Run 8: transcription garbled a word, Riya invented a product for it.

    "Deliver sounds like an interesting project management tool." -- she had
    never heard of any such tool; the transcript had simply misheard the
    caller. Narrowed to "sounds like a/an/the" so the legitimate paraphrase
    "sounds like you have a lot of experience" still passes.
    """
    from voice_agent.orchestrator import _GUESSING

    assert _GUESSING.search(
        "Deliver sounds like an interesting project management tool.")
    assert not _GUESSING.search("Sounds like you have a lot of experience with Python.")


def test_a_deflected_topic_is_only_raised_when_the_caller_raised_it():
    """The most repeated bad line in runs 8, 9 and 10, all unprompted.

    The brief says to REDIRECT salary/rate/interview questions to the
    recruitment team. A 3B model reads that as "salary is a topic I should
    bring up", and opened run 10 with "Our recruitment team will discuss
    salary details later..." -- rejected as a repeat at turn 5, rejected
    again at turn 7. Nobody had asked.
    """
    from voice_agent.orchestrator import _raises_deflected_topic

    bad = ("Our recruitment team will discuss salary details later can you "
           "tell me how many years of experience you have?")
    assert _raises_deflected_topic(bad, "")
    assert _raises_deflected_topic("What is the salary for this role?", "")

    # The caller asked -- so the deflect wording is exactly right.
    assert not _raises_deflected_topic(
        "Our recruitment team will discuss salary details separately.",
        "What is the salary?")
    assert not _raises_deflected_topic(
        "I will let our recruitment team talk about the notice period.",
        "My notice period is 60 days.")

    # Ordinary turns never touch these words.
    assert not _raises_deflected_topic(
        "How many years of experience do you have?", "")


def test_asking_where_someone_works_is_not_claiming_they_work_there():
    """_EMPLOYER_CLAIM keys on a capitalised word after "at/for/with".

    That is a company name when Riya asserts one -- and a programming language
    when she asks: "Do you currently work with Python?" matched, and was
    rejected as a fabricated employer. The distinction is not the noun, it is
    assert vs. ask, so a question-shaped opener exempts the match.
    """
    from voice_agent.orchestrator import _invents_employer

    for question in ("Do you currently work with Python?",
                     "Are you currently working with React?",
                     "Which company do you currently work for?",
                     "Can you tell me where you currently work?",
                     "I was wondering whether you currently work at a product firm?"):
        assert not _invents_employer(question), question

    # Still catches the assertion, in both word orders and both cases.
    for claim in ("You currently work at Infosys",
                  "You are currently employed at Wipro",
                  "You now work with TCS isn't it",
                  "So you currently work at Google, is that right?"):
        assert _invents_employer(claim), claim

    # A question that CLOSES before the claim leaves a plain assertion.
    assert _invents_employer("Can you confirm? You currently work at Infosys.")


def test_a_rejection_tells_the_model_which_rule_it_broke():
    """correction_note() returned "" for fabrication and off_topic.

    Both of those are new guard kinds added for runs 8-10, and neither had a
    branch -- so the model was sent back to regenerate with no note at all,
    which is just "try again" and reliably produced the same failure. A retry
    costs roughly a full extra LLM pass, so it has to say what went wrong.
    """
    from voice_agent.prompts import correction_note

    for kind, detail in (("fabrication", "You currently work at Infosys"),
                         ("off_topic", "salary"),
                         ("repeat", "How many years of experience?")):
        note = correction_note(kind, detail)
        assert note.strip(), kind
        assert detail.split()[0].lower() in note.lower(), (kind, note)


class _ScriptedLLM(FakeLLM):
    """A different reply per generation, so the retry can be observed."""

    def __init__(self, replies):
        super().__init__(replies[0] if replies else "", delay_s=0.0)
        self.replies = list(replies)
        self.attempts = 0

    def stream(self, stop_event=None, extra_system=None, stats=None):
        self.reply = self.replies.pop(0) if self.replies else ""
        self.attempts += 1
        return super().stream(stop_event, extra_system, stats)


@pytest.mark.asyncio
async def test_a_rejected_turn_costs_exactly_two_generations():
    """Run 11 is why the retry is NOT guarded. Two passes, never three.

    Guarding the retry looked obviously right -- the unguarded retry is where
    runs 6-10's fabrications came from. But run 11 measured the result:

      - six of nine turns were rejected twice in a row, so the guarded retry
        rejected again every single time and caught nothing new;
      - each one then fell through to a third, fully ungated pass;
      - avg TTFA went 1284 ms -> 2546 ms, definition of done back to FAIL;
      - the third-pass replies were the worst of the three, e.g. "What
        technologies specifically does our recruitment team use at our health
        care company?"

    A guarded retry costs a whole extra generation and buys no quality. What
    makes the unguarded second pass safe instead is correction_note(), which
    now names the actual failure for every kind -- including fabrication and
    off_topic, which previously sent the model back with no note at all.
    """
    agent = make_agent("placeholder")
    agent._loop = asyncio.get_running_loop()
    agent._regenerate = True               # opt in: this tests the retry path
    agent.transcriber = _FakeTranscriber(
        ["I have been working in data engineering for a while."])
    agent.llm = _ScriptedLLM([
        # 1. rejected: invents an employer, which nobody mentioned.
        #    Deliberately NOT the old "our recruitment team will discuss
        #    salary" line -- since run 35 that one is repaired in place and
        #    costs no second generation at all, so it no longer exercises
        #    this path. A fabrication is not a shape the repairs can cut.
        "You currently work at Infosys.",
        # 2. the steered retry -- this is what the caller should hear.
        "Got it. So can you tell me about your qualification and field of study?",
        # 3. must never be reached.
        "You currently work at Infosys, is that right?",
    ])

    m = await _drive_turn(agent, None)

    assert agent.llm.attempts == 2, (
        f"{agent.llm.attempts} generation(s) -- a third pass costs another "
        "full prompt evaluation and, measured on run 11, is the worst of the three"
    )
    assert m.retried
    assert "Infosys" not in m.response, m.response
    assert "qualification" in m.response, m.response
    assert agent.speaker.played, "the retry produced no audio"
    # The third reply was never consumed.
    assert len(agent.llm.replies) == 1


@pytest.mark.asyncio
async def test_an_off_agenda_reply_is_spoken_not_regenerated():
    """Run 13 turn 2 paid 3046 ms because Riya asked a different question.

    Asking a reasonable-but-not-live screening question is a misprioritisation,
    not a defect the caller notices. Regenerating for it costs a whole extra
    prompt evaluation and, on that turn, produced the single slowest reply in
    the call. It is now logged and spoken.
    """
    agent = make_agent("placeholder")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["Oh very busy."])
    off = ("You seem quite interested in getting into data engineering roles "
           "soon can I help with that")
    agent.llm = _ScriptedLLM([off, "Can you tell me about your education?"])

    m = await _drive_turn(agent, None)

    assert agent.llm.attempts == 1, (
        f"{agent.llm.attempts} generation(s) -- off-agenda should warn, not regenerate"
    )
    assert not m.retried
    assert agent.speaker.played
    # NOTE: this reply no longer reaches the speaker. It is 16 words and
    # carries a first-person trailing question ("...soon can I help with
    # that"), so the reaction gate drops it -- the gate is deliberately strict
    # about a line that is about Riya rather than about the candidate. What
    # this test exists to prove is the part that still holds and still costs
    # real latency: off-agenda WARNS, it does not buy a second generation.
    assert "interested" not in m.response, m.response


@pytest.mark.asyncio
async def test_a_fabrication_is_warned_and_spoken_when_regeneration_is_off():
    """Run 19 invented "Years completed are 4 years" -- nobody had said that.

    Nothing caught it: checking_open was gated on _regenerate, so with
    regeneration off every check was dead code. Checks now always run, but
    with regeneration off they must WARN rather than veto -- a veto with no
    retry available leaves the agent silent, which is worse than a flawed
    sentence and is exactly what this guards against.
    """
    agent = make_agent("placeholder")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["Oh yes I'm free."])
    fabricated = "You currently work at Infosys."
    agent.llm = _ScriptedLLM([fabricated, "Can you tell me about your education?"])

    events: list[str] = []
    agent.on_event = lambda kind, text: events.append(str(text))

    m = await _drive_turn(agent, None)

    assert agent.llm.attempts == 1, "detection must not cost a generation"
    assert not m.retried
    # Spoken, not silenced.
    assert "Infosys" in m.response, m.response
    assert agent.speaker.played
    # ...and visible in the log rather than failing silently.
    assert any("fabrication" in e for e in events), events


@pytest.mark.asyncio
async def test_a_guard_rejection_does_not_trigger_a_second_generation():
    """The shipped default: one generation per turn, whatever the guard says.

    Runs 11-14, in order: guarding the retry added a THIRD pass (2546 ms);
    removing it left two (1629 ms) and the retries were still worse than the
    originals -- run 14's second pass produced "No such position exists at my
    current company" and "We currently have several positions available;
    however, those are internal roles mostly", both fabrications, both spoken.
    A 3B model does not repair itself from a correction note, it pivots.

    So the guard now filters malformed output and otherwise warns. Flip
    `regenerate_on_guard_rejection` in config.yaml to restore the old path,
    which test_a_rejected_turn_costs_exactly_two_generations still covers.
    """
    from voice_agent.config import load_config

    assert load_config().agent.get("regenerate_on_guard_rejection") is False

    agent = make_agent("placeholder")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["I work with Python mostly."])
    bad = "Our recruitment team will discuss salary details later."
    agent.llm = _ScriptedLLM([bad, "Can you tell me about your education?"])

    m = await _drive_turn(agent, None)

    assert agent.llm.attempts == 1, (
        f"{agent.llm.attempts} generation(s) -- a rejection must not regenerate"
    )
    assert not m.retried
    assert agent.speaker.played, "the reply must still reach the caller"


@pytest.mark.asyncio
async def test_malformed_output_is_still_filtered_without_regeneration():
    """The one thing the guard still vetoes, because stripping it is mechanical.

    A hallucinated tool-call tag is noise, not speech -- Piper would read it
    aloud. It costs no generation to drop.
    """
    agent = make_agent('Sure <function name="lookup"> let me check.')
    agent._loop = asyncio.get_running_loop()

    m = TurnMetrics(turn_id=1)
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    from voice_agent.llm import GenerationStats

    thread = threading.Thread(
        target=agent._generate_worker,
        args=(queue, GenerationStats(), None, m, stop_event),
        daemon=True,
    )
    thread.start()
    spoken, rejected = await agent._consume_chunks(queue, m, stop_event, guard=True)
    thread.join(timeout=2)

    assert rejected is not None and rejected[0] == "artifact"


@pytest.mark.asyncio
async def test_a_rejected_turn_never_goes_silent():
    """The one thing the retry must always guarantee.

    Guarding the retry unconditionally meant two rejections in a row spoke
    NOTHING -- confirmed live, turns 2 and 3 went out with zero audio chunks.
    Silence reads as a broken line, so the second pass always reaches the
    speaker whatever it contains.
    """
    agent = make_agent("placeholder")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["I work with Python mostly."])
    agent.llm = _ScriptedLLM([
        "Our recruitment team will discuss salary details later.",
        "Got it. So can you tell me about your qualification and field of study?",
    ])

    m = await _drive_turn(agent, None)

    assert m.response.strip(), "the turn produced no reply at all"
    assert agent.speaker.played, "the turn produced no audio at all"


def test_the_note_is_caught_even_buried_in_a_long_reply():
    """Run 12 turn 7: "Four years is quite extensive ... Read back and confirm?:
    What was that".

    The length test cannot see this -- fourteen words, so not short -- and
    there is no third-person pronoun, so neither existing tell fired and the
    checklist note went out verbatim mid-sentence. Compare clause by clause
    instead, splitting on sentence boundaries only so that a comma-wrapped
    "So, years of experience, how many?" is still one clause and still passes.
    """
    from voice_agent.orchestrator import _echoes_agenda
    from voice_agent.prompts import DEFAULT_CALL, build_agenda

    agenda = build_agenda(DEFAULT_CALL)
    for bad in ("Four years is quite extensive for an entry-level role. "
                "Read back and confirm?: What was that",
                "Read back and confirm?: What was that",
                "Got it, thanks. Read back and confirm? Let me know."):
        assert _echoes_agenda(bad, agenda), bad

    for good in ("How many years of experience do you have?",
                 "So, years of experience, how many?",
                 "Python Java SQL are common tools in data engineering. Which "
                 "of these have you worked with?"):
        assert not _echoes_agenda(good, agenda), good


def test_the_agenda_only_advances_on_an_item_riya_actually_asked():
    """Run 11 exhausted a five-item checklist by turn 4 of a nine-turn call.

    The pointer advanced on the caller's utterance alone, so a reply that had
    drifted off the live item still marked it done. Everything after that was
    a wrap-up turn, where the guard rejects any question at all -- which is
    what pushed six of nine turns into extra generations.
    """
    from voice_agent.prompts import AgendaTracker

    t = AgendaTracker(["qualification and field?", "years of experience?",
                       "tech stack -- which ones?"])

    # She asked about qualifications -- the live item. Covered.
    assert t.record("I studied computer science.", covered=True)
    assert t.current == "years of experience?"

    # She drifted off and asked something else. Not covered: the item stays
    # live instead of being silently marked done.
    assert not t.record("Four years.", covered=False)
    assert t.current == "years of experience?"

    # Now she asks it properly.
    assert t.record("Four years.", covered=True)
    assert t.current == "tech stack -- which ones?"


# --------------------------------------------------------------- run 31


def test_the_redirect_cut_does_not_leave_an_orphan_in_front():
    """Run 31 turn 15 went out as "Separately Thank you again for speaking...".

    Three separate seam artefacts, all from the same cut: `_REDIRECT_STOP`
    stopped on the comma BEFORE the clause's closing adverb, so the adverb
    survived and opened the reply; the cleanup stripped " ,;:-" but not a
    leading period; and when no stop was found at all the cut ate everything
    else and left the adverb on its own.
    """
    from voice_agent.orchestrator import _repair_chunk

    bad = [
        "Our recruitment team will discuss openings later, separately thank "
        "you again for your time.",
        "Sure. Our recruitment team will discuss openings later, separately. "
        "Thank you again for speaking with me today.",
        "Our recruitment team will discuss salary separately. Thank you again "
        "for speaking with me today.",
        "Our recruitment team will discuss salary with you. Separately thank "
        "you again for speaking with me today.",
        "Separately our recruitment team will discuss salary Thank you again "
        "for speaking with me today.",
    ]
    for text in bad:
        out, cut = _repair_chunk(text, "")
        assert cut, f"nothing was cut from {text!r}"
        assert not out.lower().startswith(("separately", ".", ",")), (
            f"orphan left in front: {out!r} <- {text!r}"
        )
        assert out[0].isupper(), f"not re-capitalised: {out!r}"
        assert "recruitment team" not in out.lower(), f"redirect survived: {out!r}"


def test_a_question_after_the_redirect_still_survives_the_cut():
    """The run-26 reason the stop search exists. Closing the orphans above
    must not re-open this: the cut has to stop before a real question, with
    or without punctuation in front of it."""
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        "Our recruitment team will discuss salary later, can you confirm your "
        "notice period?",
        "Our recruitment team will discuss salary later can you confirm your "
        "notice period?",
        "Our recruitment team will discuss interview slots then. What is your "
        "notice period?",
    ):
        out, cut = _repair_chunk(text, "")
        assert cut, f"nothing was cut from {text!r}"
        assert "?" in out, f"the real question was eaten: {out!r}"


def test_the_cut_does_not_strand_the_tail_of_the_redirect():
    """Run 35 turn 6: the promise went but its adverb stayed.

    "You are ready to move forward; our recruitment team will discuss suitable
    openings with you separately now." cut to "You are ready to move forward;
    now." -- a semicolon and an adverb with no clause left to modify. It is
    the same family as run 31's orphans, one word further out.
    """
    from voice_agent.orchestrator import _repair_chunk

    text = ("You are ready to move forward; our recruitment team will discuss "
            "suitable openings with you separately now.")
    out, cut = _repair_chunk(text, "i have a masters and eight years")
    assert cut, f"nothing was cut from {text!r}"
    assert "recruitment team" not in out, out
    assert not out.rstrip(".").endswith("now"), f"orphan tail left: {out!r}"


def test_a_topic_volunteered_in_an_unrecognised_wording_is_still_cut():
    """Run 35 turn 1: "No salary discussion today When needed."

    No recruitment team, no "will discuss" -- every redirect pattern missed
    it, and it is the same defect. Keying on the topic rather than the
    sentence shape is what stops this needing a new pattern per run.
    """
    from voice_agent.orchestrator import _repair_chunk

    out, cut = _repair_chunk("No salary discussion today When needed.", "")
    assert cut, "the salary denial went out"
    assert "salary" not in out.lower(), out
    # The stranded connective has to go with it, or the caller hears
    # "When needed." followed by the question.
    assert not out.strip(), out


def test_narrating_the_call_back_to_the_caller_is_cut():
    """Run 38 turn 6, answering "Is there any opening in AI and ML?".

    "No specific openings were mentioned during our call regarding AI & ML
    roles -- our recruitment team will discuss that with you separately."
    She is describing the conversation to the person she is having it with,
    and "no specific openings were mentioned" is a claim about the openings
    book she cannot make.

    Every branch stops at a clause boundary, not at a character count: a
    greedy 60-character tail ran past the "--" and ate the "our" of the
    redirect that follows, leaving "Team will discuss that with you
    separately."
    """
    from voice_agent.orchestrator import _repair_chunk

    text = ("No specific openings were mentioned during our call regarding "
            "AI & ML roles -- our recruitment team will discuss that with you "
            "separately.")
    out, cut = _repair_chunk(text, "is there any opening in ai and ml")
    assert "narrated the call" in cut, cut
    # The redirect is the brief's own instruction and has to survive whole.
    assert out.startswith("Our recruitment team"), out
    assert "separately" in out, out

    # With no redirect clause after it, the whole reply is the defect. An
    # empty reaction is safe -- the templated question still goes out.
    out, cut = _repair_chunk(
        "No specific openings were mentioned during our call regarding AI & "
        "ML roles.", "is there any opening in ai and ml")
    assert "narrated the call" in cut, cut
    assert not out.strip(), out

    # The other wordings of the same defect.
    out, cut = _repair_chunk("As we discussed, the role is based in Pune.", "")
    assert "narrated the call" in cut, cut
    assert out.strip() == "The role is based in Pune.", out

    # ...but the redirect on its own is left alone. The caller raised
    # openings, so deflecting to the recruitment team is the brief's own
    # line -- see test_a_redirect_on_a_topic_the_caller_raised_is_left_alone.
    out, cut = _repair_chunk(
        "Our recruitment team will discuss that with you separately.",
        "is there any opening in ai and ml")
    assert not cut, (out, cut)


def test_every_wording_of_a_claim_about_the_call_is_cut():
    """Run 40 turns 4 and 5: "No information on SQL skills was mentioned."

    The third wording of the run-38 defect, and the one that proved the old
    rule was keyed on length rather than on the invariant. Its gap was
    `[a-z\\s]{0,24}`; "information on SQL skills" is 26 characters, so the
    sentence went out, the repeat guard then caught it on turn 5 -- because it
    duplicated turn 4 verbatim -- and since `repeat` is in regenerate_kinds the
    turn paid 1785 ms for a second generation. The guard was right; the
    sentence should never have been spoken.

    The rule is now keyed on (discourse subject) x (discourse verb) with NO
    length window, in both subject shapes the model uses: "no <noun> ..." and
    "nothing about <x> ...". Widening the bound would have caught run 40 and
    missed run 41.
    """
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        # Run 38's original.
        "No specific openings were mentioned during our call regarding AI roles.",
        # Run 40's, the one that slipped a 24-character window.
        "No information on SQL skills was mentioned.",
        # The same sentence stretched further -- what the next run would have
        # produced had the bound simply been raised to fit run 40.
        "No detailed information about your background with SQL frameworks "
        "was mentioned.",
        # The "nothing about <x>" shape, which names no noun at all.
        "Nothing about Spark was mentioned.",
        "Nothing about your educational background has been discussed.",
        # Other discourse subjects and verbs from the same family.
        "No further details were raised.",
        "No mention of expected salary was made.",
        "No indication of your notice period was given.",
        "Nothing specific regarding your experience was noted.",
        "No comment on that was given.",
        "No such thing was mentioned.",
    ):
        out, cut = _repair_chunk(text, "my skills are sql and python")
        # Which rule wins is not the point -- rule 1 fires first on "expected
        # salary" because the caller never raised it. The point is that the
        # claim never reaches the speaker.
        assert not out.strip(), f"left {out!r} behind (labels: {cut})"

    # The rule must not swallow ordinary replies -- a reaction that simply
    # contains "no" or names the caller's own skills is not a claim about the
    # conversation.
    for text in (
        "That is a solid background.",
        "No problem at all.",
        "Nothing to worry about there.",
        "No worries, we can move on.",
        "The role needs someone with SQL skills.",
        "I have four years of experience with Python.",
    ):
        out, cut = _repair_chunk(text, "my skills are sql and python")
        assert "narrated the call" not in cut, (text, cut)
        assert out.strip() == text, (text, out)


@pytest.mark.asyncio
async def test_a_claim_about_the_call_costs_no_regeneration():
    """The structural point of run 40: this defect family is now free.

    Because "narrated the call" is in _MAY_EMPTY, the offending sentence is
    removed deterministically instead of being caught by the repeat guard and
    paid for with a second generation. The caller still hears the templated
    question, so an empty reaction is a complete turn.
    """
    agent = make_agent("No information on SQL skills was mentioned.")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["my skills are sql and python"])

    m = await _drive_turn(agent, None)

    assert not getattr(m, "retried", False), "the repair should not need a retry"
    assert "No information" not in agent._last_spoken, agent._last_spoken
    # The turn still happened: the templated question carried it.
    assert agent._last_spoken.strip(), "the caller must still hear a line"
    assert agent.speaker.played


def test_a_claim_about_the_conversation_is_cut_not_just_the_call():
    """Run 44 turn 3: "...during our CONVERSATION so far" slipped the rule.

    Branch 1 required the literal word `call`. The caller said "conversation",
    so the narration went out: "No information about personal appearance has
    been requested during our conversation so far". The identical sentence with
    "call" matched -- one missing token cost a spoken sentence.

    Two things had to be fixed together, and the first attempt only did one:
    adding the noun made branch 1 match the TAIL and left "No information about
    personal appearance has been requested" standing as its own sentence. The
    verb "requested" also had to join branch 3's list, so the whole sentence is
    one match and empties cleanly.
    """
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        "No information about personal appearance has been requested during "
        "our conversation so far.",
        "No information about personal appearance has been requested during "
        "our call so far.",
        "No information about personal appearance was requested.",
        "No details about your notice period were requested in this chat.",
        "Nothing about salary was asked during our discussion.",
    ):
        out, cut = _repair_chunk(text, "what colour underwear")
        # Which rule wins is not the point -- rule 1 fires first on "notice
        # period" because the caller never raised it. What matters is that the
        # narration never reaches the speaker.
        assert not out.strip(), f"left {out!r} behind (labels: {cut})"


def test_narrating_her_own_previous_reply_is_cut():
    """Run 44 turn 12: "My previous response should have simply stated: Light blue."

    She is describing her own output to the person she is talking to, and the
    caller has no view of it. It also admits the earlier answer was wrong,
    which invites exactly the probing that produced it.

    The lookahead deliberately does NOT stop at ":". Stopping there cut the
    narration and left "Light blue." behind -- the answer being narrated about.
    """
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        "My previous response should have simply stated: Light blue.",
        "My last answer should have been clearer.",
        "I should have simply said no.",
        "That response should have been shorter.",
        "My earlier reply was wrong.",
    ):
        out, cut = _repair_chunk(text, "what colour underwear")
        assert "narrated her own reply" in cut, (text, cut)
        assert not out.strip(), f"left {out!r} behind"

    # Ordinary uses of the same words are not self-narration: "previous" and
    # "should have" appear in real answers about the caller's own history.
    for text in (
        "My previous role used Python heavily.",
        "I should have mentioned I know Spark.",
        "That is a solid background.",
    ):
        out, cut = _repair_chunk(text, "how many years")
        assert "narrated her own reply" not in cut, (text, cut)
        assert out.strip(), f"wrongly emptied {text!r}"


@pytest.mark.asyncio
async def test_echoing_the_caller_back_costs_no_regeneration():
    """Run 42 turn 6, verbatim: (rejected parrot; regenerating) 'Oh yes you are right.'

    That turn paid 584 ms of TTFT to be told the caller's own words were the
    caller's own words. There is no better version of that sentence for a
    second generation to find -- it is wrong because it is an echo, so the
    repair is to not say it.

    The point of this test is `retried` being False. Plenty of tests assert
    bad text does not reach the speaker; this one asserts the turn did not pay
    for a second generation, which is the only thing the run-42 measurement
    was actually about.
    """
    agent = make_agent("Oh yes you are right.")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["Oh yes you are right"])

    m = await _drive_turn(agent, None)

    assert not getattr(m, "retried", False), "an echo must not cost a second generation"
    assert "Oh yes you are right" not in agent._last_spoken, agent._last_spoken
    # ...and the turn is not lost: the templated question carried it.
    assert agent._last_spoken.strip(), "the caller must still hear a line"
    assert agent.speaker.played


@pytest.mark.asyncio
async def test_repeating_her_own_line_costs_no_regeneration():
    """Run 42 turn 5: (rejected repeat; regenerating) on the opening greeting.

    That turn paid 745 ms of TTFT -- by far the worst in the run -- to
    regenerate a line she had already spoken. Cutting is free here because the
    reaction and the question are independent: the reaction goes, the
    templated question still goes out.
    """
    line = "Thank you for sharing those details."
    agent = make_agent(line)
    agent._loop = asyncio.get_running_loop()
    # She already said it, so the guard has something to compare against.
    agent.llm.add_assistant(line)
    agent.transcriber = _FakeTranscriber(["my skills are python and sql"])

    m = await _drive_turn(agent, None)

    assert not getattr(m, "retried", False), "a repeat must not cost a second generation"
    assert "Thank you for sharing" not in agent._last_spoken, agent._last_spoken
    assert agent._last_spoken.strip(), "the caller must still hear a line"
    assert agent.speaker.played


def test_a_warn_only_kind_is_still_spoken():
    """The cut list must not grow to swallow ordinary replies.

    `repeat`, `parrot` and `agenda` are cut because their text is unusable by
    definition. Everything else still only warns -- a false positive there
    would silently delete a real reply, which is worse than saying a flawed
    one. This pins that the cut path is narrow, not a general mute.
    """
    from voice_agent.orchestrator import _CUT_KINDS

    assert _CUT_KINDS == frozenset({"repeat", "parrot", "agenda"})
    # A fabricated rate and a wrong-person slip stay warn-only: retrying them
    # has never helped, but neither does deleting them without judgement.
    for kind in ("fabrication", "person", "off_topic", "question", "contact"):
        assert kind not in _CUT_KINDS, kind


def test_the_reaction_gate_keeps_reactions_about_the_candidate():
    """The gate is a cut-list of SUBJECTS, not of wordings.

    Runs 44, 45, 46 and 47 each produced a new defective reaction and each
    was chased with another regex. Every one of them is a reaction about the
    call, about Riya, about the agency, about a technology, or about how the
    caller spoke. This pins that those subjects are refused, and -- the part
    that is easy to get wrong -- that ordinary reactions about the candidate
    are kept, including ones sharing no words with the caller's utterance.
    """
    from voice_agent.orchestrator import _reaction_is_worth_saying

    # Kept: about the candidate. The last four are real replies from runs
    # 46/47 and two of them share nothing with what the caller said, which is
    # why an overlap test was tried and removed.
    for good in (
        "You've completed an undergraduate degree in Computer Science.",
        "Six years of work experience sounds impressive.",
        "You've completed some education in Computer Science.",
        "Four years sounds good.",
        "That is a solid background.",
        "Got it, that is relevant.",
        "Thanks, that's helpful.",
    ):
        assert _reaction_is_worth_saying(good), good

    # Refused: about the call / Riya / the agency / a tool / their manner.
    for bad in (
        # run 46
        "Python is widely used in data analysis",
        "I'll clarify You want to know how frequently those technologies are utilized.",
        "That was quite blunt.",
        "Our team will follow up regarding suitable openings at our health care companies.",
        # run 47
        "That helps narrow down our search options.",
        "That helps clarify things",
        "No - our recruitment team will discuss suitable openings with you",
        # run 45
        "our focus today is on matching roles based on qualifications and skills.",
        "No need to reiterate our team will note down all technologies used.",
        "Yes No You have confirmed your tech stack.",
        "Sorry this conversation has strayed far off topic. Our primary goal remains noting relevant details.",
        "That was unnecessary under graduation computer science from an unknown institution after 5 years of work experience.",
        # run 44
        "My previous response should have simply stated: Light blue.",
    ):
        assert not _reaction_is_worth_saying(bad), bad


def test_the_reaction_gate_costs_no_regeneration():
    """The gate empties, it does not retry -- so a bad reaction is free.

    Same economics as `_MAY_EMPTY`: the templated question still follows, so
    dropping the model's whole line leaves the caller hearing a turn. This is
    the measurement that matters -- a gate that paid for a second generation
    would trade the content defect for the latency target.
    """
    from voice_agent.orchestrator import _reaction_is_worth_saying

    bad = "our focus today is on matching roles based on qualifications and skills."
    assert not _reaction_is_worth_saying(bad)
    # And the same call that produced run 46's worth-keeping line still keeps it.
    assert _reaction_is_worth_saying("Four years sounds good.")


def test_the_read_back_speaks_in_the_candidates_voice():
    """Run 46 turn 5: the confirm line claimed the candidate's own answers.

    `_readback()` joins the CALLER's utterances and Riya says them, so
    unflipped the line ran "Let me read that back. My highest qualification is
    under graduation and I have done it in computer science... Have I got that
    right?" -- Riya appearing to claim their degree and years.
    """
    agent = make_agent("")
    agent._answers = [
        "So my highest qualification is under graduation and uh I have done it "
        "in computer science.",
        "Currently I have total of six years of uh work experience",
    ]
    out = agent._readback()
    assert out.startswith("Your highest qualification"), out
    assert "you have done it" in out, out
    assert "you have total" in out, out
    # Nothing first-person survives -- that is the whole defect.
    for wrong in (" My ", " my ", " I "):
        assert wrong not in out, (wrong, out)
    # Agreement has to come along: "I am" is not "you am". Five content words,
    # so it clears _readable_answer's floor -- "I am a senior engineer" is
    # three and is correctly dropped as an affirmation.
    agent._answers = ["I am a senior engineer working with Python"]
    assert "you are a senior engineer" in agent._readback().lower()


def test_a_model_question_behind_a_semicolon_is_dropped():
    """Run 46 turn 5, and it was SPOKEN.

    The chunk arrived as "Python is widely used in data analysis; can you
    specifically say how often you". _strip_model_questions split on [.?!]
    only, so the semicolon glued the question onto the statement, the whole
    thing was one clause, _is_model_question said False, and a model-written
    question reached the speaker in front of the templated one.
    """
    from voice_agent.orchestrator import _strip_model_questions

    out = _strip_model_questions(
        "Python is widely used in data analysis; can you specifically say how "
        "often you"
    )
    assert "can you specifically" not in out.lower(), out
    assert "widely used" in out.lower(), out


def test_a_claim_about_what_the_client_wants_is_cut():
    """Run 39 turn 2, answering "my highest qualification is under graduation".

    "Typically our clients look for degrees above undergraduate level." Riya
    has no mandate to state the client's requirements on a first screening
    call, and phrasing it as a bar tells the caller they fall short.

    `_LECTURE` cannot see it and never will -- it is keyed on the VERB after
    "typically" (takes|lasts|is|means), so "typically our clients look for" is
    invisible to it. Keyed on the SUBJECT, one rule covers the family.
    """
    from voice_agent.orchestrator import _repair_chunk

    for text in (
        "Typically our clients look for degrees above undergraduate level.",
        "Usually our clients want a masters degree.",
        "Our clients look for degrees above undergraduate level.",
        "Most employers prefer a masters degree.",
        "Typically our client looks for a masters degree.",
    ):
        out, cut = _repair_chunk(text, "my highest qualification is under graduation")
        assert "claim about what the client wants" in cut, (text, cut)
        assert not out.strip(), f"left {out!r} behind"

    # A remark about THIS caller and THIS role is not a rule about the market,
    # and it is encouraging rather than a bar. Singular subject, no marker.
    for text in (
        "That matches what our client is looking for.",
        "Our client is looking for someone with Spark experience.",
    ):
        out, cut = _repair_chunk(text, "i work with python and spark")
        assert not cut, (text, out, cut)
        assert out.strip() == text, out


def test_a_cut_that_leaves_a_dependent_phrase_is_dropped():
    """Runs 35, 37, 38 and 39 each produced a new instance of ONE defect.

    A cut lands anywhere, so what it leaves behind is not always a sentence:
    "At another time." (t1), "Yes No." (t5) and "now that we have this
    information." (t6) are all the same failure in a new position, and each
    was patched at the cut site that caused it. This is the one rule that
    covers the family -- see _only_a_fragment.
    """
    from voice_agent.orchestrator import _only_a_fragment

    for text in (
        "At another time.",
        "At another time,.",
        "In the end.",
        "For now.",
        "Then.",
        "When needed.",
        "now that we have this information.",
        "now that we have this information,",
        "Yes No.",
    ):
        assert _only_a_fragment(text), f"fragment not recognised: {text!r}"

    # Everything here is something the model legitimately says, and every one
    # of them must survive. The opener test is what keeps them safe.
    for text in (
        "Got it.",
        "Sure.",
        "Yes.",
        "Okay, sure.",
        "That sounds good.",
        "Thanks for your time.",
        "So I have five years.",
        "In that case, let us move on.",
        "For now, that is all I need.",
        "At another time we can talk more.",
        "With five years you would be a good fit.",
        "As a recruiter I can tell you the role is in Bengaluru.",
    ):
        assert not _only_a_fragment(text), f"real reply would be dropped: {text!r}"


def test_the_second_sign_off_is_a_different_line():
    """Run 39 turns 5 and 6: the caller heard the same goodbye twice.

    "close the call" is the last checklist item and it maps to the `closing`
    topic, so once the checklist ran out the fallback resolved to the same
    template. The second sign-off has to be a DIFFERENT line rather than no
    line: the hangup only arms at farewell 2/2 (see max_farewells), so
    suppressing it would mean the call never ends.
    """
    from voice_agent.orchestrator import _is_farewell
    from voice_agent.prompts import AgendaTracker, build_agenda

    cfg = load_config()
    questions = dict(cfg.agent["questions"])
    tracker = AgendaTracker(build_agenda(dict(cfg.get("call", {}))))

    said = []
    while not tracker.finished:
        said.append(tracker.question(questions))
        tracker.advance()
    said.append(tracker.question(questions))  # checklist exhausted

    closing = questions["closing"]
    assert closing in said, f"the closing line never went out: {said}"
    assert said.count(closing) == 1, f"the same goodbye went out twice: {said}"
    # ...and the second one still has to READ as a goodbye.
    assert _is_farewell(said[-1]), said[-1]


@pytest.mark.asyncio
async def test_a_cut_that_strands_a_lead_in_leaves_only_the_question():
    """The whole turn-1 defect, end to end.

    The model wrote "At another time." as its own sentence, then a redirect;
    the redirect cut took the second sentence and the lead-in was spoken on
    its own as the reaction. It arrives as its OWN chunk, with nothing removed
    from it, which is why the dependent-phrase pass must not be gated on a cut
    having happened in that chunk.

    Note what is NOT asserted: the redirect on its own. Rule 1 is still not
    allowed to empty a reply -- only rule 1b and the span rules may (see the
    run-35 note in _repair_chunk) -- so when the redirect IS the whole chunk it
    is left in place. That asymmetry is deliberate and unchanged here.
    """
    agent = make_agent(
        "At another time. Our recruitment team will discuss the openings with "
        "you separately."
    )
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(["Um yes I'm free."])

    await _drive_turn(agent, None)

    spoken = " ".join(agent.tts.calls)
    assert "At another time" not in spoken, f"the fragment was spoken: {spoken!r}"
    assert spoken.count("?") == 1, spoken


@pytest.mark.asyncio
async def test_the_third_person_read_back_is_dropped_on_the_closing_turn():
    """Run 39 turn 6: the model summarised the caller in the third person.

    `_MODEL_READBACK` only knew the first/second-person frames ("you've said",
    "as you said", "correct about"), so this went out -- and because the
    checklist was finished, agenda_item was None, the topic resolved to None,
    and the drop was skipped entirely. It landed in front of a goodbye that is
    already a full line and pushed Piper to 760 ms (3 audio chunks).
    """
    agent = make_agent(
        "Your qualifications include an undergraduate degree in Computer "
        "Science with five years of relevant work experience."
    )
    agent._loop = asyncio.get_running_loop()
    agent.agenda.index = len(agent._agenda)  # checklist exhausted
    agent.transcriber = _FakeTranscriber(["Yes, that is right."])

    await _drive_turn(agent, None)

    spoken = " ".join(agent.tts.calls)
    assert "Your qualifications include" not in spoken, spoken
    assert spoken.strip(), "nothing was spoken at all"


@pytest.mark.asyncio
async def test_the_model_never_asks_out_loud():
    """Run 35, reported live: "why is it speaking every line for 2 times".

    `_strip_model_questions` used to run inside `_append_question`, which is
    AFTER synthesis. So the model's question was played, then deleted from the
    transcript, and the caller heard it followed by the templated question --
    two questions a turn, with a log that looked clean. The fix is to strip
    at chunk level; this pins audio == transcript so it cannot drift back.
    """
    agent = make_agent(
        "That is a solid background. How many years of experience do you have?"
    )
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(
        ["I have a masters in computer science"])

    m = await _drive_turn(agent, None)

    spoken = " ".join(agent.tts.calls)
    assert "How many" not in spoken, f"the model's question was spoken: {spoken!r}"
    assert spoken.count("?") == 1, f"more than one question went out: {spoken!r}"
    # What Piper was handed and what the transcript claims are the same text.
    assert m.response.strip() == spoken.strip(), (
        f"audio and transcript disagree:\n  audio: {spoken!r}\n  log  : {m.response!r}")


@pytest.mark.asyncio
async def test_our_question_is_not_recorded_as_something_the_model_wrote():
    """Run 38 turn 3, the worst turn of the run (705 ms TTFT, 1355 ms TTFA).

    The templated question was stored in the LLM history as an assistant
    turn, so the model saw its own previous line as one unit -- "reaction +
    question" -- and re-emitted the whole thing verbatim. The repeat guard
    caught it and answered with a second generation.

    Trimming it at comparison time would not have helped: the guard's
    startswith() test still matches the reaction the echoed line begins with.
    The question has to leave the history to stop being echoed.
    """
    agent = make_agent(
        "That is a solid background. How many years of experience do you have?"
    )
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(
        ["I have a masters in computer science"])

    await _drive_turn(agent, None)

    said = [t for role, t in agent.llm.history if role == "assistant"]
    assert said, "the model's reaction never reached the history"
    assert "?" not in said[-1], (
        f"we attributed our own question to the model: {said[-1]!r}")
    assert said[-1].strip() == "That is a solid background.", said[-1]


@pytest.mark.asyncio
async def test_the_backchannel_gate_still_sees_the_question():
    """The gate's "?" has to survive the history change.

    `_is_backchannel` refuses to call anything a backchannel when the agent
    just asked something -- otherwise "Yes." after "Have you got two
    minutes?" is dropped and the caller talks to silence. It reads the last
    AGENT line, and history no longer carries our question, so the signal now
    comes from _last_spoken. Without this the gate would silently start
    swallowing real answers, which is the exact failure the "?" branch was
    added to fix.
    """
    agent = make_agent("Got it, that is relevant.")
    agent._loop = asyncio.get_running_loop()
    agent.transcriber = _FakeTranscriber(
        ["I have a masters in computer science"])

    await _drive_turn(agent, None)

    said = [t for role, t in agent.llm.history if role == "assistant"][-1]
    assert not said.rstrip().endswith("?"), said
    assert agent._last_spoken.rstrip().endswith("?"), agent._last_spoken
    assert _is_backchannel("Yes.", agent._last_spoken) is False


def test_the_read_back_is_a_summary_not_a_transcript():
    """Run 35 turn 4: "it's just giving me back what I said".

    Three answers at 12 words each is a 36-word verbatim replay of the
    caller's own speech, STT noise included. A read-back confirms the key
    facts; it is not a transcript.
    """
    agent = make_agent("ok")
    agent._answers = [
        "I have done my undergraduation in computer science from Manipal",
        "I have total four years of experience working in data engineering",
        "I work with Python and SQL and Spark on a daily basis mostly",
    ]
    back = agent._readback()
    assert back, "the read-back was empty"
    assert len(back.split()) <= 36, f"too long to sound like a summary: {back!r}"
    # ...and it does not stop mid-phrase on a connective.
    assert back.split()[-1].strip(".,").lower() not in (
        "and", "of", "the", "in", "with"), back
    # Answers are included whole: the per-answer cap this replaced cut "and
    # that is in computer science" down to "under graduation", losing the one
    # fact the read-back exists to confirm.
    assert "computer science" in back, back


def test_a_redirect_on_a_topic_the_caller_raised_is_left_alone():
    """Deflecting a question the caller actually asked is the job. Only the
    unsolicited ones get cut."""
    from voice_agent.orchestrator import _repair_chunk

    text = ("Our recruitment team will discuss salary later, can you confirm "
            "your notice period?")
    out, cut = _repair_chunk(text, "what is the salary for this role")
    assert not cut and out == text


@pytest.mark.asyncio
async def test_the_retry_pass_still_repairs_even_though_it_cannot_veto():
    """Run 32 turn 1.

    The retry runs with guard=False so a second rejection cannot force a third
    generation. That used to switch off the deterministic repairs along with
    the veto, so the regenerated reply -- the one most likely to be broken --
    was the single reply in the call that reached the caller untouched. It read
    the checklist item aloud:

      "Can I confirm how many years of total work experience do you currently
       have?

       Qualification and field?"

    and the echoed item then fooled _invents_question into scoring the turn as
    on-agenda, so the pointer advanced over a question Riya never asked and the
    whole five-item checklist ran out by turn 6 of a 17-turn call.

    Vetoing and cutting are different decisions: a retry has no veto left to
    spend, but just as much bad text to fix.
    """
    from voice_agent.orchestrator import _SENTINEL

    async def _run(**kwargs):
        agent = make_agent("", max_sentences=0)
        agent._loop = asyncio.get_running_loop()
        m = TurnMetrics(turn_id=1)
        stop_event = threading.Event()
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(
            "Can I confirm how many years of total work experience do you "
            "currently have?"
        )
        queue.put_nowait("Qualification and field?")
        queue.put_nowait(_SENTINEL)
        return await agent._consume_chunks(queue, m, stop_event, **kwargs)

    spoken, rejected = await _run(guard=False, repair=True)
    assert rejected is None, "a retry must not veto -- no third generation"
    assert "qualification and field" not in spoken.lower(), spoken
    # The model's question is dropped too, not just the echoed item: Python
    # asks the templated one, so a question reaching the speaker since run 34
    # means the caller hears two (run 35's "every line twice").
    assert "how many years" not in spoken.lower(), (
        f"the model's question reached the speaker: {spoken!r}")

    # The flag is what buys that; with repairs off the note goes out verbatim.
    spoken, _ = await _run(guard=False, repair=False)
    assert "qualification and field" in spoken.lower(), spoken


def test_a_question_the_model_closed_with_a_period_is_still_dropped():
    """Run 34 turn 4.

    "Can you name any other tools specifically used by our recruitment team
    when matching candidates like." is interrogative in every way except the
    final character, so the "?" test let it through and it went out in front of
    the templated read-back. Shape has to count, not just punctuation.
    """
    from voice_agent.orchestrator import _strip_model_questions

    reaction = (
        "Python is a popular language; Java can also handle large projects "
        "like health data analysis MySQL is often used for database "
        "management.\n\n"
        "Can you name any other tools specifically used by our recruitment "
        "team when matching candidates like."
    )
    out = _strip_model_questions(reaction)
    assert "recruitment team" not in out, out
    assert out.startswith("Python is a popular language"), out

    # Statements are not questions, short exclamations included.
    for keep in ("You have completed your undergraduate degree in Computer "
                 "Science.", "That sounds good. How nice.", "Got it, thanks."):
        assert _strip_model_questions(keep) == keep, keep


def test_the_read_back_skips_utterances_that_are_not_answers():
    """Run 34: the caller's opening "Use um three" (Parakeet's guess at
    "Yeah, I'm free") advanced the checklist, so it was read back as though it
    were a qualification -- "Use um three, I have done my undergraduation in
    computer science and I have totaled four years". An affirmation is not
    screening data."""
    agent = make_agent("")
    agent._answers = [
        "Use um three",
        "I have done my undergraduation in computer science",
        "I have totaled four years of work experience",
    ]
    out = agent._readback()
    assert "um three" not in out.lower(), out
    assert "computer science" in out.lower(), out
    assert "four years" in out.lower(), out

    # Run 36: two more affirmations that beat the old filters. "Oh you sound
    # free" is FOUR words, so a length floor of >3 let it through, and only
    # one of its words is an acknowledgement so the all-acknowledgements test
    # passed too. "Oh yes you are right" has the same shape. A length test
    # cannot separate these from an answer; counting content words can.
    agent._answers = [
        "Oh you sound free.",
        "Oh yes you are right.",
        "My highest qualification is under graduation and that is in computer "
        "science",
    ]
    out = agent._readback()
    assert "sound free" not in out.lower(), out
    assert "are right" not in out.lower(), out
    assert "computer science" in out.lower(), out

    # Run 37: the answer opened "Oh so my highest qualification is...". "oh"
    # is a filler but "so" is not, so the confirm line read "So my highest
    # qualification is...", which sounds like a fragment of something else.
    agent._answers = [
        "Oh so my highest qualification is uh under graduation in computer "
        "science",
    ]
    # Run 46: the answers are the CANDIDATE's words, so they have to be put
    # into the candidate's voice before Riya says them. Unflipped, the confirm
    # line was "Let me read that back. My highest qualification is under
    # graduation and I have done it in computer science..." -- Riya appearing
    # to claim the candidate's degree and years.
    assert agent._readback().startswith("Your highest"), agent._readback()
    # ...and "I" becomes "you", so "I have five years" keeps its subject
    # without Riya claiming it.
    agent._answers = ["I have five years of experience"]
    assert agent._readback().startswith("You have"), agent._readback()

    # Every answer dismissed: fall back rather than read back nothing.
    agent._answers = ["Yeah", "Sure."]
    assert agent._readback() == ""
