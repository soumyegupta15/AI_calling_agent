"""Backend tests.

The llama-server backend is exercised against a real HTTP server (a stub
llama-server implemented with http.server) rather than a mock, so the SSE
parsing, cancellation-by-disconnect, and /tokenize paths are actually executed.
No GGUF and no llama.cpp binary needed.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.llm import BaseLLM, GenerationStats, LlamaServerLLM, Turn, build_llm  # noqa: E402


# --------------------------------------------------------------------------
#  Stub llama-server
# --------------------------------------------------------------------------

REPLY_WORDS = ["Got ", "it. ", "So ", "you ", "need ", "a ", "senior ", "DevOps ",
               "person. ", "Let ", "me ", "check. "]


class _StubHandler(BaseHTTPRequestHandler):
    delay_s = 0.0
    disconnects: list = []
    tokenize_calls = 0

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/tokenize":
            # One "token" per whitespace-separated word. Counted so tests can
            # assert that the hot path never reaches this endpoint.
            type(self).tokenize_calls += 1
            self._json({"tokens": list(range(len(body.get("content", "").split())))})
            return

        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        self.server.last_payload = body

        if not body.get("stream"):
            self._json({"choices": [{"message": {"content": "ok"}}]})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        sent = 0
        try:
            for word in REPLY_WORDS:
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                sent += 1
                if self.delay_s:
                    time.sleep(self.delay_s)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # This is what a barge-in looks like from the server's side.
            type(self).disconnects.append(sent)

    def _json(self, obj):
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def stub_server():
    _StubHandler.delay_s = 0.0
    _StubHandler.disconnects = []
    _StubHandler.tokenize_calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.last_payload = None
    # Expose the counter on the fixture object for readability in tests.
    type(server).tokenize_calls = property(lambda _self: _StubHandler.tokenize_calls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


LLM_CFG = {
    "n_ctx": 3072,
    "n_batch": 256,
    "n_threads": 4,
    "temperature": 0.65,
    "top_p": 0.9,
    "max_tokens": 150,
    "repeat_penalty": 1.1,
    "stop": ["\nUser:"],
    "history_token_budget": 1800,
    "min_kept_turns": 4,
    "backend": "server",
}


def make_client(server, tmp_path) -> LlamaServerLLM:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub")
    cfg = dict(LLM_CFG)
    cfg["server"] = {
        "host": "127.0.0.1",
        "port": server.server_address[1],
        "auto_start": False,          # the stub is already listening
        "startup_timeout_s": 5,
        "binary": None,
    }
    return LlamaServerLLM(model, cfg)


# --------------------------------------------------------------------------
#  Server backend
# --------------------------------------------------------------------------

def test_server_backend_streams_deltas(stub_server, tmp_path):
    llm = make_client(stub_server, tmp_path)
    llm.add_user("Any update on the DevOps role?")

    stats = GenerationStats()
    out = "".join(llm.stream(stats=stats))

    assert out == "".join(REPLY_WORDS)
    assert stats.tokens == len(REPLY_WORDS)
    assert stats.ttft_ms > 0
    assert stats.total_ms >= stats.ttft_ms
    assert not stats.interrupted


def test_server_backend_sends_the_briefs_sampling_settings(stub_server, tmp_path):
    llm = make_client(stub_server, tmp_path)
    llm.add_user("hello")
    list(llm.stream())

    sent = stub_server.last_payload
    assert sent["temperature"] == 0.65
    assert sent["top_p"] == 0.9
    assert sent["max_tokens"] == 150
    assert sent["repeat_penalty"] == 1.1
    assert sent["stop"] == ["\nUser:"]
    assert sent["stream"] is True
    assert sent["cache_prompt"] is True
    assert sent["messages"][0]["role"] == "system"
    assert "Riya" in sent["messages"][0]["content"]
    # The standing per-turn reminder rides on the last user message.
    assert sent["messages"][-1]["role"] == "user"
    assert sent["messages"][-1]["content"].startswith("hello")


def test_per_turn_note_rides_on_the_user_message_not_the_system_prompt(stub_server, tmp_path):
    """Notes must NOT touch the system message.

    llama-server caches the KV prefix; mutating byte one of the system prompt
    invalidates the whole persona and forces a re-ingest. The note fires on
    barge-in turns -- exactly the turns that most need to be fast -- so putting
    it there meant every interruption paid for a cold prefix. Pinning it to the
    final user message keeps the prefix warm, and a small model weights
    end-of-prompt instructions more heavily anyway.
    """
    llm = make_client(stub_server, tmp_path)
    llm.add_user("sorry, go on")
    list(llm.stream(extra_system="[The caller interrupted you mid-sentence.]"))

    messages = stub_server.last_payload["messages"]
    system = messages[0]["content"]
    assert system.startswith("You are Riya")
    assert "interrupted you mid-sentence" not in system, "note polluted the cached prefix"

    last = messages[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("sorry, go on")
    assert "interrupted you mid-sentence" in last["content"]


def test_system_prefix_is_byte_identical_across_turns(stub_server, tmp_path):
    """Whatever else changes, the cached prefix must not."""
    llm = make_client(stub_server, tmp_path)
    seen = []
    for i, note in enumerate([None, "[note one]", None, "[note two]"]):
        llm.add_user(f"turn {i}")
        list(llm.stream(extra_system=note))
        seen.append(stub_server.last_payload["messages"][0]["content"])
        llm.add_assistant(f"reply {i}")
    assert len(set(seen)) == 1


def test_server_backend_stops_on_stop_event(stub_server, tmp_path):
    """Barge-in: the generator must break, and the server must see a disconnect."""
    _StubHandler.delay_s = 0.02
    llm = make_client(stub_server, tmp_path)
    llm.add_user("tell me everything")

    stats = GenerationStats()
    stop = threading.Event()
    received = []
    for piece in llm.stream(stop_event=stop, stats=stats):
        received.append(piece)
        if len(received) == 3:
            stop.set()

    assert len(received) <= 4, "generation kept going after the stop event"
    assert stats.interrupted
    # Give the server thread a moment to notice the closed socket.
    deadline = time.time() + 2
    while not _StubHandler.disconnects and time.time() < deadline:
        time.sleep(0.02)
    assert _StubHandler.disconnects, "server never saw the client disconnect"


def test_warmup_primes_the_cache_with_the_real_system_prompt(stub_server, tmp_path):
    """Regression: warmup used to send a throwaway prompt, so the ~800-token
    system prefix was ingested for the first time on the caller's first
    question -- 20+ seconds on a small CPU."""
    llm = make_client(stub_server, tmp_path)
    llm.warmup()

    sent = stub_server.last_payload
    assert "Riya" in sent["messages"][0]["content"], "warmup did not send the real system prompt"
    assert sent["cache_prompt"] is True
    assert sent["max_tokens"] == 1, "warmup should not generate a real reply"


def test_compact_system_prompt_is_cheaper_but_still_in_character():
    """The compact prompt is the default; "full" adds objection-handling depth.

    Both are static prefixes cached at warmup, so length costs startup time
    rather than per-turn latency.
    """
    from voice_agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_COMPACT

    assert len(SYSTEM_PROMPT_COMPACT) < len(SYSTEM_PROMPT)
    assert "Riya" in SYSTEM_PROMPT_COMPACT
    assert "markdown" in SYSTEM_PROMPT_COMPACT
    assert "never invent" in SYSTEM_PROMPT_COMPACT.lower()
    # Anti-chatbot guardrails.
    assert "never an assistant" in SYSTEM_PROMPT_COMPACT.lower()
    assert "how can I assist" in SYSTEM_PROMPT_COMPACT
    # Guidance on the common moments, without any speaker labels (see below).
    assert "if they ask who you are" in SYSTEM_PROMPT_COMPACT.lower()
    assert "not interested" in SYSTEM_PROMPT_COMPACT


def test_prompt_frames_the_call_as_outbound():
    """The agent dialled THEM.

    The inbound phrasing ("are you calling about a role?") made the agent ask a
    person it had just phoned why they were ringing, and a live test caller
    replied "no no, you called me". Nothing in the prompt may ask the other
    party why the call is happening.
    """
    from voice_agent.prompts import SYSTEM_PROMPT_COMPACT as p

    assert "YOU dialled THEM" in p
    assert "you called them" in p
    assert "Are you calling about" not in p
    assert "you need filled" not in p


def test_prompt_examples_carry_no_stealable_facts():
    """A 1.7B copies concrete details out of examples verbatim.

    A live caller who said "I'm a data engineer" got back an entire example
    line -- "Not since Friday, I'm chasing the hiring manager today" -- because
    it was the nearest matching text in context.
    """
    from voice_agent.prompts import SYSTEM_PROMPT_COMPACT as p

    for leak in ("Friday", "Monday", "hiring manager today", "$", "per hour"):
        assert leak not in p

    # Run 5, same failure mode, three more times: the model recited a degree it
    # had never been told ("B.Tech in Computer Science"), a stack that was never
    # mentioned ("React, Node.js"), and volunteered "Email addresses typically
    # end in .com." to a caller who had not raised email. Every one of those
    # strings was sitting in the instructions as a worked example. An example is
    # not neutral to a small model -- naming a thing makes it a topic to raise.
    for leak in (
        "B.Tech",
        "React",
        "Node.js",
        "Computer Science",
        "email",
        "E-mail",
    ):
        assert leak not in p, f"{leak!r} is back in the prompt as an example"


def test_the_prompt_gives_one_sentence_budget_not_two():
    """The voice rules said "1-2 sentences"; the screening spec said "1-3".

    Two different numbers for the same thing is worse than either number on a
    3B model -- it picks one at random per turn, and half the time the stream
    then gets cut at max_sentences: 2, leaving a sentence with no ending.
    The enforced cap is 2, so 2 is the number the prompt has to say.
    """
    from voice_agent.prompts import SYSTEM_PROMPT_COMPACT as p

    assert "(1-3" not in p
    assert "(1-2" in p
    # And the guardrail that enforces it must agree.
    import yaml

    raw = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml",
                              encoding="utf-8"))
    assert raw["llm"]["max_sentences"] == 2


def test_the_read_back_rule_does_not_contradict_the_anti_echo_rule():
    """One rule says never repeat the caller; the flow says read it back.

    The screening brief makes "confirm what you noted" step 4 of the flow, but
    the voice rules flatly forbade repeating the caller's words. Asked to do
    something it had just been told never to do, the model produced garbled
    half-echoes instead of a clean confirmation.
    """
    from voice_agent.prompts import SYSTEM_PROMPT_COMPACT as p

    low = p.lower()
    assert "never parrot the caller" in low
    # The exception has to be spelled out, or step 4 of the flow is unwritable.
    assert "confirmation turn" in low


def test_prompt_contains_no_speaker_labels_to_copy():
    """Examples must not be formatted as a transcript.

    When the examples were written as "Them: ... / You: ...", the model started
    prefixing its replies with a literal "You:" -- which went straight to the
    speaker: "You: I called about a Senior Data Engineer role at [Client Name]."
    It copied the FORMAT rather than the content, so removing concrete facts was
    not enough; the labels themselves had to go.
    """
    from voice_agent.prompts import SYSTEM_PROMPT_COMPACT as p

    # It is the LINE-LEADING label that gets copied. The prompt may still name
    # "You:" inside a sentence telling the model not to produce one.
    for line in p.splitlines():
        for label in ("You:", "Them:", "Riya:", "Caller:", "Assistant:"):
            assert not line.strip().startswith(label), f"{line!r} is a copyable template"
    # And it is told, in as many words, to emit speech only.
    assert "no speaker labels" in p.lower()
    assert "[Client Name]" in p          # named as a thing NOT to produce


def test_agenda_items_are_not_speakable_sentences():
    """Checklist entries got read out verbatim when they read like sentences.

    A live turn answered "around three years of experience" with "Are they open
    to hearing about a Senior Data Engineer role at all?" -- item two, third
    person and all. Fragments cannot be spoken as-is, so the model must rewrite
    them.
    """
    from voice_agent.prompts import AGENDA_TEMPLATES, DEFAULT_CALL, build_agenda

    # Measure the template, not the interpolated skill list -- "Python, Spark,
    # AWS, and production pipeline ownership" is legitimately long, and it is
    # the surrounding phrasing that decides whether the line reads as a sentence.
    for tmpl in AGENDA_TEMPLATES:
        skeleton = [w for w in tmpl.split() if not w.startswith("{")]
        assert len(skeleton) <= 5, f"{tmpl!r} is long enough to be read aloud"

    for tmpl, item in zip(AGENDA_TEMPLATES, build_agenda(DEFAULT_CALL)):
        # The template must start lowercase; an interpolated proper noun
        # ("Bengaluru, ...") is a value, not sentence casing.
        assert not tmpl[0].isupper(), f"{tmpl!r} looks like a sentence"
        # No subject-verb opener: "are they open to..." is speakable, "open to
        # the role?" is not.
        assert not item.lower().startswith(
            ("are ", "is ", "do ", "does ", "can ", "what ", "which ", "how ")
        ), f"{item!r} reads as a question the model can speak verbatim"


def test_the_checklist_is_not_listed_in_the_system_prompt():
    """Given the whole list, a 1.7B recites it.

    The prompt now says only that a checklist exists and that one item will be
    named each turn; the items themselves arrive per-turn from AgendaTracker.
    """
    from voice_agent.prompts import DEFAULT_CALL, build_agenda
    from voice_agent.prompts import SYSTEM_PROMPT_COMPACT as p

    assert "never read them out" not in p
    assert "you are told the ONE thing to ask about" in p
    for item in build_agenda(DEFAULT_CALL):
        assert item not in p, f"{item!r} is in the prompt for the model to recite"


def test_the_tracker_names_one_item_and_moves_on():
    """The loop fix. A live call asked the same question four turns running."""
    from voice_agent.prompts import AgendaTracker

    t = AgendaTracker(["bad time?", "target rate?", "work authorization?"])
    assert t.current == "bad time?"
    # The item is no longer named in the directive at all. It used to be, and a
    # bare topic phrase sitting one line from the generation point is exactly
    # what a small model reads as a topic to raise -- which is how it kept
    # asking the item it had just been told not to ask. Python appends the
    # question after generation now; the directive only forbids writing one.
    assert "bad time?" not in t.directive()
    assert "target rate?" not in t.directive()
    assert "do not ask a question" in t.directive().lower()
    assert "added for you" in t.directive()

    assert t.record("no, go ahead")
    assert t.current == "target rate?"
    assert "do not ask a question" in t.directive().lower()


def test_a_short_clarification_does_not_advance_the_agenda():
    """"Specific in what" is the caller asking back, not answering.

    record() used to inspect only the FIRST word, so this advanced the
    checklist past an item the caller had not answered. On a live call that
    ran a 5-item agenda out by turn 5 of a 14-turn call, and everything after
    that point was improvisation -- re-asked questions and invented interview
    slots.
    """
    from voice_agent.prompts import AgendaTracker

    for clarification in ("Specific in what", "like what", "in what way",
                          "what kind"):
        t = AgendaTracker(["tech stack -- which ones?", "read back and confirm?"])
        assert not t.record(clarification), clarification
        assert t.current == "tech stack -- which ones?"

    # The length guard is what stops this over-firing: a real answer that
    # happens to contain an interrogative must still count as an answer.
    for answer in ("I know how to tune Spark jobs",
                   "I have four years of experience"):
        t = AgendaTracker(["years of experience?", "tech stack -- which ones?"])
        assert t.record(answer), answer


def test_a_hedged_non_answer_does_not_advance_the_agenda():
    """"I don't think" is not an answer, but "I don't have a degree" is.

    Run 5 turn 1: the caller opened with "I don't think" and the checklist
    walked straight off the end of the agenda, so by mid-call Riya was
    improvising. The hedge is only a hedge in a SHORT utterance -- six words or
    fewer, with nothing after it to qualify as content -- because "I don't know
    Python but I do Spark every day" is a genuine answer that happens to open
    with the same three words.
    """
    from voice_agent.prompts import AgendaTracker

    for hedge in ("I don't think", "I don't think so", "not sure",
                  "I have no idea", "hard to say", "dunno",
                  "I do not recall"):
        t = AgendaTracker(["years of experience?", "tech stack -- which ones?"])
        assert not t.record(hedge), hedge
        assert t.current == "years of experience?", hedge

    # Same opening words, but these say something. The length guard is the only
    # thing separating them, so it is asserted directly.
    for answer in ("I don't have a degree",
                   "I don't know Python, but I do Spark"):
        t = AgendaTracker(["years of experience?", "tech stack -- which ones?"])
        assert t.record(answer), answer


def test_a_question_back_does_not_advance_the_agenda():
    """"Sorry, what?" leaves the item open; advancing would skip it silently."""
    from voice_agent.prompts import AgendaTracker

    for deflection in ("sorry?", "what was that", "why are you calling",
                       "Can you repeat that?", "hello?"):
        t = AgendaTracker(["target rate?", "work authorization?"])
        assert not t.record(deflection), deflection
        assert t.current == "target rate?"


def test_the_tracker_closes_the_call_when_the_checklist_runs_out():
    from voice_agent.prompts import AgendaTracker

    t = AgendaTracker(["only item"])
    t.record("sure, ninety an hour")
    assert t.finished
    # The closing is a fixed line too, so the model must not write one -- it
    # used to improvise a goodbye and a promise ("my colleague will reach out
    # shortly to schedule an interview slot") on top of it.
    note = t.directive().lower()
    assert "do not say goodbye" in note
    assert "do not ask anything" in note
    assert "closing line is added for you" in note


def test_call_brief_retargets_the_whole_prompt():
    from voice_agent.prompts import build_system_prompt

    p = build_system_prompt(
        {"role": "Staff SRE", "client": "a fintech", "candidate_name": "Satyam"}
    )
    assert "Staff SRE" in p
    assert "a fintech" in p
    assert "speaking with Satyam" in p
    assert "Senior Data Engineer" not in p


def test_backend_selects_the_compact_prompt_from_config(stub_server, tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub")
    cfg = dict(LLM_CFG, system_prompt="compact")
    cfg["server"] = {
        "host": "127.0.0.1", "port": stub_server.server_address[1],
        "auto_start": False, "startup_timeout_s": 5, "binary": None,
    }
    llm = build_llm(cfg, model)
    llm.add_user("hi")
    list(llm.stream())

    system = stub_server.last_payload["messages"][0]["content"]
    from voice_agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_COMPACT

    assert system == SYSTEM_PROMPT_COMPACT
    assert len(system) < len(SYSTEM_PROMPT)


def test_full_system_prompt_is_opt_in(stub_server, tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub")
    cfg = dict(LLM_CFG, system_prompt="full")
    cfg["server"] = {
        "host": "127.0.0.1", "port": stub_server.server_address[1],
        "auto_start": False, "startup_timeout_s": 5, "binary": None,
    }
    llm = build_llm(cfg, model)
    llm.add_user("hi")
    list(llm.stream())
    from voice_agent.prompts import SYSTEM_PROMPT

    assert stub_server.last_payload["messages"][0]["content"] == SYSTEM_PROMPT


def test_history_counting_never_hits_the_network(stub_server, tmp_path):
    """add_user/add_assistant must not call /tokenize.

    They run between the caller finishing a sentence and the first token coming
    back. llama-server answers /tokenize on the same worker that generates, so a
    round trip issued while a previous turn is still streaming does not cost
    5 ms -- it queues behind the entire generation. The count only decides when
    old turns get dropped, so a local estimate is good enough.
    """
    llm = make_client(stub_server, tmp_path)
    before = stub_server.tokenize_calls
    llm.add_user("a b c")
    llm.add_assistant("d e f")
    assert stub_server.tokenize_calls == before, "/tokenize called on the hot path"
    assert llm.history[-1].tokens >= 1

    # The exact tokenizer is still available for startup-time work.
    assert llm._count_exact("one two three four") == 4


def test_server_backend_refuses_when_nothing_is_listening(tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub")
    cfg = dict(LLM_CFG)
    cfg["server"] = {"host": "127.0.0.1", "port": 1, "auto_start": False, "binary": None}
    with pytest.raises(RuntimeError, match="auto_start is off"):
        LlamaServerLLM(model, cfg)


def test_missing_model_file_is_reported_clearly(tmp_path):
    cfg = dict(LLM_CFG)
    cfg["server"] = {"host": "127.0.0.1", "port": 1, "auto_start": False, "binary": None}
    with pytest.raises(FileNotFoundError, match="download_models"):
        LlamaServerLLM(tmp_path / "nope.gguf", cfg)


def test_missing_server_binary_is_reported_clearly(tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub")
    cfg = dict(LLM_CFG)
    cfg["server"] = {
        "host": "127.0.0.1", "port": 1, "auto_start": True,
        "binary": str(tmp_path / "nowhere" / "llama-server.exe"),
    }
    with pytest.raises(FileNotFoundError, match="llama-server binary not found"):
        LlamaServerLLM(model, cfg)


# --------------------------------------------------------------------------
#  Factory
# --------------------------------------------------------------------------

def test_factory_rejects_an_unknown_backend(tmp_path):
    cfg = dict(LLM_CFG, backend="inprocess")
    with pytest.raises(ValueError, match="Unknown llm.backend"):
        build_llm(cfg, tmp_path / "fake.gguf")


def test_factory_default_is_the_server_backend(tmp_path):
    cfg = dict(LLM_CFG)
    cfg.pop("backend")
    cfg["server"] = {"host": "127.0.0.1", "port": 1, "auto_start": False, "binary": None}
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub")
    # Reaches LlamaServerLLM (fails on connect, not on backend selection).
    with pytest.raises(RuntimeError, match="auto_start is off"):
        build_llm(cfg, model)


# --------------------------------------------------------------------------
#  Shared history logic (backend-independent)
# --------------------------------------------------------------------------

class _PlainLLM(BaseLLM):
    """BaseLLM with the char/4 token estimate, no model attached."""

    _system_tokens = 400


def test_history_trimming_lives_in_the_base_class():
    llm = _PlainLLM(dict(LLM_CFG, history_token_budget=1000, min_kept_turns=4))
    llm.history = [Turn("user" if i % 2 == 0 else "assistant", f"turn {i}", 200) for i in range(20)]
    kept = llm._trimmed_history()
    assert sum(t.tokens for t in kept) <= 1000
    assert kept[-1].content == "turn 19"


def test_context_tokens_includes_the_system_prompt():
    llm = _PlainLLM(dict(LLM_CFG))
    llm.history = [Turn("user", "hi", 10)]
    assert llm.context_tokens == 410


def test_blank_assistant_turns_are_not_recorded():
    llm = _PlainLLM(dict(LLM_CFG))
    llm.add_assistant("   ")
    llm.add_assistant("real answer")
    assert len(llm.history) == 1


# --------------------------------------------------------------------------
#  Cancellation must not wedge the next turn
# --------------------------------------------------------------------------

def test_aborting_frees_the_client_for_the_next_turn(stub_server, tmp_path):
    """The 10.5 s time-to-first-token bug, pinned down.

    Old shape: stream() held the client lock for the whole SSE loop and only
    noticed stop_event between arriving lines. On barge-in the orchestrator
    cancelled the asyncio task, but the generation worker was a daemon thread
    parked in a blocking socket read -- and llama-server sends nothing while it
    is still ingesting the prompt, so nothing ever unblocked it. The dead turn
    kept the lock; the live turn queued behind it and reported a 10518 ms first
    token. Aborting the connection is what frees both.
    """
    _StubHandler.delay_s = 0.05
    llm = make_client(stub_server, tmp_path)
    llm.add_user("tell me everything")

    stop = threading.Event()
    started = threading.Event()
    finished = threading.Event()

    def worker():
        started.set()
        for _ in llm.stream(stop_event=stop):
            pass
        finished.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert started.wait(2)
    time.sleep(0.12)                      # let it park in a read

    llm.abort()
    assert finished.wait(2.0), "aborted generation never returned"

    # The lock must be free immediately, not after the old request drains.
    t0 = time.perf_counter()
    llm.add_user("next question")
    pieces = list(llm.stream())
    elapsed = time.perf_counter() - t0

    assert pieces, "the follow-up turn produced nothing"
    assert elapsed < 2.0, f"next turn waited {elapsed:.1f}s on the previous one"


def test_stop_event_alone_unblocks_a_stalled_read(stub_server, tmp_path):
    """The watchdog, not the loop condition, is what breaks a blocking read."""
    _StubHandler.delay_s = 0.05
    llm = make_client(stub_server, tmp_path)
    llm.add_user("go")

    stop = threading.Event()
    done = threading.Event()

    def worker():
        for _ in llm.stream(stop_event=stop):
            pass
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    time.sleep(0.12)
    stop.set()
    assert done.wait(2.0), "stop_event did not interrupt the stream"


# --------------------------------------------------------------------------
#  Prompt-cache stability (the flat ~1940 ms TTFT regression)
# --------------------------------------------------------------------------

def _common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_every_turn_reuses_the_previous_turns_prefix(stub_server, tmp_path):
    """The whole conversation so far must be re-sent byte-identically.

    llama-server reuses the cached KV for the longest common prefix. Attaching
    the standing reminder to only the LAST user message quietly destroyed that:
    turn N sent [system][u1+R][a1], turn N+1 sent [system][u1][a1]... -- u1 no
    longer carried R, so the common prefix ended at u1 and the entire
    conversation was re-ingested every single turn. Time-to-first-token went
    flat at ~1940 ms no matter how little had changed.
    """
    llm = make_client(stub_server, tmp_path)
    previous: list | None = None

    for i in range(5):
        llm.add_user(f"question number {i}")
        list(llm.stream())
        sent = stub_server.last_payload["messages"]
        if previous is not None:
            shared = _common_prefix(previous, sent)
            assert shared == len(previous), (
                f"turn {i}: only {shared} of {len(previous)} cached messages "
                "were reused; the prompt cache is being thrown away"
            )
        previous = sent
        llm.add_assistant(f"answer number {i}")


def test_the_reminder_is_on_every_user_turn_not_just_the_last(stub_server, tmp_path):
    from voice_agent.prompts import TURN_REMINDER

    llm = make_client(stub_server, tmp_path)
    for i in range(3):
        llm.add_user(f"turn {i}")
        llm.add_assistant(f"reply {i}")
    llm.add_user("latest")
    list(llm.stream())

    users = [m for m in stub_server.last_payload["messages"] if m["role"] == "user"]
    assert len(users) == 4
    for m in users:
        assert TURN_REMINDER in m["content"], "uneven reminder breaks the cached prefix"


def test_the_reminder_never_enters_the_transcript(stub_server, tmp_path):
    """History must stay clean; the reminder is a rendering detail."""
    from voice_agent.prompts import TURN_REMINDER

    llm = make_client(stub_server, tmp_path)
    llm.add_user("hello")
    list(llm.stream())
    assert all(TURN_REMINDER not in t.content for t in llm.history)


def test_stats_survive_a_consumer_that_stops_early(stub_server, tmp_path):
    """The sentence ceiling stops iterating mid-stream.

    That closes the generator at the yield, so anything assigned after the loop
    never runs -- turns were reported as "0 gen tok" while two chunks of audio
    had already played.
    """
    llm = make_client(stub_server, tmp_path)
    llm.add_user("go")

    stats = GenerationStats()
    got = 0
    for _piece in llm.stream(stats=stats):
        got += 1
        if got == 3:
            break

    assert stats.tokens == 3, f"stats.tokens was {stats.tokens} after 3 deltas"
    assert stats.total_ms > 0


def test_dropping_an_abandoned_user_turn(stub_server, tmp_path):
    llm = make_client(stub_server, tmp_path)
    llm.add_user("first half of a sentence")
    assert llm.drop_last_user("first half of a sentence")
    assert not llm.history
    # Refuses when the last turn is not the one being retracted.
    llm.add_user("something")
    llm.add_assistant("a reply")
    assert not llm.drop_last_user("something")
    assert len(llm.history) == 2
