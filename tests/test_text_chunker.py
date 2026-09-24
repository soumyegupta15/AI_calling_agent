"""Unit tests for the pure-logic pieces. No models required.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.text_chunker import SentenceChunker, chunk_text, normalize_for_tts  # noqa: E402


# --------------------------------------------------------------------------
#  Sentence chunking
# --------------------------------------------------------------------------

def test_first_chunk_emits_early():
    """Latency rule: the first chunk should not wait for a long sentence."""
    c = SentenceChunker(first_chunk_min_chars=12, chunk_min_chars=45)
    out = c.push("Got it, sure. ")
    assert out == ["Got it, sure."]


def test_streaming_deltas_produce_sentences_in_order():
    c = SentenceChunker(first_chunk_min_chars=12, chunk_min_chars=20)
    emitted = []
    for delta in ["Got it", ", so you need", " a senior DevOps person.",
                  " Contract to hire", ", remote okay?", " Let me check."]:
        emitted += c.push(delta)
    tail = c.flush()
    if tail:
        emitted.append(tail)
    assert emitted == [
        "Got it, so you need a senior DevOps person.",
        "Contract to hire, remote okay?",
        "Let me check.",
    ]


def test_decimals_do_not_split():
    assert chunk_text("He has 3.5 years of experience with Kubernetes.") == [
        "He has 3.5 years of experience with Kubernetes."
    ]


def test_abbreviations_do_not_split():
    chunks = chunk_text("I spoke with Mr. Patel this morning about the role.")
    assert chunks == ["I spoke with Mr. Patel this morning about the role."]


def test_long_run_on_splits_at_a_natural_pause():
    text = (
        "So the client wants somebody who can own the whole pipeline end to end "
        "and also mentor the two juniors they just hired which is a lot to ask"
    )
    chunks = chunk_text(text, chunk_max_chars=90)
    assert len(chunks) >= 2
    # Nothing is lost or duplicated.
    assert " ".join(chunks).split() == text.split()


def test_flush_returns_trailing_fragment():
    c = SentenceChunker()
    c.push("No period here")
    assert c.flush() == "No period here"
    assert c.flush() is None


def test_no_text_is_dropped_across_a_multi_sentence_response():
    text = "Sure. I'll check the submission status. Give me one second, okay?"
    assert " ".join(chunk_text(text)).split() == text.split()


# --------------------------------------------------------------------------
#  Spoken-form normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected_fragment",
    [
        ("It's a C2H role.", "contract to hire"),
        ("Are they open to C2C?", "corp to corp"),
        ("W2 only on this one.", "W two"),
        ("They're on 1099 right now.", "ten ninety nine"),
        ("I need an RTR before I submit.", "right to represent"),
        ("He's on H1B.", "H one B"),
        ("She has a GC.", "green card"),
        ("We're a Tier 2 vendor.", "tier two"),
    ],
)
def test_staffing_acronyms_are_spoken_out(raw, expected_fragment):
    assert expected_fragment in normalize_for_tts(raw)


def test_money_range_with_rate():
    out = normalize_for_tts("Bill rate is $85-95/hr.")
    assert "eighty five to ninety five dollars an hour" in out
    assert "$" not in out


def test_single_money_amount():
    assert "seventy dollars an hour" in normalize_for_tts("They're at $70/hr.")


def test_salary_in_thousands():
    assert "one hundred forty thousand dollars" in normalize_for_tts("Around $140k base.")


def test_bare_rate_without_dollar_sign():
    assert "ninety an hour" in normalize_for_tts("We were thinking 90/hr.")


def test_requisition_number_is_read_digit_by_digit():
    out = normalize_for_tts("Req #4521 is still open.")
    assert "requisition" in out
    assert "four five two one" in out


def test_percent():
    assert "twenty two percent" in normalize_for_tts("Our markup is 22%.")


def test_slash_pairs_become_or():
    assert "remote or hybrid" in normalize_for_tts("They're open to remote/hybrid.")


def test_markdown_is_stripped():
    out = normalize_for_tts("- **Senior** DevOps `engineer`")
    assert "*" not in out and "`" not in out and not out.startswith("-")
    assert "Senior DevOps engineer" in out


def test_em_dash_becomes_a_pause():
    assert normalize_for_tts("Got it — remote is fine.") == "Got it, remote is fine."


def test_normalizer_is_idempotent_on_plain_speech():
    plain = "So how soon do you need this person to start?"
    assert normalize_for_tts(plain) == plain


def test_empty_input_is_safe():
    assert normalize_for_tts("") == ""
    assert normalize_for_tts("   ") == ""


# --------------------------------------------------------------------------
#  Reasoning-block stripping (Qwen3 and other hybrid-reasoning models)
#
#  Qwen3 emits <think>...</think> before its answer unless thinking is disabled.
#  llama.cpp's enable_thinking:false is not reliable across builds, so the
#  chunker strips the blocks too -- otherwise the agent reads its own scratchpad
#  down the phone and time-to-first-audio doubles.
# --------------------------------------------------------------------------

def test_a_complete_think_block_is_removed():
    assert chunk_text("<think>The user wants a role.</think>Got it, what's the role?") == [
        "Got it, what's the role?"
    ]


def test_think_block_split_across_streaming_deltas_is_removed():
    c = SentenceChunker(first_chunk_min_chars=8, chunk_min_chars=20)
    emitted = []
    for delta in ["<th", "ink>Let me con", "sider the rate.", "</thi", "nk>",
                  "Sure, what's ", "the bill rate?"]:
        emitted += c.push(delta)
    tail = c.flush()
    if tail:
        emitted.append(tail)
    joined = " ".join(emitted)
    assert "<think>" not in joined and "</think>" not in joined
    assert "consider" not in joined.lower()
    assert "bill rate" in joined


def test_reasoning_text_never_reaches_a_chunk():
    c = SentenceChunker(first_chunk_min_chars=4, chunk_min_chars=10)
    out = c.push("<think>I should ask about the engagement type. That is the key question.</think>")
    assert out == []
    assert c.flush() is None


def test_answer_after_a_long_think_block_still_streams():
    c = SentenceChunker(first_chunk_min_chars=8, chunk_min_chars=20)
    out = c.push("<think>" + "reasoning " * 50 + "</think>Contract to hire, got it. ")
    tail = c.flush()
    produced = " ".join(out + ([tail] if tail else []))
    assert "reasoning" not in produced
    assert "Contract to hire" in produced


def test_unterminated_think_block_produces_nothing():
    """A truncated reasoning block means no real answer; speaking the fragment
    would be worse than silence."""
    c = SentenceChunker()
    assert c.push("<think>I am still thinking about") == []
    assert c.flush() is None


def test_plain_text_is_unaffected_by_the_stripper():
    text = "Got it. So you need a senior DevOps person, contract to hire."
    assert " ".join(chunk_text(text)).split() == text.split()


def test_reset_clears_think_state():
    c = SentenceChunker()
    c.push("<think>partial")
    c.reset()
    assert chunk_text("Hello there.") == ["Hello there."]
    out = c.push("Now a normal sentence here. ")
    assert out == ["Now a normal sentence here."]


# --------------------------------------------------------------------------
#  Orphan reasoning tags
# --------------------------------------------------------------------------

def test_a_closing_think_tag_without_an_opener_is_stripped():
    """A live turn spoke the literal characters "</think>" down the phone.

    llama.cpp's chat template can consume the opening <think> before the text
    ever reaches us, so the model's output begins mid-reasoning and ends with a
    bare closing tag. The stripper only looked for a matched pair, so everything
    -- reasoning and tag alike -- went straight to Piper.
    """
    c = SentenceChunker(18, 55, 200)
    out = c.push("Let me check our availability. Is there a date?</think>\n")
    tail = c.flush()
    if tail:
        out.append(tail)
    joined = " ".join(out)
    assert "</think>" not in joined
    # Everything before the orphan tag was reasoning, so it goes too.
    assert "Let me check our availability" not in joined


def test_text_after_an_orphan_closing_tag_survives():
    c = SentenceChunker(18, 55, 200)
    out = c.push("thinking out loud</think>So, how many years on Spark? ")
    tail = c.flush()
    if tail:
        out.append(tail)
    joined = " ".join(out)
    assert "how many years on Spark" in joined
    assert "thinking out loud" not in joined
    assert "</think>" not in joined


def test_a_closing_tag_split_across_deltas_is_still_caught():
    c = SentenceChunker(18, 55, 200)
    out = []
    for delta in ("reasoning</thi", "nk>Right, what rate are you after? "):
        out += c.push(delta)
    tail = c.flush()
    if tail:
        out.append(tail)
    joined = " ".join(out)
    assert "</think>" not in joined and "</thi" not in joined
    assert "what rate are you after" in joined


def test_run_together_sentences_split_without_a_terminator():
    """The model omits the separator entirely: "...roles Can you please...".

    Measured over a live 7-turn call, 5 of 7 replies were run together like
    this. With no period to find, the chunker waited for the first-chunk
    deadline and emitted ONE 66-95 char chunk. Piper time tracks first-chunk
    length closely, so those turns cost 557-636 ms against a 550 ms ceiling,
    while a punctuated reply with a 17-char first chunk cost 229 ms.
    """
    c = SentenceChunker(12, 45, 180, max_sentences=2)
    out = []
    for delta in ("Five plus years typically qualifies us for certain ",
                  "roles Can you please confirm how much work experience ",
                  "you've accumulated over time?"):
        out += c.push(delta)
    tail = c.flush()
    if tail:
        out.append(tail)

    assert len(out) >= 2, out
    assert out[0].startswith("Five plus years")
    assert "Can you please confirm" in out[1]
    # The point of the exercise: chunk one is much shorter than the whole.
    assert len(out[0]) < 60


def test_proper_nouns_do_not_trigger_the_missing_period_split():
    """The whitelist of openers is what keeps "Acme Corp" in one piece.

    A bare lower-case-to-capital transition would split company names and
    technology lists mid-phrase -- "Python | Spark" read aloud with a full
    stop between them is worse than a late first chunk.
    """
    for text in (
        "Our client Acme Corp is hiring right now.",
        "You work with Python Spark and AWS.",
        "The role is based in Bengaluru, Hybrid three days.",
    ):
        c = SentenceChunker(12, 45, 180, max_sentences=2)
        out = []
        for i in range(0, len(text), 6):
            out += c.push(text[i:i + 6])
        tail = c.flush()
        if tail:
            out.append(tail)
        assert len(out) == 1, (text, out)
