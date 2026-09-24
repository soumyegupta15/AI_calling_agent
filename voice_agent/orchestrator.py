"""The async orchestrator: overlapping STT / TTT / TTS with barge-in.

Concurrency model (brief section 7)
-----------------------------------
One asyncio event loop owns coordination and never blocks. Everything that calls
into C (Silero, Whisper, llama.cpp, Piper) runs off-loop:

  PortAudio input thread  -> pushes 32 ms frames into an asyncio.Queue
  _vad_loop (event loop)  -> Silero is ~0.2 ms/frame, cheap enough to run inline;
                             it is also the supervisor that detects barge-in
  _stt_pool (1 thread)    -> distil-Whisper partials and the final pass
  LLM thread (per turn)   -> llama.cpp token stream -> SentenceChunker -> queue
  _tts_pool (1 thread)    -> Piper synthesis, one sentence-chunk at a time
  PortAudio output thread -> drains the speaker deque

Nothing runs strictly end-to-end: chunk 1 is already playing while the LLM is
still generating chunk 2 and Piper is synthesizing it.

Cancellation
------------
Every turn carries a turn id and a threading.Event. On barge-in we (1) interrupt
playback, which takes effect within one output block (~20 ms), (2) set the event
so the llama.cpp generator breaks on its next token instead of burning cores, and
(3) cancel the asyncio turn task, which drops any queued chunks. Late results
from the old turn are discarded by comparing turn ids.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import re

import numpy as np

from .audio_io import MicStream, SpeakerStream
from .config import Section, model_path
from .llm import GenerationStats, build_llm
from .metrics import MetricsLog, TurnMetrics
from .prompts import (
    INTERRUPTION_NOTE,
    AgendaTracker,
    agenda_topic,
    build_agenda,
    build_user_message,
    correction_note,
)
from .stt import StreamingTranscriber, build_stt
from .text_chunker import SentenceChunker
from .tts import KokoroTTS, build_tts, confirm_kokoro_streaming
from .vad import SileroVAD, TurnDetector, VADEvent

_SENTINEL = object()

# Words that mean "I have not finished my sentence". A caller who trails off on
# one of these ("...and I have two years of experience, so") has not yielded the
# turn, but a fixed 400 ms silence timer cannot tell the difference between a
# thinking pause and a finished thought. Seeing one of these at the end of the
# transcript is the cheap signal that we should wait a beat longer.
_DANGLING = {
    "so", "and", "but", "or", "because", "cause", "coz", "if", "then", "than",
    "with", "for", "to", "of", "in", "on", "at", "like", "that", "which",
    "um", "uh", "erm", "hmm", "well", "actually", "basically", "means",
    "my", "the", "a", "an", "is", "are", "was", "were", "i", "im", "ive",
}

# True continuers only: sounds that mean "I'm still listening, go on".
#
# This list used to include "yes", "yeah", "okay", "sure", "right", "correct",
# "got it" and "I see" -- and that killed the conversation stone dead. Riya asks
# a question almost every turn by design, so the caller's reply is usually one
# of exactly those words. "Have you got two minutes?" / "Yes." was silently
# discarded and the agent said nothing back, on the second turn of the call.
#
# An acknowledgement is only a backchannel when nothing was asked. See
# _is_backchannel, which will not classify anything as one if the agent's last
# line was a question.
_BACKCHANNELS = {
    "mm hmm", "mmhmm", "mhm", "mm", "hmm", "hm", "uh huh", "uhhuh", "huh",
    "aha", "ah", "mhmm", "mm hm",
}

# Terms Parakeet reliably clips or mangles on a phone-band mic. The keys are
# what actually comes back; "AWS" losing its leading vowel to "WS" was the most
# common. The VAD pre-roll fix removes most of the cause, but a phone line still
# eats leading consonants and these are the words that matter in this domain.
_STT_FIXUPS = [
    (r"\b(?:a\s*)?w\s*s\b", "AWS"),
    (r"\bdev\s*ops\b", "DevOps"),
    (r"\bdata\s*bricks\b", "Databricks"),
    (r"\bk\s*8\s*s\b|\bkubernetes\b", "Kubernetes"),
    (r"\bc\s*2\s*c\b|\bc\s*to\s*c\b|\bcorp\s*to\s*corp\b", "C2C"),
    (r"\bc\s*2\s*h\b|\bcontract\s*to\s*hire\b", "C2H"),
    (r"\bw\s*2\b|\bw\s*two\b", "W2"),
    (r"\bh\s*1\s*b\b|\bh\s*one\s*b\b", "H1B"),
    (r"\bopt\b|\bo\s*p\s*t\b", "OPT"),
    (r"\bgreen\s*card\b|\bg\s*c\b", "Green Card"),
    (r"\brtr\b|\br\s*t\s*r\b", "RTR"),
    (r"\bsnowflake\b", "Snowflake"),
    (r"\bpy\s*spark\b", "PySpark"),
    (r"\bair\s*flow\b", "Airflow"),
    (r"\bterra\s*form\b", "Terraform"),
    (r"\bsee\s*i\s*/?\s*see\s*d\b|\bci\s*cd\b", "CI/CD"),
    # Nemotron often hears the recruiter's name as "Rhea" / "Rea".
    (r"\b(?:rhea|rea)\b", "Riya"),
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For comparisons only."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip().lower()


def _apply_stt_fixups(text: str) -> str:
    for pattern, replacement in _STT_FIXUPS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # "a WS engineer" becomes "AWS engineer" above, which swallows the article.
    # Put it back so the sentence still parses for the model.
    text = re.sub(r"(?<![\w])(I am|I'm|a|an)\s+AWS\b", r"\1 AWS", text)
    text = re.sub(r"\ba\s+(AWS|RTR|OPT)\b", r"an \1", text)
    return text


def _ends_dangling(text: str) -> bool:
    words = _normalize(text).split()
    if not words:
        return False
    # A sentence that ends on real punctuation has been finished deliberately.
    if text.rstrip().endswith((".", "?", "!")) and words[-1] not in _DANGLING:
        return False
    return words[-1] in _DANGLING


def _is_backchannel(text: str, last_agent_line: str = "") -> bool:
    """True only for a continuer that answers nothing.

    The gate matters more than the word list. If the agent just asked something,
    whatever comes back is the answer -- even "mm-hmm", which after "Have you got
    two minutes?" plainly means yes. Dropping it leaves the caller talking to
    silence, which is a far worse failure than answering a stray grunt.
    """
    if last_agent_line.rstrip().endswith("?"):
        return False
    return _normalize(text) in _BACKCHANNELS


# Phrases that mean the model has fallen out of the persona and back into its
# instruct prior. By turn 14 of a live call, "hello" was answered with "Hello
# there! How can I assist you today?" plus an emoji -- every one of which the
# system prompt forbids in plain words. The instruction was still in context,
# just far away and outvoted. Catching the reply is more reliable than asking
# a 1.7B more nicely.
_ASSISTANT_TELLS = (
    "how can i assist", "how may i assist", "how can i help", "how may i help",
    "what can i help", "i'm here to help", "im here to help", "happy to help",
    "let me clarify", "let's get started", "lets get started", "as an ai",
    "i'm an ai", "im an ai", "language model", "is there anything else",
    "feel free to ask", "please state that clearly", "do you have any questions",
)

# Riya talking as if SHE is the job candidate instead of the recruiter who
# placed the call. Observed live, twice, verbatim: "Okay so where can I
# apply?" and "That seems unrelated to my query regarding application
# opportunities..." -- the model inverting who called whom. _wrong_person
# catches third-person slips ("they've confirmed interest"); this catches the
# opposite direction, first-person claims that only make sense from the
# candidate's side of the conversation.
_CANDIDATE_VOICE_TELLS = (
    "where can i apply", "my application", "my query regarding application",
    "i'm applying", "im applying", "my resume", "my cv", "should i apply",
    "can i apply", "hope to hear back", "did i get the job",
)

# Anything outside basic Latin punctuation that Piper would either mispronounce
# or read as a symbol. Emoji are the obvious case; a live turn ended with a star.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿️]"
)


def _has_emoji(text: str) -> bool:
    return bool(_EMOJI.search(text))


# Template scaffolding that is not speech. Each of these reached the speaker in
# a live call:
#   "You: I called about a Senior Data Engineer role at [Client Name]."
#   "...works better for you?</think>"
# The first is the model copying the "Them:/You:" labels out of the style
# examples; the second is a reasoning tag with no opening partner. The prompt and
# the chunker each handle their own case now, but a reply carrying any of this is
# malformed regardless of how it got there, so it never goes to Piper.
_ARTIFACTS = (
    re.compile(r"^\s*(?:you|them|riya|caller|user|assistant)\s*:", re.I),
    re.compile(r"\[[^\]]{2,40}\]"),          # [Client Name], [insert rate]
    re.compile(r"</?think>", re.I),
    re.compile(r"^\s*\d+\.\s"),              # numbered list, i.e. the checklist
    re.compile(r"\{[a-z_]+\}", re.I),        # an uninterpolated template field
    re.compile(r"<function[\s>]", re.I),     # hallucinated tool-call syntax
    re.compile(r"<!\[CDATA\[", re.I),
)


def _has_artifact(text: str) -> str:
    for pattern in _ARTIFACTS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return ""


def _sounds_like_a_chatbot(text: str) -> str:
    """Return the offending phrase, or "" if the line is in character."""
    low = _normalize(text)
    for tell in _ASSISTANT_TELLS:
        if _normalize(tell) in low:
            return tell
    for tell in _CANDIDATE_VOICE_TELLS:
        if _normalize(tell) in low:
            return tell
    if _has_emoji(text):
        return "emoji"
    return ""


def _parrots(reply: str, caller_said: str) -> bool:
    """True when the reply is just the caller's own words handed back.

    A live turn answered "Uh working with the big team" with "Working with the
    big team." -- six tokens, no question, no progress. It is not a repeat of
    anything Riya said, so the repetition guard missed it entirely.
    """
    r, c = _normalize(reply), _normalize(caller_said)
    if not r or not c:
        return False
    rw, cw = r.split(), c.split()
    if len(rw) > 12:
        return False
    # Near-verbatim echo, or wholly contained in what they just said.
    if r in c or c in r:
        return True
    overlap = len(set(rw) & set(cw)) / max(1, len(set(rw)))
    return overlap >= 0.8 and "?" not in reply


_WRONG_PERSON = re.compile(
    r"\b(they|them|their|they're|they've|we|we're|we've|our"
    # Singular third person about the caller. These were named in this
    # function's own docstring as the failures it exists to catch -- "does he
    # have", "has he worked" -- but were missing from the pattern, so it could
    # never have caught them.
    r"|he|she|him|his|her)\b", re.I,
)
_SECOND_PERSON = re.compile(r"\b(you|your|you're|you've)\b", re.I)

# Phrases that mean Riya has finished the call.
_FAREWELL = re.compile(
    r"\b(goodbye|good\s?bye|bye|take care|talk soon|speak soon|"
    r"talk to you later|speak to you later|"
    r"have a (?:great|good|wonderful|lovely|nice) "
    r"(?:day|evening|afternoon|morning|week|weekend|one)|"
    r"thank(?:s| you)(?: very much)? for "
    r"(?:your time|taking|calling|chatting|speaking with me)|"
    r"(?:lovely|nice|great) (?:speaking|talking|chatting) with you)\b",
    re.I,
)


def _is_farewell(text: str) -> bool:
    """True when Riya just said goodbye.

    A line that ENDS IN A QUESTION MARK is not a goodbye. Run 27 hung up on a
    caller mid-sentence because "can I thank you for taking time today?" and
    "Can I thank you for chatting with me today though?" both matched
    `thank you for (taking|chatting)` and counted as two farewells. Those are
    offers, not sign-offs. Erring toward "not a goodbye" only ever costs one
    more turn; erring the other way hangs up on someone mid-sentence.

    There was no end-of-call handling anywhere in this agent: she said goodbye
    at turn 8 and then kept taking turns through turn 18. Those eleven extra
    turns produced most of that call's worst output ("That information is
    confidential they don't require access to client data", a meta-apology, a
    chatbot tell) and dragged every latency average down with them. A real
    recruiter ends the call.
    """
    stripped = (text or "").strip()
    if not stripped or stripped.endswith("?"):
        # Ending on a question means she is still asking something. Erring
        # toward "not a goodbye" only ever costs one more turn; erring the
        # other way hangs up on someone mid-sentence.
        return False
    return bool(_FAREWELL.search(stripped))


# Giving up mid-call. Run 25 turn 6: the caller had just answered ("I said My
# SQL and post Gray SQL") and Riya replied "Sorry it seems we didn't quite
# match thanks anyway." -- she tried to end the call over a transcription
# error. It is not a goodbye, so _FAREWELL never saw it and the farewell
# counter never moved, but to the caller it reads exactly like being hung up
# on. A recruiter who cannot parse an answer asks for a repeat; only the
# CALLER gets to end the call.
_BAILOUT = re.compile(
    r"\b(?:didn'?t|doesn'?t|did\s+not|does\s+not|couldn'?t|could\s+not)\s+"
    r"(?:quite\s+|really\s+)?(?:match|work\s+out|fit)"
    r"|\bnot\s+(?:a\s+)?(?:good\s+|great\s+)?(?:match|fit)\b"
    r"|\b(?:don'?t|do\s+not)\s+seem\s+to\s+be\s+(?:a\s+)?(?:good\s+|great\s+)?"
    r"(?:match|fit)\b"
    r"|\bthanks?\s+anyway\b"
    r"|\b(?:we|i)'?ll\s+(?:leave|let)\s+it\s+(?:there|at\s+that)\b",
    re.I,
)

# When the caller is the one backing out, "not a match" is the correct answer
# and must survive. Only their own words can authorise ending the call.
# "no" on its own is NOT a decline -- a caller answering a yes/no question
# says it constantly, and reading that as "I want off this call" would let the
# bail-out stand on every negative answer. The refusal has to be explicit.
_DECLINE = re.compile(
    r"\b(?:not\s+interested"
    r"|no\s+(?:thank|thanks)(?:\s+you)?"
    r"|not\s+(?:looking|available|for\s+me)"
    r"|don'?t\s+(?:call|contact|bother)\s+me"
    r"|remove\s+(?:me|my\s+number)"
    r"|stop\s+calling"
    r"|leave\s+me\s+alone)\b",
    re.I,
)

# A read-back that never asks "is that right" is not a read-back. Run 25 turn
# 4 echoed the tech stack ("You've mentioned Python, Java and SQL so far") and
# then went straight on to another question; the checklist marked "read back
# and confirm?" done anyway, so years and education were never confirmed and
# the agenda ran out two turns early -- which is the frame turn 6's bail-out
# was generated in.
_READBACK_CONFIRM = re.compile(
    r"(?:sounds?\s+(?:right|correct|accurate)"
    r"|(?:is|was|does|did)\s+that\s+(?:right|correct|accurate)"
    r"|(?:did|have)\s+i\s+(?:get|got|captured)\s+(?:that|it|this)"
    r"|any\s+corrections?"
    r"|anything\s+(?:to\s+add|else|i\s+missed)"
    r"|(?:is|are)\s+there\s+anything\s+(?:to\s+add|i\s+missed))",
    re.I,
)
# Any run of 7+ digits (allowing spaces/dashes, as spoken phone numbers have)
# is phone-number-shaped. The call config never contains a phone number, so
# there is nothing legitimate this could match -- if it fires, it's invented.
# Written for: "Rachel can be reached at 94455 1234."
_DIGIT_RUN = re.compile(r"[\d][\d\-\s]{6,}[\d]")


def _invents_contact(reply: str) -> str:
    """Catch a fabricated phone number read aloud as if it were real."""
    for m in _DIGIT_RUN.finditer(reply):
        if len(re.sub(r"\D", "", m.group())) >= 7:
            return m.group().strip()
    return ""


def _wrong_person(reply: str) -> str:
    """Catch a reply that talks ABOUT the caller instead of TO them.

    Observed live: "They've confirmed interest.", "Yes, we're available.",
    "We're glad to hear that." Riya is on the phone with one person; that person
    is "you". These slipped past every other guard because they echo nothing and
    invent nothing -- they are simply in the wrong grammatical person.

    Only the FIRST sentence is checked, and only when it contains no "you". That
    keeps legitimate uses alive: "They're flexible on rate - does that work for
    you?" is about the client, and says "you", so it passes.
    """
    first = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0]
    if not first or _SECOND_PERSON.search(first):
        return ""
    match = _WRONG_PERSON.search(first)
    if not match:
        return ""
    # First-person plural is NEVER worth a regeneration.
    #
    # It used to be blocked unless a company-ish word appeared nearby, and
    # that heuristic failed in both directions. Run 13 rejected three replies
    # for it ("we", "our", "We") and the regenerations were worse than the
    # originals -- turn 13's retry opened "I shouldn't say 'we'.", the model
    # narrating the rule it had just been handed instead of answering the
    # caller. Turn 9's retry still said "our" anyway. So the check spent
    # three full generations (the single largest share of that call's 8
    # rejections) to make the output worse.
    #
    # "we"/"our" from a recruiter about her own side is normal speech and a
    # style preference, not a defect the caller will notice. Third person
    # about the caller IS a defect -- "they have confirmed interest" is
    # confusing in a way "we will follow up" is not -- so that still blocks.
    if match.group(1).lower() in {"we", "us", "our", "we're", "we've"}:
        return ""
    return match.group(0)


# Words that mean the caller asked something rather than stated something.
_QUESTION_WORDS = (
    "what", "why", "how", "who", "when", "where", "is", "are", "does", "do",
    "can", "could", "will", "would", "should",
)


def _is_direct_question(text: str) -> bool:
    """True when the caller asked something rather than stated something."""
    t = _normalize(text)
    if not t:
        return False
    if text.strip().endswith("?"):
        return True
    return t.split()[0] in _QUESTION_WORDS


def _unanswered_question(reply: str, caller_said: str) -> str:
    """Catch a reply that skips straight to the checklist question instead of
    answering what the caller just asked.

    Rule 2 in the system prompt says to answer a direct question in one short
    line before asking the checklist item -- but nothing enforced it, so "what
    is five times three" got buried inside an unrelated interview-scheduling
    question, three turns running, in a live call. This does not verify the
    answer is CORRECT, only that the reply does not open with another
    question -- which is the shape every observed miss actually took.
    """
    if not _is_direct_question(caller_said):
        return ""
    first = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0]
    if first and first.strip().endswith("?"):
        return caller_said.strip()[:80]
    return ""


def _echoes_agenda(reply: str, agenda: list[str]) -> str:
    """Catch a checklist item being read out instead of turned into a question.

    Even as fragments, the model sometimes inflates an item back into the
    third-person note it came from: "Are they open to hearing about a Senior
    Data Engineer role at all?" is item two, spoken at the candidate about the
    candidate. A recruiter says "you", never "they", so overlap with an agenda
    item plus third-person framing is a reliable tell.
    """
    words = set(_normalize(reply).split())
    if not words:
        return ""
    third_person = bool(words & {"they", "them", "their"})
    for item in agenda:
        item_words = set(_normalize(item).split())
        if len(item_words) < 2:
            continue
        overlap = len(words & item_words) / len(item_words)
        if overlap < 0.8:
            continue
        # The item spoken more or less AS IS. Run 9 did this twice: the agent
        # said "Years of experience?" and later "Read back and confirm?" --
        # the checklist note, verbatim, in the recruiter's own voice. Third
        # person used to be the only tell, but that let the verbatim copy
        # through, and a note read aloud is exactly what the terse fragments
        # exist to prevent.
        #
        # Length is what separates it from a correct rewrite: "How many years
        # of experience do you have?" shares all three item words but is eight
        # words long, so it has clearly been turned into a question.
        if len(words) < len(item_words) + 3:
            return item
        # ...and the note quoted as its own clause inside a longer reply, which
        # the length test structurally cannot see. Run 12 turn 7: "Four years
        # is quite extensive for an entry-level role. Read back and confirm?:
        # What was that" -- fourteen words, so not short, and no third-person
        # pronoun, so neither tell fired and the checklist went out verbatim.
        #
        # Compare clause by clause instead. Splitting on sentence boundaries
        # only (not commas) is what keeps this exact: "So, years of experience,
        # how many?" stays one clause and is not clause-identical to the item,
        # while "Read back and confirm?" is.
        if _normalize(item) in (
            _normalize(s)
            for s in re.split(r"[.?!:;]|\n", reply)
            if s.strip()
        ):
            return item
        # Third person is still a tell on a longer reply. AgendaTracker names
        # the live item in the turn note, so a correctly phrased question is
        # *supposed* to resemble it -- "years hands-on with Python, Spark, AWS"
        # and "How many years have you been hands-on with Python and Spark?"
        # overlap almost completely, and rejecting the second would reject
        # exactly the rewrite we asked for. What still cannot be right is
        # talking about the caller as "they" while talking to them.
        if third_person:
            return item
    return ""


def _strip_agenda_suffix(reply: str, agenda: list[str]) -> str:
    """Cut a checklist note that got bolted onto the END of a reply.

    Run 18 spoke the live item verbatim twice, appended after a clause with no
    separator at all: "...when you're available again qualification and field?"
    and "...job openings at our clients years of experience?".

    _echoes_agenda cannot see these. Its clause test splits on [.?!:;] and
    compares whole clauses, but the note is fused to the clause before it, so
    the clause it lands in is "when you're available again qualification and
    field" -- never equal to the item. Its length test needs the whole reply
    to be short, which a full sentence is not.

    This is a suffix test instead, and it repairs rather than rejects: the
    leak is mechanical, so it is cut out of the chunk in place. Rejecting
    would cost a second generation, which is exactly what
    regenerate_on_guard_rejection=false exists to avoid.

    Run 24: this cut too much. The sentence chunker split one sentence across
    two chunks -- "...how you use" then "it?\nRead back and confirm?" -- so the
    leftover on the second chunk was "it?", three characters, under the old
    four-character floor. The whole chunk was dropped and Riya stopped
    mid-sentence. A short leftover is still speech; only a leftover with no
    letters or digits in it at all is genuinely nothing but the note.
    """
    out = reply.rstrip()
    if not out.strip():
        return reply
    # Compare against the reply with any trailing terminator removed, so the
    # note matches whether or not the model put a "?" after it.
    tail = out.rstrip(".?!:;").rstrip()
    for item in sorted(agenda, key=len, reverse=True):
        bare = item.strip().rstrip(".?!:;").rstrip()
        if len(bare) < 4:
            continue
        if not tail.lower().endswith(bare.lower()):
            continue
        kept = tail[: len(tail) - len(bare)].rstrip(" \t\r\n,;:-")
        if not any(ch.isalnum() for ch in kept):
            # The whole chunk was the note -- nothing worth speaking is left.
            return ""
        # Do not cut when the note is genuinely part of the question. "Can you
        # tell me about your qualification and field?" is a correct rewrite
        # that happens to end on the item; amputating it would leave "Can you
        # tell me about your." A word that binds to what follows means the
        # model is asking, not quoting.
        before = kept.split()[-1].strip(".,!?;:").lower() if kept.split() else ""
        if before in _AGENDA_BINDERS:
            continue
        # The chunker splits mid-sentence, so what is left of a real sentence
        # can already carry its own terminator ("...how you use" / "it?").
        # Appending another one there produced "it?.".
        return kept if kept[-1:] in ".?!:;" else kept + "."

    # Run 26 turn 3: "Can you give me approximately how many years of total
    # experience you have? Qualification" -- the note is a single topic WORD,
    # not the whole item, so the suffix test above cannot see it. Only a word
    # standing on its own after a terminator counts: "How many years?" must
    # keep its "years", which is why the character before it has to be
    # punctuation rather than another word.
    for item in sorted(agenda, key=len, reverse=True):
        word = re.split(r"\W+", item.strip())[0] if item.strip() else ""
        if len(word) < 5:
            continue
        if not tail.lower().endswith(word.lower()):
            continue
        kept = tail[: len(tail) - len(word)].rstrip(" \t\r\n,;:-")
        if not kept or kept[-1:] not in ".?!:;":
            continue
        if not any(ch.isalnum() for ch in kept):
            return ""
        return kept

    # Run 26 turn 12: "...now let me clarify something read back and confirm
    # what technology languages frameworks..." -- the note sits in the middle
    # of the sentence this time, so no suffix test can reach it. Same binder
    # gate as above: "Can you tell me about your qualification and field?" is
    # a real question and must survive.
    for item in sorted(agenda, key=len, reverse=True):
        bare = item.strip().rstrip(".?!:;").rstrip()
        if len(bare) < 8:
            continue
        at = out.lower().find(bare.lower())
        if at <= 0:
            continue
        head = out[:at].rstrip()
        if not head:
            continue
        before = head.split()[-1].strip(".,!?;:").lower()
        if before in _AGENDA_BINDERS:
            continue
        pruned = (head + " " + out[at + len(bare):].lstrip()).strip()
        if not any(ch.isalnum() for ch in pruned):
            return ""
        return re.sub(r"\s{2,}", " ", pruned)
    return reply


# Words that, sitting immediately before a checklist item, mean the item is
# part of a real question rather than a note being quoted. Used only to avoid
# amputating a correct rewrite -- see _strip_agenda_suffix.
_AGENDA_BINDERS = {
    "a", "an", "the", "your", "my", "their", "our", "its", "his", "her",
    "about", "of", "in", "on", "for", "with", "to", "some", "any", "this",
    "that", "which", "what", "tell", "me", "know", "confirm", "is", "are",
    "was", "were", "and", "or", "so", "let",
}


# --------------------------------------------------------------------------
#  Deterministic repairs for the four failure shapes run 23 showed.
#
#  All four are mechanical: the offending span is identifiable, and what is
#  left is still a good reply. That is the test for doing this in code rather
#  than asking the model to behave -- nine prompt rules did not stop any of
#  them, and a rejection would cost a second generation.
# --------------------------------------------------------------------------

# Topics the screening brief forbids volunteering. Naming one the caller never
# raised produced "Our recruitment team will discuss salary details
# separately" as the OPENING LINE of runs 20 and 23, when nobody had mentioned
# salary. The redirect is correct behaviour when they raise it, so it is only
# cut when they did not.
_FORBIDDEN_TOPICS = {
    "salary", "salaries", "rate", "rates", "pay", "compensation", "ctc",
    "interview", "interviews", "notice", "joining", "offer",
}

# The same list as a WORD matcher.
#
# Run 35: the check above was a substring test (`t in clause`), and "rate" is
# a substring of "separately" -- so "will discuss suitable openings with you
# separately" fired the redirect rule for a topic that was never mentioned,
# while the real thing it should have looked at ("openings") was not on the
# list at all. Substring matching is wrong in both directions: it fires on
# "accurate", "corporate" and "generate", and it silently stops firing the
# moment the model picks a synonym. Word boundaries fix the first and the
# vocabulary below fixes the second.
_TOPIC_WORD = re.compile(
    r"\b(?:salaries|salary|compensation|ctc|rates|rate|pay|interviews|interview"
    r"|notice|joining|offers|offer|openings|opening|vacancies|vacancy"
    r"|positions|position)\b",
    re.I,
)
# The narrow half of that list: topics that are wrong to volunteer in ANY
# wording, not just inside a redirect. "openings" is deliberately not in here
# -- "We will follow up regarding suitable openings" is the brief's own
# closing shape, so only the "our recruitment team will..." promise is
# checked for it.
_FORBIDDEN_WORD = re.compile(
    r"\b(?:salaries|salary|compensation|ctc|rates|rate|pay|interviews|interview"
    r"|notice|joining|offers|offer)\b",
    re.I,
)

# Run 26: the same line came back with a different last word -- "Our
# recruitment team will discuss salary details LATER". Ending the pattern on a
# fixed list of closers is what let it through, so the tail is now a list of
# closers AND a fallback: if none of them appears, the cut stops at the next
# punctuation mark or at the next question ("...later can you confirm...").
_TEAM_REDIRECT = re.compile(
    r"(?:our|the)\s+recruitment\s+team\s+will\s+"
    r"(?:discuss|handle|cover|go\s+over|talk\s+about|follow\s+up\s+on|share"
    r"|finali[sz]e|confirm|take\s+care\s+of|let\s+you\s+know)"
    r"[^.?!]{0,60}?"
    r"(?:separately|with\s+you|later|shortly|soon|then|afterwards|directly"
    r"|at\s+the\s+end|in\s+a\s+(?:moment|bit|while|minute)"
    r"|with\s+(?:them|the\s+(?:hiring|client|team)))",
    re.I,
)
# Same sentence as above, no recognised closer -- used as a fallback so the
# redirect is cut on the forbidden topic itself rather than on the wording
# around it.
_TEAM_REDIRECT_HEAD = re.compile(
    r"(?:our|the)\s+recruitment\s+team\s+will\s+"
    r"(?:discuss|handle|cover|go\s+over|talk\s+about|follow\s+up\s+on|share"
    r"|finali[sz]e|confirm|take\s+care\s+of|let\s+you\s+know)",
    re.I,
)
# Where an unclosed redirect clause must stop: punctuation, or the next real
# question. Without this the cut would swallow the question that follows.
_REDIRECT_STOP = re.compile(r"[.?!,;]|\s+(?:can|could|shall|would|will|let)\b", re.I)

# The words a redirect clause ends on. Run 31: "...will discuss openings
# later, separately thank you again..." -- the stop landed on the comma BEFORE
# the closing adverb, so the cut left the reply opening with a bare
# "Separately". If one of these follows the punctuation the stop search found,
# that punctuation is mid-clause and has to be skipped.
_REDIRECT_CLOSER = re.compile(
    r"(?:separately|with\s+you|later|shortly|soon|then|afterwards|directly"
    r"|at\s+the\s+end|in\s+a\s+(?:moment|bit|while|minute)"
    r"|with\s+(?:them|the\s+(?:hiring|client|team)))\b",
    re.I,
)
# Of those, the ones that cannot open a reply. "Then" and "later" can start a
# sentence ("Then we can move on"); these three are only ever the tail of a
# clause that a cut removed, so if the cut leaves one in front it is dropped.
_ORPHAN_CLOSER = re.compile(r"^(?:separately|afterwards|thereafter)\b[.,;:]*", re.I)

# Narrating the call itself instead of making it. The dominant defect of run
# 26: "You're correct that we already covered that earlier", "You are correct
# we should confirm your previous answers instead of repeating them", "No we
# strayed from our original purpose", "I don't know how long ago your
# education ended", "Sorry to hear that I'm being unhelpful apologize for any
# frustration caused". None of these carries a question, so the agenda does not
# advance either -- the checklist moved 5 times in 14 turns and the call ran
# on. The clause is cut; whatever question follows survives.
_META_COMMENT = re.compile(
    r"(?:"
    r"(?:you(?:'re|\s+are)\s+)?(?:correct|right)[,.]?\s+(?:that\s+)?"
    r"(?:we|I)\s+(?:'?ve\s+|already\s+)*(?:covered|discussed|went\s+over|asked)\s+"
    r"(?:that|this|it|those)\s*(?:earlier|before|already)?"
    # Run 26 turn 9: "You are correct we should confirm your previous answers
    # instead of repeating them..." -- the agreement itself is the dead part;
    # the lookahead keeps this from firing on "You're correct, five years."
    r"|(?:you(?:'re|\s+are)\s+)?(?:correct|right)\b(?=\s+(?:we|I)\s+"
    r"(?:should|need|must|strayed|didn'?t|'?ve))"
    # Run 36 turn 5: "You're correct about our notes so far You've got an
    # Undergrad Computer Science from a four-year program..." -- the same dead
    # agreement in a third wording ("correct about X" rather than "correct
    # that/we"), and it went out uncut with the real content trailing after
    # it. The lookahead ends the cut at the next clause so "You've got an
    # Undergrad..." survives.
    # Run 37 turn 5: "You're correct on both points." Same agreement, now with
    # "on". The preposition list is open rather than fixed -- chasing them one
    # at a time is what made runs 27-33 feel endless.
    r"|(?:you(?:'re|\s+are)\s+)?(?:correct|right)\s+"
    r"(?:about|on|regarding|concerning)\s+"
    r"(?:our|the|your|my|this|that|it|both|those|these|all)\s+[^.?!]{0,40}?"
    r"(?=\s+(?:you|i|we|so|and|but|now|let|that|it|this)\b|[.?!]|$)"
    r"|(?:we|I)\s+(?:should|need\s+to|must)\s+(?:confirm|not\s+repeat|move\s+on"
    r"|avoid\s+repeating)[^.?!]{0,60}?(?:instead\s+of\s+repeating|repeating\s+them"
    r"|repeating\s+it|again)(?:\s+them)?"
    r"|(?:we|I)\s+(?:'?ve\s+)?(?:strayed|drifted|gone\s+off)[^.?!]{0,40}?"
    r"(?=\s+(?:now|so|let|but|and|I|we)\b|[.?!]|$)"
    r"|(?:let\s+me|let's)\s+(?:clarify|rephrase|re-?ask)[^.?!]{0,30}?"
    r"(?=\s+(?:what|which|how|read|just|so|now)\b|[.?!]|$)"
    r"|I\s+don'?t\s+know[^.?!]{0,60}?\s+ended"
    # Both of these run up to a question or a stop, never into it: without the
    # lookahead "I'm being unhelpful..." swallowed the "shall I close this
    # call?" that followed and left "Sorry to hear that l now since ."
    r"|(?:I'?m|I\s+am)\s+being\s+unhelpful[^.?!]{0,80}?"
    r"(?=\s+(?:shall|can|could|let|so|now|but)\b|[.?!]|$)"
    r"|apolog(?:ise|ize)\s+for\s+any\s+(?:frustration|confusion|misunderstanding)"
    r"[^.?!]{0,40}"
    # "since we didn't quite cover everything" -- taking the conjunction with
    # it, or the cut leaves a dangling "since".
    r"|(?:since|as|because)\s+(?:we|I)\s+didn'?t\s+(?:quite\s+)?cover[^.?!]{0,40}"
    r"|(?:we|I)\s+didn'?t\s+(?:quite\s+)?cover[^.?!]{0,40}"
    r")",
    re.I,
)

# Explaining a term back to the person who asked about it. Run 26 turn 11: the
# candidate said "What is data warehouse management system, what the fuck are
# you talking about?" and got "Data warehouse management systems refer to
# software tools that help manage large datasets for analysis." Run 24 had the
# same shape ("AI/ML is a machine learning framework"). A screener does not
# lecture; the whole definitional sentence goes, keeping whatever follows.
_DEFINITION = re.compile(
    r"\b\w[\w/+'.-]*\s+(?:systems?|tools?|frameworks?|terms?|platforms?|stacks?)?\s*"
    r"(?:refer(?:s|ence)?\s+to|means|is\s+defined\s+as"
    r"|is\s+a\s+[\w\s-]{0,24}?"
    r"(?:kind|type|sort|way|tool|system|software|set|database|language|framework"
    r"|technology|platform|process))\b[^.?!\n]{0,140}",
    re.I,
)

# Narrating the caller back to them. Run 23: "You're having trouble
# answering", "Your response was unclear", "You hung up on me". A recruiter
# does not diagnose the person on the phone, and turn 3's line is what later
# tripped the repeat guard and cost a second generation.
_CALLER_COMMENT = re.compile(
    r"\b(?:you(?:'re|\s+are)?\s+(?:having\s+trouble|struggling|not\s+answering|confused)"
    r"|your\s+response\s+(?:was|is|seems)\s+(?:unclear|confusing|not\s+clear|vague)"
    r"|you\s+hung\s+up(?:\s+on\s+(?:me|us))?"
    r"|you\s+cut\s+(?:us|me)\s+off"
    r"|you(?:'re|\s+are)\s+not\s+(?:helping|being\s+clear)"
    # Run 26 turn 16: "You're ending this conversation early thanks for your
    # time goodbye." Commenting on what the caller is doing is not ours to say.
    r"|you(?:'re|\s+are)\s+ending\s+(?:this|the)\s+(?:conversation|call)"
    r"(?:\s+(?:early|soon|now|already))?"
    r"|you(?:'re|\s+are)\s+(?:wasting|repeating\s+yourself))\b",
    re.I,
)

# Apologising for the machinery. Run 23's retry opened "Sorry I'm Riya from IT
# Staffing Solutions Hi sorry I forgot my name earlier" -- narrating its own
# mistake instead of just carrying on. Deliberately narrow: a bare "sorry" is
# often a genuine apology and must survive.
_META_APOLOGY = re.compile(
    r"\b(?:I\s+(?:forgot|misspoke|apologise|apologize|made\s+a\s+mistake)"
    r"|sorry,?\s+I\s+(?:forgot|misspoke))\b",
    re.I,
)

# Claiming to know the openings book. Run 23: "No I don't currently have an
# opening for an A Engineer". She cannot know that; the brief says redirect.
_OPENINGS_CLAIM = re.compile(
    r"(?:"
    # "we don't have an opening" / "I don't currently have any openings".
    # The leading "No"/"Sorry" is part of the same breath; without it the cut
    # leaves "No for an A Engineer".
    r"\b(?:no|sorry|yes)?[,\s]{0,3}"
    r"(?:we|I)\s+(?:don'?t|do\s+not)\s+(?:currently\s+)?have\s+"
    # "have an opening" and "have any openings" both occur; without the article
    # the run-23 line "No I don't currently have an opening for an A Engineer"
    # did not match at all.
    r"(?:any\s+|an\s+|a\s+)?"
    r"(?:opening|openings|position|positions|role|roles|vacanc\w*)"
    r"|"
    # Run 24 turn 11: "No immediate openings but our team will follow up
    # regarding suitable roles." Same claim, different grammar -- no verb, so
    # the pattern above never saw it. Kept narrow on purpose: the qualifier is
    # required for position/role, because "no positions that match" is not a
    # claim about the openings book and must survive.
    r"\bno\s+(?:immediate|current|present|open|available|suitable|relevant)\s+"
    r"(?:openings?|positions?|roles?|vacanc\w+)"
    r"|"
    r"\bno\s+(?:immediate|current|present|open|available|suitable|relevant)?\s*"
    r"(?:openings|vacanc\w+)"
    r")",
    re.I,
)


def _clause_end(text: str, end: int) -> int:
    """Where a cut should stop, given the offending phrase ends at `end`.

    These replies are frequently unpunctuated, so cutting only the phrase
    leaves a fragment: "You're having trouble answering" cut at the phrase
    leaves "answering That helps clarify things". If what follows begins with a
    lowercase letter we are mid-clause, so run forward to the next capitalised
    word, which is where the next sentence actually starts.

    This cannot be a `(?=[A-Z][a-z])` lookahead inside the patterns above,
    because they are compiled with re.I -- under re.I [A-Z] also matches
    lowercase, so the lookahead was always satisfied and the tail cut nothing.
    Case sensitivity has to be done here, in Python.
    """
    i = end
    while i < len(text) and text[i] in " ,;:-":
        i += 1
    if i >= len(text) or not text[i].islower():
        # Next thing is already the start of a sentence -- nothing to extend.
        return i
    j = i
    while j < len(text):
        if text[j].isupper():
            k = j
            while k < len(text) and text[k].isalpha():
                k += 1
            # A capital is only a sentence start if it opens a word that can
            # actually begin a clause. "for an A Engineer You're looking at..."
            # has two capitals and only the second is the sentence; matching any
            # capital left "Engineer You're looking at...".
            if len(text[j:k]) > 2 and text[j:k].lower() in _CLAUSE_OPENERS:
                return j
        j += 1
    # No clause opener ahead, so this phrase ran to the end of the chunk.
    return i


# Words that can begin a clause. Used only to decide where a cut should stop;
# deliberately excludes articles and single letters, which are capitals inside
# a clause ("A Engineer") rather than the start of the next one.
_CLAUSE_OPENERS = frozenset({
    "and", "any", "are", "can", "could", "did", "do", "does", "good", "great",
    "has", "have", "how", "is", "it", "just", "let", "now", "okay", "please",
    "shall", "so", "sorry", "sure", "thank", "thanks", "that", "the", "then",
    "this", "was", "we", "what", "when", "where", "which", "who", "why", "will",
    "would", "yes", "you", "your", "our", "i",
})


def _word(w: str) -> str:
    return w.lower().strip(",.;:!?-\"'")


def _eat_greeting(chunk: str, remaining: list[str]) -> tuple[int, str]:
    """Consume leading greeting words from a chunk.

    Run 32 turn 1 and run 33 turn 3 both opened a mid-call reply with the
    whole greeting; the repeat guard rejected it and the run paid for a second
    generation -- 1723 ms on run 33 turn 3, its worst turn. The greeting is a
    fixed line in config, so this is knowable in advance and should never have
    cost anything.

    Word-by-word, carrying the remainder across chunks, because the chunker
    can cut the greeting anywhere: a partial greeting in chunk one and the
    rest in chunk two have to be recognised as one thing.
    """
    got = chunk.split()
    i = 0
    while i < len(got) and i < len(remaining) and _word(got[i]) == _word(remaining[i]):
        i += 1
    return i, " ".join(got[i:]).strip()


def _split_runon(piece: str) -> list[str]:
    """Split a question clause off an unpunctuated run-on.

    "That sounds good Now did you complete an undergraduate degree?" has to
    lose only the second clause. Splitting at ANY capital does not work --
    "Do you know Python?" would become "Do you know" -- so the capital has to
    open a word that can actually begin a clause. "Now" and "How" can;
    "Python" cannot.
    """
    out: list[str] = []
    start = 0
    for m in re.finditer(r"\s+(?=[A-Z])", piece):
        word = re.match(r"[A-Za-z']+", piece[m.end():])
        if not word or len(word.group(0)) < 3:
            continue  # "I" and "A" are capitals inside a clause, not before one
        if word.group(0).lower() not in _CLAUSE_OPENERS:
            continue
        head = piece[start: m.start()].strip()
        if head:
            out.append(head)
        start = m.start()
    tail = piece[start:].strip()
    if tail:
        out.append(tail)
    return out or ([piece] if piece.strip() else [])


# Opens a question. Used on shape rather than punctuation -- see
# _is_model_question. Deliberately excludes is/are/was/were/have/has, which
# open ordinary statements far more often than questions.
_QUESTION_OPENER = re.compile(
    r"^(?:what|which|how|when|where|why|who|whom|whose|can|could|would|will"
    r"|shall|should|do|does|did|may|any|tell)\b",
    re.I,
)

# Utterances that acknowledge rather than answer. The read-back must not
# repeat these back as though they were screening data -- see _readback.
_ACK_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "right", "correct",
    "true", "fine", "good", "great", "thanks", "thank", "um", "uh", "hmm",
    "mm", "hmm", "oh", "ah", "free", "available", "go", "ahead", "please",
    "hello", "hi", "bye", "no", "not", "really", "absolutely", "definitely",
})

# Words that carry no screening content: pronouns, articles, auxiliaries and
# the fillers Parakeet sprinkles into a short reply. See _readable_answer.
_ANSWER_NOISE = frozenset({
    "i", "im", "i'm", "am", "is", "are", "was", "were", "be", "been",
    "you", "your", "yours", "me", "my", "we", "us", "our", "it", "its",
    "a", "an", "the", "this", "that", "these", "those",
    "and", "or", "but", "so", "if", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "about", "as", "like", "just", "very", "all",
    "have", "has", "had", "do", "does", "did",
    "sound", "sounds", "sound's", "mean", "means", "guess", "know", "think",
    "sorry", "saying", "said", "say", "tell", "told",
})

# Discourse markers that can only ever start an answer, never carry one.
# Stripped from the FRONT of an answer only -- see _readable_answer.
_LEADING_MARKERS = frozenset({
    "so", "oh", "well", "ok", "okay", "yeah", "yes", "and", "but", "actually",
    "basically", "like", "um", "uh", "hmm", "mm", "er", "erm", "right",
})


def _readable_answer(answer: str) -> str:
    """The part of an utterance worth reading back, or "" if there is none.

    Run 36 turn 2 read back "Oh you sound free" as though it were a
    qualification -- Parakeet's guess at a four-word grunt. Two separate
    filters were supposed to stop that and neither did: the floor was >3
    words and it is exactly four, and only one of its words is an
    acknowledgement so the all-acknowledgements test passed as well.

    Disfluencies come out first, then CONTENT words are counted. That is what
    separates an affirmation from an answer: "Oh yes you are right" still has
    four words after "oh" goes, so a length test alone cannot drop it, but it
    has zero content words. "I have five years of experience" keeps exactly
    three, so the floor is 3.
    """
    words: list[str] = []
    for w in answer.strip().rstrip(".").split():
        if w.strip(".,!?;:").lower() in _FILLER:
            continue
        # "...and uh and..." -> "...and..."
        if w.lower() == "and" and words and words[-1].lower() == "and":
            continue
        words.append(w)
    # Start on the fact, not on how they got to it. Run 37's caller opened
    # with "Oh so my highest qualification is...", and the confirm line read
    # "So my highest qualification is..." -- "oh" is a filler but "so" is not,
    # and a read-back that opens on a connective sounds like a fragment.
    # "i" is deliberately NOT here: it would turn "I have five years" into
    # "have five years".
    while words and words[0].strip(".,!?;:").lower() in _LEADING_MARKERS:
        words.pop(0)
    content = [
        w for w in words
        if w.strip(".,!?;:").lower() not in _ACK_WORDS
        and w.strip(".,!?;:").lower() not in _ANSWER_NOISE
    ]
    if len(content) < 3:
        return ""
    # The WHOLE answer, deliberately. Capping each one at N words cut the
    # substance out of the most important answer of the call: "my highest
    # qualification is under graduation and that is in computer science"
    # became "...under graduation", losing the field. The total is bounded by
    # the caller of this function instead.
    return " ".join(words)


def _is_model_question(clause: str) -> bool:
    """Is this clause a question the model wrote, so Python should drop it?

    Run 34 turn 4: "Can you name any other tools specifically used by our
    recruitment team when matching candidates like." -- interrogative in every
    way except the final character. The model closed it with a period, so the
    "?" test missed it and it went out in front of the templated question.
    Shape has to count, not just punctuation. The length floor keeps short
    exclamations ("How nice", "What a day") from being read as questions.
    """
    if clause.rstrip().endswith("?"):
        return True
    words = clause.split()
    return len(words) > 4 and bool(_QUESTION_OPENER.match(clause.strip()))


def _strip_model_questions(text: str) -> str:
    """Drop the question the model wrote. Python asks it instead.

    One rule that covers a whole family of defects we have been chasing
    separately since run 24: the checklist item read aloud, two questions in a
    turn, a question already answered, a question invented off-agenda, and
    "Can our recruitment team help with finding suitable openings based on my
    Qualifications?" (run 33 turn 3, verbatim).

    Splits on CLAUSES, not sentences, because these replies are usually
    unpunctuated: "That sounds good Now did you complete an undergraduate
    degree?" has to lose only the second clause, or it loses everything.
    Returns "" when the model wrote nothing but a question -- correct here,
    because the caller is about to hear the real one.
    """
    if not text or not text.strip():
        return text
    clauses: list[str] = []
    for part in re.split(r"(?<=[.?!])\s+|\n+", text):
        piece = part.strip()
        if not piece:
            continue
        if piece.endswith("?"):
            clauses.extend(_split_runon(piece))
        else:
            clauses.append(piece)
    kept = [c for c in clauses if not _is_model_question(c)]
    if len(kept) == len(clauses):
        return text  # no question in it; leave the reply alone
    if not any(any(ch.isalnum() for ch in c) for c in kept):
        return ""
    out = re.sub(r"\s{2,}", " ", " ".join(kept)).strip(" \t\r\n,;:-")
    # A clause cut mid-sentence can leave a dangling connective.
    out = re.sub(r"\s+(?:and|now|so|but|then)$", "", out, flags=re.I).strip(" ,;:-")
    return out


def _redirect_stop(text: str, start: int) -> int:
    """Where an unclosed redirect clause ends.

    Earliest of: real punctuation, the next question, or the start of the next
    clause. A comma or semicolon is skipped when a closing adverb follows it,
    because there the clause is still running -- that skip is what keeps
    "...later, separately thank you" from answering "Separately thank you".
    """
    best = len(text)
    for found in _REDIRECT_STOP.finditer(text, start):
        if found.group(0) in ",;":
            after = text[found.end(): found.end() + 40].lstrip()
            if _REDIRECT_CLOSER.match(after):
                continue
        best = found.start()
        break
    return min(best, _clause_end(text, start))


def _redirect_tail(text: str, stop: int) -> int:
    """Walk a stop forward over the rest of the clause it landed inside.

    The stop usually lands on the first closing adverb, and the model very
    often staples a second one on after a comma. "...discuss openings later,
    separately thank you" stopped on "later", which left the reply beginning
    "Separately thank you". Both belong to the redirect, so both go.
    """
    i = stop
    while True:
        j = i
        while j < len(text) and text[j] in " \t,;":
            j += 1
        found = _REDIRECT_CLOSER.match(text, j)
        if not found:
            return i
        i = found.end()


def _clean_seam(text: str) -> str:
    """Clean the front of a reply after a clause was cut out of it.

    Run 31 spoke three separate artefacts of the redirect cut: a leading
    period (". Thank you again..."), a doubled terminator ("Sure.,
    separately. Thank you...") and an orphan adverb ("Separately thank
    you..."). All three are seam junk, so they are all removed here rather
    than at each cut site.
    """
    out = text
    while True:
        before = out
        # "Sure." + ", separately." -> "Sure. separately."
        out = re.sub(r"([.?!])(?:[ \t]*[.,;:])+", r"\1", out)
        # "Sure." + ". Thank you" -> "Sure. Thank you"
        out = re.sub(r"([.?!])(?:[ \t]*[.?!])+", r"\1 ", out)
        out = out.lstrip(" \t\r\n,;:.-")
        orphan = _ORPHAN_CLOSER.match(out)
        if orphan:
            out = out[orphan.end():].lstrip(" \t\r\n,;:.")
        if out == before:
            return out


# A cut can strand the tail of the clause it removed at the END of the reply.
# Run 35 turn 6: "You are ready to move forward; our recruitment team will
# discuss suitable openings with you separately now." cut the promise and left
# "You are ready to move forward; now." -- a semicolon and an adverb with no
# clause left to modify. Scoped to the cut that causes it, NOT applied to
# every reply: a general "strip the last adverb" rule would eat perfectly good
# lines like "Great, talk soon."
_ORPHAN_TAIL = re.compile(
    r"[;,:]?\s*\b(?:now|then|afterwards|thereafter|separately|shortly|soon"
    r"|later|today)\b\s*[.?!]*$",
    re.I,
)

# Speech disfluency. These are Parakeet faithfully transcribing hesitation,
# and they are never part of an answer -- reading "and uh and" back makes the
# whole confirmation sound broken.
_FILLER = frozenset({
    "uh", "um", "umm", "uhh", "oh", "ah", "er", "erm", "mm", "hmm", "mmhmm",
    "yeah", "yep", "yup", "okay", "ok", "hmm", "like", "basically",
})

# Ceiling on the read-back, in words, across all answers. It replaced a
# per-answer cap, which bounded the length the same way but chopped the middle
# out of the answers to do it.
_READBACK_WORDS = 35

# Words a truncated read-back must not be left dangling on.
_TRAILING_JUNK = frozenset({
    "and", "of", "the", "in", "with", "from", "to", "a", "an", "for", "on",
    "at", "by", "that", "is", "are", "was", "were", "my", "our", "as", "but",
})

# The model writing its own read-back. On the confirm turn Python reads the
# answers back from the template, so this is always a duplicate. Run 36 turn 5
# did the same thing on the CLOSING turn -- 38 tokens of "You've got an
# Undergrad Computer Science from a four-year program followed by five years
# of work experience..." in front of a goodbye that is already a full line.
_MODEL_READBACK = re.compile(
    r"^(?:you(?:'ve|\s+have)?\s+(?:said|mentioned|told|stated|got)"
    r"|as\s+you\s+(?:said|mentioned)"
    r"|like\s+you\s+said"
    r"|(?:you(?:'re|\s+are)\s+)?(?:correct|right)\s+about"
    # Run 39 turn 6, the closing turn: "Your qualifications include an
    # undergraduate degree in Computer Science with 5 years of relevant Work
    # experience primarily using Python and Java." A THIRD-person summary of
    # the caller, which none of the frames above cover -- they are all
    # first/second person ("you've said", "as you said"). The read-back is
    # Python's, so this is always a duplicate; on the closing turn it landed in
    # front of a goodbye that is already a full line and pushed Piper to 760 ms
    # (3 audio chunks) on a turn that should have been short.
    r"|your\s+(?:qualifications?|experience|background|profile|skills?"
    r"|details?|education|degree|history|answers?)\s+"
    r"(?:include|includes|cover|covers|consist\s+of|consists\s+of|are|is)\b)",
    re.I,
)

# Cuts that may leave the reply empty. Everything else keeps the "never
# amputate the whole reply" guard, which still holds for a SPAN cut out of a
# longer reply. These five are different: the offending text is the whole
# reply, so leaving it means saying the one thing we just decided not to say.
# Empty is safe because the repairs only run in persona mode and the templated
# question is appended afterwards -- the caller always hears a line.
# "narrated the call" joined them for run 38 turn 6's defect: "No specific
# openings were mentioned during our call regarding AI & ML roles." is a claim
# about the openings book she cannot make, and with no redirect clause after it
# to keep, the whole reply is the defect.
# "claim about what the client wants" joined for run 39 turn 2, whose whole
# reaction was "Typically our clients look for degrees above undergraduate
# level." -- and the sentence-start cut leaves nothing but that claim.
#
# This list is the reason a whole defect FAMILY costs zero regenerations
# instead of one. Run 40 turn 5 was the demonstration: the reaction was "No
# information on SQL skills was mentioned.", the repair had not been able to
# see it, and the repeat guard caught the line only because it happened to
# duplicate turn 4 -- `repeat` is in regenerate_kinds, so it paid 1785 ms for
# a second generation. Both turns 4 and 5 said that sentence; the cost was
# charged to a guard doing its job correctly, because NOTHING EARLIER HAD
# REMOVED THE SENTENCE. A label in this set removes it for free, and the
# templated question still goes out, so the caller still hears a turn.
_MAY_EMPTY = frozenset({
    "commitment she cannot make",
    "narrated the machinery",
    "narrated the call",
    "commentary about the call",
    "claim about what the client wants",
})

# Guard kinds that are CUT rather than retried or merely warned about.
#
# Run 42 is the measurement. It failed the definition of done at 1348 ms avg,
# and the summary printed the cause itself: "2/6 turns regenerated after a
# guard rejection (each roughly doubles that turn's LLM time)". Those two
# turns logged TTFT 745 ms and 584 ms against 360-441 ms on the turns that did
# not retry. Every other stage had actually IMPROVED on run 41 -- STT 470 ->
# 345, TTS 731 -> 512 -- so the entire regression was the retry policy.
#
# The reasoning that put repeat/parrot/agenda in `regenerate_kinds` was that
# those three are "mechanical", so a correction note gives the model something
# concrete to avoid. That reasoning is now obsolete, and run 42 proves it: on
# turn 6 the rejected text was 'Oh yes you are right.' -- the caller's own
# words, echoed back. There is nothing to regenerate there. The reply is wrong
# because it is a duplicate, and the fix is to not say it.
#
# Cutting is free because of the templated question: the reaction is dropped
# and the question still goes out, so the caller hears a turn either way. This
# is _MAY_EMPTY applied to the guard rather than to the repair.
_CUT_KINDS = frozenset({"repeat", "parrot", "agenda"})

# Sentinel kind meaning "drop this text, do not retry". Compared with `==`
# rather than `is` so it survives being returned through a tuple.
_CUT = "cut"

# Promising something Riya has no authority to promise. Run 36 turn 6, after
# the caller asked about openings: "I'll check on any relevant job openings at
# our client companies soon." She has no client list and no mandate; the
# config says no commitment -- no callback, no date, no "we'll be in touch".
_COMMITMENT = re.compile(
    r"\b(?:I(?:'ll|\s+will|\s+shall)|we(?:'ll|\s+will)|let\s+me)\s+"
    r"(?:check|look|see|find|confirm|get\s+back|follow\s+up|let\s+you\s+know"
    r"|call\s+you|reach\s+out|update|share|send|arrange|search|review|forward"
    r"|ask)\b[^.?!]{0,60}",
    re.I,
)

# Narrating the machinery. Run 37 turn 7, straight after a barge-in: "It seems
# we lost connection mid-sentence." The caller had only repeated themselves.
# Nobody on a phone call says this, and the caller cannot act on it.
# Each branch runs to the end of its clause. Stopping at the verb left
# "Connection." -- _clause_end only walks forward from where the match ended,
# and "connection" is lowercase with no clause opener after it.
# The leading interjection is part of the same breath, exactly as in
# _OPENINGS_CLAIM: without it "Sorry, you cut out for a moment." cut to
# "Sorry,." -- the apology survives the cut that removed what it apologised
# for.
# Narrating what the call has and has not contained. Run 38 turn 6, answering
# "Is there any opening in AI and ML?": "No specific openings were mentioned
# during our call regarding AI & ML roles". She is describing the conversation
# to the person she is having it with, and "no specific openings were
# mentioned" is a claim about the openings book that she cannot make -- the
# brief says redirect that, which the rest of the same sentence then does.
# Every branch stops at a clause boundary, not at a character count: a greedy
# 60-character tail ran past the "--" and ate the "our" of the redirect that
# follows, leaving "Team will discuss that with you separately."
#
# KEYED ON SUBJECT + PREDICATE, WITH NO LENGTH WINDOW. Run 40 turn 4/5 added
# the third wording of this same defect -- "No information on SQL skills was
# mentioned." -- and it slipped through the {-bounded} gap below, which was
# `[a-z\s]{0,24}`: "information on SQL skills" is 26 characters, two more than
# the window allowed. "No specific openings" (18) and "Nothing about Spark"
# (17) had fitted, so the rule looked correct for three runs while being one
# character away from the next failure. Widening the bound chases the model
# forever -- the 3B's next noun phrase is always a little longer than the last.
#
# The invariant across all three wordings is not the length of the middle. It
# is that the SUBJECT is the conversation (information / openings / details /
# anything / nothing ...) and the PREDICATE is a discourse verb (mentioned /
# noted / discussed / covered / raised). So there is no window at all here: a
# bounded gap between the two is what keeps breaking, and the fix is to stop
# writing one. Same move as _CLIENT_REQUIREMENT -- key on the invariant, not
# on the phrasing the model reached for last time.
_CALL_NARRATION = re.compile(
    r"\b(?:during|in)\s+(?:our|this|the)\s+call\b[^.?!]{0,60}?"
    r"(?=\s*(?:--|,|;|\.|!|\?|$))"
    r"|\b(?:as|like)\s+(?:we|i)\s+(?:discussed|mentioned|talked\s+about)\b"
    r"[^.?!]{0,60}?(?=\s*(?:--|,|;|\.|!|\?|$))"
    # The conversation as subject, any discourse verb. "no information on SQL
    # skills was mentioned", "nothing about your background has been
    # discussed", "no details were raised" -- one rule, unbounded middle.
    #
    # Two subject shapes, because the model uses two. "No information on X"
    # names the thing and qualifies it afterwards; "Nothing about X" names no
    # thing at all and goes straight to the about-phrase. Requiring the noun
    # slot in both is what missed "Nothing about Spark was mentioned" on the
    # first pass at this rule.
    r"|\b(?:no|nothing|not\s+any|any)\s+"
    r"(?:specific\s+|detailed\s+|particular\s+|further\s+)?"
    r"(?:"
    r"(?:information|openings?|details?|mention|discussion|comment"
    r"|indication|word|reference|record|note|answer|response|feedback"
    r"|confirmation|thing|things|such\s+thing)\b"
    r"(?:\s+[\w'\-]+){0,8}?"
    r"|\b(?:about|regarding|concerning|on)\b(?:\s+[\w'\-]+){0,8}?"
    r")"
    r"\s+(?:was|were|is|are|has\s+been|have\s+been|had\s+been)\s+"
    r"(?:mentioned|noted|discussed|covered|raised|brought\s+up|stated|said"
    r"|given|provided|shared|received|made|recorded|confirmed)\b"
    r"[^.?!]{0,60}?(?=\s*(?:--|,|;|\.|!|\?|$))"
    r"|\b(?:earlier\s+in\s+the\s+call|in\s+this\s+conversation)\b",
    re.I,
)

_MACHINERY_NARRATION = re.compile(
    r"\b(?:sorry|oh|well|no|yes)?[,\s]{0,3}(?:"
    r"(?:it\s+seems|it\s+looks\s+like|looks\s+like|sounds\s+like|seems\s+like)"
    r"\s+(?:we|I|you)\s+(?:lost|dropped|missed)\b[^.?!]{0,40}"
    r"|(?:we|I|you)\s+(?:lost|dropped)\s+(?:the\s+)?connection\b[^.?!]{0,40}"
    r"|(?:we|I|you)\s+(?:were|was|got)\s+cut\s+off\b[^.?!]{0,40}"
    r"|mid-?sentence\b"
    r"|(?:you|we)\s+(?:cut\s+out|broke\s+up|went\s+quiet|disappeared)\b[^.?!]{0,40}"
    r")",
    re.I,
)

# Lecturing about how the world works. Run 36 turn 2: "Undergraduate degree in
# Computer Science typically takes 3 years so how many years." Same defect as
# _DEFINITION with different grammar -- a screener does not teach.
_LECTURE = re.compile(
    r"\b(?:typically|usually|generally|normally|ordinarily)\s+"
    r"(?:takes|lasts|runs|is|are|means)\b[^.?!]{0,40}",
    re.I,
)

# Stating the client's requirements. Run 39 turn 2, answering "my highest
# qualification is under graduation in computer science": "Typically our
# clients look for degrees above undergraduate level." Riya has no mandate to
# state what the client wants on a first screening call, and phrasing it as a
# bar ("above undergraduate level") tells the caller they fall short.
#
# _LECTURE cannot see this and never will: it is keyed on the VERB after
# "typically" (takes|lasts|is|means), so "typically our clients look for" is
# invisible to it. Keyed on the SUBJECT instead, one rule covers the family
# whatever verb the model reaches for next -- the same move as rule 1b.
#
# The plural/generic subject is the line that keeps this off "That matches
# what our client is looking for." A PLURAL subject ("our clients", "most
# employers") is a rule about the market, which is the defect. A SINGULAR one
# ("our client") is a remark about this role and this caller, which is
# encouraging and must survive -- so a singular subject is only cut when the
# sentence is explicitly generalised ("typically our client looks for").
_CLIENT_REQUIREMENT = re.compile(
    r"\b(?:"
    # Plural/generic subject: a statement about the market or the client base,
    # never about this role or this caller. ANY verb follows. Listing the verbs
    # is the mistake that was just made here -- the rule went in keyed on
    # want|need|look for|prefer|require and the very first replay produced
    # "our clients often MATCH candidates", which it could not see. The subject
    # is the invariant; the verb is whatever the model reaches for next.
    r"(?:our|the|most|many|some)\s+(?:clients|companies|employers|firms"
    r"|organi[sz]ations)\b"
    # Singular subject, but only behind a generalisation marker. "That matches
    # what our client is looking for." is a remark about this caller and must
    # survive; "typically our client looks for a masters" is a rule about the
    # market and must not.
    r"|(?:typically|usually|generally|normally|ordinarily|in\s+general)"
    r"\s*,?\s*(?:our|the)\s+(?:client|company|employer|firm|organi[sz]ation)\b"
    r"\s+(?:is\s+|are\s+|will\s+)?"
    r")"
    r"[^.?!]{0,60}",
    re.I,
)

# Openers of a stranded fragment: words that cannot begin a complete turn,
# because they start something that needs a main clause to lean on. Two kinds,
# and they need DIFFERENT tests -- see _only_a_fragment.
#
# A subordinator opens a clause, and a clause carries its own verb: "now that
# we HAVE this information" has one. So a verb test cannot see that it is a
# fragment; what it lacks is the main clause, and the comma that would have
# separated the two is the tell.
_FRAGMENT_SUBORDINATOR = re.compile(
    r"^(?:when|whenever|if|unless|as|now|once|while|after|before|afterwards"
    r"|since|until|because|though|although|whether)\b",
    re.I,
)

# A lead-in opens a phrase with no clause at all, so there is no comma to look
# for -- the verb test is what separates "At another time." (run 39 turn 1)
# from "In that case, let us move on.". "then" and "so" live here rather than
# with the subordinators because both are ordinary sentence openers: "So I
# have five years." and "Then we can move on." carry verbs and survive, while
# a bare "Then." does not.
_FRAGMENT_LEAD_IN = re.compile(
    r"^(?:at|in|on|for|with|by|from|about|during|under|over|between|against"
    r"|without|within|regarding|concerning|to|of|into|through|across|then|so"
    r"|today)\b",
    re.I,
)

# How long a phrase that opens on one of the above may be and still count as a
# fragment. Longer than that and it is more likely a real sentence the opener
# test misjudged -- "As a recruiter I can tell you the role is in Bengaluru."
# opens on a subordinator and has no comma, and must survive.
_FRAGMENT_MAX_WORDS = 6

# Does the lead-in phrase carry a clause of its own? Contractions count:
# "that's" is a verb as much as "that is" is.
_FINITE_VERB = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|have|has|had|do|does|did"
    r"|can|could|will|would|shall|should|may|might|must|let"
    r"|get|gets|got|give|gives|gave|take|takes|took|make|makes|made"
    r"|go|goes|went|come|comes|came|see|sees|saw|say|says|said"
    r"|know|knows|knew|think|thinks|thought|mean|means|meant"
    r"|want|wants|wanted|need|needs|needed|like|likes|liked"
    r"|look|looks|looked|work|works|worked|help|helps|helped|keep|keeps"
    r"|sound|sounds|sounded|match|matches|matched|cover|covers|covered"
    r"|include|includes|included|prefer|prefers|preferred|expect|expects"
    r"|require|requires|discuss|discusses|discussed|confirm|confirms"
    r"|complete|completed|graduate|graduated|study|studied|manage|managed"
    r"|use|uses|used|build|built|lead|led|run|ran|own|owned|speak|speaks"
    r"|find|finds|found|check|checks|review|reviews|share|shares|tell|told"
    r"|hear|heard|agree|agrees|appreciate|appreciates)\b"
    r"|\w+(?:'s|'re|'ve|'ll|'d|'m)\b",
    re.I,
)

# Bare polarity. A reply made of nothing but these, carrying BOTH a yes and a
# no, is a yes/no answer the model emitted without deciding which. Run 39
# turn 5: the caller said "Oh yes, you are right." and the reaction came out
# "Yes No." -- what was left of "Yes No \nCan you name any frameworks..."
# once _strip_model_questions took the question. Deliberately requires the
# contradiction: "Okay, sure." and "Yes." are things people say.
_AFFIRMATIVE = frozenset({"yes", "yeah", "yep", "yup", "correct", "right",
                          "indeed", "affirmative", "certainly"})
_NEGATIVE = frozenset({"no", "nope", "nah", "negative", "never"})
_POLARITY = _AFFIRMATIVE | _NEGATIVE | {"ok", "okay", "sure", "exactly",
                                        "absolutely", "definitely"}


def _strip_orphan_tail(text: str) -> str:
    """Drop a clause tail left dangling at the end of a reply by a cut."""
    out = text
    while True:
        before = out
        out = _ORPHAN_TAIL.sub("", out).strip(" \t\r\n,;:-")
        if out == before or not out:
            return out


def _only_a_fragment(text: str) -> bool:
    """Is what is left of a reply a dependent phrase, not a complete turn?

    Three shapes, and every one of them is what a CUT leaves behind rather
    than anything the model wrote on its own:

    - "At another time." -- a prepositional lead-in whose clause the redirect
      cut removed (run 39 turn 1).
    - "now that we have this information." -- a subordinate clause with no
      main clause left to lean on (run 39 turn 6, same cut).
    - "Yes No." -- both polarities and no verb (run 39 turn 5, left by the
      question strip).

    This replaces the per-site fragment handling that runs 35, 37, 38 and 39
    each added another patch to (`_strip_orphan_tail`, the old four-word
    `_only_a_fragment`, the `emptied` guard, `_clean_seam`). A cut can land
    anywhere, so a fix scoped to the cut that produced one fragment never
    covers the next one -- the same lesson as rule 1b in _repair_chunk.

    Deliberately narrow, because dropping a reaction is only free while the
    line being dropped was not worth keeping. The opener test is what makes it
    safe: "Got it.", "Sure.", "That sounds good." and "Thanks for your time."
    do not open on a lead-in or a subordinator and are never touched.
    """
    stripped = text.strip()
    if not stripped or "?" in stripped:
        return False
    words = stripped.split()

    # Both polarities and nothing else: a yes/no answer the model emitted
    # without deciding which (run 39 turn 5).
    bare = [w.strip(".,!;:").lower() for w in words]
    if (
        len(bare) >= 2
        and all(w in _POLARITY for w in bare)
        and any(w in _AFFIRMATIVE for w in bare)
        and any(w in _NEGATIVE for w in bare)
    ):
        return True

    if len(words) > _FRAGMENT_MAX_WORDS:
        return False
    # A comma is where the main clause would have started. Strip a trailing one
    # first: a cut that stopped at the comma leaves "now that we have this
    # information," -- still a fragment, and the comma is not evidence of a
    # main clause.
    body = stripped.strip(" \t,;:.")
    if not body:
        return False

    if _FRAGMENT_SUBORDINATOR.match(body):
        return "," not in body and ";" not in body

    if _FRAGMENT_LEAD_IN.match(body):
        return "," not in body and not _FINITE_VERB.search(body)

    return False



def _repair_chunk(
    text: str,
    caller_corpus: str,
    asked: list[bool] | None = None,
) -> tuple[str, list[str]]:
    """Cut the mechanical failure shapes. Returns (text, what was cut).

    `asked` is a one-element per-turn flag: [True] once a question has gone
    out this turn. It lets rule 5 below see a second question that streamed in
    a later chunk than the first.
    """
    out = text
    cut: list[str] = []
    caller_words = set(re.findall(r"[a-z']+", caller_corpus.lower()))

    def _note(label: str) -> None:
        if label not in cut:
            cut.append(label)

    # Run 26's replies carried two or three defects at once in a single
    # ramble -- meta-commentary, then a definition, then a second question.
    # One pass removes only the first thing each pattern finds, which left the
    # rest in. Three passes is enough for anything observed and cannot loop
    # forever because every pass either changes the text or breaks.
    for _pass in range(3):
        before = out

        # 1. A forbidden topic redirected before the caller ever raised it.
        match = _TEAM_REDIRECT.search(out)
        head = _TEAM_REDIRECT_HEAD.search(out)
        if head:
            clause = out[head.start(): head.start() + 80].lower()
            raised = {m.group(0).lower() for m in _TOPIC_WORD.finditer(clause)}
            unsolicited = bool(raised) and not (raised & caller_words)
            if unsolicited:
                if match and match.start() == head.start():
                    # A recognised closer ("separately", "with you", "later").
                    # That is a genuine clause boundary even though the model
                    # often starts the next clause with a lowercase letter;
                    # extending would eat the real question that follows.
                    stop = match.end()
                else:
                    # No closer: stop at the next punctuation, the next
                    # question, or the next clause start, whichever comes
                    # first. Without the clause start the cut swallowed the
                    # rest of an unpunctuated reply -- run 31 turn 15's
                    # "...will discuss salary Thank you again" became
                    # "Separately." because nothing but the final period was
                    # ahead of it.
                    stop = _redirect_stop(out, head.end())
                # Whatever the stop landed on, the tail of the clause may
                # still be in front of us (a comma plus a second adverb).
                stop = _redirect_tail(out, stop)
                candidate = _strip_orphan_tail(
                    _clean_seam(out[: head.start()] + out[stop:])
                )
                # Same rule as below: if the redirect was the whole reply,
                # leave it. Silence costs more than one wrong sentence, and
                # the off-topic guard still gets its chance to reject it.
                if any(ch.isalnum() for ch in candidate):
                    out = candidate
                    _note("unsolicited redirect on a topic they never raised")

        # 1b. The same topic volunteered in any OTHER wording.
        #
        # Run 35 turn 1 spoke "No salary discussion today When needed." --
        # no recruitment team, no "will discuss", nothing the redirect rule
        # recognises, and it is the same defect. One rule per phrasing is
        # what made runs 27-33 feel endless, so this one keys on the TOPIC
        # rather than the sentence shape: any clause that names a forbidden
        # topic the caller never raised goes, whatever words carry it.
        found = _FORBIDDEN_WORD.search(out)
        if found:
            topic = found.group(0).lower()
            if topic not in caller_words:
                # From the start of the sentence the topic sits in, not from
                # the start of the reply -- a good sentence before it is worth
                # keeping.
                edge = -1
                for sep in (".", "!", "?", "\n", ";"):
                    at = out.rfind(sep, 0, found.start())
                    if at > edge:
                        edge = at
                start = edge + 1
                # The whole SENTENCE, not just up to the next clause. Cutting
                # at the next clause start amputates the front of a sentence
                # that only mentions the topic in passing -- "What rate are you
                # targeting for this one?" became "Are you targeting for this
                # one?", which is worse than either leaving it or dropping it.
                stop = len(out)
                for j in range(found.end(), len(out)):
                    if out[j] in ".!?\n":
                        stop = j + 1
                        break
                # A QUESTION naming the topic is not a volunteered topic --
                # it is at least addressed to the caller, and the question
                # machinery (which drops everything the model writes and asks
                # the templated one) deals with it. Without this the cut ate
                # "can you confirm your notice period?" that survived the
                # redirect in run 26.
                if "?" not in out[start:stop]:
                    candidate = _clean_seam(out[:start] + out[stop:])
                    # All that is left can be the tail of the clause that
                    # went ("No salary discussion today When needed." ->
                    # "When needed."). A stranded connective is not a
                    # sentence.
                    if _only_a_fragment(candidate):
                        candidate = ""
                    # Unlike rule 1 this MAY empty the reply. With the
                    # questions templated the caller still hears one --
                    # Python appends it -- so silence is no longer what a cut
                    # costs here.
                    out = candidate
                    _note(f"volunteered {topic} when they never raised it")

        # 2-5. Spans that are never right, whoever raised them. These run to
        # the end of the clause, not just the end of the phrase.
        for label, pattern in (
            ("openings claim", _OPENINGS_CLAIM),
            ("commitment she cannot make", _COMMITMENT),
            ("lecture about how something works", _LECTURE),
            ("claim about what the client wants", _CLIENT_REQUIREMENT),
            ("comment on the caller", _CALLER_COMMENT),
            ("apology for the machinery", _META_APOLOGY),
            ("narrated the machinery", _MACHINERY_NARRATION),
            ("narrated the call", _CALL_NARRATION),
            ("commentary about the call", _META_COMMENT),
            ("definition of a term", _DEFINITION),
        ):
            found = pattern.search(out)
            if found:
                start, stop = found.start(), _clause_end(out, found.end())
                if label in ("definition of a term", "claim about what the client wants"):
                    # The clause starts at the beginning of its sentence, not
                    # at the subject or the verb. "Data warehouse management
                    # systems refer to software tools that..." has to lose the
                    # subject too, or it lands as "Data warehouse ." -- and
                    # "Typically our clients look for degrees above
                    # undergraduate level" (run 39 turn 2) has to lose the
                    # "Typically", or the cut leaves a bare "Typically." as the
                    # whole reaction. Take the full stop with it, since these
                    # are often separated by a blank line.
                    edge = -1
                    for sep in (".", "!", "?", "\n"):
                        at = out.rfind(sep, 0, start)
                        if at > edge:
                            edge = at
                    start = edge + 1
                    if stop < len(out) and out[stop] == ".":
                        stop += 1
                candidate = out[:start] + out[stop:]
                # Never amputate the whole reply. Run 26 turn 7's entire line
                # was meta-commentary, and cutting it left a bare "." -- Piper
                # would have read that as silence anyway, and a flawed
                # sentence beats no sentence. Leave it and let the next pass,
                # or the caller's own ears, deal with it.
                #
                # Exception: a reply that is ONLY a promise. Run 36 turn 6 was
                # nothing but "I'll check on any relevant job openings at our
                # client companies soon", and the guard would have waved it
                # through on the reasoning above. Silence no longer costs
                # anything -- the templated line follows -- and a promise she
                # cannot keep is the one thing worse than a flawed sentence.
                if any(ch.isalnum() for ch in candidate) or label in _MAY_EMPTY:
                    out = candidate
                    _note(label)

        if out == before:
            break
        out = re.sub(r"\s{2,}", " ", out).strip(" ,;:-")

    # 6. One question per turn. Voice rule 1 says it and the model ignores it
    # -- run 24 turns 1, 2 and 9 each asked twice. Asking nicely never held,
    # so the second question is dropped the same way as the spans above.
    # Deliberately does not touch a single question followed by a statement
    # ("Are you free to talk? I'll be quick."), which is fine as it stands.
    q = out.find("?")
    if q != -1:
        if asked is not None and asked[0]:
            # A question already went out this turn, so this one is the second
            # even if it is the only "?" in this chunk. Drop the sentence the
            # question lives in, not just the question -- "Great. And what is
            # your tech stack?" has to lose the whole clause, or it lands as
            # "...and what is your tech stack" with no ending. Any complete
            # sentence before it ("Sure." in "Sure. What's your stack?") is
            # worth keeping, so the cut starts after the last terminator.
            edge = -1
            for sep in (".", "!", "\n", ";"):
                found_at = out.rfind(sep, 0, q)
                if found_at > edge:
                    edge = found_at
            out = out[: edge + 1].rstrip(" \t\r\n,;:-") if edge >= 0 else ""
            cut.append("second question")
        elif out.count("?") > 1:
            out = out[: q + 1]
            cut.append("second question")
    if "?" in out and asked is not None:
        asked[0] = True

    if not cut:
        return text, cut
    # Close the seams the cuts leave behind.
    out = re.sub(r"\s{2,}", " ", out).strip(" ,;:-")
    # A cut that ended on a sentence boundary can leave its punctuation
    # orphaned ("...shall I close this call now ."). Pull it back onto the
    # word it belongs to.
    out = re.sub(r"\s+([.?!,;])", r"\1", out)
    out = re.sub(r"\.\s*(?=\.)", "", out).strip()
    # Any cut can leave punctuation in front ("...salary separately. Thank
    # you" -> ". Thank you"); Piper reads that leading period as silence.
    out = _clean_seam(out)
    # Cutting a leading clause leaves the rest starting mid-sentence ("can you
    # confirm how many years..."). Re-capitalise so Piper does not read it as a
    # fragment continuing something else.
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out, cut


# Function words carry no topic. Comparing raw word sets let four askings of the
# same question slip through: "Did you work on production pipelines before?" and
# "Have you ever managed production pipelines in your career?" share only 43% of
# their words, because do/did/have/any/ever/before/your dominate the count. Strip
# them and both reduce to {work, production, pipeline} vs {manage, production,
# pipeline} -- obviously the same question.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "if", "then", "of", "to", "in",
    "on", "at", "for", "with", "by", "from", "about", "into", "over", "as",
    "i", "you", "your", "yours", "me", "my", "we", "us", "our", "it", "its",
    "he", "she", "they", "them", "their", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "any", "some", "ever", "before", "after", "much", "many", "more", "most",
    "there", "here", "just", "also", "very", "really", "please", "okay", "ok",
    "got", "get", "like", "know", "think", "want", "need", "tell", "say",
    "yes", "no", "not", "up", "out", "back", "now", "today",
}

# Crude suffix stripping. A real stemmer is not worth a dependency here; the job
# is only to make "work" / "worked" / "working" / "works" compare equal.
#
# Plurals are handled BEFORE the verb suffixes and separately from them. Folding
# "es" in with the rest turned "pipelines" into "pipelin" while "pipeline" stayed
# whole, so two questions about production pipelines did not match on the one
# word that mattered.
_VERB_SUFFIXES = ("ing", "edly", "ed", "ment", "ance", "ence")


def _stem(word: str) -> str:
    if len(word) > 3:
        if word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith(("ses", "xes", "zes", "ches", "shes")):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
    for suffix in _VERB_SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    return {
        _stem(w) for w in _normalize(text).split()
        if w not in _STOPWORDS and len(w) > 1
    }


# _invents_question used to compare the reply's question against the content
# words of the live agenda item itself. That stopped working when
# AGENDA_TEMPLATES were cut down to terse fragments -- deliberately, so the
# model could not read a checklist entry aloud. A fragment is a hopeless
# basis for comparison: "tech stack -- which ones?" has the content words
# {tech, stack, one}, so "Which programming languages do you work with?"
# shared ZERO words with it and was rejected.
#
# Measured against a real 7-turn call, 6 of 9 perfectly good questions scored
# under the 0.25 threshold -- including two the agent actually spoke. Each
# rejection discards a good answer and pays for a whole extra generation, and
# it is the regenerated answer that produced the fabrications ("Five plus
# years typically qualifies us...", "You currently work with Java and
# Python"). The guard was throwing away the right reply and keeping the
# invented one.
#
# So the guard carries its own topic vocabulary, keyed off the item. The terse
# fragment still goes to the model; this never does.
_AGENDA_TOPIC_WORDS = {
    "qualification": (
        "qualification degree study studied field major education college "
        "university bachelors masters academic subject stream course diploma "
        "graduation graduated school trained certification"
    ),
    "experience": (
        "years year experience experienced working worked work tenure total "
        "long since"
    ),
    "tech": (
        "tech stack technologies technology tools tool languages language "
        "frameworks framework programming coding code software platforms "
        "platform libraries library stack using use build building day daily "
        "currently hands-on developing develop"
    ),
    "confirm": (
        "confirm correct right noted note read recap verify accurate spelling "
        "double-check"
    ),
}


# Words that are decisive on their own. The shared-word test above dilutes:
# run 8 rejected "Four years sounds good Can you confirm how many years
# specifically?" because only "year" overlapped out of six content words
# (0.17 against a 0.25 threshold). One unmistakable topic word is stronger
# evidence than a ratio over a long question, so an anchor short-circuits it.
_AGENDA_ANCHOR_WORDS = {
    "qualification": (
        # "educational" is listed alongside "education" because _stem does not
        # strip "-al", so the two forms do not compare equal without it. Run 8
        # asked "details about your educational background" and was rejected
        # as off-agenda purely on that missing suffix.
        "qualification degree study studied education educational college "
        "university field major graduation graduated academic diploma course "
        "subject"
    ),
    "experience": "years year experience experienced tenure total",
    "tech": (
        "languages language frameworks framework tools tool technologies "
        "technology stack programming coding code software libraries library "
        "platforms platform"
    ),
    "confirm": "correct right confirm confirmed noted recap verify accurate",
}
# Stemmed once at import; both sides of every comparison go through
# _content_words, so the anchors must be stemmed the same way.
_AGENDA_ANCHOR_SETS = {
    k: _content_words(v) for k, v in _AGENDA_ANCHOR_WORDS.items()
}

# What a closing turn is still allowed to ask about.
_CLOSING_ALLOWED_QUESTION = re.compile(
    r"\bfollow\s*-?\s*up\b|\b(?:our|recruitment|hiring)\s+team\b|\bopenings\b"
    r"|\bcall\s+(?:you|back)\b|\breach\s+out\b|\bin\s+touch\b|\bget\s+back\b"
    r"|\bsuitable\b",
    re.I,
)


# Which topic a live agenda item belongs to. Lives in prompts.py next to the
# checklist it describes -- it is the same key used for agent.questions, and
# prompts cannot import this module.
_agenda_topic = agenda_topic


def _invents_question(reply: str, live_item: str | None, caller_said: str = "") -> str:
    """Catch a plausible-sounding question that is NOT the live checklist item.

    Observed live: AgendaTracker named "years hands-on with Python, Spark,
    AWS?" as the live item, and the model asked "Any experience with
    healthcare analytics projects?" instead -- fluent, on-domain, and entirely
    invented. _echoes_agenda cannot catch this because it only flags a REAL
    agenda item spoken in the third person; this item was never in the
    checklist at all.

    Only checked when the reply actually ends in a question (there is
    something to compare), and only against the ONE topic the model was told
    to ask -- not the whole checklist, since a good rewrite of the live item
    can look quite different in raw words but should still share real topic
    vocabulary with it.
    """
    if not live_item:
        return ""
    sentences = re.split(r"(?<=[.!?])\s", reply.strip())
    question_sentences = [s for s in sentences if s.strip().endswith("?")]
    if not question_sentences:
        return ""
    asked = question_sentences[-1]
    asked_words = _content_words(asked)
    if len(asked_words) < 2:
        return ""

    topic = _agenda_topic(live_item)
    if topic is None:
        return ""

    if topic == "closing":
        # A closing turn should carry no question at all beyond arranging the
        # follow-up. Asking anything else here is how the agent ends up
        # re-opening a closed call: run 6 asked for the qualification again
        # after wrap-up, run 7 asked for education, run 8 asked the candidate
        # to opine on whether AI/ML "can be applied generally across different
        # types of projects" -- three runs, three backwards questions, none of
        # them caught by a topic blacklist because each shares no vocabulary
        # with the checklist. So: allow only follow-up questions.
        if _CLOSING_ALLOWED_QUESTION.search(asked):
            return ""
        return asked.strip()

    # One decisive topic word settles it. See _AGENDA_ANCHOR_WORDS -- run 8
    # rejected a correct "how many years specifically?" for scoring 0.17.
    if asked_words & _AGENDA_ANCHOR_SETS.get(topic, frozenset()):
        return ""

    wanted = _content_words(_AGENDA_TOPIC_WORDS[topic])
    # The confirmation turn reads back what the CALLER said, so their own
    # words are legitimately on-topic. "So you have five years of experience
    # and work with Python -- is that right?" shares nothing with the terse
    # item "read back and confirm?" on its own.
    if topic == "confirm" and caller_said:
        wanted |= _content_words(caller_said)

    shared = asked_words & wanted
    if len(shared) >= 2:
        return ""
    if shared and len(shared) / len(asked_words) >= 0.25:
        return ""
    return asked.strip()


def _content_bigrams(text: str) -> set[tuple[str, str]]:
    """Adjacent stemmed content-word pairs ("production pipeline").

    A reworded repeat almost always keeps its domain phrase and swaps the
    verbs around it: "Did you work on production pipelines before?" and "Have
    you ever managed production pipelines in your career?" share only 2 of 4
    content words (0.5), well under any threshold that does not also fire on
    two DIFFERENT questions that merely share domain vocabulary ("experience
    with Python" vs "experience with Spark", also 0.5 on 2 of 4).

    The shared bigram separates the two cases: the pipeline pair keeps
    (production, pipeline) intact, the Python/Spark pair shares no pair at all.
    """
    words = [
        _stem(w) for w in _normalize(text).split()
        if w not in _STOPWORDS and len(w) > 1
    ]
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def _too_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """Topic-overlap similarity. Catches verbatim repeats AND rewordings.

    Was 0.6. On a recruiting screen every reply legitimately reuses the same
    domain words ("experience", "role", "years", "AI engineer") turn after
    turn -- that is not a repeat, it is the subject matter. At 0.6 that
    coincidental overlap was tripping the guard on genuinely new replies,
    each rejection costing roughly a full extra LLM generation. Raised to
    0.75 so it still catches an actual reworded repeat but stops firing on
    two different sentences that just happen to share domain vocabulary.
    """
    ca, cb = _content_words(a), _content_words(b)
    if len(ca) < 2 or len(cb) < 2:
        return False
    if len(ca & cb) / max(1, min(len(ca), len(cb))) >= threshold:
        return True
    # Same domain phrase, different verbs -- a reworded repeat. See
    # _content_bigrams for why raw overlap cannot carry this on its own.
    #
    # A shared bigram alone is NOT enough: two long replies can share one
    # incidental pair while being about different things, and that over-fired
    # badly in live testing. Require some raw overlap underneath it as well.
    # The pipeline pairs score 0.5; unrelated long replies sharing one phrase
    # score far lower.
    if len(ca & cb) / max(1, min(len(ca), len(cb))) < 0.4:
        return False
    return bool(_content_bigrams(a) & _content_bigrams(b))


# Numbers/specifics the model has no business stating as fact unless they
# came from the call brief or the caller themselves. Rule 8 says "never
# invent a rate -- say you will check", but nothing enforced that: turns
# fabricating a salary ("$120k annually", "$167 K", "$150 K per annum"), a
# notice period ("six months' notice"), or a named competing candidate all
# reached the caller unchallenged in live testing.
# Spelled-out numbers. Run 6's worst fabrication was "Five plus years
# typically qualifies us for certain roles" -- and it sailed straight through,
# because every branch below used to match DIGITS only. The caller had never
# mentioned five years; they said it two turns LATER. The model simply pulled
# the figure out of the air, and stating it as a qualification threshold is
# worse than stating a wrong rate, because the candidate hears it as a real
# requirement.
#
# Numbers the caller themselves said are exempted at the call site, so the
# read-back turn can still repeat "five years of experience" back to them.
_NUMWORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|several|a\s+few)"
)
_SPECIFIC_CLAIM = re.compile(
    r"\$\s?\d[\d,]*\s?(k|/hr|/yr|per\s+hour|per\s+year|annually|per\s+annum)?"  # money
    r"|\b\d+\s*(%|percent)\b"                                        # percentages
    r"|\b\d+[\s-]*(month|year|week|day)s?\s*(of\s+)?(notice|experience)\b"
    # Same quantity claim, spelled out instead of written as a digit.
    r"|\b" + _NUMWORD + r"\s+(?:plus\s+)?[\w\s-]{0,12}?"
    r"(?:month|year|week|day)s?\s*(?:of\s+)?(?:notice|experience)?\b",
    re.I,
)

# Visa/work-authorization eligibility stated as a determination rather than
# asked about. Worse than a wrong salary figure -- misinforming a real
# candidate about their own legal eligibility has real consequences for
# them. Observed live: "You're eligible under H4 visa status since family
# members are U.S.-citizens" -- stated as settled fact, invented entirely,
# to a question the model was never given any real answer to.
_VISA_TERMS = r"(h-?1b|h-?4|l-?1|tn\s+visa|opt|cpt|green\s+card|work\s+authorization)"
_ELIGIBILITY_ASSERTION = re.compile(
    r"\b(you'?re|you\s+are)\s+eligible\b"
    r"|\bthat\s+qualifies\s+you\b"
    r"|\byou\s+qualify\b"
    r"|\byou'?re\s+authorized\b"
    r"|\byour\s+status\s+(allows|qualifies)\b",
    re.I,
)

# Hedges that mean the model is correctly deferring rather than asserting.
_HEDGES = (
    "i'll check", "i will check", "let me check", "let me confirm",
    "i'll confirm", "i don't have that", "i dont have that",
    "not sure yet", "i'd have to check", "i will find out", "i'll find out",
)


# Placeholder company names the model reaches for when it has nothing real to
# say. Run 7: "You currently work at XYZ company can you tell me some
# technologies or tools like SQL Server Reporting Services?" -- an employer
# the call brief has never mentioned, named as a placeholder, stated as fact.
_PLACEHOLDER_NAME = re.compile(
    r"\b(?:XYZ|ABC|Foo(?:bar)?|Company\s+Name|Your\s+Company|"
    r"Employer\s+Name)\b", re.I,
)

# Asserting where the caller works. Two guards against false positives:
# "currently"/"now" keeps it off "Do you work at Acme?", and requiring a
# CAPITALIZED name after the preposition keeps it off the equally legitimate
# "which company you currently work for?" -- which has no name to assert.
# No re.I here: the trailing [A-Z] must stay case-SENSITIVE, and re.I would
# silently make it match any letter. It did -- run 9 rejected the reply
# "you currently work with data..." because "d" satisfied [A-Z] under re.I.
# The subject and verb are spelled out in both cases instead.
_EMPLOYER_CLAIM = re.compile(
    r"\b[Yy]ou\s+(?:are\s+)?(?:[Cc]urrently|[Nn]ow)\s+(?:are\s+)?"
    r"(?:work|working|employed)\s+(?:at|for|with)\s+[A-Z]"
)

# "Do you currently work with Python?" hits _EMPLOYER_CLAIM: a capitalised
# word follows the preposition, and a tool name looks exactly like a company
# name to a regex. It is a question, though -- and asking is the opposite of
# asserting. So a question-shaped opener in the few words before the match
# exempts it. The fabricated line this guard exists for is always declarative:
# "You currently work at Infosys", "You currently work at XYZ company".
_EMPLOYER_QUESTION = re.compile(
    r"(?:do|does|did|are|is|was|were|can|could|would|will|have|has|"
    # Only bare words may sit between the auxiliary and "you": a "?" in there
    # means the question already closed and this is a fresh declarative
    # sentence ("Can you confirm? You currently work at Infosys.").
    r"which|what|where|whether|tell\s+me|know\s+if)\s+(?:[A-Za-z']+\s+){0,2}$",
    re.I,
)


# Volunteering an interpretation of a misheard word. Run 8: the transcript
# garbled "...currently Deliver for me" and Riya answered "Deliver sounds like
# an interesting project management tool." She invented a product out of a
# transcription error and described it back to the candidate -- and because the
# reply ended in "?", the numeric-fact check skipped it as a question.
#
# Narrowed to "sounds like a/an/the" so it cannot fire on the legitimate
# paraphrase "sounds like you have a lot of experience".
#
# Still not narrow enough. Run 22 flagged "That sounds like a solid foundation
# next can you give me an idea of how many years..." -- an entirely correct
# acknowledgement and follow-up, reported as fabrication. What separates the
# real failure is the SUBJECT: run 8's bad line was "Deliver sounds like an
# interesting project management tool", where "Deliver" is the garbled word
# being inflated into a product. When the subject is a generic pronoun -- or
# absent -- the sentence is vague praise ("sounds like a good fit"), not an
# invented fact. So the subject is captured and checked.
_GUESSING = re.compile(r"\b(\S+)?\s*sounds?\s+like\s+(?:a|an|the)\b", re.I)

# Subjects that mean the sentence is an acknowledgement, not a fabrication.
_GENERIC_SOUND_SUBJECTS = {"that", "it", "this", "there", "which", "those", "these"}


# Topics the screening brief tells Riya to DEFLECT rather than discuss.
#
# They are named in the brief, and there is the trap: a small model treats a
# noun in its instructions as a topic it is supposed to raise. So Riya started
# raising salary unprompted -- "Our recruitment team will discuss salary
# details later can you tell me how many years..." opened run 10, was rejected
# as a repeat at turn 5, rejected as a repeat AGAIN at turn 7, and the same
# shape opened run 9. Nobody had asked about salary in any of them.
#
# The word is only legitimate when the caller raised it first, which is what
# the caller_said argument is for.
_DEFLECTED_TOPICS = re.compile(
    r"\b(salary|salaries|compensation|remuneration|payroll|ctc|"
    r"notice\s+period|interview|shortlist|offer\s+letter)\b"
    r"|\bpay\s+(?:rate|scale|band)\b|\brate\s+card\b", re.I,
)


def _raises_deflected_topic(reply: str, caller_said: str) -> str:
    """Catch Riya bringing up a deflected topic the caller never asked about.

    The brief says to redirect salary/rate/interview questions "to our
    recruitment team". That is a rule for when the CALLER asks -- not an
    invitation to open with it, which is what the model started doing.
    """
    hit = _DEFLECTED_TOPICS.search(reply)
    if not hit:
        return ""
    # They asked -- so the deflect-the-question wording is correct.
    if caller_said and _DEFLECTED_TOPICS.search(caller_said):
        return ""
    return hit.group(0)


def _invents_employer(reply: str) -> str:
    """Catch a placeholder employer, or an employer asserted out of nowhere.

    Unlike _invents_specifics this ignores whether the sentence ends in a
    question. Run 7's line ran a fabricated employer straight into a question
    with no terminator between them, so the whole thing was classified as a
    question and skipped by the numeric-fact check.
    """
    hit = _PLACEHOLDER_NAME.search(reply)
    if hit:
        return hit.group(0)
    hit = _EMPLOYER_CLAIM.search(reply)
    if hit:
        # Asking where someone works is not the same as stating it.
        if _EMPLOYER_QUESTION.search(reply[: hit.start()]):
            return ""
        return hit.group(0)
    return ""


def _caller_said_it(claim: str, caller_said: str) -> bool:
    """True when the figure in `claim` was introduced by the caller.

    Exists so the read-back turn can do its job. Confirmation means repeating
    the caller's own numbers -- "So that's five years and Python, is that
    right?" -- and without this exemption every confirmation would be
    rejected as a fabrication, which costs a full regeneration on the one turn
    where the agent most needs to be accurate.
    """
    if not caller_said:
        return False
    num = re.search(r"\d+|" + _NUMWORD, claim, re.I)
    if not num:
        return False
    return num.group(0).lower() in _normalize(caller_said)


def _invents_specifics(reply: str, caller_said: str = "") -> str:
    """Catch a declarative numeric/factual claim not hedged as unconfirmed.

    `caller_said` is everything the caller has said so far, not just this turn.
    A number they introduced is not an invention when Riya repeats it back, so
    without this the read-back turn ("So that's five years and Python -- is
    that right?") would be rejected for quoting the caller's own figure.

    Only fires on the FIRST sentence carrying a number, and only when none
    of the hedge phrases are present anywhere in that sentence -- "I don't
    have the exact rate, but I'll check and get back to you" must pass.
    Observed live: "$120k annually", "$167 K", "$150 K per annum", and a
    fabricated H4-visa eligibility determination all stated as fact -- none
    of the other guards were designed to catch a false STATEMENT rather
    than a malformed or off-persona one.

    Questions are skipped entirely: "Is $150K acceptable?" or "Do you hold
    an H1B?" are the model correctly ASKING, not asserting -- exactly what
    it should be doing with rate/visa questions on this agenda.
    """
    for sentence in re.split(r"(?<=[.!?])\s", reply.strip()):
        if sentence.strip().endswith("?"):
            continue
        low = _normalize(sentence)
        if any(h in low for h in _HEDGES):
            continue
        if re.search(_VISA_TERMS, sentence, re.I) and _ELIGIBILITY_ASSERTION.search(sentence):
            return sentence.strip()
        match = _SPECIFIC_CLAIM.search(sentence)
        if match and not _caller_said_it(match.group(0), caller_said):
            return match.group(0).strip()
        # Claims about what the client or the company has DECIDED. The brief
        # names the client; it says nothing about decisions they have made, so
        # the model has no basis for these. Observed live: "The client has
        # confirmed interest in moving forward with next steps" and "Our
        # company does bill on pay margin initially" -- invented, stated as
        # settled fact, and the kind of thing that misleads a real candidate
        # worse than a wrong number does.
        if re.search(
            r"\b(?:(?:the|our|my)\s+(?:client|company|hiring\s+manager|team|agency|firm)"
            r"|we)\b"
            r"[^.!?]{0,40}?\b(?:ha(?:s|ve)\s+(?:confirmed|approved|agreed|accepted|"
            r"shortlisted|selected|decided)|is\s+(?:interested|keen|eager|impressed)"
            r"|(?:will|does|do)\s+(?:bill|invoice))\b",
            sentence, re.I,
        ):
            return sentence.strip()
    return ""


class VoiceAgent:
    def __init__(self, cfg: Section, on_event: Optional[Callable[[str, str], None]] = None) -> None:
        self.cfg = cfg
        self.models_dir = Path(cfg.paths.models_dir)
        self.on_event = on_event or self._default_event
        self.metrics = MetricsLog()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frames: Optional[asyncio.Queue] = None
        self._running = False
        self._turn_id = 0
        self._active_turn: Optional[asyncio.Task] = None
        self._active_metrics: Optional[TurnMetrics] = None
        # Replaced with a fresh Event each turn; workers capture their own
        # reference so a new turn can never accidentally un-cancel an old one.
        self._stop_generation = threading.Event()
        self._pending_interruption = False
        self._speech_start_t = 0.0
        # Last thing actually said out loud, including the greeting (which is
        # spoken without going through the LLM).
        self._last_spoken = ""
        # The questions Riya asks, read from config and NOT written by the
        # model -- see the long comment on agent.questions in config.yaml.
        self._questions: dict = dict(cfg.agent.get("questions") or {})
        # The opener, spoken once at startup. Kept so a mid-call reply that
        # re-opens with it can be cut instead of regenerated.
        self._greeting = str(cfg.agent.get("greeting") or "").strip()
        # What the caller told us, one entry per answered checklist item. The
        # read-back turn is built from these instead of the model improvising
        # a summary it then gets wrong or truncated.
        self._answers: list[str] = []
        # End-of-call. Riya says goodbye and then keeps talking unless we stop
        # her -- see _is_farewell. One grace turn so a caller who starts
        # talking right after the first goodbye still gets answered.
        self._farewells = 0
        self._max_farewells = max(1, int(cfg.agent.get("max_farewells", 2)))
        # After the last goodbye, wait this long before actually dropping the
        # call. Riya saying goodbye is not the same as the caller being
        # finished -- see the SPEECH_START branch in run(), which cancels it.
        self._hangup_grace_s = float(
            cfg.agent.get("hangup_grace_ms", 2500)
        ) / 1000.0
        self._hangup_at = 0.0
        # Set when a bail-out was replaced with a repair line this turn, so the
        # agenda does not credit Riya with having asked the live item.
        self._blocked_bail = False
        # Consecutive turns spent on "read back and confirm?" without an actual
        # confirmation. Two strikes and we take whatever she gave, rather than
        # letting the pointer stick and the call never end -- see
        # _READBACK_CONFIRM.
        self._confirm_misses = 0
        # A caller sentence that trailed off on a connector, waiting to be
        # completed by the next segment (or flushed if they never continue).
        self._carry_text = ""
        self._carry_rounds = 0
        self._max_carry_rounds = 2
        self._carry_task: Optional[asyncio.Task] = None
        self._carry_grace_s = float(cfg.vad.get("continuation_grace_ms", 900)) / 1000.0
        # _absorb_unspoken_turn (below) merges new speech into an in-flight turn
        # on the assumption that the caller is still mid-sentence. That's a fair
        # bet for a second or two, but a slow turn (e.g. one that got guard-
        # rejected and is regenerating) can sit "thinking" for many seconds --
        # long enough for the caller to start a genuinely NEW, unrelated
        # thought, which then gets silently glued onto the old fragment instead
        # of being treated as its own turn. Past this cap, stop absorbing.
        self._max_absorb_wait_s = float(cfg.vad.get("max_absorb_wait_ms", 2500)) / 1000.0
        # Whether a persona/system-prompt is active. When it's not (generic,
        # unprompted mode), the recruiter agenda checklist is meaningless --
        # injecting it as a per-turn note and then judging replies against it
        # just produces false guard rejections, each of which roughly doubles
        # that turn's LLM latency (see _consume_chunks / _echoes_agenda etc).
        # Whether a guard rejection triggers a second generation. Off by
        # default -- see the long note at `checking_open` in _consume_chunks.
        self._regenerate = bool(
            cfg.agent.get("regenerate_on_guard_rejection", False)
        )
        # Failure kinds that always get a second generation, even when
        # `regenerate_on_guard_rejection` is false. See `regenerate_kinds` in
        # config.yaml for the full reasoning; short version: these three are
        # mechanical (the model said something it already said, echoed the
        # caller, or read a checklist note), the correction note names the
        # exact offending text, and a re-roll reliably produces something
        # different. The judgment failures (fabrication, person, off_agenda)
        # are excluded on purpose -- every live run showed those retries
        # inventing something worse.
        # `or [...]` here was a trap: an empty list is falsy, so setting
        # `regenerate_kinds: []` in config.yaml silently fell back to the
        # DEFAULT and kept retrying. Emptying the list is the whole point of
        # the run-42 change, and it had no effect until this became an
        # explicit None check.
        _rk = cfg.agent.get("regenerate_kinds")
        if _rk is None:
            _rk = ["repeat", "parrot", "agenda"]
        self._regen_kinds = {str(k).strip().lower() for k in _rk}
        self._has_persona = str(cfg.llm.get("system_prompt", "compact")).lower() not in (
            "none", "generic", "off", "no", "false"
        )
        # The checklist, so the guard can tell when the model reads one out
        # instead of turning it into a question. Built regardless of persona
        # (greeting_covers / agenda.advance() still run safely) -- it's simply
        # not USED for prompting or guarding below when _has_persona is False.
        self._agenda = build_agenda(dict(cfg.get("call", {}) or {}))
        # Which checklist item is live. Kept in Python, not in the model's head.
        self.agenda = AgendaTracker(self._agenda)
        # 200 frames of 32 ms = ~6.4 s of slack before the oldest audio is dropped.
        self._frame_queue_size = 200
        self._dropped_frames = 0
        self._warned_about_drops = False

        self._stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

        # Optional per-turn live-data block (brief section 3).
        self.context_provider: Optional[Callable[[str], Optional[str]]] = None

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        """Load all four models. Blocking; warmup also primes the LLM prefix cache."""
        rt = self.cfg.runtime

        self.on_event("system", "Loading Silero VAD...")
        vad = SileroVAD(
            model_path(self.cfg, self.cfg.vad["model_file"]),
            sample_rate=int(self.cfg.audio["input_sample_rate"]),
            num_threads=int(rt["vad_threads"]),
        )
        self.turn_detector = TurnDetector(
            vad,
            self.cfg.vad,
            self.cfg.barge_in,
            frame_samples=int(self.cfg.audio["frame_samples"]),
            sample_rate=int(self.cfg.audio["input_sample_rate"]),
        )

        engine = str(self.cfg.stt.get("engine", "whisper")).lower()
        self.on_event("system", f"Loading STT ({engine})...")
        self.stt = build_stt(
            self.cfg.stt,
            self.models_dir,
            num_threads=int(rt["stt_threads"]),
            on_event=self.on_event,
        )
        self.transcriber = StreamingTranscriber(
            self.stt, self.cfg.stt, sample_rate=int(self.cfg.audio["input_sample_rate"])
        )

        backend_name = str(self.cfg.llm.get("backend", "server")).lower()
        self.on_event(
            "system",
            f"Starting LLM backend '{backend_name}'...",
        )
        self.llm = build_llm(
            self.cfg.llm,
            model_path(self.cfg, self.cfg.llm["model_file"]),
            call=dict(self.cfg.get("call", {}) or {}),
        )
        # Unambiguous proof of what actually got built -- the banner above is
        # a config read, not a guarantee. Remove this once you've confirmed
        # the Ollama swap is really in effect.
        print(
            f"[llm backend] class={type(self.llm).__name__} "
            f"model={getattr(self.llm, 'model_name', getattr(self.llm, 'model_path', '?'))} "
            f"base_url={getattr(self.llm, 'base_url', '?')}"
        )

        self.on_event("system", f"Loading TTS ({self.cfg.tts.get('engine', 'piper')})...")
        self.tts = build_tts(
            self.cfg.tts,
            self.models_dir,
            num_threads=int(rt["tts_threads"]),
            on_event=self.on_event,
        )

        self.on_event("system", "Warming up LLM...")
        llm_ms = self.llm.warmup()

        if isinstance(self.tts, KokoroTTS):
            self.tts = confirm_kokoro_streaming(
                self.tts,
                self.cfg.tts,
                self.models_dir,
                num_threads=int(rt["tts_threads"]),
                on_event=self.on_event,
            )

        self.mic = MicStream(
            on_frame=self._on_mic_frame,
            sample_rate=int(self.cfg.audio["input_sample_rate"]),
            frame_samples=int(self.cfg.audio["frame_samples"]),
            device=self.cfg.audio.get("input_device"),
        gain=float(self.cfg.audio.get("mic_gain", 2.0)),)
        self.speaker = SpeakerStream(
            sample_rate=self.tts.sample_rate,
            blocksize=int(self.cfg.audio["output_blocksize"]),
            device=self.cfg.audio.get("output_device"),
        )

        self.on_event("system", "Warming up TTS...")
        tts_ms = self.tts.warmup()
        self.on_event("system", f"Warmup done (llm {llm_ms:.0f} ms, tts {tts_ms:.0f} ms). Ready.")

    # ------------------------------------------------------------------- mic
    def _on_mic_frame(self, frame: np.ndarray) -> None:
        """PortAudio thread -> event loop. Must be fast and must not block."""
        loop = self._loop
        if loop is None or not self._running:
            return
        try:
            loop.call_soon_threadsafe(self._enqueue_frame, frame)
        except RuntimeError:
            pass  # loop closing

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        """Runs on the event loop. Never raises -- a live mic cannot block.

        If the queue is full the event loop has fallen behind (heavy synthesis on
        a small CPU, say). Dropping the OLDEST frame is the right trade for a
        real-time stream: stale audio is worthless, and raising QueueFull out of
        a call_soon_threadsafe callback just spams the console while the frame is
        lost anyway.
        """
        queue = self._frames
        if queue is None:
            return
        while queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._dropped_frames += 1
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:  # pragma: no cover - defensive
            self._dropped_frames += 1

    # ------------------------------------------------------------------- run
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._frames = asyncio.Queue(maxsize=self._frame_queue_size)
        self._running = True
        self._dropped_frames = 0

        self.speaker.start()
        self.mic.start()
        # Say which device won. A stale config index silently lands on the
        # fallback, and without this line you cannot tell that the mic you
        # tuned your gain for is not the one being used.
        self.on_event("system", f"Mic: {self.mic.device_label}")

        # Start listening BEFORE speaking the greeting, not after. Two reasons:
        # the VAD loop has to be draining the mic queue while the TTS synthesizes
        # (otherwise frames pile up and overflow), and the caller should be able
        # to talk over the greeting like any other agent speech.
        vad_task = asyncio.create_task(self._vad_loop())

        greeting = self.cfg.agent.get("greeting")
        if greeting:
            try:
                await self._speak_standalone(greeting)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.on_event("error", f"greeting failed: {exc}")
            else:
                # The model has to know it already introduced itself. Without
                # this the greeting is invisible to it, so turn one it says
                # "I'm Riya, an IT staffing recruiter..." all over again -- and
                # the backchannel gate cannot see that a question was just
                # asked, because history is empty.
                self.llm.add_assistant(greeting)
                # The greeting already asks the first checklist item ("have you
                # got two minutes?" IS "bad time?"). Leaving the pointer at zero
                # meant turn one was spent asking a question the caller had just
                # answered, and the model papered over it by inventing one
                # instead ("How long have you been considering roles in
                # healthcare analytics?").
                for _ in range(int(self.cfg.agent.get("greeting_covers", 1))):
                    self.agenda.advance()

        try:
            await vad_task
        finally:
            vad_task.cancel()
            await self.shutdown()

    async def shutdown(self) -> None:
        self._running = False
        await self._cancel_turn(wait=True)
        try:
            self.mic.stop()
        except Exception:
            pass
        try:
            self.speaker.stop()
        except Exception:
            pass
        # Stops the llama-server subprocess if we were the ones who started it.
        llm = getattr(self, "llm", None)
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        self._stt_pool.shutdown(wait=False)
        self._tts_pool.shutdown(wait=False)

    # -------------------------------------------------------------- vad loop
    async def _vad_loop(self) -> None:
        """Supervisor: segments turns and watches for barge-in every 32 ms."""
        show_partials = bool(self.cfg.agent.get("show_partials", True))
        partial_task: Optional[asyncio.Task] = None

        while self._running:
            frame = await self._frames.get()

            # Dropped frames corrupt VAD timing and clip transcripts, so say so
            # once rather than failing quietly.
            if self._dropped_frames and not self._warned_about_drops:
                self._warned_about_drops = True
                self.on_event(
                    "error",
                    f"dropped {self._dropped_frames} mic frame(s) -- the event loop is "
                    "falling behind. Try a smaller stt.model_name or llm.model_file.",
                )

            # Keep the detector's sensitivity profile in sync with playback.
            self.turn_detector.agent_speaking = self.speaker.is_speaking or self._turn_active

            # The goodbye has been said and the caller has not objected. Only
            # fire once her audio has actually drained -- otherwise the grace
            # window expires while "goodbye" is still playing and we cut off
            # her own sentence.
            if (
                self._hangup_at
                and time.monotonic() >= self._hangup_at
                and not self.speaker.is_speaking
                and self.speaker.pending_seconds <= 0
            ):
                self.on_event("system", "(call complete; hanging up)")
                self._running = False
                break

            result = self.turn_detector.process(frame)

            if result.event is VADEvent.SPEECH_START:
                # They started talking after Riya's goodbye -- so they are not
                # finished. Drop the pending hang-up and reset the counter, or
                # the next goodbye would immediately end the call again.
                if self._hangup_at:
                    self._hangup_at = 0.0
                    self._farewells = 0
                    self.on_event("system", "(caller spoke; staying on the call)")
                self.transcriber.reset()
                # They resumed -- the held fragment will be completed by this
                # segment, so stop the grace timer from firing underneath us.
                self._cancel_carry_flush()
                heard_something = (
                    self.speaker.is_speaking or self.speaker.pending_seconds > 0
                )
                if heard_something:
                    await self._barge_in()
                elif self._turn_active:
                    # The caller carried on while we were still thinking. Nothing
                    # has been played, so there is nothing to interrupt: this is
                    # one utterance split by a pause, not an interruption. The
                    # old code logged "BARGE | dropped 0.00s" and binned the
                    # turn, leaving its transcript in history with no reply --
                    # nine turns vanished that way in one call. Fold the text
                    # into the next turn instead.
                    #
                    # BUT only while the turn is still recent. A turn stuck
                    # regenerating after a guard rejection can take many
                    # seconds -- long enough that what follows is a genuinely
                    # new, unrelated thought, not a continuation. Past the cap,
                    # let the slow turn finish and treat this as its own turn
                    # rather than silently gluing unrelated fragments together.
                    thinking_s = (
                        time.perf_counter() - self._active_metrics.speech_end_t
                        if self._active_metrics is not None else 0.0
                    )
                    if thinking_s <= self._max_absorb_wait_s:
                        await self._absorb_unspoken_turn()
                    else:
                        # Don't just leave the stale turn running while a new
                        # one starts on SPEECH_END below -- that races two
                        # turns against the same shared state (self._active_turn,
                        # LLM history, metrics), which is what produced two
                        # separate "(rejected ...; regenerating)" lines for what
                        # looked like one turn, and one of the two replies never
                        # made it to the speaker. Cancel it properly, same as
                        # _absorb_unspoken_turn does, but drop its transcript
                        # instead of merging it -- past the cap it's probably
                        # not the same thought anymore.
                        m = self._active_metrics
                        said = m.transcript.strip() if m is not None else ""
                        self._stop_generation.set()
                        self.llm.abort()
                        await self._cancel_turn(wait=False)
                        if said:
                            self.llm.drop_last_user(said)
                        self.on_event(
                            "vad",
                            f"previous turn still thinking ({thinking_s:.1f}s) -- "
                            "dropped it, treating this as a new turn",
                        )
                self.on_event("vad", "speech start")
                self._speech_start_t = time.perf_counter()

            elif result.event in (VADEvent.SPEECH_END, VADEvent.SPEECH_TIMEOUT):
                if partial_task and not partial_task.done():
                    # Let an in-flight partial finish; it may make the final pass free.
                    try:
                        await asyncio.wait_for(partial_task, timeout=0.35)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                self.on_event("vad", f"speech end ({result.duration_ms:.0f} ms)")
                self._start_turn(result.audio, result.duration_ms)

            elif self.turn_detector.is_speaking and self.transcriber.supports_streaming:
                # Vosk decodes as we go, so finalize() is nearly free (~90 ms vs
                # ~3 s batch). Feeding costs ~6 ms per 32 ms frame -- 20% of the
                # frame budget, too much to run inline on the event loop. The
                # pool has exactly one worker, so frames stay in order and the
                # later finalize() naturally queues behind them.
                self._loop.run_in_executor(self._stt_pool, self.transcriber.feed_frame, frame)

            elif self.turn_detector.is_speaking and self.transcriber.due_for_partial():
                snapshot = self.turn_detector.current_audio()
                if partial_task is None or partial_task.done():
                    partial_task = asyncio.create_task(self._run_partial(snapshot, show_partials))

    async def _run_partial(self, audio: np.ndarray, show: bool) -> None:
        text = await self._loop.run_in_executor(self._stt_pool, self.transcriber.run_partial, audio)
        if show and text:
            self.on_event("partial", text)

    # ------------------------------------------------------------ turn control
    @property
    def _turn_active(self) -> bool:
        return self._active_turn is not None and not self._active_turn.done()

    def _arm_carry_flush(self) -> None:
        """If the caller never resumes, answer the fragment anyway.

        Without this, someone who says "I'm a data engineer, so..." and then goes
        quiet waits forever, because the held text only gets released by the next
        speech segment.
        """
        self._cancel_carry_flush()
        self._carry_task = asyncio.create_task(self._flush_carry_later())

    def _cancel_carry_flush(self) -> None:
        task, self._carry_task = self._carry_task, None
        if task is not None and not task.done():
            task.cancel()

    async def _flush_carry_later(self) -> None:
        try:
            await asyncio.sleep(self._carry_grace_s)
        except asyncio.CancelledError:
            return
        # Nothing awaits this task, so an exception here would vanish silently
        # and the caller would be left holding a sentence the agent never
        # answered. Report it and release the fragment regardless.
        try:
            if not self._carry_text:
                return
            if self._turn_active or getattr(self.turn_detector, "is_speaking", False):
                return
        except Exception as exc:  # pragma: no cover - defensive
            self.on_event("error", f"carry flush check failed: {exc}")
        if not self._carry_text:
            return
        # Force it through: audio=None means "no STT, use the carried text".
        self._carry_rounds = self._max_carry_rounds
        self._start_turn(None, 0.0)

    def _start_turn(self, audio: Optional[np.ndarray], duration_ms: float) -> None:
        self._turn_id += 1
        m = TurnMetrics(turn_id=self._turn_id)
        m.speech_end_t = time.perf_counter()
        m.user_audio_ms = duration_ms
        m._vad_hold_ms = float(self.cfg.vad["min_silence_ms"])
        self._active_metrics = m
        stop_event = threading.Event()
        self._stop_generation = stop_event
        self._active_turn = asyncio.create_task(
            self._handle_turn(audio, m, self._turn_id, stop_event)
        )

    async def _barge_in(self) -> None:
        """Must complete in well under 200 ms -- so it never awaits model work."""
        dropped = self.speaker.interrupt()          # takes effect within one 20 ms block
        self._stop_generation.set()                 # llama.cpp breaks on next token
        # Setting the event is not enough on its own: the generation thread may
        # be parked in a blocking socket read with nothing arriving. Closing the
        # connection frees both that thread and llama-server's slot immediately.
        self.llm.abort()
        self.turn_detector.start_cooldown()
        if self._active_metrics is not None:
            self._active_metrics.interrupted = True
        await self._cancel_turn(wait=False)
        self._pending_interruption = True
        self.on_event(
            "bargein", f"cut playback, dropped {dropped / self.speaker.sample_rate:.2f}s of audio"
        )

    async def _absorb_unspoken_turn(self) -> None:
        """Cancel a turn nothing was heard from, and keep what the caller said.

        Without this the abandoned turn's transcript stays in history as a user
        message that never got an answer, so the next prompt carries two or three
        consecutive user turns. That is a malformed conversation, and a 1.7B
        handles it badly -- it was part of why replies drifted off-topic.
        """
        m = self._active_metrics
        said = m.transcript.strip() if m is not None else ""
        self._stop_generation.set()
        self.llm.abort()
        await self._cancel_turn(wait=False)
        if said:
            self.llm.drop_last_user(said)
            self._carry_text = f"{self._carry_text} {said}".strip()
            self.on_event("vad", f"caller still talking; holding {said!r}")

    async def _cancel_turn(self, wait: bool = True) -> None:
        task = self._active_turn
        self._active_turn = None
        if task is None or task.done():
            return
        task.cancel()
        if not wait:
            # Do not block the VAD loop waiting for an in-flight executor call.
            return
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # --------------------------------------------------------------- the turn
    async def _handle_turn(
        self,
        audio: np.ndarray,
        m: TurnMetrics,
        turn_id: int,
        stop_event: threading.Event,
    ) -> None:
        try:
            # ---- 1. finalize the transcript -------------------------------
            if audio is None:
                # A carried fragment being flushed on its own; no audio to decode.
                transcript = None
                text = ""
                m.stt_done_t = time.perf_counter()
            else:
                transcript = await self._loop.run_in_executor(
                    self._stt_pool, self.transcriber.finalize, audio
                )
                m.stt_done_t = time.perf_counter()
                m.stt_cached = transcript.from_cache
                text = _apply_stt_fixups(transcript.text.strip())

            # Stitch on anything the caller trailed off mid-sentence last time.
            if self._carry_text:
                text = f"{self._carry_text} {text}".strip()
                self._carry_text = ""

            m.transcript = text

            if not text:
                # Always visible. A turn that produces no reply must say why --
                # silently discarding one is indistinguishable from a hang, and
                # that is exactly how the backchannel bug presented.
                if transcript is not None and transcript.rejected:
                    self.on_event("note", f"ignored, no content: {transcript.rejected}")
                else:
                    self.on_event("note", "no speech recognized")
                # Ask them to repeat instead of going quiet. To the caller,
                # silence here is indistinguishable from a dropped line --
                # they have no way to know whether they were heard. This is a
                # canned line, so it costs no LLM turn.
                try:
                    await self._speak_standalone(
                        self.cfg.agent.get(
                            "no_speech_prompt",
                            "Sorry, I didn't catch that. Could you say it again?",
                        )
                    )
                except Exception as exc:
                    self.on_event("error", f"re-prompt failed: {exc}")
                return

            # ---- 1a. is the caller actually done talking? -------------------
            # "...and I have two years of experience, so" is not a finished turn.
            # Hold it, wait a beat, and let the next segment complete it rather
            # than answering half a sentence and then getting barged in on.
            if (
                self._carry_rounds < self._max_carry_rounds
                and _ends_dangling(text)
                and audio is not None
            ):
                self._carry_text = text
                self._carry_rounds += 1
                self._arm_carry_flush()
                self.on_event("vad", f"holding unfinished turn: {text!r}")
                return

            self._carry_rounds = 0

            # ---- 1b. backchannels are not turns ----------------------------
            # "Mm-hmm" while the agent is mid-thought means "go on", not "answer
            # me". But only when nothing was asked -- see _is_backchannel.
            #
            # The test needs the line the CALLER HEARD, not the last thing in
            # history. History now holds only what the model wrote (the
            # question is ours -- see the note at the _append_question call
            # site), so it no longer ends in the "?" that decides whether an
            # acknowledgement is an answer or just a continuer. Without this
            # the gate would classify "Yes." as a backchannel right after a
            # question and drop the caller's actual answer -- the exact bug
            # the "?" branch of _is_backchannel exists to prevent.
            last_lines = self.llm.recent_assistant_lines(1)
            last_agent_line = self._last_spoken or (
                last_lines[0] if last_lines else ""
            )
            if last_lines and _is_backchannel(text, last_agent_line):
                self.on_event("user", text)
                self.on_event("note", "backchannel - waiting for them to continue")
                return

            self.on_event("user", text)

            # ---- 2. assemble the prompt -----------------------------------
            context = self.context_provider(text) if self.context_provider else None
            self.llm.add_user(build_user_message(text, context))

            # Name the single live checklist item. This goes on the newest user
            # message only, so it never touches the cached prefix.
            note = self.agenda.directive() if self._has_persona else None
            if self._pending_interruption:
                note = f"{INTERRUPTION_NOTE}\n{note}" if note else INTERRUPTION_NOTE
                self._pending_interruption = False

            # ---- 3. generate, with one retry if the guard rejects it -------
            live_item = self.agenda.current if self._has_persona else None
            spoken_text, stats, rejected = await self._generate_and_speak(
                m, turn_id, stop_event, note, caller_said=text, agenda_item=live_item
            )

            # All retry policy now lives in _flag() inside _consume_chunks: it
            # returns a veto ONLY when a retry is wanted (regeneration on, or
            # the kind is in regenerate_kinds, or it is a malformed artifact).
            # So `rejected` being set already means "retry this" -- no second
            # gate needed here, and no risk of speaking nothing after a veto.
            if rejected and not stop_event.is_set():
                # Nothing reached the speaker -- the guard caught it before Piper
                # ran. Retry once, naming the specific failure rather than
                # restating the rule it just broke.
                kind, detail = rejected
                # The detail is the offending text itself. Logging only the
                # kind made rejections undiagnosable -- a run showing "4/7
                # rejected" gave no way to tell a correct rejection from a
                # guard firing on a good reply.
                self.on_event(
                    "system",
                    f"(rejected {kind}; regenerating) {str(detail)[:90]!r}",
                )
                m.first_audio_ready_t = 0.0
                m.playback_start_t = 0.0
                m.chunks = 0
                # Reset the LLM mark too, or the retry's whole generation gets
                # charged to the TTS stage: tts_ms is measured from first token
                # to first audio, so a rejected turn reported "Piper first audio
                # 17151 ms" when Piper had done nothing of the sort. The retry
                # cost belongs to the LLM stage.
                m.llm_first_token_t = 0.0
                m.retried = True
                retry_note = correction_note(kind, detail)
                if note:
                    retry_note = f"{note}\n{retry_note}"
                # The retry is UNGUARDED, and that is a measurement, not an
                # oversight. Guarding it was tried (runs 9/10) and reverted
                # here after run 11 measured the result:
                #
                #   - 6 of 9 turns hit two rejections in a row, so the guarded
                #     retry rejected again EVERY time. It caught nothing that
                #     the first pass had not already caught.
                #   - Each rejection then fell through to a third, fully
                #     ungated pass. Average TTFA went 1284 ms -> 2546 ms and
                #     the definition of done flipped back to FAIL.
                #   - The third-pass replies were the worst of the three:
                #     "What technologies specifically does our recruitment
                #     team use at our health care company?"
                #
                # So a guarded retry costs a full extra generation and buys
                # no quality. Two passes is the ceiling: the first is checked,
                # and the second is steered by correction_note(), which now
                # names the actual failure for every rejection kind --
                # including fabrication and off_topic, which used to send the
                # model back with no note at all and so reliably reproduced
                # the same mistake. That steering is what makes the unguarded
                # second pass safe; the extra guard was not.
                spoken_text, stats, _ = await self._generate_and_speak(
                    m, turn_id, stop_event, retry_note, guard=False,
                    agenda_item=live_item,
                    # No veto on the retry -- a second rejection would mean a
                    # third generation. Repairs still run: this is the reply
                    # most likely to need them.
                    repair=True,
                )

            m.llm_done_t = time.perf_counter()
            m.llm_tokens = stats.tokens
            m.llm_prompt_tokens = stats.prompt_tokens
            m.response = spoken_text

            # The model wrote the reaction; the question is a fixed line.
            # Appended here rather than inside generation so the guards, the
            # repairs and the retry pass never have to know about it.
            spoken_text, own_line = await self._append_question(
                spoken_text, stop_event, m
            )
            m.response = spoken_text

            if spoken_text.strip():
                # History gets the model's OWN line, not the line the caller
                # heard. Storing "reaction + question" as an assistant turn
                # tells the 3B that it wrote a question -- on every single
                # turn -- which is a demonstration directly contradicting the
                # standing "do not write the question" directive, and it is
                # the thing the model copies. Run 38 turn 3 re-emitted its
                # previous "reaction + question" verbatim; the repeat guard
                # caught it, and the retry cost a second generation (705 ms
                # TTFT, 1355 ms TTFA -- the worst turn of the run, and enough
                # to fail the headline metric on its own).
                #
                # Trimming the templated question at COMPARISON time instead
                # would not have helped: the guard's startswith() test still
                # matches the reaction that the echoed line begins with. The
                # question has to leave the history to stop being echoed.
                #
                # Skipped when the reaction was cut to nothing: the model
                # said nothing, so there is no assistant turn to record, and
                # the caller still heard the templated question.
                if own_line.strip():
                    self.llm.add_assistant(own_line)
                self._last_spoken = spoken_text
                if _is_farewell(spoken_text):
                    self._farewells += 1
                    if self._farewells >= self._max_farewells:
                        # Do not drop the line the instant she stops talking.
                        # Run 27 and run 30 both ended while the caller was
                        # mid-sentence -- the goodbye is hers, the decision to
                        # stop talking is theirs. Arm a grace window instead;
                        # the listen loop cancels it if they start speaking.
                        self._hangup_at = time.monotonic() + self._hangup_grace_s
                        self.on_event(
                            "system",
                            f"(call complete; hanging up in "
                            f"{self._hangup_grace_s:.0f}s unless they speak)",
                        )
                    else:
                        self.on_event(
                            "system",
                            f"(farewell {self._farewells}/{self._max_farewells}; "
                            "one more turn, then hang up)",
                        )
                self.on_event("agent", spoken_text)
                # The caller answered the item that was live when they spoke, so
                # move the pointer on -- but only if Riya actually ASKED that
                # item. Deflections ("sorry?", "why?") do not count either; see
                # AgendaTracker.record.
                #
                # Without the `covered` test the pointer advanced on the
                # caller's utterance alone, so an off-script reply that the
                # guard had rejected still marked the item done. The checklist
                # then ran out while the call was still going, and every later
                # turn became a wrap-up turn in which any question is
                # off-agenda -- reliably two rejections, and a third pass.
                covered = True
                if live_item and spoken_text.strip():
                    off_item = _invents_question(spoken_text, live_item, text)
                    # A closing turn is meant to carry no question, so do not
                    # demand one there or the pointer sticks on the last item.
                    asked = ("?" in spoken_text
                             or _agenda_topic(live_item) == "closing")
                    # Run 25 turn 4: "You've mentioned Python, Java and SQL so
                    # far Can you tell me specifically which databases..." ends
                    # in a question, so `asked` was satisfied and "read back
                    # and confirm?" was marked done -- but nothing was ever
                    # confirmed, and years and education were never read back.
                    # The checklist then ran out two turns early. A read-back
                    # has to actually ask for confirmation. Two misses and we
                    # accept the reply rather than stick on the item forever.
                    if asked and _agenda_topic(live_item) == "confirm":
                        asked = bool(_READBACK_CONFIRM.search(spoken_text))
                        if not asked:
                            self._confirm_misses += 1
                            if self._confirm_misses >= 2:
                                asked = True
                        else:
                            self._confirm_misses = 0
                    covered = not off_item and asked and not self._blocked_bail
                if self.agenda.record(text, covered=covered):
                    # Keep what they said, so the read-back is built from real
                    # answers instead of the model's improvised summary.
                    self._answers.append(text)
                    nxt = self.agenda.current or "wrap-up"
                    self.on_event("vad", f"agenda -> {nxt}")
                # Checklist done: make the next closing firmer so the call ends
                # instead of the agent improvising until the caller gives up.
                if self.agenda.finished:
                    self.agenda.note_closing()

            self.metrics.add(m)
            if self.cfg.agent.get("show_metrics", True):
                self.on_event("metrics", m.render())

        except asyncio.CancelledError:
            stop_event.set()
            m.interrupted = True
            # Keep whatever the agent actually said out loud in the history, so
            # the next turn knows what it already covered (prompt: "pick it back
            # up briefly only if relevant").
            if m.response.strip():
                self.llm.add_assistant(m.response)
            self.metrics.add(m)
            raise

    async def _generate_and_speak(
        self,
        m: TurnMetrics,
        turn_id: int,
        stop_event: threading.Event,
        note: str | None,
        guard: bool = True,
        caller_said: str = "",
        agenda_item: str | None = None,
        repair: bool = True,
    ) -> tuple[str, GenerationStats, tuple[str, str] | None]:
        """One generation pass. Returns (spoken, stats, rejection).

        `rejection` is (kind, detail) when the guard stopped the reply before any
        audio was synthesized: a repeat of something already said, a slip into
        assistant voice, or the caller's own words handed back.

        `repair` is independent of `guard` -- see _consume_chunks.
        """
        chunk_queue: asyncio.Queue = asyncio.Queue()
        stats = GenerationStats()
        thread = threading.Thread(
            target=self._generate_worker,
            args=(chunk_queue, stats, note, m, stop_event),
            name=f"llm-turn-{turn_id}",
            daemon=True,
        )
        thread.start()
        spoken, repeated = await self._consume_chunks(
            chunk_queue, m, stop_event, guard, caller_said, agenda_item, repair
        )
        if repeated:
            # Stop the model finishing a reply we have already decided to bin.
            # Abort the socket rather than setting stop_event: that event is
            # shared with barge-in, and clearing it again afterwards could
            # un-cancel a genuine interruption that landed in between.
            self.llm.abort()
        await self._loop.run_in_executor(None, thread.join, 5.0)
        return spoken, stats, repeated

    def _generate_worker(
        self,
        chunk_queue: asyncio.Queue,
        stats: GenerationStats,
        extra_system: str | None,
        m: TurnMetrics,
        stop_event: threading.Event,
    ) -> None:
        """llama.cpp token stream -> sentence chunks -> asyncio queue."""
        chunker = SentenceChunker(
            first_chunk_min_chars=int(self.cfg.tts["first_chunk_min_chars"]),
            chunk_min_chars=int(self.cfg.tts["chunk_min_chars"]),
            chunk_max_chars=int(self.cfg.tts["chunk_max_chars"]),
            max_sentences=int(self.cfg.llm.get("max_sentences", 2)),
            first_chunk_max_ms=int(self.cfg.tts.get("first_chunk_max_ms", 280)),
            emergency_min_chars=int(self.cfg.tts.get("emergency_min_chars", 10)),
        )
        loop = self._loop

        def put(item) -> None:
            try:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
            except RuntimeError:
                pass

        # The chunker enforces the sentence ceiling (see SentenceChunker). Once
        # it reports done, breaking out of the loop closes the HTTP response,
        # which makes llama-server drop the slot -- so we also stop burning CPU
        # generating sentences the caller was never going to hear.
        try:
            for delta in self.llm.stream(
                stop_event=stop_event, extra_system=extra_system, stats=stats
            ):
                if m.llm_first_token_t == 0.0:
                    m.llm_first_token_t = time.perf_counter()
                for chunk in chunker.push(delta):
                    put(chunk)
                if chunker.done:
                    break
            if not stop_event.is_set():
                tail = chunker.flush()
                if tail:
                    put(tail)
        except Exception as exc:  # surface, don't deadlock the consumer
            put(f"__ERROR__{exc}")
        finally:
            put(_SENTINEL)

    async def _consume_chunks(
        self,
        chunk_queue: asyncio.Queue,
        m: TurnMetrics,
        stop_event: threading.Event,
        guard: bool = True,
        caller_said: str = "",
        agenda_item: str | None = None,
        repair: bool = True,
    ) -> tuple[str, str]:
        """Synthesize and play chunks as they arrive.

        Returns (spoken_text, reason). A non-empty reason means the reply was
        rejected and should be regenerated.

        `guard` is the veto: may this pass reject the reply and ask for
        another? `repair` is the scalpel: may it cut bad spans out of what it
        already has? They are separate on purpose -- a retry has no veto left
        to spend but just as much bad text to fix. Run 32 turn 1.

        The guard stays open until GUARD_CHARS of the reply have been seen, then
        closes so it never delays the body of a good answer. One chunk is not
        enough: "Hello there! How can I assist you today?" splits with the tell
        in chunk two, and the first chunk alone looks harmless.
        """
        # Was 60. Too small: the behavioural checks closed as soon as the
        # first sentence was done, so a question in the SECOND sentence was
        # never examined. Observed in run 7 -- the reply
        #   "Years of relevant professional experience are important here.
        #    Are there 5+ years of relevant professional experience included
        #    in your current job title?"
        # was an experience question asked while the live item was "read back
        # and confirm?". Sentence one is 61 characters, so the window shut
        # before the question arrived and it went to the caller unchecked.
        # The window costs nothing to keep open -- these are regex scans that
        # run before synthesis is queued, so a larger window does not delay
        # audio -- and 140 covers a normal two-sentence reply.
        GUARD_CHARS = 140
        parts: list[str] = []
        # One-element mutable flag so _repair_chunk can carry "a question has
        # already gone out this turn" across streamed chunks -- the two
        # questions the model asked in run 24 did not always land in one piece.
        asked: list[bool] = [False]
        # Has a bail-out already been replaced this turn? Once it has, the rest
        # of the reply is dropped -- see the block below.
        bailed: list[bool] = [False]
        # Did the previous chunk get dropped entirely? The chunker splits on
        # clause boundaries, so a defect can land across two chunks -- the
        # topic in one and its stranded tail ("When needed.", "experience.")
        # in the next, where there is nothing left to judge it against. See
        # its use below.
        emptied = False
        self._blocked_bail = False
        # Greeting words still unmatched this turn, and how many have been
        # matched so far. Reset every pass: see _eat_greeting.
        self._greeting_left = self._greeting.split()
        self._greeting_eaten = 0
        # Was 4. Turns 6+ apart in the conversation kept landing outside this
        # window and slipping the repeat check entirely -- observed live: the
        # exact same two lines recurred verbatim at turn 4/12 and turn 5/14,
        # neither catch triggered because recent_assistant_lines(4) had long
        # since aged them out. Widened to 12 so a repeat anywhere in a normal
        # call length still gets caught, at the cost of a few more content
        # words to compare per check -- negligible next to LLM/TTS latency.
        recent = self.llm.recent_assistant_lines(12) if guard else []
        # Everything the caller has said so far, not just this turn. The
        # read-back turn repeats the caller's own figures and words, which
        # looks exactly like fabrication unless the guard can see where they
        # came from.
        # The current utterance is joined on separately as well as being in the
        # history: history is where it normally lives, but a reply repaired
        # before the turn reached the transcript has to see it too, or the
        # topic rules below treat the caller's own words as fabricated.
        try:
            caller_corpus = " ".join(
                [t.content for t in self.llm.history if t.role == "user"]
                + ([caller_said] if caller_said else [])
            )
        except Exception:
            caller_corpus = caller_said
        # Controls only the behavioral checks (agenda/repetition/etc), which
        # close after GUARD_CHARS. Artifact/tell checks below use `guard`
        # itself and stay open the whole turn -- see the note at their use.
        # Regeneration is OFF, and that is the single most important latency
        # decision in this file. Measured across runs 11-14:
        #
        #   cost   - a retry added ~850 ms in run 12 and ~1800 ms in run 14
        #            (t12: 4553 ms for two passes against a 343 ms clean turn).
        #   quality- the retry is consistently WORSE than what it replaced.
        #            Run 14: "No such position exists at my current company"
        #            and "We currently have several positions available;
        #            however, those are internal roles mostly" -- both
        #            fabrications, both spoken, both the SECOND attempt.
        #            Run 13's retry opened "I shouldn't say 'we'."
        #
        # A 3B model cannot repair itself from a correction note; it pivots to a
        # new failure. Meanwhile the replies that were let straight through
        # (run 14 turns 7, 13, 15) read fine. So the guard now warns instead of
        # vetoing. Only `artifact` -- malformed tool-call scaffolding, which is
        # noise rather than speech -- still blocks, because stripping it is
        # mechanical and costs no generation.
        #
        # Flip `regenerate_on_guard_rejection` in config.yaml to restore it.
        # DETECTION IS ALWAYS ON; only the VETO is gated on _regenerate.
        #
        # This used to be `guard and self._regenerate`. With regeneration off
        # that made checking_open permanently False, so every check below was
        # dead code -- _echoes_agenda, _invents_employer, _parrots,
        # _wrong_person, _invents_specifics, _GUESSING, _unanswered_question,
        # _invents_contact, _raises_deflected_topic. Runs 16/17 looked clean
        # because the model behaved, not because anything was watching; run 18
        # then read a checklist note aloud, and run 19 invented "4 years" of
        # experience the caller never claimed.
        #
        # So: checks always run. When regeneration is enabled they veto exactly
        # as before. When it is not, they warn once per kind and let the reply
        # through -- a rejected reply with no retry available would be silence,
        # which is worse than a flawed sentence, and warning costs nothing.
        checking_open = guard
        warned_off_agenda = False
        warned_kinds: set[str] = set()

        def _flag(kind: str, detail) -> tuple | None:
            """Decide what a guard hit costs: a retry, a cut, or a warning.

            Three outcomes now, not two. The middle one is new:

              retry  - the caller of this function returns a veto and the turn
                       pays for a second generation. Only for kinds the user
                       has explicitly opted into via `regenerate_kinds`, which
                       after run 42 is none of them.
              cut    - drop the offending text, speak nothing of it, and do
                       NOT regenerate. Cheaper than a retry and strictly better
                       than a warning, because a warning lets the text through.
              warn   - report once and let it through.

            Cut exists because retrying is the wrong instrument for these
            three kinds. Run 42 turn 6 rejected 'Oh yes you are right.' -- the
            caller's own words. There is no better version of that sentence for
            a second generation to find; the sentence is wrong because it is an
            echo, and the correct repair is silence plus the templated
            question. A retry paid 584 ms of TTFT to discover the same thing.

            The three were chosen because their text is unusable BY
            DEFINITION -- it already appeared (repeat), it is the caller's
            words (parrot), or it is a checklist note (agenda) -- so there is
            no judgement call about whether cutting destroys something worth
            keeping. Everything else still warns, because a false positive
            there would silently delete a real reply.
            """
            if self._regenerate or kind in self._regen_kinds:
                return (kind, detail)
            if kind in _CUT_KINDS:
                self.on_event("system", f"(cut {kind}: {str(detail)[:60]!r})")
                return (_CUT, detail)
            if kind not in warned_kinds:
                warned_kinds.add(kind)
                self.on_event("system", f"(guard: {kind}) {str(detail)[:70]!r}")
            return None
        while True:
            item = await chunk_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, str) and item.startswith("__ERROR__"):
                self.on_event("error", item[len("__ERROR__") :])
                break
            if stop_event.is_set():
                continue  # drain, don't synthesize audio nobody will hear

            # Artifact / chatbot-tell checks run for the WHOLE reply, not just
            # until GUARD_CHARS closes the window below. They're a cheap regex
            # scan, and a malformed tag (a hallucinated tool-call, a stray
            # </think>, template scaffolding) can land anywhere in a reply --
            # observed live: the first sentence alone ran past GUARD_CHARS, so
            # a `<function name="...">` tag in the SECOND sentence sailed
            # straight past the old guard and reached Piper unfiltered. Only
            # gated by the call-level `guard` param (off for retries, where a
            # second veto would mean a third generation, and for the --text
            # harness). The repairs below are NOT gated on it -- see `repair`.
            if guard:
                artifact = _has_artifact(item)
                if artifact:
                    return "", ("artifact", artifact)
                tell = _sounds_like_a_chatbot(item)
                if tell and self._regenerate:
                    return "", ("assistant", tell)

            # Deterministic repair -- outside `checking_open` below (that block
            # is gated on _regenerate, which is false) AND outside `guard`
            # above. A checklist note read aloud is mechanical rather than a
            # judgement failure, so it is cut out of the chunk instead of paid
            # for with a second generation. Costs no LLM time.
            #
            # Why `guard` must not gate this: run 32 turn 1. The retry pass
            # runs with guard=False so it cannot veto a second time, but that
            # used to disable the repairs along with the veto -- so the
            # regenerated reply, the one most likely to be broken, was the one
            # reply in the call that went out completely unfiltered. It spoke
            # the checklist item "Qualification and field?" aloud, and the
            # echoed item then fooled _invents_question into scoring the turn
            # as on-agenda, so the pointer advanced over a question she never
            # asked and the whole checklist ran out by turn 6 of 17.
            #
            # Vetoing and cutting are different decisions. A retry has no veto
            # to spend; it still has bad text to fix.
            if repair and self._has_persona:
                # Before anything else: she has re-opened with the greeting.
                # Cutting it costs nothing; regenerating cost 1723 ms in run 33.
                if self._greeting_left:
                    eaten, rest = _eat_greeting(item, self._greeting_left)
                    if eaten == 0:
                        # Diverged from the greeting -- it is not one.
                        self._greeting_left = []
                    else:
                        self._greeting_left = self._greeting_left[eaten:]
                        self._greeting_eaten += eaten
                        # Four words is past coincidence; "Hi, this is Riya"
                        # cannot start any other turn of this call.
                        if self._greeting_eaten >= 4:
                            rest = _clean_seam(rest)
                            if rest != item:
                                self.on_event("system", "(greeting repeated; cut)")
                                if not rest.strip():
                                    continue
                                item = rest
                pruned = _strip_agenda_suffix(item, self._agenda)
                if pruned != item:
                    self.on_event(
                        "system",
                        f"(agenda note stripped: {item[len(pruned):].strip()[:60]!r})",
                    )
                    if not pruned.strip():
                        continue
                    item = pruned

                # Run 35, reported live: "why is it speaking every line for 2
                # times". _strip_model_questions only ran inside
                # _append_question, which is AFTER synthesis -- so the
                # transcript was cleaned and the audio was NOT. The model
                # still writes a question despite the directive, so the caller
                # heard it and then heard the templated one. Worse, the strip
                # above had already cut the checklist words out of it, so
                # "How many years of experience do you have?" went out as
                # "How many do you have?". Audio and transcript have to be the
                # same text, so this runs here, before the chunk is spoken.
                #
                # Removing a clause can leave junk in front of it. Run 39 turn
                # 5: "Yes No \nCan you name any frameworks..." came back as
                # "Yes No" and was spoken as the reaction. The dependent-phrase
                # pass below is what deals with that.
                pruned = _strip_model_questions(item)
                if pruned != item:
                    self.on_event(
                        "system", f"(model question dropped: {item[:60]!r})"
                    )
                    if not pruned.strip():
                        continue
                    item = pruned

                # On the confirm turn the template IS the read-back. The model
                # writing one as well means the caller hears their own answers
                # twice -- run 35 turn 4.
                #
                # A finished checklist counts as a closing turn too. When the
                # checklist has run out, agenda_item is None, so the topic
                # resolves to None and this test used to be skipped entirely --
                # which is why run 39 turn 6's third-person summary
                # ("Your qualifications include...") went out in front of the
                # goodbye.
                closing_turn = _agenda_topic(agenda_item or "") in (
                    "confirm",
                    "closing",
                ) or (self._has_persona and self.agenda.finished)
                if closing_turn and _MODEL_READBACK.match(item):
                    self.on_event(
                        "system", "(model read-back dropped; template reads back)"
                    )
                    emptied = True
                    continue

                # Run 23's four mechanical failure shapes: an unsolicited
                # redirect onto a forbidden topic, a claim about openings,
                # a comment about the caller, or an apology for the
                # machinery. Same reasoning as the note above -- cut the
                # span, do not pay for a second generation.
                repaired, cut = _repair_chunk(item, caller_corpus, asked)
                if cut:
                    self.on_event("system", f"(cut: {', '.join(cut)})")
                    if not repaired.strip():
                        emptied = True
                        continue
                    item = repaired

                # One pass for the whole "a dependent phrase left standing
                # where a sentence should be" family. Runs 35, 37, 38 and 39
                # each produced a new instance in a new position, and each was
                # patched at the cut site that caused it -- four patches for
                # one defect, because a cut can land anywhere.
                #
                # NOT gated on something having been removed from this chunk.
                # The chunker splits on sentence boundaries, so the fragment
                # often arrives as its own chunk with nothing removed from it:
                # run 39 turn 1's "At another time." was written by the model
                # as its own sentence and the redirect cut took the sentence
                # AFTER it. A gate here let it straight through.
                if _only_a_fragment(item):
                    self.on_event(
                        "system", f"(dropped a dependent phrase: {item[:40]!r})"
                    )
                    emptied = True
                    continue

                # The chunker splits on clause boundaries, so one defect can
                # arrive as two chunks and only the first is recognisable. Run
                # 37: "No salary discussion today" / "When needed.". Run 38
                # turn 5: a model read-back split as "...and six years of" /
                # "experience." -- the drop took the first, and the tail went
                # out on its own in front of the templated question.
                #
                # A tail is short, because there is not much of a clause left
                # once its head is gone. Deliberately NOT applied after the
                # model-question drop: the model asks a question on almost
                # every turn, so that path is routine and a genuinely short
                # reaction could follow it.
                if emptied and len(item.split()) <= 3 and "?" not in item:
                    self.on_event(
                        "system", f"(dropped stranded tail: {item[:40]!r})"
                    )
                    continue
                emptied = False

                # Run 25 turn 6. Ending the call is the caller's decision;
                # if Riya could not follow an answer she asks for a repeat,
                # which is already the canned line used when nothing was
                # recognised at all. Substituting rather than cutting,
                # because the give-up IS the whole utterance -- cutting
                # leaves "Sorry it seems thanks anyway."
                if bailed[0]:
                    continue
                if _BAILOUT.search(item) and not (
                    _FAREWELL.search(caller_said) or _DECLINE.search(caller_said)
                ):
                    bailed[0] = True
                    self._blocked_bail = True
                    item = str(
                        self.cfg.agent.get(
                            "no_speech_prompt",
                            "Sorry, I didn't catch that. Could you say it again?",
                        )
                    )
                    self.on_event(
                        "system", "(bail-out blocked; asking for a repeat)"
                    )

            if checking_open:
                seen = " ".join(parts + [item])

                # Agenda/persona-specific checks -- meaningless with no system
                # prompt, since there is no recruiter script or "you are Riya
                # on a call" framing for the model to have drifted away from.
                # Running these in generic mode is what was causing the false
                # rejections (guard fires -> regenerate -> ~2x LLM latency).
                if self._has_persona:
                    echoed = _echoes_agenda(item, self._agenda)
                    if echoed:
                        flagged = _flag("agenda", echoed)
                        if flagged:
                            if flagged[0] == _CUT:
                                emptied = True
                                continue
                            return "", flagged
                    # contact/person/off_agenda were gated to `if not parts:`
                    # -- checked ONCE against the raw first streamed fragment
                    # only. Confirmed live: turn 1's REGENERATED reply still
                    # said "does he have", turn 2's REGENERATED reply still
                    # said "has he worked" -- both should have tripped
                    # `person`, neither did, because the pronoun streamed in
                    # after that first tiny fragment and the check never ran
                    # again. Same fix as the fabrication check: run on `seen`
                    # (the growing accumulated text) every iteration the
                    # window is open, not once on a fragment that might be
                    # two words long.
                    contact = _invents_contact(seen)
                    if contact:
                        flagged = _flag("contact", contact)
                        if flagged:
                            return "", flagged
                    person = _wrong_person(seen)
                    if person:
                        flagged = _flag("person", person)
                        if flagged:
                            return "", flagged
                    # Off-agenda is logged, not regenerated. Asking a
                    # different-but-reasonable screening question is a
                    # misprioritisation, not a defect the caller will notice,
                    # and a regeneration costs a full extra prompt evaluation.
                    # Run 13 turn 2 paid 3046 ms for it. Warn once -- the check
                    # re-runs every iteration the window is open, so without
                    # the latch it would log on every chunk.
                    if not warned_off_agenda:
                        invented = _invents_question(seen, agenda_item, caller_corpus)
                        if invented:
                            warned_off_agenda = True
                            self.on_event(
                                "system",
                                f"(off-agenda, not regenerating: {invented!r})",
                            )
                    employer = _invents_employer(seen)
                    if employer:
                        flagged = _flag("fabrication", employer)
                        if flagged:
                            return "", flagged
                    guess = _GUESSING.search(seen)
                    # Only a real subject counts. "That sounds like a solid
                    # foundation" and a bare "Sounds like a good fit" are
                    # acknowledgements; "Deliver sounds like an interesting
                    # project management tool" inflates a misheard word.
                    subject = (guess.group(1) or "").strip(".,!?;:").lower() if guess else ""
                    if guess and subject and subject not in _GENERIC_SOUND_SUBJECTS:
                        flagged = _flag("fabrication", guess.group(0))
                        if flagged:
                            return "", flagged
                    deflected = _raises_deflected_topic(seen, caller_corpus)
                    if deflected:
                        flagged = _flag("off_topic", deflected)
                        if flagged:
                            return "", flagged

                unanswered = _unanswered_question(seen, caller_said)
                if unanswered:
                    flagged = _flag("question", unanswered)
                    if flagged:
                        return "", flagged
                if caller_said and _parrots(seen, caller_said):
                    flagged = _flag("parrot", caller_said)
                    if flagged:
                        # Echoing the caller back at them. Cut it: the
                        # templated question still goes out, so the turn is
                        # not lost, and a retry cannot produce a better echo.
                        if flagged[0] == _CUT:
                            emptied = True
                            continue
                        return "", flagged
                # Was gated to `if not parts:` like the checks above, so it
                # only ever saw the FIRST raw streamed fragment -- observed
                # live: it caught a fabricated rate once, but the regenerated
                # reply's rate landed in a LATER fragment (the dollar figure
                # streamed in after "The company pays between") and sailed
                # through uncaught on the very next turn. Runs on `seen`
                # (the growing accumulated text, same variable _echoes_agenda
                # already uses above) every iteration the window is open,
                # instead of a single raw fragment once.
                invented_fact = _invents_specifics(seen, caller_corpus)
                if invented_fact:
                    flagged = _flag("fabrication", invented_fact)
                    if flagged:
                        return "", flagged
                if not parts:
                    opening = _normalize(item)
                    if len(opening) >= 12:
                        # Cannot `continue` straight out of the `for`: that
                        # would only advance to the next `recent` line and the
                        # offending text would then be spoken. Latch the
                        # decision and drop the item after the loop.
                        drop_repeat = False
                        for previous in recent:
                            if _normalize(previous).startswith(opening) or _too_similar(
                                item, previous
                            ):
                                flagged = _flag("repeat", previous)
                                if flagged:
                                    if flagged[0] == _CUT:
                                        drop_repeat = True
                                        break
                                    return "", flagged
                        if drop_repeat:
                            emptied = True
                            continue
                if len(seen) >= GUARD_CHARS:
                    checking_open = False


            speech = await self._loop.run_in_executor(self._tts_pool, self.tts.synthesize, item)
            if speech is None or stop_event.is_set():
                continue

            if m.first_audio_ready_t == 0.0:
                m.first_audio_ready_t = time.perf_counter()
                m.playback_start_t = m.first_audio_ready_t
            m.chunks += 1
            parts.append(item)
            m.response = " ".join(p.strip() for p in parts if p).strip()
            self.speaker.play(speech.samples)

            # Backpressure: more than ~2.5 s of queued audio buys nothing, and
            # holding back keeps the TTS thread from competing with llama.cpp
            # for cores while chunk 1 is still playing.
            while self.speaker.pending_seconds > 2.5 and not stop_event.is_set():
                await asyncio.sleep(0.05)

        return " ".join(p.strip() for p in parts if p).strip(), None

    # ------------------------------------------------------- standalone speech
    def _readback(self) -> str:
        """The caller's own answers, joined for the confirm template.

        Built from what they actually said rather than left to the model,
        which either summarised wrongly or ran out of tokens mid-sentence --
        run 33 turn 6 got as far as "Can I confirm that before" and stopped.
        """
        parts: list[str] = []
        for answer in self._answers:
            # Run 34's "Use um three" and run 36's "Oh you sound free" both
            # advanced the checklist and were then read back as screening
            # data. _readable_answer drops disfluencies, counts CONTENT words
            # and caps the length -- see it for why each part is needed.
            bit = _readable_answer(answer)
            if not bit:
                continue
            # Answers are included whole or not at all. Truncating one cuts
            # facts mid-clause; the budget bounds the total instead, which is
            # what "read back the key facts" actually needs.
            if parts and len(" ".join(parts + [bit]).split()) > _READBACK_WORDS:
                break
            # Never end the read-back on a dangling connective.
            while bit.split() and (
                bit.split()[-1].strip(".,").lower() in _TRAILING_JUNK
            ):
                bit = " ".join(bit.split()[:-1])
            parts.append(bit)
        if not parts:
            return ""
        joined = parts[0] if len(parts) == 1 else (
            ", ".join(parts[:-1]) + " and " + parts[-1]
        )
        # It lands mid-sentence in the confirm template, so it has to read as
        # the start of a clause: "Let me read that back. Your background..."
        return joined[0].upper() + joined[1:]

    async def _append_question(
        self, spoken: str, stop_event, metrics=None
    ) -> tuple[str, str]:
        """Speak the fixed question for the live item after Riya's line.

        The model wrote this for runs 1-33 and produced a new failure
        phrasing every single run. It writes the reaction; Python asks.

        Returns `(spoken, own_line)`. `spoken` is the whole line the caller
        hears -- reaction plus the templated question -- and it is what the
        transcript, `_last_spoken` and the agenda bookkeeping must see.
        `own_line` is only the part the MODEL wrote, and it is what belongs in
        the LLM history: the question is Python's, and storing it as an
        assistant turn is what caused run 38 turn 3 (see the call site).

        `metrics` is the turn's TurnMetrics. It is set here because on a turn
        whose reaction was cut to nothing THIS is the first audio the caller
        hears, and only _consume_chunks used to timestamp first audio -- see
        the note at the assignment.
        """
        if not self._has_persona or not self._questions:
            return spoken, spoken
        # "Could you say it again?" is the whole point of the canned line --
        # stripping it would leave Riya asking something new instead.
        canned = str(self.cfg.agent.get("no_speech_prompt", "") or "").strip()
        if canned and spoken.strip() == canned:
            return spoken, spoken

        answers = self._readback() or str(
            self.cfg.agent.get("readback_fallback", "your background so far")
        )
        question = self.agenda.question(self._questions, answers=answers)
        if not question:
            return spoken, spoken
        reaction = _strip_model_questions(spoken)
        if stop_event.is_set():
            return reaction, reaction
        speech = await self._loop.run_in_executor(
            self._tts_pool, self.tts.synthesize, question
        )
        if speech is None or stop_event.is_set():
            return reaction, reaction
        # Run 37 turn 1: the reaction was cut to nothing, so no streamed chunk
        # played, so _consume_chunks never stamped first audio -- and TTFA
        # reported 0 ms for a turn that really took most of a second. A 0 ms
        # turn drags the average down and makes a PASS mean nothing, so the
        # templated question stamps it when it is the first thing out.
        if metrics is not None and metrics.first_audio_ready_t == 0.0:
            metrics.first_audio_ready_t = time.perf_counter()
            metrics.playback_start_t = metrics.first_audio_ready_t
        self.speaker.play(speech.samples)
        self.on_event("system", f"(question from template: {question!r})")
        reaction = reaction.rstrip()
        if reaction and reaction[-1] not in ".?!":
            reaction += "."
        return f"{reaction} {question}".strip(), reaction

    async def _speak_standalone(self, text: str) -> None:
        """Speak a fixed line (greeting) without involving the LLM."""
        # No sentence ceiling here: the greeting is a fixed line we wrote, and
        # it is three sentences on purpose (who I am, why I rang, may I speak).
        # The ceiling exists to rein in the model, not us.
        chunker = SentenceChunker(
            first_chunk_min_chars=int(self.cfg.tts["first_chunk_min_chars"]),
            chunk_min_chars=int(self.cfg.tts["chunk_min_chars"]),
            chunk_max_chars=int(self.cfg.tts["chunk_max_chars"]),
            max_sentences=0,
        )
        chunks = chunker.push(text)
        tail = chunker.flush()
        if tail:
            chunks.append(tail)
        self.on_event("agent", text)
        self._last_spoken = text
        for chunk in chunks:
            speech = await self._loop.run_in_executor(self._tts_pool, self.tts.synthesize, chunk)
            if speech is not None:
                self.speaker.play(speech.samples)

    # ------------------------------------------------- text-only test harness
    async def say_text(self, user_text: str) -> TurnMetrics:
        """Run one full LLM->TTS turn from typed input (no mic). Used by --text."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._turn_id += 1
        m = TurnMetrics(turn_id=self._turn_id)
        m.speech_end_t = m.stt_done_t = time.perf_counter()
        m.transcript = user_text
        stop_event = threading.Event()
        self._stop_generation = stop_event
        self._active_metrics = m

        context = self.context_provider(user_text) if self.context_provider else None
        self.llm.add_user(build_user_message(user_text, context))
        chunk_queue: asyncio.Queue = asyncio.Queue()
        stats = GenerationStats()
        thread = threading.Thread(
            target=self._generate_worker,
            args=(chunk_queue, stats, None, m, stop_event),
            daemon=True,
        )
        thread.start()
        spoken, _repeated = await self._consume_chunks(chunk_queue, m, stop_event, guard=False)
        await self._loop.run_in_executor(None, thread.join, 10.0)

        m.llm_done_t = time.perf_counter()
        m.llm_tokens = stats.tokens
        m.llm_prompt_tokens = stats.prompt_tokens
        m.response = spoken
        if spoken.strip():
            self.llm.add_assistant(spoken)
        self.metrics.add(m)
        return m

    async def scripted_turn(self, user_text: str) -> TurnMetrics:
        """One turn from SCRIPTED input: the real turn path minus the audio.

        say_text() is not faithful enough to evaluate prompt changes -- it runs
        with guard=False and passes no agenda note, so neither the output guard
        nor the checklist pointer is exercised. That left one noisy live call
        as the only way to score a change, and a 3B at temperature 0.4 fails
        differently every time, so single samples were being read as signal.

        This mirrors the voice turn: same system prompt, same per-turn agenda
        note, guard ON, and the same advance rule. Only the mic and speaker
        are absent. replay.py drives it to score quality offline.
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._turn_id += 1
        m = TurnMetrics(turn_id=self._turn_id)
        m.speech_end_t = m.stt_done_t = time.perf_counter()
        m.transcript = user_text
        stop_event = threading.Event()
        self._stop_generation = stop_event
        self._active_metrics = m

        self.llm.add_user(build_user_message(user_text, None))
        note = self.agenda.directive() if self._has_persona else None
        live_item = self.agenda.current if self._has_persona else None

        spoken, stats, _rejected = await self._generate_and_speak(
            m, self._turn_id, stop_event, note,
            caller_said=user_text, agenda_item=live_item,
        )

        m.llm_done_t = time.perf_counter()
        m.llm_tokens = stats.tokens
        m.llm_prompt_tokens = stats.prompt_tokens
        m.response = spoken
        if spoken.strip():
            self.llm.add_assistant(spoken)
        if self._has_persona and spoken.strip():
            covered = "?" in spoken or _agenda_topic(live_item) == "closing"
            self.agenda.record(user_text, covered=covered)
        self.metrics.add(m)
        return m

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _default_event(kind: str, text: str) -> None:
        print(f"[{kind}] {text}")
