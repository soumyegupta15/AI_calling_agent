"""Incremental sentence chunking + spoken-form text normalization.

Two independent pure-Python pieces (both unit-tested in tests/):

  SentenceChunker    Fed LLM token deltas, emits speakable chunks the moment one
                     is complete. The first chunk is allowed to be very short so
                     audio starts fast (brief section 2 rule 1); later chunks are
                     longer because by then playback is already running and bigger
                     chunks synthesize with better prosody. If no terminator has
                     arrived by chunk_max_chars, it splits at a comma or
                     conjunction rather than stalling.

  normalize_for_tts  Piper reads raw text literally, so "C2H at $85-95/hr" comes
                     out as letter soup. This expands staffing shorthand, money,
                     rates and dates into the words a recruiter would actually say.
"""

from __future__ import annotations

import re
import time
from typing import Iterator

# --------------------------------------------------------------------------
#  Sentence chunking
# --------------------------------------------------------------------------

_TERMINATORS = ".!?"
# Splitting after these would be wrong ("Mr. Patel", "3.5 years", "U.S. based").
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "corp",
    "co", "approx", "dept", "est", "min", "max", "no", "govt", "u.s", "u.k",
}
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_SOFT_BREAK = re.compile(r",\s+(?=(?:and|but|so|because|which|while|then|though)\b)", re.I)

# The model reliably writes " - " where it should have written a period --
# "That sounds good - how many years?" -- despite the prompt telling it not to.
# Because _find_terminator only accepts .!?, a reply like that produced NO
# terminator until the very end, so the whole reply arrived as ONE chunk.
# Measured live: most turns logged "1 audio chunk(s)", i.e. streaming TTS was
# effectively off and first audio waited for the entire generation. Spaced
# dashes are a hard break; unspaced hyphens ("12-month", "contract-to-hire")
# are deliberately not matched.
_DASH_BREAK = re.compile(r"\s+[\-\u2013\u2014]\s+")

# Worse than the dash: the model simply OMITS the separator and runs two
# sentences together -- "Five plus years typically qualifies us for certain
# roles Can you please confirm how much work experience you've accumulated
# over time?". No period, no dash, one 131-character run-on.
#
# Measured over a live 7-turn call, 5 of 7 replies were run together like
# this. With no terminator to find, the chunker fell through to the
# first-chunk deadline and emitted ONE chunk of 66-95 chars. Piper time
# tracks first-chunk length at roughly 120 ms + 6.2 ms per char, so those
# turns cost 557-636 ms against a 550 ms target, while the one punctuated
# reply (17-char first chunk) cost 229 ms. That single difference is most of
# the gap between a passing and a failing time-to-first-audio.
#
# Detecting it needs care, because "Acme Corp", "Python Spark" and "Bengaluru,
# Hybrid" are also lower-case-to-capital transitions and are NOT sentence
# boundaries. So the capitalized word has to be a plausible sentence opener --
# a wh- word, an auxiliary, or a discourse marker. "Corp" and "Spark" are not
# in that set, so they never trigger a split.
_SENTENCE_OPENERS = frozenset("""
can could would will should shall do does did are is was were have has had
what which who whom when where why how
thanks thank now so also great perfect sure okay alright got right let
please well i we you it that this there here just actually
""".split())
_MISSING_PERIOD = re.compile(r"(?<=[a-z,;)\]])\s+(?=[A-Z][a-z])")

# Emoji, pictographs, dingbats and arrows. Piper reads these as symbol names or
# drops them unpredictably, so they never reach it.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF"      # emoji, pictographs, symbols
    "\U00002600-\U000027BF"       # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"       # regional indicators (flags)
    "\U00002B00-\U00002BFF"       # arrows and stars
    "\U0000FE00-\U0000FE0F"       # variation selectors
    "\U0000200D]"                 # zero-width joiner
)

# Typographic characters the model emits that Piper mispronounces.
_PUNCT_MAP = {
    "’": "'", "‘": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    # NOTE: em/en dashes are deliberately absent. A later rule already turns
    # them into a comma, which is the right pause for speech; mapping them
    # here would beat that rule to it and leave a spoken hyphen.
    "…": "...",                        # ellipsis
    " ": " ",                          # non-breaking space
}


class SentenceChunker:
    """Accumulates text deltas and yields complete speakable chunks.

    Also strips reasoning blocks. Qwen3 and other hybrid-reasoning models emit
    <think>...</think> before their actual answer; llama.cpp's
    `enable_thinking: false` is not reliable across builds, so we defend here
    too. Without this the agent would read its own scratchpad down the phone.
    """

    def __init__(
        self,
        first_chunk_min_chars: int = 12,
        chunk_min_chars: int = 45,
        chunk_max_chars: int = 180,
        max_sentences: int = 0,
        first_chunk_max_ms: int = 280,
        emergency_min_chars: int = 10,
    ):
        self.first_min = first_chunk_min_chars
        self.min_chars = chunk_min_chars
        self.max_chars = chunk_max_chars
        # Deadline for the FIRST chunk, measured from the first token. The
        # 3B model often emits long unpunctuated runs, and waiting for a
        # terminator that never comes is what pushed first-audio to ~770 ms.
        # Past this deadline we cut at a word boundary regardless. 0 disables.
        self.first_chunk_max_ms = int(first_chunk_max_ms)
        # Shortest buffer the deadline is allowed to cut. Below this, keep
        # waiting -- emitting a two-word fragment sounds worse than waiting.
        self.emergency_min_chars = int(emergency_min_chars)
        # Hard ceiling on sentences per turn; 0 disables it.
        #
        # This has to live here rather than in the caller. Counting terminators
        # in emitted chunks is too coarse: chunk_min_chars means three short
        # questions arrive glued into ONE chunk, so by the time a caller could
        # count them they have all been queued for synthesis. Enforcing it at
        # the split point is the only place that can cut mid-chunk.
        self.max_sentences = int(max_sentences)
        self._buf = ""
        self._emitted = 0
        self._sentences = 0
        self._done = False
        self._in_think = False
        # Wall clock at the first push(); the first-chunk deadline runs from it.
        self._t_start: float | None = None
        # Holds a tag that arrived split across two deltas ("<thi" + "nk>").
        self._carry = ""

    # ------------------------------------------------------------------ public
    @property
    def done(self) -> bool:
        """True once max_sentences has been emitted; the caller should stop."""
        return self._done

    def reset(self) -> None:
        self._buf = ""
        self._emitted = 0
        self._sentences = 0
        self._done = False
        self._in_think = False
        self._t_start = None
        self._carry = ""

    def _strip_reasoning(self, delta: str) -> str:
        """Return only the text outside <think>..</think>, streaming-safe.

        Tags can be split across deltas, so a partial tag at the end of the
        input is held in self._carry rather than emitted.
        """
        buf = self._carry + delta
        self._carry = ""
        kept: list[str] = []

        while buf:
            if self._in_think:
                end = buf.find(_THINK_CLOSE)
                if end == -1:
                    # Still inside. Retain only what could be a partial closing
                    # tag; everything else is reasoning and gets dropped.
                    self._carry = buf[-(len(_THINK_CLOSE) - 1):]
                    buf = ""
                else:
                    buf = buf[end + len(_THINK_CLOSE):]
                    self._in_think = False
            else:
                # An ORPHAN closing tag: the model emitted </think> without ever
                # opening one, which happens when llama.cpp's chat template has
                # already swallowed the opening tag. A live turn spoke the
                # literal characters "</think>" down the phone. Everything before
                # it was reasoning, so drop that too and keep only what follows.
                orphan = buf.find(_THINK_CLOSE)
                start = buf.find(_THINK_OPEN)
                if orphan != -1 and (start == -1 or orphan < start):
                    kept.clear()
                    buf = buf[orphan + len(_THINK_CLOSE):]
                    continue

                if start == -1:
                    # A partial tag at the tail must not be spoken. Either tag
                    # could be arriving split across deltas.
                    hold = 0
                    for tag in (_THINK_OPEN, _THINK_CLOSE):
                        for k in range(1, min(len(tag), len(buf)) + 1):
                            if buf.endswith(tag[:k]):
                                hold = max(hold, k)
                    if hold:
                        self._carry = buf[-hold:]
                        kept.append(buf[:-hold])
                    else:
                        kept.append(buf)
                    buf = ""
                else:
                    kept.append(buf[:start])
                    buf = buf[start + len(_THINK_OPEN):]
                    self._in_think = True

        return "".join(kept)

    def push(self, delta: str) -> list[str]:
        """Add generated text; return zero or more chunks ready to synthesize."""
        if self._done:
            return []
        if self._t_start is None:
            self._t_start = time.monotonic()
        self._buf += self._strip_reasoning(delta)
        out: list[str] = []
        while not self._done:
            chunk = self._try_split()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self) -> str | None:
        """Emit whatever is left at end of generation."""
        # An unterminated <think> block means the model never produced a real
        # answer; speaking the fragment would be worse than saying nothing.
        # Likewise once the sentence ceiling is hit: the tail is the start of a
        # sentence we have already decided not to say.
        rest = "" if (self._in_think or self._done) else self._buf.strip()
        self._buf = ""
        self._carry = ""
        self._in_think = False
        if rest:
            self._emitted += 1
            return rest
        return None

    # ----------------------------------------------------------------- private
    @property
    def _threshold(self) -> int:
        return self.first_min if self._emitted == 0 else self.min_chars

    def _try_split(self) -> str | None:
        buf = self._buf
        if not buf.strip():
            return None

        # 0. First-chunk deadline. If the model is emitting a long run with no
        # terminator and no dash, the caller would otherwise hear nothing until
        # generation finishes. Cut at a word boundary so audio starts on time.
        if (
            self._emitted == 0
            and self.first_chunk_max_ms > 0
            and self._t_start is not None
            and (time.monotonic() - self._t_start) * 1000.0 >= self.first_chunk_max_ms
        ):
            forced = self._force_split()
            if forced:
                return forced

        # 1. Hard sentence terminator followed by whitespace.
        idx = self._find_terminator(buf)
        if idx is not None:
            candidate = buf[: idx + 1].strip()
            # At the ceiling, split here no matter how short the chunk is. This
            # is the cut that stops "Do you mean Python? Or Spark? Or AWS?" from
            # reaching the speaker as a single chunk.
            at_ceiling = (
                self.max_sentences and self._sentences + 1 >= self.max_sentences
            )
            long_enough = len(candidate) >= self._threshold
            first_chunk_exception = self._emitted == 0 and len(candidate) >= 4
            if at_ceiling or long_enough or first_chunk_exception:
                self._buf = buf[idx + 1 :].lstrip()
                self._emitted += 1
                self._sentences += 1
                if self.max_sentences and self._sentences >= self.max_sentences:
                    self._done = True
                    self._buf = ""
                return candidate

        # 1b. Dash-joined clause. See _DASH_BREAK: the model uses " - " where a
        # period belongs, and without this a two-clause reply with a single
        # final "?" arrived as one chunk and never streamed.
        dash = _DASH_BREAK.search(buf)
        if dash and len(buf[: dash.start()].strip()) >= 8:
            candidate = buf[: dash.start()].strip()
            self._buf = buf[dash.end() :].lstrip()
            self._emitted += 1
            self._sentences += 1
            if self.max_sentences and self._sentences >= self.max_sentences:
                self._done = True
                self._buf = ""
            return candidate

        # 1c. Run-together sentences with no separator at all. See
        # _MISSING_PERIOD for the measurement and for why the opener has to be
        # whitelisted. This is a real sentence boundary, so unlike _force_split
        # it counts against max_sentences.
        gap = _MISSING_PERIOD.search(buf)
        if gap:
            tail = buf[gap.end():]
            words = tail.split()
            opener = words[0].strip(".,!?;:").lower() if words else ""
            head = buf[: gap.start()].strip()
            if opener in _SENTENCE_OPENERS and len(head) >= self._threshold:
                self._buf = tail.lstrip()
                self._emitted += 1
                self._sentences += 1
                if self.max_sentences and self._sentences >= self.max_sentences:
                    self._done = True
                    self._buf = ""
                return head

        # 2. Running long with no terminator -> break at a natural pause.
        if len(buf) >= self.max_chars:
            match = None
            for m in _SOFT_BREAK.finditer(buf[: self.max_chars]):
                match = m
            if match and match.start() >= self._threshold:
                self._buf = buf[match.start() + 1 :].lstrip()
                self._emitted += 1
                return buf[: match.start() + 1].strip()
            # Fall back to the last comma, then the last space.
            for sep in (",", " "):
                pos = buf.rfind(sep, self._threshold, self.max_chars)
                if pos > 0:
                    self._buf = buf[pos + 1 :].lstrip()
                    self._emitted += 1
                    return buf[: pos + 1].strip()

        return None

    def _force_split(self) -> str | None:
        """Deadline fallback: emit the buffer at its last word boundary.

        Deliberately does NOT increment _sentences -- this is a mid-sentence
        cut, not a completed sentence, and the sentence ceiling should not
        count it. Returns None when there is too little to speak, so the
        deadline simply re-arms on the next push.
        """
        buf = self._buf
        if len(buf.strip()) < self.emergency_min_chars:
            return None
        pos = buf.rfind(" ")
        # No usable word boundary yet -- wait for one rather than clipping
        # a word in half, which sounds worse than a few extra ms.
        if pos < 4:
            return None
        self._buf = buf[pos + 1 :].lstrip()
        self._emitted += 1
        return buf[:pos].strip()

    @staticmethod
    def _find_terminator(buf: str) -> int | None:
        for i, ch in enumerate(buf):
            if ch not in _TERMINATORS:
                continue
            # Must be followed by whitespace (or be at a clear boundary).
            if i + 1 >= len(buf):
                continue
            if not buf[i + 1].isspace():
                continue
            if ch == ".":
                # Decimal number: "3.5"
                if i > 0 and buf[i - 1].isdigit() and i + 2 < len(buf) and buf[i + 2].isdigit():
                    continue
                # Known abbreviation.
                word = re.split(r"[\s(]", buf[:i])[-1].lower().rstrip(".")
                if word in _ABBREVIATIONS:
                    continue
                # Single initial: "J. Smith"
                if len(word) == 1 and word.isalpha():
                    continue
            return i
        return None


# --------------------------------------------------------------------------
#  Spoken-form normalization
# --------------------------------------------------------------------------

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _say_number(n: int) -> str:
    if n < 0:
        return "minus " + _say_number(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        out = f"{_ONES[hundreds]} hundred"
        return out + (f" {_say_number(rest)}" if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        out = f"{_say_number(thousands)} thousand"
        return out + (f" {_say_number(rest)}" if rest else "")
    millions, rest = divmod(n, 1_000_000)
    out = f"{_say_number(millions)} million"
    return out + (f" {_say_number(rest)}" if rest else "")


# Staffing / IT shorthand -> how a recruiter actually says it out loud.
_ACRONYMS: list[tuple[str, str]] = [
    (r"\bC2H\b", "contract to hire"),
    (r"\bC2C\b", "corp to corp"),
    (r"\bW-?2\b", "W two"),
    (r"\b1099\b", "ten ninety nine"),
    (r"\bRTR\b", "right to represent"),
    (r"\bSOW\b", "S O W"),
    (r"\bMSP\b", "M S P"),
    (r"\bVMS\b", "V M S"),
    (r"\bH-?1B\b", "H one B"),
    (r"\bOPT\b", "O P T"),
    (r"\bEAD\b", "E A D"),
    (r"\bGC\b", "green card"),
    (r"\bUSC\b", "U S citizen"),
    (r"\bTN\b", "T N"),
    (r"\bFTE\b", "full time employee"),
    (r"\bPTO\b", "P T O"),
    (r"\bSRE\b", "S R E"),
    (r"\bQA\b", "Q A"),
    (r"\bCI/CD\b", "C I C D"),
    (r"\bK8s\b", "kubernetes"),
    (r"\bIaC\b", "infrastructure as code"),
    (r"\bAWS\b", "A W S"),
    (r"\bGCP\b", "G C P"),
    (r"\bSAP\b", "sap"),
    (r"\bERP\b", "E R P"),
    (r"\bBA\b", "business analyst"),
    (r"\bPM\b", "project manager"),
    (r"\bJD\b", "job description"),
    (r"\bATS\b", "A T S"),
    (r"\bYOE\b", "years of experience"),
    (r"\bNDA\b", "N D A"),
    (r"\bReq\b\.?", "requisition"),
    (r"\bTier\s*([123])\b", lambda m: f"tier {_ONES[int(m.group(1))]}"),
]

_MONEY_RANGE = re.compile(r"\$\s?(\d[\d,]*)\s?(?:-|–|to)\s?\$?\s?(\d[\d,]*)\s?(?:/|per\s)?(hr|hour|k|K)?\b")
_MONEY = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{2}))?\s?(?:/|per\s)?(hr|hour|k|K|yr|year)?\b")
_PERCENT = re.compile(r"\b(\d[\d,]*)\s?%")
_HASH_NUM = re.compile(r"#\s?(\d+)")
_PER_HR = re.compile(r"\b(\d[\d,]*)\s?/\s?(hr|hour|yr|year)\b", re.I)


def _rate_word(unit: str | None) -> str:
    if not unit:
        return ""
    unit = unit.lower()
    if unit in ("hr", "hour"):
        return " an hour"
    if unit in ("yr", "year"):
        return " a year"
    if unit == "k":
        return " thousand"
    return ""


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def normalize_for_tts(text: str) -> str:
    """Rewrite a chunk into something the TTS will pronounce like a person."""
    if not text:
        return text

    out = text

    # Emoji and pictographs. The prompt forbids them; a 1.7B emits them anyway
    # (a live turn ended in a star), and Piper either mangles them or reads a
    # symbol name aloud. Strip unconditionally -- this is the backstop for the
    # instruction, not a substitute for it.
    out = _EMOJI.sub("", out)

    # Typographic punctuation the model picks up from its training data. Piper
    # handles the ASCII forms far more predictably: a curly apostrophe can come
    # out as a pause, and an em dash as a spoken word.
    for fancy, plain in _PUNCT_MAP.items():
        out = out.replace(fancy, plain)

    # Strip any markdown the model leaked despite the prompt -- it would be
    # read aloud as literal asterisks and hyphens.
    out = re.sub(r"^\s*[#>*\-•]+\s*", "", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"\*(.+?)\*", r"\1", out)
    out = re.sub(r"`(.+?)`", r"\1", out)

    # Doubled sentence terminators. Rule 15 forbids "?." but the 3B emits it
    # anyway -- run 17 ended four separate turns on "have you?." / "SQL?.".
    # Piper reads the trailing "." as a second, shorter sentence, so the
    # question never settles; it lands as a stumble. Collapse to one real mark.
    # Deterministic rather than another prompt rule: the model has been told
    # once already and did not comply.
    out = re.sub(r"([?!])\.+", r"\1", out)
    out = re.sub(r"\.\s*\?", "?", out)

    # Money ranges first ("$85-95/hr" -> "eighty five to ninety five dollars an hour").
    def _money_range(m: re.Match) -> str:
        lo, hi = _int(m.group(1)), _int(m.group(2))
        unit = m.group(3)
        if unit and unit.lower() == "k":
            return f"{_say_number(lo)} to {_say_number(hi)} thousand dollars"
        return f"{_say_number(lo)} to {_say_number(hi)} dollars{_rate_word(unit)}"

    out = _MONEY_RANGE.sub(_money_range, out)

    def _money(m: re.Match) -> str:
        whole = _say_number(_int(m.group(1)))
        cents = m.group(2)
        unit = m.group(3)
        if unit and unit.lower() == "k":
            return f"{whole} thousand dollars"
        base = f"{whole} dollars"
        if cents:
            base += f" {_say_number(int(cents))}"
        return base + _rate_word(unit)

    out = _MONEY.sub(_money, out)
    out = _PER_HR.sub(lambda m: f"{_say_number(_int(m.group(1)))}{_rate_word(m.group(2))}", out)
    out = _PERCENT.sub(lambda m: f"{_say_number(_int(m.group(1)))} percent", out)
    out = _HASH_NUM.sub(lambda m: " number " + " ".join(_ONES[int(d)] for d in m.group(1)), out)

    for pattern, repl in _ACRONYMS:
        out = re.sub(pattern, repl, out)

    # Slashes read badly: "remote/hybrid" -> "remote or hybrid".
    out = re.sub(r"(?<=[A-Za-z])\s?/\s?(?=[A-Za-z])", " or ", out)
    # Em/en dashes -> comma pause.
    out = re.sub(r"\s*[—–]\s*", ", ", out)
    # Collapse whitespace.
    out = re.sub(r"\s+", " ", out).strip()
    return out


def chunk_text(text: str, **kwargs) -> list[str]:
    """Convenience: split a complete string the same way streaming would."""
    chunker = SentenceChunker(**kwargs)
    chunks = chunker.push(text)
    tail = chunker.flush()
    if tail:
        chunks.append(tail)
    return chunks
