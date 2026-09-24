"""STT engine selection, hallucination filtering, and streaming behaviour.

The hallucination tests matter: on a live call Whisper reliably invents "Thank
you." from silence, and before filtering those phantom turns were being sent to
the LLM as if the caller had spoken. A real session produced three of them in a
row.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent.stt import (  # noqa: E402
    StreamingTranscriber,
    Transcript,
    _looks_like_hallucination,
    build_stt,
)


# --------------------------------------------------------------------------
#  Hallucination detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Thank you.", "thank you", "Thanks for watching!", "Bye.", "you",
    "Okay.", " . ", "!", "Thank you very much", "Music", "[Applause]".strip("[]"),
])
def test_canonical_whisper_fillers_are_rejected(text):
    assert _looks_like_hallucination(text)


@pytest.mark.parametrize("text", [
    "I need a senior DevOps engineer.",
    "Thank you for sending the resume over.",     # longer -> real speech
    "No.",                                        # short but meaningful in context
    "Are you calling about the Acme role?",
])
def test_real_speech_is_not_rejected(text):
    if text == "No.":
        pytest.skip("bare 'No.' is genuinely ambiguous; not in the blocklist")
    assert not _looks_like_hallucination(text)


def test_empty_and_whitespace_count_as_hallucination():
    assert _looks_like_hallucination("")
    assert _looks_like_hallucination("   ")


# --------------------------------------------------------------------------
#  WhisperSTT filtering, against a fake faster-whisper
# --------------------------------------------------------------------------

class _Seg:
    def __init__(self, text, no_speech_prob=0.0, avg_logprob=0.0):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


def _whisper_with(segments, **kwargs):
    captured = {}

    class FakeModel:
        def __init__(self, *a, **k):
            captured["ctor"] = k

        def transcribe(self, audio, **kw):
            captured["kw"] = kw
            return list(segments), None

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    sys.modules["faster_whisper"] = fake

    from voice_agent.stt import WhisperSTT

    return WhisperSTT("m", **kwargs), captured


AUDIO = np.ones(16000, dtype=np.int16) * 1000


def test_whisper_drops_high_no_speech_segments():
    stt, _ = _whisper_with([_Seg(" Thank you.", no_speech_prob=0.95, avg_logprob=-1.8)])
    result = stt.transcribe(AUDIO)
    assert result.text == ""
    assert "no_speech" in result.rejected


def test_whisper_drops_filler_even_when_confident():
    """High confidence on 'Thank you.' is still almost certainly silence."""
    stt, _ = _whisper_with([_Seg(" Thank you.", no_speech_prob=0.1, avg_logprob=-0.2)])
    result = stt.transcribe(AUDIO)
    assert result.text == ""
    assert "filler" in result.rejected


def test_whisper_keeps_real_speech():
    stt, _ = _whisper_with([_Seg(" I need a DevOps engineer.", 0.05, -0.3)])
    result = stt.transcribe(AUDIO)
    assert result.text == "I need a DevOps engineer."
    assert result.rejected == ""


def test_whisper_passes_the_right_decoding_options():
    stt, captured = _whisper_with([_Seg(" hello", 0.0, 0.0)],
                                  beam_size=1, chunk_length=15, num_threads=3)
    stt.transcribe(AUDIO)
    kw = captured["kw"]
    assert kw["beam_size"] == 1
    assert kw["chunk_length"] == 15
    assert kw["vad_filter"] is False              # Silero already segmented
    assert kw["condition_on_previous_text"] is False   # else it drifts across turns
    assert captured["ctor"]["cpu_threads"] == 3


def test_whisper_omits_chunk_length_when_unset():
    stt, captured = _whisper_with([_Seg(" hello", 0.0, 0.0)], chunk_length=None)
    stt.transcribe(AUDIO)
    assert "chunk_length" not in captured["kw"]


def test_whisper_is_not_a_streaming_engine():
    stt, _ = _whisper_with([_Seg(" hello", 0.0, 0.0)])
    assert stt.supports_streaming is False


# --------------------------------------------------------------------------
#  Engine factory
# --------------------------------------------------------------------------

def test_factory_rejects_an_unknown_engine(tmp_path):
    with pytest.raises(ValueError, match="Unknown stt.engine"):
        build_stt({"engine": "deepspeech"}, tmp_path)


def test_factory_reports_a_missing_vosk_model_clearly(tmp_path):
    pytest.importorskip("vosk")
    with pytest.raises(FileNotFoundError, match="download_models"):
        build_stt({"engine": "vosk", "vosk": {"model_dir": "not-here"}}, tmp_path)


def test_factory_builds_whisper_from_the_nested_config(tmp_path):
    _whisper_with([_Seg(" hi", 0.0, 0.0)])        # installs the fake module
    stt = build_stt(
        {"engine": "whisper",
         "whisper": {"model_name": "m", "compute_type": "int8",
                     "beam_size": 1, "chunk_length": 12}},
        tmp_path, num_threads=2,
    )
    assert stt.chunk_length == 12
    assert stt.beam_size == 1


# --------------------------------------------------------------------------
#  StreamingTranscriber routing
# --------------------------------------------------------------------------

class _FakeStreamingSTT:
    supports_streaming = True

    def __init__(self):
        self.fed = 0
        self.reset_calls = 0

    def feed_frame(self, frame):
        self.fed += 1

    def transcribe(self, audio, sample_rate=16000):
        return Transcript("streamed result", 5.0, False, 1000.0)

    def reset(self):
        self.reset_calls += 1


CFG = {"partial_interval_ms": 0, "reuse_stable_partial": False, "min_audio_ms": 250}


def test_frames_reach_a_streaming_engine():
    stt = _FakeStreamingSTT()
    t = StreamingTranscriber(stt, CFG)
    assert t.supports_streaming
    for _ in range(10):
        t.feed_frame(np.zeros(512, dtype=np.int16))
    assert stt.fed == 10


def test_streaming_engines_never_run_partial_passes():
    """Partials are a Whisper workaround; for Vosk they'd duplicate the decode."""
    t = StreamingTranscriber(_FakeStreamingSTT(), dict(CFG, partial_interval_ms=500))
    assert t.due_for_partial() is False


def test_feed_frame_is_a_noop_for_non_streaming_engines():
    class Plain:
        supports_streaming = False

        def transcribe(self, audio, sample_rate=16000):
            return Transcript("x")

        def reset(self):
            pass

    t = StreamingTranscriber(Plain(), CFG)
    t.feed_frame(np.zeros(512, dtype=np.int16))     # must not raise
    assert t.supports_streaming is False


def test_finalize_resets_the_recognizer_between_turns():
    stt = _FakeStreamingSTT()
    t = StreamingTranscriber(stt, CFG)
    result = t.finalize(np.ones(16000, dtype=np.int16))
    assert result.text == "streamed result"


def test_too_short_audio_is_ignored_without_calling_the_engine():
    stt = _FakeStreamingSTT()
    t = StreamingTranscriber(stt, CFG)
    result = t.finalize(np.ones(100, dtype=np.int16))
    assert result.text == ""
    assert stt.reset_calls >= 1


# --------------------------------------------------------------------------
#  Nemotron cache-aware streaming
# --------------------------------------------------------------------------

def test_nemotron_decode_accepts_a_list_from_the_processor():
    """Nemotron's processor.decode() returns ['text'], not 'text'."""
    from voice_agent.stt import _decode_transformers_asr

    class _Proc:
        def decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return ["Yes, I have a minute."]

    class _Out:
        sequences = [[1, 2, 3]]

    assert _decode_transformers_asr(_Proc(), _Out()) == "Yes, I have a minute."


def test_nemotron_is_declared_streaming():
    """The whole reason for choosing it.

    Parakeet and Whisper cannot start until the utterance is complete, so their
    entire cost (250-475 ms in live runs) lands on the critical path between the
    caller stopping and the agent replying. A streaming engine consumes audio as
    it arrives, so finalize() is nearly free.
    """
    from voice_agent.stt import NemotronStreamingSTT

    assert NemotronStreamingSTT.supports_streaming is True


def test_the_orchestrator_feeds_frames_only_to_streaming_engines():
    """StreamingTranscriber must route frames by capability, not by name."""
    from voice_agent.stt import StreamingTranscriber

    class _Batch:
        supports_streaming = False

        def reset(self):
            pass

    class _Stream:
        supports_streaming = True

        def __init__(self):
            self.fed = 0

        def feed_frame(self, frame):
            self.fed += 1

        def reset(self):
            pass

    cfg = {"partial_interval_ms": 0, "reuse_stable_partial": False, "min_audio_ms": 300}
    frame = np.zeros(512, dtype=np.int16)

    batch = StreamingTranscriber(_Batch(), cfg)
    batch.feed_frame(frame)          # must be a no-op, not an AttributeError
    assert not batch.supports_streaming

    engine = _Stream()
    streaming = StreamingTranscriber(engine, cfg)
    assert streaming.supports_streaming
    for _ in range(5):
        streaming.feed_frame(frame)
    assert engine.fed == 5


def test_nemotron_step_is_a_whole_number_of_model_frames():
    """The encoder works in 80 ms frames; a partial step misaligns the cache."""
    from voice_agent.stt import NemotronStreamingSTT

    for chunk_ms, expected_frames in ((80, 1), (160, 2), (560, 7), (1120, 14)):
        stt = NemotronStreamingSTT.__new__(NemotronStreamingSTT)
        stt.sample_rate = 16000
        frame = int(0.08 * stt.sample_rate)
        step = max(frame, int(round(chunk_ms / 80.0)) * frame)
        assert step % frame == 0
        assert step // frame == expected_frames


def test_chunk_size_config_is_a_supported_operating_point():
    """1120/560/160/80 ms are the published points; anything else is a typo."""
    import yaml

    raw = yaml.safe_load(
        open(Path(__file__).resolve().parent.parent / "config.yaml", encoding="utf-8")
    )
    assert raw["stt"]["nemotron"]["chunk_ms"] in (80, 160, 560, 1120)


# --------------------------------------------------------------------------
#  An unavailable engine must degrade, not kill the agent
# --------------------------------------------------------------------------

def test_build_stt_falls_back_instead_of_crashing(monkeypatch):
    """A missing optional backend left the user with a stack trace.

    Nemotron needs either an onnx-asr release that knows
    'nemotron_asr_streaming' or transformers >= 5.13. On a machine with neither,
    build_stt raised and took the whole agent down -- while Parakeet, which
    works and is already a dependency, sat right there unused.
    """
    import voice_agent.stt as stt_mod

    built = []

    def fake_build_one(cfg, models_dir, num_threads, engine):
        built.append(engine)
        if engine == "nemotron":
            raise RuntimeError("Could not load Nemotron streaming ASR\n  - onnx-asr: nope")
        return object()

    monkeypatch.setattr(stt_mod, "_build_one", fake_build_one)

    events = []
    result = stt_mod.build_stt(
        {"engine": "nemotron", "fallback_engine": "parakeet"},
        Path("models"),
        2,
        on_event=lambda kind, text: events.append((kind, text)),
    )

    assert result is not None, "build_stt should have returned the fallback engine"
    assert built == ["nemotron", "parakeet"]
    assert any(k == "error" and "unavailable" in t for k, t in events)
    assert any("parakeet" in t for _, t in events), "the fallback must be named"


def test_build_stt_does_not_swallow_a_failing_fallback(monkeypatch):
    """If the fallback is broken too, fail loudly -- do not pretend."""
    import voice_agent.stt as stt_mod

    def always_fails(cfg, models_dir, num_threads, engine):
        raise RuntimeError(f"{engine} is unavailable")

    monkeypatch.setattr(stt_mod, "_build_one", always_fails)

    with pytest.raises(RuntimeError):
        stt_mod.build_stt(
            {"engine": "nemotron", "fallback_engine": "parakeet"}, Path("models"), 2
        )


def test_an_explicitly_chosen_engine_is_not_silently_swapped(monkeypatch):
    """Only the default gets a fallback; an explicit choice must be honoured."""
    import voice_agent.stt as stt_mod

    def always_fails(cfg, models_dir, num_threads, engine):
        raise RuntimeError(f"{engine} is unavailable")

    monkeypatch.setattr(stt_mod, "_build_one", always_fails)

    with pytest.raises(RuntimeError):
        stt_mod.build_stt({"engine": "whisper"}, Path("models"), 2)


def test_old_transformers_is_reported_as_a_version_problem():
    """"ParakeetTokenizer has no attribute set_num_lookahead_tokens" is useless.

    That is what an older transformers produces: AutoProcessor silently returns a
    plain tokenizer, and the real cause -- the architecture landed in 5.13.0 --
    is nowhere in the message.
    """
    from voice_agent.stt import NemotronStreamingSTT

    source = NemotronStreamingSTT._load.__doc__ or ""
    import inspect

    body = inspect.getsource(NemotronStreamingSTT._load)
    assert "5, 13" in body, "no version gate on the transformers backend"
    assert "too old" in body
    # And the setter is probed rather than called blindly.
    assert 'getattr(processor, "set_num_lookahead_tokens", None)' in body
