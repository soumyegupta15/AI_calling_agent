"""Speech-to-text with two interchangeable engines.

  WhisperSTT  faster-whisper / CTranslate2, int8. More accurate, but Whisper pads
              every clip to a fixed mel window, so a 1 s utterance costs about the
              same as a 10 s one -- a flat tax on every turn that you cannot
              overlap with anything.

  VoskSTT     Kaldi-based streaming recognizer. Consumes audio AS IT ARRIVES, so
              by the time the caller stops talking the decode is essentially done
              and finalize() is nearly free. Less accurate than Whisper on hard
              audio, dramatically better for turn latency.

Select with `stt.engine: whisper | vosk`, or `run_agent.py --stt-engine vosk`.

Both expose the same interface:

    supports_streaming -> bool
    feed_frame(int16)  -> None      (streaming engines only)
    transcribe(int16)  -> Transcript
    reset()            -> None

Whisper hallucination filtering
-------------------------------
Whisper reliably invents text from silence or near-silence -- "Thank you.",
"Thanks for watching!", "Bye." are its favourites. On a live call this happens
constantly, because the VAD occasionally opens on a cough or on the agent's own
voice leaking through the speakers. Those turns then get sent to the LLM as if
the caller had said something. We drop them using the model's own no_speech_prob
and avg_logprob, plus a blocklist of the canonical phrases.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Transcript:
    text: str
    latency_ms: float = 0.0
    from_cache: bool = False
    audio_ms: float = 0.0
    rejected: str = ""          # non-empty when we dropped a likely hallucination


# Canonical Whisper hallucinations on silence. Compared after lowercasing and
# stripping punctuation, so "Thank you." and "thank you" both match.
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thanks for watching!", "thank you for watching", "bye", "bye bye", "goodbye",
    "you", "yeah", "oh", "hmm", "mm", "mhm", "uh", "um", "so", ".", "!", "?",
    "please subscribe", "subscribe", "the end", "music", "applause",
    "thank you for your attention", "see you next time", "okay", "ok",
}


# What a NON-hallucinating engine should drop: nothing but punctuation, or a
# bare disfluency. Emphatically NOT "yeah" / "okay" / "yes" -- on a call where
# the agent asks a question every turn, those are the answer. Dropping them at
# the STT layer meant the caller replied and the agent sat in silence.
_NOISE_ONLY = {"uh", "um", "erm", "mm", "hm", "hmm", "ah", "eh", "."}


def _looks_like_hallucination(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned in _HALLUCINATIONS or not cleaned


def _is_noise_only(text: str) -> bool:
    """Minimal filter for engines that do not invent speech from silence."""
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return not cleaned or cleaned in _NOISE_ONLY


def _decode_transformers_asr(processor, generated) -> str:
    """Turn Nemotron/Parakeet generate() output into one plain string.

    Nemotron's processor.decode() returns a list (e.g. ['Hello there.']), not a
    str. The old code called .strip() on that list, hit AttributeError, and the
    bare except in _transcribe_whole turned every turn into '' -- which is why
    the agent logged 'no speech recognized' on every VAD segment.
    """
    token_ids = getattr(generated, "sequences", generated)
    decoded = processor.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0] if decoded else ""
    return str(decoded or "").strip()


class WhisperSTT:
    """faster-whisper (CTranslate2), int8, with hallucination filtering."""

    supports_streaming = False

    def __init__(
        self,
        model_name: str = "Systran/faster-whisper-base.en",
        compute_type: str = "int8",
        num_threads: int = 2,
        beam_size: int = 1,
        chunk_length: int | None = None,
        no_speech_threshold: float = 0.6,
        min_avg_logprob: float = -1.0,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "faster-whisper is not installed. Run:\n  pip install faster-whisper"
            ) from exc

        self.model_name = model_name
        self.beam_size = int(beam_size)
        self.chunk_length = int(chunk_length) if chunk_length else None
        self.no_speech_threshold = float(no_speech_threshold)
        self.min_avg_logprob = float(min_avg_logprob)
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=int(num_threads),
            num_workers=1,
        )
        self._lock = threading.Lock()

    def reset(self) -> None:
        pass

    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> Transcript:
        audio_ms = 1000.0 * len(audio_int16) / sample_rate
        if len(audio_int16) < sample_rate // 10:
            return Transcript("", 0.0, False, audio_ms)

        audio = audio_int16.astype(np.float32) / 32768.0
        extra = {"chunk_length": self.chunk_length} if self.chunk_length else {}
        started = time.perf_counter()
        with self._lock:
            segments, _info = self.model.transcribe(
                audio,
                language="en",
                beam_size=self.beam_size,
                # Silero already segmented the turn; a second VAD would only clip
                # the edges of what we deliberately captured.
                vad_filter=False,
                # Without this the model carries context between turns and drifts,
                # inventing continuations of the previous sentence.
                condition_on_previous_text=False,
                without_timestamps=True,
                no_speech_threshold=self.no_speech_threshold,
                **extra,
            )

            kept, dropped = [], []
            for seg in segments:
                no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
                logprob = getattr(seg, "avg_logprob", 0.0) or 0.0
                if no_speech > self.no_speech_threshold and logprob < self.min_avg_logprob:
                    dropped.append(f"{seg.text.strip()!r} (no_speech={no_speech:.2f})")
                    continue
                kept.append(seg.text)
            text = " ".join(kept).strip()

        latency_ms = (time.perf_counter() - started) * 1000.0

        if text and _looks_like_hallucination(text):
            return Transcript("", latency_ms, False, audio_ms,
                              rejected=f"{text!r} (canonical Whisper filler)")
        if not text and dropped:
            return Transcript("", latency_ms, False, audio_ms, rejected="; ".join(dropped))
        return Transcript(text, latency_ms, False, audio_ms)


# Back-compat alias; the class is no longer distil-specific.
DistilWhisperSTT = WhisperSTT


class ParakeetSTT:
    """NVIDIA Parakeet TDT 0.6B v3, int8 ONNX, via the onnx-asr runtime.

    This is the accuracy pick. Parakeet tops the Open ASR leaderboard and beats
    Whisper large-v3 on word error rate at a quarter of the size, while running
    several times faster than Whisper on CPU. Two properties matter here beyond
    raw WER:

      * It is a transducer over a conv encoder, so cost scales with the actual
        length of the audio. Whisper pads everything to a fixed 30 s mel window,
        which is why a one-second "yes" cost the same as a ten-second sentence.
      * It does not hallucinate on silence. Whisper's "Thank you." / "Thanks for
        watching!" problem simply does not occur, so a mis-triggered VAD produces
        an empty transcript instead of a phantom turn.

    Not streaming: like Whisper it needs the finished utterance. Unlike Whisper,
    a short utterance is genuinely cheap.
    """

    supports_streaming = False

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: str = "int8",
        num_threads: int = 2,
    ) -> None:
        try:
            import onnxruntime as ort

            import onnx_asr
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "onnx-asr is not installed. Run:\n"
                '  pip install "onnx-asr[cpu,hub]"'
            ) from exc

        # onnx-asr builds its own sessions, so pass thread limits explicitly
        # rather than relying on an OMP_NUM_THREADS env var that may already be
        # set by another import.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(num_threads)
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model_name = model_name
        self.quantization = quantization
        try:
            self.model = onnx_asr.load_model(
                model_name,
                quantization=quantization,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {model_name!r} ({quantization}). The weights are "
                "fetched from Hugging Face on first use (~670 MB for int8); run\n"
                "  python scripts/download_models.py --only parakeet\n"
                f"Underlying error: {exc}"
            ) from exc
        self._lock = threading.Lock()

    def reset(self) -> None:
        pass

    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> Transcript:
        audio_ms = 1000.0 * len(audio_int16) / sample_rate
        if len(audio_int16) < sample_rate // 10:
            return Transcript("", 0.0, False, audio_ms)

        audio = audio_int16.astype(np.float32) / 32768.0
        started = time.perf_counter()
        with self._lock:
            text = (self.model.recognize(audio, sample_rate=sample_rate) or "").strip()
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Parakeet does not invent speech from silence, so it must NOT get
        # Whisper's blocklist. That list contains "yeah", "okay" and "you" --
        # perfectly good answers to a recruiter's question, which were being
        # thrown away before the orchestrator ever saw them. Only strip results
        # that carry no content at all.
        if text and _is_noise_only(text):
            return Transcript("", latency_ms, False, audio_ms, rejected=f"{text!r} (no content)")
        return Transcript(text, latency_ms, False, audio_ms)


class NemotronStreamingSTT:
    """NVIDIA Nemotron cache-aware streaming ASR (FastConformer-RNNT, 0.6B).

    This is the one engine here that is genuinely STREAMING, and that is the
    whole point of it. Parakeet and Whisper cannot start until the utterance is
    finished, so their entire cost lands on the critical path between the caller
    stopping and the agent replying -- 250-475 ms in live runs. Cache-aware
    streaming consumes audio as it arrives, reusing cached encoder state instead
    of recomputing an overlapping window, so when end-of-speech fires the decode
    is essentially already done and finalize() is nearly free.

    Accuracy is not the trade-off it usually is with streaming models. NVIDIA
    report 6.93% average WER on the Open ASR Leaderboard sets at a 1.12 s chunk
    and 7.07% at 0.56 s, with punctuation and capitalisation built in.

    `chunk_ms` picks the point on the latency/accuracy curve at RUNTIME -- no
    retraining, and no reload. Smaller chunks respond sooner and cost accuracy:
    1120 ms -> 6.93%, 560 ms -> 7.07%, 160 ms -> 7.67%, 80 ms -> 8.43%. For a
    phone call 560 ms is the sweet spot, because the VAD is already holding
    400 ms of silence before it declares the turn over.

    Backends are tried in order of how well they suit a small CPU box. The
    weights are ~630 MB int8 / q8_0.
    """

    supports_streaming = True

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        model_path: str | Path | None = None,
        quantization: str = "int8",
        chunk_ms: int = 560,
        num_threads: int = 2,
        sample_rate: int = 16000,
    ) -> None:
        self.model_name = model_name
        self.quantization = quantization
        self.chunk_ms = int(chunk_ms)
        self.sample_rate = int(sample_rate)
        self.num_threads = int(num_threads)
        self.backend = ""
        self._lock = threading.Lock()
        self._pending: list[np.ndarray] = []
        self._fed_samples = 0
        self._text_so_far = ""

        # Samples per streaming step. The model works in 80 ms frames, so round
        # to a whole number of them or the cache lines up wrong.
        frame = int(0.08 * self.sample_rate)
        self._step_samples = max(frame, int(round(self.chunk_ms / 80.0)) * frame)

        self._impl = self._load(model_path)

    # ------------------------------------------------------------------ load
    def _load(self, model_path: str | Path | None):
        errors: list[str] = []

        # 1. onnx-asr, if a release ever ships this architecture. Cheapest
        #    dependency by far and matches how Parakeet is already loaded.
        try:
            import onnx_asr
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.num_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model = onnx_asr.load_model(
                self.model_name,
                path=str(model_path) if model_path else None,
                quantization=self.quantization,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.backend = "onnx-asr"
            return model
        except Exception as exc:
            errors.append(f"onnx-asr: {exc}")

        # 2. HF Transformers. This architecture landed in 5.13.0; older releases
        #    fall back to a plain ParakeetTokenizer, and the failure surfaces
        #    much later as "ParakeetTokenizer has no attribute
        #    set_num_lookahead_tokens", which says nothing about the real cause.
        #    Check the version up front and say what is actually wrong.
        try:
            import transformers

            version = getattr(transformers, "__version__", "0")
            major, minor = (int(p) for p in (version.split(".") + ["0", "0"])[:2])
            if (major, minor) < (5, 13):
                raise RuntimeError(
                    f"transformers {version} is too old for this architecture "
                    "(needs >= 5.13.0)"
                )

            from transformers import AutoModelForRNNT, AutoProcessor

            processor = AutoProcessor.from_pretrained(self.model_name)
            # Lookahead tokens ARE the chunk size: each is one 80 ms frame.
            # lookahead = max(0, self._step_samples // int(0.08 * self.sample_rate) - 1)
            lookahead = max(0, self._step_samples // int(0.08 * self.sample_rate) - 1)
            setter = getattr(processor, "set_num_lookahead_tokens", None)
            if not callable(setter):
                raise RuntimeError(
                    f"{type(processor).__name__} has no streaming API -- this build of "
                    "transformers loaded a plain tokenizer instead of the Nemotron "
                    "processor"
                )
            setter(lookahead)
            model = AutoModelForRNNT.from_pretrained(self.model_name)
            model.eval()
            self.backend = "transformers"
            return (processor, model)
        except Exception as exc:
            errors.append(f"transformers: {exc}")

        raise RuntimeError(
            "Could not load Nemotron streaming ASR through any backend.\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nOptions, cheapest first:\n"
            "  1. Stay on Parakeet. It is batch-only but fast and accurate, and it\n"
            "     is what stt.fallback_engine already does automatically.\n"
            "  2. pip install \"transformers>=5.13.0\" torch --index-url \\\n"
            "       https://download.pytorch.org/whl/cpu\n"
            "     NOTE: this backend decodes the finished utterance, so you get the\n"
            "     accuracy but NOT the streaming latency win, and it pulls ~2.5 GB\n"
            "     of dependencies.\n"
            "  3. NeMo-Speech.cpp with the q8_0 GGUF -- the only lightweight route\n"
            "     to real cache-aware streaming on CPU, and the closest analogue to\n"
            "     how llama.cpp is already used here:\n"
            "       https://github.com/NVIDIA/NeMo-Speech.cpp\n"
            "  4. Wait for onnx-asr to add 'nemotron_asr_streaming' support.\n"
        )

    # ------------------------------------------------------------- streaming
    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._fed_samples = 0
            self._text_so_far = ""
            reset = getattr(self._impl, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass

    def feed_frame(self, frame_int16: np.ndarray) -> None:
        """Consume one 32 ms VAD frame while the caller is still speaking.

        Frames are buffered until a whole streaming step is available, then the
        encoder advances by exactly that step. This is what keeps finalize()
        cheap: by end-of-speech only a partial step is left to flush.
        """
        with self._lock:
            self._pending.append(np.asarray(frame_int16, dtype=np.int16))
            self._fed_samples += len(frame_int16)
            buffered = sum(len(f) for f in self._pending)
            if buffered < self._step_samples:
                return
            block = np.concatenate(self._pending)
            self._pending = [block[self._step_samples:]] if len(block) > self._step_samples else []
            self._advance(block[: self._step_samples])

    def _advance(self, block: np.ndarray) -> None:
        """One streaming step. Must not raise -- a dropped step is recoverable."""
        try:
            audio = block.astype(np.float32) / 32768.0
            if self.backend == "onnx-asr":
                step = getattr(self._impl, "recognize_stream", None) or getattr(
                    self._impl, "accept_waveform", None
                )
                if step is None:
                    return
                out = step(audio, sample_rate=self.sample_rate)
                if isinstance(out, str) and out:
                    self._text_so_far = (self._text_so_far + " " + out).strip()
            # The transformers path decodes in finalize(); its streaming
            # generator wants the whole feature stream, not incremental calls.
        except Exception:
            pass

    def partial_text(self) -> str:
        with self._lock:
            return self._text_so_far.strip()

    # ------------------------------------------------------------------ final
    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> Transcript:
        audio_ms = 1000.0 * len(audio_int16) / sample_rate
        started = time.perf_counter()

        with self._lock:
            streamed = self._text_so_far.strip()

        # If streaming already produced the text, finalizing is free -- this is
        # the entire latency argument for this engine.
        if streamed and self._fed_samples >= 0.8 * len(audio_int16):
            self.reset()
            latency_ms = (time.perf_counter() - started) * 1000.0
            if _is_noise_only(streamed):
                return Transcript("", latency_ms, True, audio_ms, rejected=f"{streamed!r} (no content)")
            return Transcript(streamed, latency_ms, True, audio_ms)

        text = self._transcribe_whole(audio_int16, sample_rate)
        self.reset()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if text and _is_noise_only(text):
            return Transcript("", latency_ms, False, audio_ms, rejected=f"{text!r} (no content)")
        return Transcript(text, latency_ms, False, audio_ms)

    def _transcribe_whole(self, audio_int16: np.ndarray, sample_rate: int) -> str:
        """Fallback: decode the buffered utterance in one pass."""
        if len(audio_int16) < sample_rate // 10:
            return ""
        audio = audio_int16.astype(np.float32) / 32768.0
        try:
            if self.backend == "onnx-asr":
                return (self._impl.recognize(audio, sample_rate=sample_rate) or "").strip()
            processor, model = self._impl
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                return_dict_in_generate=True,
            )
            return _decode_transformers_asr(processor, out)
        except Exception as exc:
            # Previously a bare `except Exception: return ""` here silently
            # ate every failure (including the .decode()-returns-a-list bug
            # described above) and surfaced only as "no speech recognized"
            # on the caller's side, with zero way to tell WHY. Print it so
            # the real cause is visible instead of guessed at.
            import traceback
            print(f"[nemotron _transcribe_whole] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return ""


class VoskSTT:
    """Streaming Kaldi recognizer. Decodes while the caller is still speaking.

    This is the structural fix for turn latency: Whisper cannot start until the
    utterance is complete, so its cost lands entirely on the critical path. Vosk
    consumes 32 ms frames as the VAD produces them, so when speech ends the
    remaining work is just flushing the final hypothesis.
    """

    supports_streaming = True

    def __init__(self, model_path: str | Path, sample_rate: int = 16000, log_level: int = -1) -> None:
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover
            raise ImportError("vosk is not installed. Run:\n  pip install vosk") from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found at {model_path}. "
                "Run: python scripts/download_models.py --only vosk"
            )

        vosk.SetLogLevel(log_level)          # silence Kaldi's stderr chatter
        self._vosk = vosk
        self.model_path = model_path
        self.model = vosk.Model(str(model_path))
        self.sample_rate = int(sample_rate)
        self._rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self._rec.SetWords(False)
        self._lock = threading.Lock()
        self._fed_samples = 0
        self._stream_started = 0.0

    # ------------------------------------------------------------- streaming
    def reset(self) -> None:
        with self._lock:
            self._rec = self._vosk.KaldiRecognizer(self.model, self.sample_rate)
            self._rec.SetWords(False)
            self._fed_samples = 0
            self._stream_started = 0.0

    def feed_frame(self, frame_int16: np.ndarray) -> None:
        """Push one frame while the caller is still talking. Cheap (~1 ms)."""
        with self._lock:
            if self._stream_started == 0.0:
                self._stream_started = time.perf_counter()
            self._rec.AcceptWaveform(frame_int16.astype(np.int16).tobytes())
            self._fed_samples += len(frame_int16)

    def partial_text(self) -> str:
        with self._lock:
            try:
                return json.loads(self._rec.PartialResult()).get("partial", "").strip()
            except Exception:
                return ""

    # ---------------------------------------------------------------- final
    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> Transcript:
        """Finalize. Feeds anything not already streamed, then flushes."""
        audio_ms = 1000.0 * len(audio_int16) / sample_rate
        started = time.perf_counter()
        with self._lock:
            # Frames the orchestrator never streamed (batch mode, or the pre-roll).
            if self._fed_samples < len(audio_int16):
                tail = audio_int16[self._fed_samples:]
                if len(tail):
                    self._rec.AcceptWaveform(tail.astype(np.int16).tobytes())
            try:
                text = json.loads(self._rec.FinalResult()).get("text", "").strip()
            except Exception:
                text = ""
            self._rec = self._vosk.KaldiRecognizer(self.model, self.sample_rate)
            self._rec.SetWords(False)
            self._fed_samples = 0
            self._stream_started = 0.0

        latency_ms = (time.perf_counter() - started) * 1000.0
        # Vosk, like Parakeet, does not hallucinate speech out of silence, so it
        # gets the minimal filter rather than Whisper's blocklist -- which would
        # otherwise discard "yeah" and "okay", i.e. most answers to a yes/no
        # question.
        if text and _is_noise_only(text):
            return Transcript("", latency_ms, False, audio_ms, rejected=f"{text!r} (no content)")
        return Transcript(text, latency_ms, False, audio_ms)


# ==========================================================================
#  Factory
# ==========================================================================

def build_stt(cfg, models_dir: Path, num_threads: int = 2, on_event=None):
    """Construct the engine named by cfg['engine'], degrading rather than dying.

    An unavailable optional backend must not take the whole agent down. Nemotron
    needs either an onnx-asr release that knows the architecture or
    transformers >= 5.13; on a machine with neither, hard-failing here left the
    user with a stack trace and no phone agent, when Parakeet was sitting right
    there and works. Same policy as TTS, which measures Kokoro and quietly drops
    to Piper.
    """
    say = on_event or (lambda kind, text: None)
    engine = str(cfg.get("engine", "nemotron")).lower()
    fallback = str(cfg.get("fallback_engine", "parakeet")).lower()

    if engine == "nemotron" and fallback and fallback != "nemotron":
        try:
            return _build_one(cfg, models_dir, num_threads, "nemotron")
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc) else exc
            say("error", f"Nemotron STT unavailable: {first}")
            say("error", f"Falling back to stt.fallback_engine '{fallback}'. "
                         "Run with -v for the full explanation.")
            say("system", str(exc))
            return _build_one(cfg, models_dir, num_threads, fallback)

    return _build_one(cfg, models_dir, num_threads, engine)


def _build_one(cfg, models_dir: Path, num_threads: int, engine: str):
    """Construct exactly the engine named. Raises if it cannot be built."""
    engine = str(engine).lower()

    if engine == "nemotron":
        nm = cfg.get("nemotron", {}) or {}
        model_dir = nm.get("model_dir")
        if model_dir:
            model_dir = Path(model_dir)
            if not model_dir.is_absolute():
                model_dir = Path(models_dir) / model_dir
            if not model_dir.exists():
                model_dir = None
        return NemotronStreamingSTT(
            nm.get("model_name", "nvidia/nemotron-speech-streaming-en-0.6b"),
            model_path=model_dir,
            quantization=nm.get("quantization", "int8"),
            chunk_ms=int(nm.get("chunk_ms", 560)),
            num_threads=num_threads,
        )

    if engine == "parakeet":
        pk = cfg.get("parakeet", {}) or {}
        return ParakeetSTT(
            pk.get("model_name", "nemo-parakeet-tdt-0.6b-v3"),
            quantization=pk.get("quantization", "int8"),
            num_threads=num_threads,
        )

    if engine == "vosk":
        vosk_cfg = cfg.get("vosk", {}) or {}
        model_dir = Path(vosk_cfg.get("model_dir", "vosk-model-small-en-us-0.15"))
        if not model_dir.is_absolute():
            model_dir = Path(models_dir) / model_dir
        return VoskSTT(model_dir)

    if engine != "whisper":
        raise ValueError(
            f"Unknown stt.engine {engine!r}; expected 'nemotron', 'parakeet', "
            "'vosk' or 'whisper'"
        )

    whisper_cfg = cfg.get("whisper", {}) or {}
    return WhisperSTT(
        whisper_cfg.get("model_name", "Systran/faster-whisper-base.en"),
        compute_type=whisper_cfg.get("compute_type", "int8"),
        num_threads=num_threads,
        beam_size=whisper_cfg.get("beam_size", 1),
        chunk_length=whisper_cfg.get("chunk_length"),
        no_speech_threshold=whisper_cfg.get("no_speech_threshold", 0.6),
        min_avg_logprob=whisper_cfg.get("min_avg_logprob", -1.0),
    )


class StreamingTranscriber:
    """Wraps an engine; streams frames into it when the engine supports that."""

    def __init__(self, stt, cfg, sample_rate: int = 16000) -> None:
        self.stt = stt
        self.sample_rate = sample_rate
        self.partial_interval_s = float(cfg.get("partial_interval_ms", 0)) / 1000.0
        self.reuse_stable = bool(cfg.get("reuse_stable_partial", False))
        self.min_samples = int(sample_rate * float(cfg.get("min_audio_ms", 250)) / 1000.0)

        self._last_partial_at = 0.0
        self._partials: list[str] = []
        self._cached_text = ""
        self._cached_samples = 0
        self._busy = False

    @property
    def supports_streaming(self) -> bool:
        return bool(getattr(self.stt, "supports_streaming", False))

    def feed_frame(self, frame) -> None:
        """Called per 32 ms frame while the caller speaks. No-op for Whisper."""
        if self.supports_streaming:
            self.stt.feed_frame(frame)

    def reset(self) -> None:
        self._last_partial_at = 0.0
        self._partials.clear()
        self._cached_text = ""
        self._cached_samples = 0
        self._busy = False
        try:
            self.stt.reset()
        except Exception:
            pass

    # ---------------------------------------------------------------- partials
    def due_for_partial(self, now: float | None = None) -> bool:
        if self.supports_streaming or self.partial_interval_s <= 0 or self._busy:
            return False
        now = now if now is not None else time.perf_counter()
        return (now - self._last_partial_at) >= self.partial_interval_s

    def run_partial(self, audio_int16: np.ndarray) -> str:
        self._busy = True
        try:
            if len(audio_int16) < self.min_samples:
                return self._cached_text
            result = self.stt.transcribe(audio_int16, self.sample_rate)
            self._partials.append(result.text)
            self._cached_text = result.text
            self._cached_samples = len(audio_int16)
            return result.text
        finally:
            self._busy = False
            self._last_partial_at = time.perf_counter()

    @property
    def stable(self) -> bool:
        return (
            len(self._partials) >= 2
            and self._partials[-1] == self._partials[-2]
            and bool(self._partials[-1].strip())
        )

    # ------------------------------------------------------------------- final
    def finalize(self, audio_int16: np.ndarray) -> Transcript:
        audio_ms = 1000.0 * len(audio_int16) / self.sample_rate

        if len(audio_int16) < self.min_samples:
            self.reset()
            return Transcript("", 0.0, False, audio_ms)

        if self.reuse_stable and self.stable and not self.supports_streaming:
            new_samples = len(audio_int16) - self._cached_samples
            if new_samples <= int(0.6 * self.sample_rate):
                text = self._cached_text
                self.reset()
                return Transcript(text, 0.0, True, audio_ms)

        result = self.stt.transcribe(audio_int16, self.sample_rate)
        self._partials.clear()
        self._cached_text = ""
        self._cached_samples = 0
        self._busy = False
        return result
