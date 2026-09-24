"""Piper text-to-speech.

Synthesizes one sentence-chunk at a time so playback of chunk 1 overlaps
generation of chunk 2. Text goes through normalize_for_tts() first, otherwise
staffing shorthand like "C2H at $85-95/hr" is read out as letters and digits.

Piper was chosen over Kokoro after measurement, not preference: on a single core
Piper synthesized a short first chunk in ~120 ms against Kokoro's ~2000 ms, with
a realtime factor of 0.15 vs 2.69. A realtime factor above 1.0 means synthesis is
slower than playback, which makes streaming impossible. Kokoro is the nicer
voice; it needs more cores than this pipeline targets.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .text_chunker import normalize_for_tts


@dataclass
class Speech:
    samples: np.ndarray
    sample_rate: int
    latency_ms: float
    text: str
    spoken_text: str

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    @property
    def realtime_factor(self) -> float:
        """< 1.0 means we synthesize faster than playback consumes it."""
        return (self.latency_ms / 1000.0) / max(1e-6, self.duration_s)


def _load_piper_voice_capped(model_path: Path, threads: int):
    """Load a Piper voice with ONNX Runtime intra-op threads actually capped.

    PiperVoice.load() builds its own onnxruntime.InferenceSession internally
    with a bare `onnxruntime.SessionOptions()` -- no threads argument, no way
    to inject one. Setting OMP_NUM_THREADS/MKL_NUM_THREADS before calling
    .load() (the previous approach here) does not fix this: modern pip
    onnxruntime wheels use their own internal thread pool, not OpenMP, so
    those env vars are silently ignored for the default CPU execution
    provider. The result was Piper grabbing every core on every synthesis
    call regardless of tts_threads, confirmed live by first-audio latency
    sitting at 400-660ms instead of the ~120ms this project measured Piper
    capable of.

    This mirrors PiperVoice.load()'s own internal logic (see piper/voice.py)
    but builds the InferenceSession ourselves with intra_op_num_threads
    actually set, then constructs PiperVoice directly -- the same pattern
    KokoroTTS already uses correctly via Kokoro.from_session() below.
    """
    import json

    import onnxruntime as ort
    from piper.config import PiperConfig
    from piper.phonemize_espeak import ESPEAK_DATA_DIR
    from piper.voice import PiperVoice

    config_path = f"{model_path}.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )

    return PiperVoice(
        config=PiperConfig.from_dict(config_dict),
        session=session,
        espeak_data_dir=Path(ESPEAK_DATA_DIR),
        download_dir=Path.cwd(),
    )


class PiperTTS:
    def __init__(self, model_path: str | Path, cfg, num_threads: int = 2) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {model_path}. "
                "Run: python scripts/download_models.py --only piper"
            )
        try:
            from piper.voice import PiperVoice  # noqa: F401  (import-check only)
        except ImportError as exc:  # pragma: no cover
            raise ImportError("piper-tts is not installed. Run: pip install piper-tts") from exc

        threads = max(1, int(num_threads))
        self.voice = _load_piper_voice_capped(model_path, threads)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.cfg = cfg

    def _pcm(self, spoken: str) -> bytes:
        """piper-tts renamed its synthesis API at 1.3; support both shapes."""
        if hasattr(self.voice, "synthesize_stream_raw"):        # piper-tts < 1.3
            return b"".join(self.voice.synthesize_stream_raw(spoken))
        chunks = []
        for chunk in self.voice.synthesize(spoken):             # piper-tts >= 1.3
            data = getattr(chunk, "audio_int16_bytes", None)
            if data is None:
                data = getattr(chunk, "audio_int16_array", chunk)
                data = np.asarray(data, dtype=np.int16).tobytes()
            chunks.append(data)
        return b"".join(chunks)

    def synthesize(self, text: str) -> Speech | None:
        """Blocking synthesis of one chunk. Call from an executor thread."""
        spoken = normalize_for_tts(text)
        if not spoken or not any(c.isalnum() for c in spoken):
            return None
        started = time.perf_counter()
        pcm = self._pcm(spoken)
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return Speech(samples, self.sample_rate, latency_ms, text, spoken)

    def warmup(self) -> float:
        """First call pays graph init; do it before the call, not during."""
        started = time.perf_counter()
        try:
            self.synthesize("Ready.")
        except Exception:
            pass
        return (time.perf_counter() - started) * 1000.0

    @property
    def available_voices(self) -> list[str]:
        return [str(self.cfg.get("voice", ""))]


class KokoroTTS:
    """Kokoro-82M via ONNX Runtime. Much better voice, much more compute.

    Kokoro is worth the trouble on a machine that can feed it: it is the best
    open TTS in this size class by a wide margin. But it is roughly an order of
    magnitude more work per second of audio than Piper, and what matters for a
    live call is not quality-per-second, it is the REALTIME FACTOR -- synthesis
    time divided by the duration of audio produced. Above 1.0 the synthesiser
    cannot keep ahead of the speaker and streaming is impossible; you get a
    stall mid-sentence.

    Measured on this project: Piper ~0.15, Kokoro ~2.69 on a single core. Public
    int8 ONNX numbers put Kokoro near ~1100 ms for a short phrase on a modern
    8-core laptop chip. So it is a genuine upgrade from about 4 physical cores
    up, and unusable below that -- which is exactly what `warmup()` measures
    rather than assumes.
    """

    def __init__(
        self,
        model_path: str | Path,
        voices_path: str | Path,
        cfg,
        num_threads: int = 2,
    ) -> None:
        model_path, voices_path = Path(model_path), Path(voices_path)
        for path, what in ((model_path, "model"), (voices_path, "voices file")):
            if not path.exists():
                raise FileNotFoundError(
                    f"Kokoro {what} not found at {path}. "
                    "Run: python scripts/download_models.py --only kokoro"
                )
        try:
            import onnxruntime as rt
            from kokoro_onnx import Kokoro
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "kokoro-onnx is not installed. Run:\n"
                "  pip install kokoro-onnx espeakng-loader"
            ) from exc

        threads = max(1, int(num_threads))
        # kokoro-onnx's default session ignores thread caps; build our own so ORT
        # does not grab every core and starve llama.cpp during the overlap window.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = rt.InferenceSession(
            str(model_path),
            opts,
            providers=["CPUExecutionProvider"],
        )
        self.kokoro = Kokoro.from_session(sess, str(voices_path))
        self.cfg = cfg
        self.voice = str(cfg.get("voice", "af_heart"))
        self.speed = float(cfg.get("speed", 1.0))
        self.lang = str(cfg.get("lang", "en-us"))
        self.sample_rate = int(cfg.get("sample_rate", 24000))
        self.measured_rtf: float | None = None

    def synthesize(self, text: str) -> Speech | None:
        spoken = normalize_for_tts(text)
        if not spoken or not any(c.isalnum() for c in spoken):
            return None
        started = time.perf_counter()
        samples, sample_rate = self.kokoro.create(
            spoken, voice=self.voice, speed=self.speed, lang=self.lang
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples = np.asarray(samples, dtype=np.float32)
        self.sample_rate = int(sample_rate)
        return Speech(samples, self.sample_rate, latency_ms, text, spoken)

    def measure_rtf(self, phrase: str = "Ready when you are.") -> float | None:
        """Synthesise twice; return the best (lowest) realtime factor."""
        best: float | None = None
        for _ in range(2):
            speech = self.synthesize(phrase)
            if speech is None:
                continue
            rtf = speech.realtime_factor
            best = rtf if best is None else min(best, rtf)
        self.measured_rtf = best
        return best

    def warmup(self) -> float:
        started = time.perf_counter()
        try:
            self.measure_rtf()
        except Exception:
            pass
        return (time.perf_counter() - started) * 1000.0

    @property
    def available_voices(self) -> list[str]:
        try:
            return sorted(self.kokoro.get_voices())
        except Exception:
            return [self.voice]


def _load_kokoro(cfg, models_dir: Path, num_threads: int) -> KokoroTTS:
    kokoro_cfg = cfg.get("kokoro", {}) or {}
    return KokoroTTS(
        models_dir / kokoro_cfg.get("model_file", "kokoro-v1.0.int8.onnx"),
        models_dir / kokoro_cfg.get("voices_file", "voices-v1.0.bin"),
        {**cfg, **kokoro_cfg},
        num_threads,
    )


def confirm_kokoro_streaming(
    tts: KokoroTTS,
    cfg,
    models_dir: Path,
    num_threads: int = 2,
    on_event=None,
) -> KokoroTTS | PiperTTS:
    """Measure Kokoro after the rest of the stack has warmed up.

    Measuring while llama-server is still loading the LLM skews the realtime
    factor high and triggers a Piper fallback on machines that can feed Kokoro
    once the CPU settles.
    """
    say = on_event or (lambda kind, text: None)
    ceiling = float(cfg.get("max_realtime_factor", 0.6))
    fallback = bool(cfg.get("fallback_to_piper", True))
    rtf = tts.measure_rtf()
    if rtf is None:
        if fallback:
            say("error", "Kokoro warmup failed; falling back to Piper.")
            return PiperTTS(models_dir / cfg["model_file"], cfg, num_threads)
        say("error", "Kokoro warmup failed.")
        return tts
    if rtf <= ceiling:
        say("system", f"Kokoro realtime factor {rtf:.2f} -- streaming OK.")
        return tts
    if not fallback:
        say(
            "system",
            f"Kokoro realtime factor {rtf:.2f} (target < {ceiling:.2f}) -- "
            "keeping Kokoro as configured; long replies may pause briefly on this CPU.",
        )
        return tts
    say(
        "error",
        f"Kokoro measured realtime factor {rtf:.2f} (needs < {ceiling:.2f} "
        "to stay ahead of playback on this machine) -- falling back to "
        "Piper. Set tts.fallback_to_piper: false to keep Kokoro, "
        "tts.engine: piper to silence this, or run on a machine with more "
        "physical cores.",
    )
    return PiperTTS(models_dir / cfg["model_file"], cfg, num_threads)


def build_tts(cfg, models_dir: Path, num_threads: int = 2, on_event=None):
    """Construct the TTS engine. Kokoro streaming is confirmed after LLM warmup."""
    models_dir = Path(models_dir)
    engine = str(cfg.get("engine", "piper")).lower()
    say = on_event or (lambda kind, text: None)

    if engine == "kokoro":
        try:
            return _load_kokoro(cfg, models_dir, num_threads)
        except Exception as exc:
            if bool(cfg.get("fallback_to_piper", True)):
                say("error", f"Kokoro unavailable ({exc}); falling back to Piper.")
            else:
                raise

    return PiperTTS(models_dir / cfg["model_file"], cfg, num_threads)
