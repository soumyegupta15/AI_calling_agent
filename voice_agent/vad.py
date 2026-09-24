"""Silero VAD (ONNX) plus turn / barge-in detection.

Two layers:

  SileroVAD      Thin onnxruntime wrapper. Handles both the v5 signature
                 (single `state` tensor) and the older v4 one (`h` + `c`), so a
                 model file grabbed from either release works.

  TurnDetector   Stateful hysteresis on top of the raw probabilities. Emits
                 SPEECH_START / SPEECH_END events, keeps the pre-roll buffer so
                 the first consonant is never clipped, and applies the stricter
                 barge-in thresholds while the agent is talking.

Per the brief (section 6), VAD is used only for end-of-turn and barge-in
detection -- it never gates whether the STT is listening.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


class VADEvent(enum.Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    # Emitted when max_speech_ms is hit; the utterance is force-closed.
    SPEECH_TIMEOUT = "speech_timeout"


@dataclass
class TurnResult:
    event: VADEvent = VADEvent.NONE
    probability: float = 0.0
    audio: np.ndarray | None = None       # int16, only on SPEECH_END/TIMEOUT
    duration_ms: float = 0.0


# Silero v5 conditions each frame on a tail of the previous one. The ONNX graph
# declares input as [None, None], so it accepts a bare 512-sample frame without
# complaint -- and returns near-zero probabilities for even the clearest speech.
# The context must be prepended explicitly: 64 samples at 16 kHz, 32 at 8 kHz.
_CONTEXT_SAMPLES = {16000: 64, 8000: 32}


class SileroVAD:
    """Frame-level speech probability. Requires exactly 512 samples @ 16 kHz."""

    def __init__(self, model_path: str | Path, sample_rate: int = 16000, num_threads: int = 1) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                "Run: python scripts/download_models.py"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The VAD runs 31x/second; sequential execution has lower jitter than
        # spinning up a thread pool per call.
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._v5 = "state" in self._input_names
        self._context_samples = _CONTEXT_SAMPLES.get(sample_rate, 64) if self._v5 else 0
        self._sr_tensor = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, self._context_samples), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame_int16: np.ndarray) -> float:
        """Return P(speech) for one 512-sample int16 frame."""
        x = (frame_int16.astype(np.float32) / 32768.0).reshape(1, -1)
        if self._v5:
            # Prepend the tail of the previous frame, then carry the tail of THIS
            # combined input forward. Omitting this does not error -- it silently
            # returns ~0.0 for everything, including obvious speech.
            x = np.concatenate([self._context, x], axis=1)
            out, self._state = self.session.run(
                None, {"input": x, "state": self._state, "sr": self._sr_tensor}
            )
            self._context = x[:, -self._context_samples:]
        else:
            out, self._h, self._c = self.session.run(
                None, {"input": x, "h": self._h, "c": self._c, "sr": self._sr_tensor}
            )
        return float(np.asarray(out).reshape(-1)[0])


class TurnDetector:
    """Turn segmentation with separate sensitivity for idle vs. agent-speaking."""

    def __init__(self, vad: SileroVAD, vad_cfg, barge_cfg, frame_samples: int = 512, sample_rate: int = 16000):
        self.vad = vad
        self.frame_samples = frame_samples
        self.sample_rate = sample_rate
        self.frame_ms = 1000.0 * frame_samples / sample_rate

        self.threshold = float(vad_cfg["threshold"])
        self.neg_threshold = float(vad_cfg["neg_threshold"])
        self.min_silence_frames = self._frames(vad_cfg["min_silence_ms"])
        self.min_speech_frames = self._frames(vad_cfg["min_speech_ms"])
        self.max_speech_frames = self._frames(vad_cfg["max_speech_ms"])
        self.pad_frames = self._frames(vad_cfg["speech_pad_ms"])

        self.barge_enabled = bool(barge_cfg["enabled"])
        self.barge_threshold = float(barge_cfg["threshold"])
        self.barge_min_frames = self._frames(barge_cfg["min_speech_ms"])
        self.cooldown_frames = self._frames(barge_cfg["cooldown_ms"])

        # The pre-roll must hold speech_pad_ms of audio *in addition to* every
        # frame the trigger consumed while it was making up its mind. It takes
        # min_speech_frames (or barge_min_frames, which is larger) of consecutive
        # speech to fire, and those frames land in the pre-roll first. Sizing the
        # deque at pad_frames alone meant the only thing left in it at trigger
        # time WAS the leading speech -- zero padding, and under the stricter
        # barge-in profile the first ~130 ms of the word was evicted outright.
        # That is why "AWS" arrived as "WS" and "so" as "C".
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, self.pad_frames + max(self.min_speech_frames, self.barge_min_frames) + 1)
        )
        self._speech: list[np.ndarray] = []
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._cooldown = 0
        # Set by the orchestrator so we know which sensitivity profile to use.
        self.agent_speaking = False

    def _frames(self, ms) -> int:
        return max(1, int(round(float(ms) / (1000.0 * self.frame_samples / self.sample_rate))))

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self._speech.clear()
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0

    def start_cooldown(self) -> None:
        """Ignore the mic briefly after we cut playback (speaker ring-out)."""
        self._cooldown = self.cooldown_frames

    def process(self, frame: np.ndarray) -> TurnResult:
        prob = self.vad(frame)

        if self._cooldown > 0:
            self._cooldown -= 1
            self._preroll.append(frame)
            return TurnResult(VADEvent.NONE, prob)

        # While the agent is talking, demand a higher probability sustained for
        # longer -- otherwise the agent's own voice leaking into the mic (or a
        # single "mm-hm" backchannel) would cut it off constantly.
        on_threshold = self.barge_threshold if (self.agent_speaking and self.barge_enabled) else self.threshold
        min_start = self.barge_min_frames if (self.agent_speaking and self.barge_enabled) else self.min_speech_frames

        if not self._triggered:
            self._preroll.append(frame)
            if prob >= on_threshold:
                self._speech_frames += 1
                if self._speech_frames >= min_start:
                    self._triggered = True
                    self._silence_frames = 0
                    # Pre-roll already contains the frames that got us here.
                    self._speech = list(self._preroll)
                    self._preroll.clear()
                    return TurnResult(VADEvent.SPEECH_START, prob)
            else:
                self._speech_frames = 0
            return TurnResult(VADEvent.NONE, prob)

        # --- currently inside an utterance ---
        self._speech.append(frame)

        if prob >= self.neg_threshold:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        total_frames = len(self._speech)
        if total_frames >= self.max_speech_frames:
            return self._close(VADEvent.SPEECH_TIMEOUT, prob)

        if self._silence_frames >= self.min_silence_frames:
            return self._close(VADEvent.SPEECH_END, prob)

        return TurnResult(VADEvent.NONE, prob)

    # ----------------------------------------------------------------- private
    def _close(self, event: VADEvent, prob: float) -> TurnResult:
        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype=np.int16)
        duration_ms = 1000.0 * len(audio) / self.sample_rate
        self._speech.clear()
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._preroll.clear()
        return TurnResult(event, prob, audio, duration_ms)

    # ------------------------------------------------------- live partial view
    @property
    def is_speaking(self) -> bool:
        return self._triggered

    def current_audio(self) -> np.ndarray:
        """Snapshot of the utterance so far, for partial transcription."""
        if not self._speech:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._speech)

    @property
    def current_ms(self) -> float:
        return len(self._speech) * self.frame_ms
