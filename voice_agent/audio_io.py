"""Microphone capture and interruptible speaker playback (sounddevice / WASAPI).

Two objects:

  MicStream      16 kHz mono int16, fixed 512-sample frames, delivered to a callback
                 from the PortAudio thread. 512 samples is what Silero VAD v5
                 requires, so no re-blocking happens anywhere downstream.

  SpeakerStream  24 kHz mono float32, fed from a deque of numpy chunks. Playback is
                 driven by a PortAudio callback pulling from that deque, so
                 `interrupt()` is O(1): clear the deque and the very next callback
                 (<= 20 ms later) outputs silence. That is what makes the brief's
                 "stop within ~200 ms" requirement easy rather than tight.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


def _resolve_device(device):
    """Accept None (default), an int index, or a case-insensitive name substring."""
    if device is None or isinstance(device, int):
        return device
    needle = str(device).lower()
    for idx, info in enumerate(sd.query_devices()):
        if needle in info["name"].lower():
            return idx
    raise ValueError(f"No audio device matching {device!r}")


def _device_candidates(device) -> list:
    """Input-device indices to try, most preferred first.

    A pinned index is a trap. Windows renumbers devices whenever a Bluetooth
    headset connects or drops, so an index that pointed at a microphone last
    session can point at a speaker today. Asking a speaker for one input
    channel is what produces PortAudioError -9998, "Invalid number of
    channels" -- the device is fine, the index is stale.

    So the configured device is a preference, not a contract: if it will not
    open, fall back to the system default, then to anything that reports an
    input channel. A name substring is safer than an index for the same
    reason, and both are accepted.
    """
    candidates = []
    if device is not None:
        if isinstance(device, int):
            candidates.append(int(device))
        else:
            needle = str(device).lower()
            candidates += [i for i, info in enumerate(sd.query_devices())
                           if needle in info["name"].lower()]

    # sd.default.device is an (input, output) pair, but not a tuple subclass,
    # so an isinstance check misses it -- just index it defensively.
    default = sd.default.device
    try:
        default = default[0]
    except (TypeError, IndexError, KeyError):
        default = None
    try:
        default = int(default) if default is not None and int(default) >= 0 else None
    except (TypeError, ValueError):
        default = None

    # Everything else with an input channel, split by expected capture quality.
    #
    # An HFP headset caps audio at a narrowband compressed codec and produced
    # garbage transcripts here ("Febrics, fazer o Febrics"); "Sound Mapper" is
    # a virtual endpoint that just forwards to whatever is default, so it
    # inherits the same problem. Anything under 44.1 kHz is the same story.
    # These are last resorts, not equal candidates -- deprioritised rather than
    # excluded, so a headset still gets used if it is the only microphone.
    good, poor = [], []
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] <= 0:
            continue
        name = info["name"].lower()
        narrowband = float(info["default_samplerate"]) < 44100
        if narrowband or any(t in name for t in ("headset", "hands-free", "sound mapper")):
            poor.append(idx)
        else:
            good.append(idx)

    # Honour the system default only when it is a genuinely good capture
    # device. If Windows has defaulted to a Bluetooth headset, a real mic is
    # the better guess even though it is not the default.
    if default in good:
        good.insert(0, default)
    candidates += good + poor
    if default is not None and default not in good:
        candidates.append(default)

    seen, ordered = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _device_label(idx) -> str:
    """'8 (Microphone Array (AMD Audio Device))' -- readable, and safe if stale."""
    try:
        return f"{idx} ({sd.query_devices()[idx]['name']})"
    except Exception:
        return str(idx)


class MicStream:
    """Continuous 16 kHz mono capture in fixed 512-sample int16 frames.

    `gain` applies software amplification to compensate for a physically
    quiet microphone (e.g. a laptop array mic still producing a weak
    signal even with Windows input volume maxed out). Applied in float32
    with clipping to the int16 range so amplified peaks saturate cleanly
    instead of wrapping around. 1.0 = no change; try 4.0-8.0 for a mic
    that reads well under 5% RMS even when speaking normally.
    """

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None],
        sample_rate: int = 16000,
        frame_samples: int = 512,
        device=None,
        gain: float = 1.0,
    ) -> None:
        self.on_frame = on_frame
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        try:
            self.device = _resolve_device(device)
        except ValueError:
            # A name that is not on this machine (unplugged, renamed, or a
            # stale config). Keep the raw value so start() can fall back and
            # say which device it wanted instead of dying here.
            self.device = device
        self.gain = gain
        self._stream: Optional[sd.InputStream] = None
        self._overflows = 0

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status and status.input_overflow:
            self._overflows += 1
        # indata is (frames, 1) int16; blocksize is pinned so frames == frame_samples.
        frame = indata[:, 0]
        if self.gain != 1.0:
            # Amplify in float32, then clip back to int16 range before
            # casting -- avoids silent integer wraparound on loud peaks.
            boosted = frame.astype(np.float32) * self.gain
            np.clip(boosted, -32768, 32767, out=boosted)
            frame = boosted.astype(np.int16)
        self.on_frame(frame.copy())

    def start(self) -> None:
        """Open the first input device that accepts 16 kHz mono.

        self.device is updated to whatever actually opened, so the caller can
        report it. If nothing opens, every attempt is listed -- a bare
        PortAudio error code does not say which device was at fault, and with
        a stale index the number in the config is the thing that is wrong.
        """
        attempts = []
        for candidate in _device_candidates(self.device):
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.frame_samples,
                    device=candidate,
                    channels=1,
                    dtype="int16",
                    latency="low",
                    callback=self._callback,
                )
                self._stream.start()
                self.device = candidate
                return
            except Exception as exc:  # PortAudioError and friends
                attempts.append(f"{_device_label(candidate)}: {str(exc).splitlines()[0]}")
        raise RuntimeError(
            f"No input device would open at {self.sample_rate} Hz mono. Tried:\n  "
            + "\n  ".join(attempts or ["<no input devices reported>"])
        )

    @property
    def device_label(self) -> str:
        """The device actually in use, for the startup log line."""
        return "not started" if self._stream is None else _device_label(self.device)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def overflow_count(self) -> int:
        return self._overflows


class SpeakerStream:
    """Queued, interruptible playback of float32 mono audio at 24 kHz."""

    def __init__(
        self,
        sample_rate: int = 24000,
        blocksize: int = 480,
        device=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = _resolve_device(device)
        self._buf: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._pending_samples = 0
        # Set while any queued audio is still being drained.
        self._speaking = threading.Event()
        # Fires each time the queue empties -- used to know a turn finished cleanly.
        self._drained = threading.Event()
        self._drained.set()
        self._interrupted = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self.device,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ---------------------------------------------------------------- callback
    def _callback(self, outdata, frames, time_info, status):  # noqa: ANN001
        filled = 0
        with self._lock:
            while filled < frames and self._buf:
                chunk = self._buf[0]
                take = min(len(chunk), frames - filled)
                outdata[filled : filled + take, 0] = chunk[:take]
                filled += take
                self._pending_samples -= take
                if take == len(chunk):
                    self._buf.popleft()
                else:
                    self._buf[0] = chunk[take:]
            empty = not self._buf

        if filled < frames:
            outdata[filled:, 0] = 0.0

        if empty:
            if self._speaking.is_set():
                self._speaking.clear()
                self._drained.set()

    # ------------------------------------------------------------------- write
    def play(self, samples: np.ndarray) -> None:
        """Enqueue mono float32 audio. Returns immediately."""
        if samples is None or len(samples) == 0:
            return
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        # Cheap clip guard against any TTS overshoot on plosives.
        np.clip(samples, -1.0, 1.0, out=samples)
        with self._lock:
            self._buf.append(samples)
            self._pending_samples += len(samples)
            self._interrupted = False
        self._drained.clear()
        self._speaking.set()

    def interrupt(self) -> int:
        """Drop all queued audio immediately. Returns samples discarded."""
        with self._lock:
            dropped = self._pending_samples
            self._buf.clear()
            self._pending_samples = 0
            self._interrupted = True
        self._speaking.clear()
        self._drained.set()
        return dropped

    # ------------------------------------------------------------------ status
    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def pending_seconds(self) -> float:
        return self._pending_samples / float(self.sample_rate)

    @property
    def was_interrupted(self) -> bool:
        return self._interrupted

    def wait_until_drained(self, timeout: float | None = None) -> bool:
        return self._drained.wait(timeout)


def list_devices() -> str:
    """Human-readable device table, for `python run_agent.py --list-devices`.

    The last column is a live probe, not an inference. A device that reports
    input channels can still refuse 16 kHz mono -- WASAPI shared mode will not
    resample for you, and WDM-KS endpoints reject most formats outright -- and
    only actually opening it tells you which. That column is the one to read
    when picking an input_device.
    """
    lines = []
    for idx, info in enumerate(sd.query_devices()):
        probe = "-"
        if info["max_input_channels"] > 0:
            try:
                stream = sd.InputStream(samplerate=16000, blocksize=512, device=idx,
                                        channels=1, dtype="int16", latency="low")
                stream.close()
                probe = "OK 16 kHz mono"
            except Exception as exc:
                probe = "no: " + str(exc).splitlines()[0].split("[")[0].strip()[:44]
        lines.append(
            f"{idx:>3}  in={info['max_input_channels']} out={info['max_output_channels']}"
            f"  sr={float(info['default_samplerate']):>6.0f}"
            f"  {info['name'][:44]:<44}  {probe}"
        )
    return "\n".join(lines)
