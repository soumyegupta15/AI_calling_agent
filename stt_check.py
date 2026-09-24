"""Measure transcription accuracy against known ground truth.

Run this instead of a full call when the agent "isn't hearing properly". It
drives the EXACT production path -- same config.yaml, same MicStream gain, same
Silero VAD gates, same Parakeet model -- but instead of a free-form
conversation it asks you to say eight specific phrases whose correct transcript
is known, and then reports where each one was lost.

For every phrase it prints three numbers that together localise the fault:

  vad_ms    Length of the segment the VAD handed over. If this never appears
            at all, the utterance never opened a turn -- vad.min_speech_ms is
            the gate that ate it, and stt.min_audio_ms is irrelevant.
  clipped%  Share of samples pinned to the int16 rail. Anything above ~1% means
            audio.mic_gain is too high and the waveform is distorted. This is
            the usual cause of "wrong words" that stay wrong every run.
  heard     What Parakeet actually returned.

If a phrase is dropped, the row says which gate dropped it, so you know which
single setting to change instead of changing three and hoping.

Usage:
    python stt_check.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from voice_agent.audio_io import MicStream
from voice_agent.config import (
    load_config,
    model_path,
    physical_core_count,
    plan_threads,
)
from voice_agent.stt import StreamingTranscriber, build_stt
from voice_agent.vad import SileroVAD, TurnDetector, VADEvent

# (what to say, why that phrase) -- chosen to hit the failures actually seen in
# logs: very short replies, the employer sentence, and the place name that kept
# coming back as "Manipal University jacket".
PROMPTS = [
    ("Yes.", "shortest possible reply"),
    ("Okay.", "short reply"),
    ("Five.", "single-word number"),
    ("No.", "shortest negative"),
    ("I work at Infosys.", "employer claim"),
    ("I'm from Jaipur.", "the place name that kept mangling"),
    ("Bachelor's in computer science.", "qualification"),
    ("Around five years.", "experience"),
]

RAIL = 32700  # int16 peak is 32767; treat anything this close as clipped


def _norm(text: str) -> list[str]:
    """Lowercase, drop punctuation, collapse whitespace, split to words.

    Apostrophes are deleted rather than turned into spaces, so "bachelor's"
    and "bachelors" normalise to the same single token -- otherwise every
    contraction scores as an extra substitution and inflates the WER.
    """
    cleaned = text.lower().replace("'", "").replace("\u2019", "")
    chars = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in cleaned]
    return " ".join("".join(chars).split()).split()


def _wer(expected: str, heard: str) -> float:
    """Word error rate, 0.0 = perfect. Pure-python edit distance on words."""
    ref, hyp = _norm(expected), _norm(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Classic Levenshtein over word lists.
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i]
        for j, hw in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[len(hyp)] / len(ref)


def main() -> int:
    cfg = load_config()
    rt = plan_threads(physical_core_count())
    sr = int(cfg.audio["input_sample_rate"])
    frame_samples = int(cfg.audio["frame_samples"])

    print("=" * 74)
    print("STT accuracy check -- production path, config.yaml as-is")
    print("=" * 74)

    engine = str(cfg.stt.get("engine", "?")).lower()
    on_event = lambda kind, msg: print(f"  [{kind}] {msg}")

    print("Loading Silero VAD...")
    vad = SileroVAD(
        model_path(cfg, cfg.vad["model_file"]),
        sample_rate=sr,
        num_threads=int(rt["vad_threads"]),
    )
    detector = TurnDetector(
        vad,
        cfg.vad,
        cfg.barge_in,
        frame_samples=frame_samples,
        sample_rate=sr,
    )

    print(f"Loading STT ({engine})...")
    stt = build_stt(cfg.stt, cfg.paths["models_dir"], num_threads=int(rt["stt_threads"]), on_event=on_event)
    transcriber = StreamingTranscriber(stt, cfg.stt, sample_rate=sr)

    min_audio_ms = float(cfg.stt.get("min_audio_ms", 250))
    min_speech_ms = float(cfg.vad.get("min_speech_ms", 250))
    min_samples = transcriber.min_samples

    frames: list[np.ndarray] = []
    result: dict = {"turn": None}

    def on_frame(frame: np.ndarray) -> None:
        res = detector.process(frame)
        if res.event in (VADEvent.SPEECH_END, VADEvent.SPEECH_TIMEOUT):
            result["turn"] = res
        elif res.event == VADEvent.SPEECH_START:
            frames.clear()
        elif detector.is_speaking():
            frames.append(frame)

    mic = MicStream(
        on_frame=on_frame,
        sample_rate=sr,
        frame_samples=frame_samples,
        device=cfg.audio.get("input_device"),
        gain=float(cfg.audio.get("mic_gain", 1.0)),
    )
    mic.start()
    print(f"\nMic up: {mic.device!r}  gain={mic.gain}x  engine={engine}")
    print(
        f"Gates:  vad.min_speech_ms={min_speech_ms:.0f}  "
        f"stt.min_audio_ms={min_audio_ms:.0f} ({min_samples} samples)\n"
    )

    rows = []
    try:
        for idx, (phrase, why) in enumerate(PROMPTS, start=1):
            input(f"[{idx}/{len(PROMPTS)}] Say: \"{phrase}\"   ({why})\n       then press Enter > ")
            result["turn"] = None
            frames.clear()
            detector.reset()
            transcriber.reset()

            print("       listening...", end="", flush=True)
            deadline = time.time() + 12.0
            while result["turn"] is None and time.time() < deadline:
                time.sleep(0.02)
            print("\r", end="")

            res = result["turn"]
            if res is None or res.audio is None or len(res.audio) == 0:
                print("       DROPPED -- VAD never opened a turn")
                print("                  -> vad.min_speech_ms is the suspect\n")
                rows.append((phrase, None, None, None, "dropped-vad"))
                continue

            audio = res.audio
            dur_ms = 1000.0 * len(audio) / sr
            peak = float(np.abs(audio).max())
            clipped = float(np.mean(np.abs(audio) >= RAIL)) * 100.0

            if len(audio) < min_samples:
                print(f"       DROPPED -- {dur_ms:.0f} ms < min_audio_ms {min_audio_ms:.0f}")
                print(f"                  peak={peak:.0f} clipped={clipped:.2f}%")
                print("                  -> stt.min_audio_ms is the suspect\n")
                rows.append((phrase, dur_ms, clipped, None, "dropped-short"))
                continue

            tr = transcriber.finalize(audio)
            heard = (tr.text or "").strip()
            if not heard:
                print(f"       DROPPED -- Parakeet returned nothing ({dur_ms:.0f} ms)")
                print(f"                  peak={peak:.0f} clipped={clipped:.2f}%")
                if tr.rejected:
                    print(f"                  rejected: {tr.rejected}")
                print("                  -> the audio reached STT; suspect gain/room noise\n")
                rows.append((phrase, dur_ms, clipped, None, "dropped-stt"))
                continue

            err = _wer(phrase, heard)
            print(f"       heard: \"{heard}\"")
            print(
                f"       vad={dur_ms:.0f} ms  peak={peak:.0f}/32767  "
                f"clipped={clipped:.2f}%  WER={err * 100:.0f}%\n"
            )
            rows.append((phrase, dur_ms, clipped, err, "ok", heard))
    except KeyboardInterrupt:
        print("\n(stopped early)")
    finally:
        mic.stop()

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)

    got = [r for r in rows if r[4] == "ok"]
    drop_vad = [r for r in rows if r[4] == "dropped-vad"]
    drop_short = [r for r in rows if r[4] == "dropped-short"]
    drop_stt = [r for r in rows if r[4] == "dropped-stt"]

    print(f"Transcribed:      {len(got)}/{len(rows)}")
    print(f"Lost at VAD:      {len(drop_vad)}   (vad.min_speech_ms too high)")
    print(f"Lost as too short:{len(drop_short)}   (stt.min_audio_ms too high)")
    print(f"Lost inside STT:  {len(drop_stt)}   (audio arrived, nothing decoded)")

    if got:
        wers = [r[3] for r in got]
        clips = [r[2] for r in got]
        print(f"\nMean WER:         {sum(wers) / len(wers) * 100:.0f}%")
        print(f"Max clipping seen:{max(clips):.2f}%")
        if max(clips) > 1.0:
            print(
                "  ^ CLIPPING. The waveform is saturating, which mangles words no\n"
                "    matter what the model does. LOWER audio.mic_gain."
            )
        elif max(clips) < 0.01 and max(r[1] for r in got) > 0:
            print(
                "  No clipping, so gain is not the problem. If words are still\n"
                "  wrong, the signal is too quiet -- raise audio.mic_gain."
            )

    worst = sorted([r for r in got if r[3] > 0], key=lambda r: -r[3])
    if worst:
        print("\nWorst mismatches:")
        for phrase, _, _, err, _, heard in worst[:5]:
            print(f"   {err * 100:>3.0f}%  said \"{phrase}\"  ->  heard \"{heard}\"")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
