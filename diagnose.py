"""Diagnose a slow or silent voice agent, one stage at a time.

Answers two questions that `run_agent.py` cannot tell you apart:

  1. Is the microphone actually capturing your voice, and does Silero agree that
     it is speech?  (If not, nothing downstream will ever fire.)
  2. Which stage is eating the time?

Run:
    python diagnose.py                 full check, ~2 minutes
    python diagnose.py --mic-only      just the microphone/VAD test
    python diagnose.py --device 2      force an input device index
    python diagnose.py --seconds 8     longer capture window
    python diagnose.py --stt-bench     sweep STT settings, print a speed table
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.config import load_config, model_path  # noqa: E402

BAR = "=" * 74


def head(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def verdict(ok: bool, text: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {text}")


# ==========================================================================
#  1. Environment
# ==========================================================================

def check_environment(cfg) -> None:
    head("1. Environment")
    cores = cfg.runtime["physical_cores"]
    print(f"  Python              {sys.version.split()[0]}")
    print(f"  Physical cores      {cores}")
    print(f"  llama.cpp threads   {cfg.llm['n_threads']}")
    print(f"  Threads             stt {cfg.runtime['stt_threads']}  tts {cfg.runtime['tts_threads']}  vad {cfg.runtime['vad_threads']}")
    print(f"  LLM model           {cfg.llm['model_file']}")
    print(f"  System prompt       {cfg.llm.get('system_prompt', 'compact')}")
    engine = str(cfg.stt.get("engine", "whisper")).lower()
    print(f"  STT engine          {engine}")
    if engine == "vosk":
        print(f"  Vosk model          {cfg.stt['vosk']['model_dir']}")
    else:
        w = cfg.stt["whisper"]
        print(f"  Whisper model       {w['model_name']} ({w['compute_type']}, "
              f"beam {w['beam_size']}, chunk {w.get('chunk_length')})")
    print(f"  TTS engine          piper / {cfg.tts['model_file']}")
    if cores < 4:
        print(f"\n  NOTE: {cores} physical cores is below the 6-8 the latency budget assumes.")
        print("  Expect this run to show the LLM and TTS stages well over target.")


# ==========================================================================
#  2. Audio devices
# ==========================================================================

def check_devices(cfg, forced_device) -> object:
    head("2. Audio devices")
    import sounddevice as sd

    try:
        devices = sd.query_devices()
    except Exception as exc:
        verdict(False, f"could not enumerate audio devices: {exc}")
        return None

    default_in, default_out = sd.default.device
    print("  idx  in/out  name")
    for i, d in enumerate(devices):
        marks = []
        if d["max_input_channels"] > 0:
            marks.append("IN")
        if d["max_output_channels"] > 0:
            marks.append("OUT")
        star = " <- default in" if i == default_in else (" <- default out" if i == default_out else "")
        print(f"  {i:>3}  {'/'.join(marks) or '--':<6} {d['name'][:52]}{star}")

    device = forced_device if forced_device is not None else cfg.audio.get("input_device")
    if device is None:
        device = default_in
    print(f"\n  Using input device: {device}")

    try:
        info = sd.query_devices(device, "input")
        print(f"    {info['name']}")
        print(f"    max input channels {info['max_input_channels']}, "
              f"default rate {info['default_samplerate']:.0f} Hz")
        verdict(info["max_input_channels"] > 0, "device has an input channel")
    except Exception as exc:
        verdict(False, f"device is not usable for input: {exc}")
        return None
    return device


# ==========================================================================
#  3. Microphone + VAD  (the important one)
# ==========================================================================

def check_microphone(cfg, device, seconds: float) -> np.ndarray | None:
    head("3. Microphone capture and voice detection")

    import sounddevice as sd

    from voice_agent.vad import SileroVAD

    sample_rate = int(cfg.audio["input_sample_rate"])
    frame_samples = int(cfg.audio["frame_samples"])

    try:
        vad = SileroVAD(model_path(cfg, cfg.vad["model_file"]), sample_rate=sample_rate)
    except Exception as exc:
        verdict(False, f"could not load Silero VAD: {exc}")
        return None

    threshold = float(cfg.vad["threshold"])
    frames: list[np.ndarray] = []
    probs: list[float] = []

    print(f"\n  Speak normally for {seconds:.0f} seconds, starting NOW.")
    print("  Say something like: 'I need a senior DevOps engineer, contract to hire.'\n")

    def callback(indata, n, t, status):  # noqa: ANN001
        frames.append(indata[:, 0].copy())

    try:
        with sd.InputStream(samplerate=sample_rate, blocksize=frame_samples, device=device,
                            channels=1, dtype="int16", callback=callback):
            start = time.perf_counter()
            while time.perf_counter() - start < seconds:
                elapsed = time.perf_counter() - start
                level = 0.0
                if frames:
                    level = float(np.abs(frames[-1]).mean()) / 32768.0
                bars = int(min(1.0, level * 40) * 40)
                print(f"\r  {elapsed:4.1f}s  level |{'#' * bars:<40}| {level * 100:5.2f}%",
                      end="", flush=True)
                time.sleep(0.05)
    except Exception as exc:
        print()
        verdict(False, f"could not open the input stream: {exc}")
        return None

    print("\n")
    if not frames:
        verdict(False, "no audio frames arrived from the microphone at all")
        return None

    audio = np.concatenate(frames)
    rms = float(np.sqrt(np.mean((audio.astype(np.float64) / 32768.0) ** 2)))
    peak = float(np.max(np.abs(audio))) / 32768.0

    # Silero over the whole capture.
    vad.reset()
    t0 = time.perf_counter()
    n_frames = len(audio) // frame_samples
    for i in range(n_frames):
        probs.append(vad(audio[i * frame_samples:(i + 1) * frame_samples]))
    vad_ms_per_frame = (time.perf_counter() - t0) * 1000.0 / max(1, n_frames)

    speech_frames = sum(1 for p in probs if p >= threshold)
    speech_ratio = speech_frames / max(1, len(probs))

    print(f"  Captured            {len(audio) / sample_rate:.1f}s ({len(probs)} frames)")
    print(f"  RMS level           {rms * 100:.2f}%   peak {peak * 100:.1f}%")
    print(f"  Silero max P(speech){max(probs) if probs else 0:6.2f}   "
          f"mean {statistics.mean(probs) if probs else 0:.2f}")
    print(f"  Frames over {threshold:.2f}     {speech_frames}/{len(probs)}  ({speech_ratio * 100:.0f}%)")
    print(f"  Silero cost         {vad_ms_per_frame:.2f} ms per 32 ms frame")
    print()

    verdict(peak > 0.005, "microphone is producing signal (not silence/muted)")
    verdict(speech_frames > 0, f"Silero detected speech in {speech_frames} frame(s)")
    verdict(vad_ms_per_frame < 10, "Silero is fast enough to keep up with the mic")

    if peak <= 0.005:
        print("\n  -> The microphone is silent. Check Windows Sound settings: the right")
        print("     device is default, it is not muted, and input volume is up. Then")
        print("     re-run with `python diagnose.py --device N` using an index above.")
    elif speech_frames == 0:
        print("\n  -> Audio is arriving but Silero never called it speech. Usually the mic")
        print("     gain is very low. Raise input volume in Windows, or lower")
        print("     `vad.threshold` in config.yaml (try 0.35).")
    elif speech_ratio > 0.9:
        print("\n  -> Almost every frame reads as speech, which usually means background")
        print("     noise. Raise `vad.threshold` (try 0.6) or the agent will never")
        print("     detect end-of-turn and will not respond.")

    return audio


# ==========================================================================
#  4. Stage timings
# ==========================================================================

def check_stt(cfg, audio) -> str | None:
    head("4. Speech to text")
    from voice_agent.stt import build_stt

    try:
        t0 = time.perf_counter()
        stt = build_stt(cfg.stt, Path(cfg.paths.models_dir),
                        num_threads=cfg.runtime["stt_threads"])
        print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        verdict(False, f"could not load the STT engine: {exc}")
        return None

    stt.transcribe(np.zeros(16000, dtype=np.int16))       # warm

    result = stt.transcribe(audio)
    print(f"  Transcript          {result.text!r}")
    print(f"  Latency             {result.latency_ms:.0f} ms for "
          f"{result.audio_ms / 1000:.1f}s of audio   target 50-400 ms")
    verdict(bool(result.text.strip()), "produced a transcript")
    verdict(result.latency_ms <= 400, "STT latency is workable")
    if not result.text.strip():
        print("\n  -> No transcript. If step 3 showed speech, the audio may be too")
        print("     quiet for Whisper. Raise the Windows input volume and retry.")
    return result.text


def check_llm(cfg, prompt: str) -> float | None:
    head("5. Qwen2.5 (language model)")
    from voice_agent.llm import GenerationStats, build_llm

    try:
        t0 = time.perf_counter()
        llm = build_llm(cfg.llm, model_path(cfg, cfg.llm["model_file"]), call=dict(cfg.get("call", {}) or {}))
        print(f"  started in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        verdict(False, f"could not start the LLM: {exc}")
        return None

    try:
        print("  warming up (first run is slower)...")
        llm.warmup()

        llm.add_user(prompt or "I need a senior DevOps engineer, contract to hire.")
        stats = GenerationStats()
        pieces = []
        t0 = time.perf_counter()
        for piece in llm.stream(stats=stats):
            pieces.append(piece)
        total = time.perf_counter() - t0

        reply = "".join(pieces).strip()
        print(f"\n  Prompt              {prompt[:60]!r}")
        print(f"  Reply               {reply[:90]!r}")
        print(f"  Prompt tokens       {stats.prompt_tokens}")
        print(f"  Time to first token {stats.ttft_ms:.0f} ms          target 150-400 ms")
        print(f"  Generation speed    {stats.tokens_per_second:.1f} tok/s ({stats.tokens} tokens)")
        print(f"  Full reply took     {total:.1f}s")
        verdict(stats.ttft_ms <= 400, "time-to-first-token within target")
        verdict(stats.tokens_per_second >= 8, "generation fast enough to stay ahead of speech")
        if stats.tokens_per_second < 8:
            print("\n  -> Below ~8 tok/s the agent cannot generate faster than it speaks,")
            print("     so audio will stutter. Use a smaller model (see --help of")
            print("     scripts/download_models.py --llm-size 0.5b).")
        return stats.ttft_ms
    finally:
        try:
            llm.close()
        except Exception:
            pass


def check_tts(cfg) -> float | None:
    head("6. Piper (text to speech)")
    from voice_agent.tts import build_tts

    try:
        t0 = time.perf_counter()
        tts = build_tts(cfg.tts, Path(cfg.paths.models_dir), num_threads=cfg.runtime["onnx_threads"])
        print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        verdict(False, f"could not load the TTS engine: {exc}")
        return None

    print("  warming up...")
    tts.warmup()

    samples = [
        "Got it.",                                                   # typical first chunk
        "So you need a senior DevOps person, contract to hire.",      # typical middle chunk
    ]
    first_ms = None
    for text in samples:
        speech = tts.synthesize(text)
        if speech is None:
            continue
        if first_ms is None:
            first_ms = speech.latency_ms
        print(f"  {len(text):>3} chars -> {speech.duration_s:4.1f}s audio in "
              f"{speech.latency_ms:6.0f} ms   (realtime factor {speech.realtime_factor:.2f})")

    print("\n  Target for the first chunk: 150-300 ms")
    verdict(first_ms is not None and first_ms <= 500, "first-chunk synthesis is workable")
    print("  Realtime factor under 1.0 means synthesis outruns playback (required).")
    if first_ms and first_ms > 500:
        print("\n  -> Piper is too slow on this CPU. Try a *-low voice instead of")
        print("     *-medium, e.g. en_US-lessac-low.")
    return first_ms


# ==========================================================================

def stt_bench(cfg, audio) -> int:
    """Sweep model / threads / chunk_length and print a table.

    STT dominated the first real measurement (3482 ms of a 4304 ms turn), and the
    right settings are hardware-specific, so measure rather than guess. Uses your
    own recorded audio when available so the accuracy column is meaningful.
    """
    head("STT sweep")
    from voice_agent.stt import build_stt

    cores = int(cfg.runtime["physical_cores"])
    models = [
        "Systran/faster-whisper-tiny.en",
        "Systran/faster-whisper-base.en",
        "Systran/faster-distil-whisper-small.en",
    ]

    # Vosk first: it is the streaming engine and the one the config defaults to,
    # so its number is the bar everything else has to beat.
    try:
        vosk_dir = Path(cfg.paths.models_dir) / cfg.stt["vosk"]["model_dir"]
        vosk = VoskSTT(vosk_dir)
        vosk.transcribe(np.zeros(16000, dtype=np.int16))
        streamed = 0.0
        vosk.reset()
        t0 = time.perf_counter()
        for i in range(len(audio) // 512):
            vosk.feed_frame(audio[i * 512:(i + 1) * 512])
        streamed = (time.perf_counter() - t0) * 1000.0
        result = vosk.transcribe(audio)
        print(f"  {'vosk (streamed)':<40} {'-':>3} {'-':>6} {result.latency_ms:>7.0f}   "
              f"{result.text[:44]!r}")
        print(f"  {'  ^ feeding cost, spread over speech':<40} {'':>3} {'':>6} "
              f"{streamed:>7.0f}   (off the critical path)\n")
    except Exception as exc:
        print(f"  vosk skipped: {exc}\n")
    chunk_lengths = [30, 15, 10, 6]
    thread_counts = sorted({1, cores})

    if audio is None:
        from voice_agent.tts import build_tts

        print("  No microphone audio; synthesizing a test utterance with Piper.")
        tts = build_tts(cfg.tts, Path(cfg.paths.models_dir), num_threads=1)
        speech = tts.synthesize("I need a senior DevOps engineer, contract to hire.")
        duration = len(speech.samples) / speech.sample_rate
        resampled = np.interp(
            np.linspace(0, duration, int(duration * 16000), endpoint=False),
            np.linspace(0, duration, len(speech.samples), endpoint=False),
            speech.samples,
        )
        audio = np.clip(resampled * 32767, -32768, 32767).astype(np.int16)

    print(f"  {len(audio) / 16000:.1f}s of audio, int8, beam 1, {cores} physical core(s)\n")
    print(f"  {'model':<40} {'thr':>3} {'chunk':>6} {'ms':>7}   transcript")
    print("  " + "-" * 96)

    best = None
    for model_name in models:
        for threads in thread_counts:
            try:
                stt = WhisperSTT(model_name, compute_type=cfg.stt["whisper"]["compute_type"],
                                 num_threads=threads, beam_size=1)
            except Exception as exc:
                print(f"  {model_name:<40} skipped: {exc}")
                break
            stt.transcribe(np.zeros(16000, dtype=np.int16))     # warm
            for chunk in chunk_lengths:
                stt.chunk_length = chunk
                result = stt.transcribe(audio)
                short = model_name.split("/")[-1]
                print(f"  {short:<40} {threads:>3} {chunk:>6} {result.latency_ms:>7.0f}   "
                      f"{result.text[:44]!r}")
                if result.text.strip() and (best is None or result.latency_ms < best[0]):
                    best = (result.latency_ms, model_name, threads, chunk)
            del stt

    if best:
        ms, model_name, threads, chunk = best
        print(f"\n  Fastest that still produced text: {ms:.0f} ms")
        print(f"    stt.model_name:   {model_name}")
        print(f"    stt.chunk_length: {chunk}")
        print("\n  Check the transcript column before adopting it -- the fastest row is")
        print("  only useful if it still got your words right.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", type=int, default=None, help="force an input device index")
    parser.add_argument("--seconds", type=float, default=6.0, help="capture window")
    parser.add_argument("--mic-only", action="store_true")
    parser.add_argument("--stt-bench", action="store_true",
                        help="sweep STT model / threads / chunk_length and print a table")
    parser.add_argument("--skip-mic", action="store_true", help="use canned text instead of your voice")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_environment(cfg)

    transcript = "I need a senior DevOps engineer, contract to hire."
    audio = None

    if not args.skip_mic:
        device = check_devices(cfg, args.device)
        if device is not None:
            audio = check_microphone(cfg, device, args.seconds)

    if args.mic_only:
        return 0

    if args.stt_bench:
        return stt_bench(cfg, audio)

    if audio is not None:
        text = check_stt(cfg, audio)
        if text and text.strip():
            transcript = text

    ttft = check_llm(cfg, transcript)
    tts_ms = check_tts(cfg)

    head("Summary")
    if ttft is not None and tts_ms is not None:
        rough = float(cfg.vad["min_silence_ms"]) + 150 + ttft + tts_ms
        print(f"  Rough time from you stopping talking to hearing audio: ~{rough / 1000:.1f}s")
        print("  Definition of done: under 1.3s (excluding the VAD hold)")
        if rough - float(cfg.vad["min_silence_ms"]) > 1300:
            print("\n  Over budget. Next levers: --llm-size 0.5b, a *-low Piper voice,")
            print("  or stt.model_name: openai/whisper-tiny.en")
    print("\n  Paste this whole output back if you want help interpreting it.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
