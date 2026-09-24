"""Entry point.

    python run_agent.py                  live mic/speaker voice call
    python run_agent.py --text           typed input, spoken output (no mic)
    python run_agent.py --list-devices   show audio devices, then exit
    python run_agent.py --check          verify models + deps, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.config import load_config, model_path  # noqa: E402

# ANSI colors work in Windows Terminal and PowerShell 7; harmless elsewhere.
_COLORS = {
    "system": "\033[90m",
    "vad": "\033[90m",
    "partial": "\033[36m",
    "user": "\033[1;36m",
    "agent": "\033[1;32m",
    "bargein": "\033[1;33m",
    "metrics": "\033[90m",
    "note": "\033[33m",
    "error": "\033[1;31m",
}
_RESET = "\033[0m"

_LABELS = {
    "user": "You   ",
    "agent": "Riya  ",
    "partial": "  ... ",
    "bargein": "BARGE ",
    # Always shown. A turn that produces no reply has to explain itself, or it
    # is indistinguishable from the agent having hung.
    "note": "  --> ",
    "error": "ERROR ",
}


def make_printer(verbose: bool):
    def emit(kind: str, text: str) -> None:
        if not verbose and kind in ("vad", "system", "metrics"):
            if kind == "metrics":
                pass  # metrics are useful even when not verbose
            else:
                return
        color = _COLORS.get(kind, "")
        if kind == "metrics":
            print(f"{color}{text}{_RESET}")
        elif kind == "partial":
            # Overwrite in place so partials don't spam the console.
            print(f"\r{color}{_LABELS['partial']}{text}{_RESET}", end="", flush=True)
        else:
            label = _LABELS.get(kind, f"{kind:<6}")
            print(f"\r{color}{label}| {text}{_RESET}")

    return emit


def preflight(cfg) -> list[str]:
    """Return a list of problems; empty means good to go."""
    problems: list[str] = []
    models_dir = Path(cfg.paths.models_dir)

    required = [
        (cfg.vad["model_file"], "Silero VAD"),
        (cfg.llm["model_file"], f"GGUF model ({cfg.llm['model_file']})"),
        # Piper is always required: it is the fallback whenever the chosen TTS
        # engine turns out to be too slow to stream on this machine.
        (cfg.tts["model_file"], "Piper voice (fallback)"),
        (cfg.tts["voices_file"], "Piper voice config"),
    ]
    for filename, label in required:
        if not (models_dir / filename).exists():
            problems.append(f"missing {label}: {models_dir / filename}")

    engine = str(cfg.stt.get("engine", "nemotron")).lower()
    modules = [
        ("sounddevice", "pip install sounddevice"),
        ("onnxruntime", "pip install onnxruntime"),
        ("piper", "pip install piper-tts"),
    ]
    if str(cfg.tts.get("engine", "piper")).lower() == "kokoro":
        modules.append(("kokoro_onnx", "pip install kokoro-onnx espeakng-loader"))
    if engine == "nemotron":
        # Either loader is acceptable; only complain if BOTH are missing.
        try:
            __import__("onnx_asr")
        except Exception:
            try:
                __import__("transformers")
            except Exception:
                problems.append(
                    "nemotron STT needs a loader  ->  "
                    'pip install "onnx-asr[cpu,hub]"  (or "transformers>=5.13.0")'
                )
    elif engine == "parakeet":
        modules.append(("onnx_asr", 'pip install "onnx-asr[cpu,hub]"'))
    elif engine == "vosk":
        modules.append(("vosk", "pip install vosk"))
        model_dir = Path(cfg.stt["vosk"]["model_dir"])
        if not model_dir.is_absolute():
            model_dir = models_dir / model_dir
        if not model_dir.exists():
            problems.append(
                f"missing Vosk model: {model_dir}"
                "  ->  python scripts/download_models.py --only vosk"
            )
    else:
        modules.append(("faster_whisper", "pip install faster-whisper"))

    # A GGUF that stopped mid-download looks present but is truncated, and
    # llama-server then fails with an unhelpful parse error.
    gguf = models_dir / cfg.llm["model_file"]
    if gguf.exists() and gguf.stat().st_size < 100_000_000:
        problems.append(
            f"{gguf.name} is only {gguf.stat().st_size / 1e6:.0f} MB -- looks like a "
            "truncated download. Delete it and re-run scripts/download_models.py"
        )

    binary = Path(cfg.llm["server"]["binary"])
    if not binary.exists():
        problems.append(
            f"missing llama-server binary: {binary}"
            "  ->  python scripts/download_models.py --only llamacpp"
        )

    for module, hint in modules:
        try:
            __import__(module)
        except ImportError:
            problems.append(f"missing package {module}  ->  {hint}")
        except Exception as exc:
            # e.g. sounddevice raises OSError when PortAudio itself is missing.
            problems.append(f"package {module} failed to load: {exc}")

    return problems


async def run_voice(agent, emit) -> None:
    emit("system", "Listening. Speak normally; interrupt any time. Ctrl+C to hang up.")
    try:
        await agent.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await agent.shutdown()


async def run_text(agent, emit) -> None:
    emit("system", "Text mode. Type a message and press Enter. Blank line or Ctrl+C to exit.")
    agent.speaker.start()
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, input, "\nYou   | ")
            if not line.strip():
                break
            m = await agent.say_text(line.strip())
            emit("agent", m.response or "(no response)")
            if agent.cfg.agent.get("show_metrics", True):
                emit("metrics", m.render())
            agent.speaker.wait_until_drained(timeout=30)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        agent.speaker.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", action="store_true", help="typed input instead of a microphone")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check", action="store_true", help="preflight only")
    parser.add_argument("--stt-engine",
                        choices=["nemotron", "parakeet", "vosk", "whisper"], default=None,
                        help="override stt.engine")
    parser.add_argument("--tts-engine", choices=["kokoro", "piper", "auto"], default=None,
                        help="override tts.engine")
    parser.add_argument("--stt", default=None, help="override the Whisper model name")
    parser.add_argument("--threads", type=int, default=None, help="override llm.n_threads")
    parser.add_argument("--no-greeting", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        from voice_agent.audio_io import list_devices

        print(list_devices())
        return 0

    cfg = load_config(args.config)
    if args.stt_engine:
        cfg["stt"]["engine"] = args.stt_engine
    if args.tts_engine:
        cfg["tts"]["engine"] = args.tts_engine
        cfg["tts"].pop("_auto_selected", None)
    if args.stt:
        cfg["stt"].setdefault("whisper", {})["model_name"] = args.stt
    if args.threads:
        cfg["llm"]["n_threads"] = args.threads
    if args.no_greeting:
        cfg["agent"]["greeting"] = ""

    rt = cfg.runtime
    print(
        f"Hardware: {rt['physical_cores']} physical / {rt['logical_cores']} logical cores, "
        f"{rt['ram_gb']:.0f} GB RAM  ->  tier '{rt['tier']}'"
    )
    print(
        f"Threads: llm {cfg.llm['n_threads']} gen / {cfg.llm['n_threads_batch']} batch, "
        f"stt {rt['stt_threads']}, tts {rt['tts_threads']}, vad {rt['vad_threads']}"
    )
    engine = str(cfg.stt.get("engine", "nemotron")).lower()
    if engine == "nemotron":
        detail = (f"{cfg.stt['nemotron']['model_name']} "
                  f"({cfg.stt['nemotron']['quantization']}, "
                  f"{cfg.stt['nemotron']['chunk_ms']}ms chunks, streaming)")
    elif engine == "parakeet":
        detail = f"{cfg.stt['parakeet']['model_name']} ({cfg.stt['parakeet']['quantization']})"
    elif engine == "vosk":
        detail = cfg.stt["vosk"]["model_dir"]
    else:
        detail = f"{cfg.stt['whisper']['model_name']} ({cfg.stt['whisper']['compute_type']})"
    tts_engine = str(cfg.tts.get("engine", "piper")).lower()
    auto = "  (auto-selected)" if cfg.tts.get("_auto_selected") else ""
    print(f"STT {engine}: {detail}")
    print(f"LLM {cfg.llm['model_file']}  |  TTS {tts_engine}{auto}")

    from voice_agent.metrics import set_engine_labels

    llm_label = cfg.llm["model_file"].split("-Instruct")[0].split(".gguf")[0][:18] or "LLM"
    set_engine_labels(stt=engine.capitalize(), llm=llm_label, tts=tts_engine.capitalize())

    problems = preflight(cfg)
    if problems:
        print("\nPreflight failed:")
        for p in problems:
            print(f"  - {p}")
        print("\nRun: python scripts/download_models.py   (and see README.md for packages)")
        return 1
    if args.check:
        print("Preflight OK.")
        return 0

    from voice_agent.orchestrator import VoiceAgent

    emit = make_printer(args.verbose)
    agent = VoiceAgent(cfg, on_event=emit)
    agent.load()

    runner = run_text if args.text else run_voice
    try:
        asyncio.run(runner(agent, emit))
    except KeyboardInterrupt:
        pass

    print(agent.metrics.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
