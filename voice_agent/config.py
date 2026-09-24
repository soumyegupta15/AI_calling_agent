"""Configuration loading and hardware detection.

Reads config.yaml into a dotted-access object and resolves anything that has to
be decided at runtime -- most importantly the physical core count for llama.cpp
(brief section 2, rule 3: oversubscribing hyperthreads measurably hurts Q4_K_M
throughput on CPU).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Section(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Section(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def total_ram_gb() -> float:
    """Best-effort physical RAM in GiB. Used only to pick a model tier."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:  # Linux / macOS
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return 8.0          # a safe, unremarkable assumption


def plan_threads(cores: int) -> dict:
    """Split the machine between the stages that actually run at the same time.

    The old code handed each stage its own maximum -- llama.cpp `cores`, STT
    `cores`, TTS `cores//2` -- which on a 2-core laptop promised 5 threads to a
    machine that has 2. Oversubscription does not fail loudly; it just makes
    everything slower and jittery, and it is why Piper was reporting 800-1400 ms
    for a first chunk that should take ~120 ms: it was fighting llama.cpp for the
    same two cores mid-generation.

    What actually overlaps:

      * STT runs WHILE the caller is speaking. Nothing else is running then, so
        it can be generous -- but with a cache-aware streaming model the cost is
        a few ms per 32 ms frame, so it needs very little.
      * The LLM runs after end-of-speech, alone at first.
      * TTS starts as soon as chunk 1 of the reply exists and then overlaps the
        rest of generation. This is the ONE contended window, so the budget that
        matters is llm + tts <= cores.

    Above 8 threads llama.cpp stops scaling on small models and starts losing to
    synchronisation overhead, so generation threads are capped.
    """
    cores = max(1, int(cores))

    if cores <= 2:
        # Nothing fits cleanly. Give generation the machine and let TTS borrow a
        # thread in bursts -- deliberately slightly oversubscribed, because the
        # alternative (llm=1) is worse for the stage that dominates the turn.
        llm, tts, stt = cores, 1, 1
    elif cores <= 4:
        llm, tts, stt = cores - 1, 1, 2
    elif cores <= 8:
        llm, tts, stt = cores - 2, 2, 2
    else:
        llm, tts, stt = min(8, cores - 3), 3, 4

    return {
        "llm_threads": max(1, llm),
        # Prompt ingest happens before any audio exists, so it is never contended
        # -- it can have the whole machine even when generation cannot.
        "llm_batch_threads": max(1, min(cores, 16)),
        "tts_threads": max(1, tts),
        "stt_threads": max(1, stt),
        "vad_threads": 1,
    }


def physical_core_count() -> int:
    """Best-effort physical (non-hyperthreaded) core count.

    Falls back through psutil -> os.cpu_count()//2 -> 4.
    """
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:
        pass

    logical = os.cpu_count() or 4
    # Assume SMT2 on anything with more than 2 logical CPUs.
    return max(1, logical // 2 if logical > 2 else logical)


def load_config(path: str | os.PathLike | None = None) -> Section:
    """Load config.yaml, resolve paths, and auto-detect thread counts."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Section(raw)

    # --- resolve models directory to an absolute path ---
    models_dir = Path(raw["paths"]["models_dir"])
    if not models_dir.is_absolute():
        models_dir = (cfg_path.parent / models_dir).resolve()
    raw["paths"]["models_dir"] = str(models_dir)

    # --- llama-server binary path resolves against models_dir ---
    server = raw.get("llm", {}).get("server")
    if server and server.get("binary"):
        binary = Path(server["binary"])
        # config.yaml is written Windows-first; drop the .exe elsewhere.
        if os.name != "nt" and binary.suffix == ".exe":
            binary = binary.with_suffix("")
        if not binary.is_absolute():
            binary = models_dir / binary
        if not binary.exists():
            # Some llama.cpp archives nest the binary (e.g. build/bin/). Find it
            # rather than making the user hand-edit the path.
            found = sorted((models_dir / "llama.cpp").rglob(binary.name)) if models_dir.exists() else []
            if found:
                binary = found[0]
        server["binary"] = str(binary)

    # --- thread policy: scale to whatever machine this is ---
    cores = physical_core_count()
    logical = os.cpu_count() or cores
    ram_gb = total_ram_gb()
    plan = plan_threads(cores)

    if raw["llm"].get("n_threads") in (None, 0, "auto"):
        raw["llm"]["n_threads"] = plan["llm_threads"]
    if raw["llm"].get("n_threads_batch") in (None, 0, "auto"):
        raw["llm"]["n_threads_batch"] = plan["llm_batch_threads"]
    raw["llm"]["_physical_cores"] = cores

    raw.setdefault("runtime", {})
    raw["runtime"].update(
        physical_cores=cores,
        logical_cores=logical,
        ram_gb=round(ram_gb, 1),
        tier=hardware_tier(cores, ram_gb),
        stt_threads=plan["stt_threads"],
        tts_threads=plan["tts_threads"],
        vad_threads=plan["vad_threads"],
        llm_batch_threads=plan["llm_batch_threads"],
    )
    # Back-compat for anything still reading the old key.
    raw["runtime"]["onnx_threads"] = raw["runtime"]["stt_threads"]

    # --- engine auto-selection ---
    # "auto" means "pick what this laptop can actually stream". Kokoro sounds
    # better than Piper but is ~20x more compute per second of audio; on two
    # cores it cannot keep ahead of playback, which makes streaming impossible.
    tts_engine = str(raw.get("tts", {}).get("engine", "piper")).lower()
    if tts_engine == "auto":
        raw["tts"]["engine"] = "kokoro" if cores >= 4 else "piper"
        raw["tts"]["_auto_selected"] = True

    return Section(raw)


def hardware_tier(cores: int, ram_gb: float) -> str:
    """Coarse label used for reporting and engine defaults."""
    if cores <= 2 or ram_gb < 6:
        return "minimal"
    if cores <= 4:
        return "low"
    if cores <= 8:
        return "mid"
    return "high"


def model_path(cfg: Section, filename: str) -> Path:
    return Path(cfg.paths.models_dir) / filename
