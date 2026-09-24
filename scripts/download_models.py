r"""One-shot model fetch. After this runs the agent makes zero network calls.

Downloads:
  models/silero_vad.onnx                     ~2 MB    Silero VAD v5
  models/Qwen_Qwen3-1.7B-Q4_K_M.gguf         ~1.3 GB  TTT
  models/en_US-amy-medium.onnx (+ .json)     ~63 MB   Piper TTS voice
  models/vosk-model-small-en-us-0.15/        ~68 MB   Vosk streaming STT (optional)
  (HF cache) parakeet-tdt-0.6b-v3 int8        ~670 MB  default STT
  models/llama.cpp/llama-server.exe          ~50 MB   prebuilt llama.cpp CPU build

distil-Whisper is not downloaded here -- faster-whisper pulls its CTranslate2
weights from Hugging Face on first use and caches them under ~/.cache/huggingface.
Pass --warm-stt to force that now instead of on the first call.

Usage:
    python scripts/download_models.py --warm-stt
    python scripts/download_models.py --only vosk --vosk-model en-us-0.22
    python scripts/download_models.py --only llamacpp   # just the server binary
    python scripts/download_models.py --llm-size 0.6b   # smaller LLM (~490 MB)

Needs ~2.1 GB free. The GGUF downloads straight into models/ (no duplicate blob
in the Hugging Face cache); Parakeet's weights do live in that cache, ~670 MB.
Run .\cleanup.ps1 -Force first if space is tight.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
# Qwen3 is the default: same speed class as Qwen2.5-1.5B on CPU, markedly better
# instruction following. Sizes are the real Q4_K_M blob sizes.
QWEN_SIZES = {
    "1.7b": ("bartowski/Qwen_Qwen3-1.7B-GGUF", "Qwen_Qwen3-1.7B-Q4_K_M.gguf"),        # 1282 MB
    "0.6b": ("bartowski/Qwen_Qwen3-0.6B-GGUF", "Qwen_Qwen3-0.6B-Q4_K_M.gguf"),        # ~490 MB
    "4b": ("bartowski/Qwen_Qwen3-4B-GGUF", "Qwen_Qwen3-4B-Q4_K_M.gguf"),              # ~2.5 GB
    "1.5b-legacy": ("bartowski/Qwen2.5-1.5B-Instruct-GGUF",
                    "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
}

# Piper TTS voice.
PIPER_BASE = ("https://huggingface.co/rhasspy/piper-voices/resolve/main"
              "/en/en_US/amy/medium")
PIPER_FILES = ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]

# Vosk streaming STT models. "small" is 40 MB and real-time on one core;
# "en-us-0.22" is 1.8 GB and noticeably more accurate.
VOSK_MODELS = {
    "small-en-us": "vosk-model-small-en-us-0.15",
    "en-us-0.22": "vosk-model-en-us-0.22",
    "small-en-in": "vosk-model-small-en-in-0.4",
    "en-in": "vosk-model-en-in-0.5",
}
VOSK_BASE = "https://alphacephei.com/vosk/models"

LLAMACPP_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
# Ranked preference for the CPU-only asset on each (system, arch). First match
# wins. Verified against release b10293, which publishes e.g.
# llama-b10293-bin-win-cpu-x64.zip (18 MB) and llama-b10293-bin-ubuntu-x64.tar.gz.
# The patterns are deliberately specific so GPU builds (cuda/hip/vulkan/sycl/
# openvino) and the separate cudart bundles never match.
LLAMACPP_ASSET_PATTERNS = {
    ("Windows", "x64"): ["bin-win-cpu-x64", "bin-win-avx2-x64", "bin-win-avx-x64"],
    ("Windows", "arm64"): ["bin-win-cpu-arm64"],
    ("Linux", "x64"): ["bin-ubuntu-x64", "bin-linux-x64"],
    ("Linux", "arm64"): ["bin-ubuntu-arm64"],
    ("Darwin", "arm64"): ["bin-macos-arm64"],
    ("Darwin", "x64"): ["bin-macos-x64"],
}


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name}  <-  {url}")

    # A \r progress bar is great in a terminal and terrible in a log file.
    interactive = sys.stdout.isatty()
    state = {"last_pct": -10.0}

    def hook(block_num: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block_num * block_size, total)
        pct = 100.0 * done / total
        if interactive:
            bar = "#" * int(pct // 3)
            print(f"\r         {bar:<34} {pct:5.1f}%  {_human(done)}/{_human(total)}", end="")
        elif pct - state["last_pct"] >= 10.0:
            state["last_pct"] = pct
            print(f"         {pct:5.1f}%  {_human(done)}/{_human(total)}")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    if interactive:
        print()
    tmp.replace(dest)
    print(f"  [ok  ] {dest.name} ({_human(dest.stat().st_size)})")
    return dest


def fetch_vad(models_dir: Path) -> None:
    print("\n-- Silero VAD --")
    download(SILERO_URL, models_dir / "silero_vad.onnx")


# Approximate download sizes, for the free-space check.
QWEN_SIZE_BYTES = {
    "1.7b": 1_282_000_000,
    "0.6b": 490_000_000,
    "4b": 2_500_000_000,
    "1.5b-legacy": 1_000_000_000,
}


def _free_bytes(path: Path) -> int:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free


def load_project_config() -> dict:
    """Read config.yaml so this script and the agent agree on which model to use.

    Without this the downloader has its own opinion: it shipped a hardcoded Qwen
    table and a --llm-size default, so switching llm.model_file in config.yaml to
    Llama-3.2-3B downloaded Qwen3-1.7B anyway and preflight then failed with
    "missing GGUF model". One source of truth.
    """
    try:
        import yaml

        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def fetch_llm(models_dir: Path, size: str | None = None) -> None:
    if size:
        repo, filename = QWEN_SIZES[size]
    else:
        llm = load_project_config().get("llm", {}) or {}
        filename = llm.get("model_file")
        repo = llm.get("hf_repo")
        if not filename or not repo:
            repo, filename = QWEN_SIZES["1.7b"]
        filename = llm.get("hf_filename", filename)
    print(f"\n-- {filename} --")
    dest = models_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({_human(dest.stat().st_size)})")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface-hub is required. Run: pip install huggingface-hub")
        raise

    models_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast and usefully rather than dying 61% into a 1.3 GB download.
    needed = QWEN_SIZE_BYTES.get(size or "", 2_200_000_000)
    free = _free_bytes(models_dir)
    if free < needed * 1.1:
        print(f"  NOT ENOUGH DISK SPACE.")
        print(f"    needed : {_human(needed)} (plus ~10% headroom)")
        print(f"    free   : {_human(free)} on {models_dir.drive or models_dir.anchor}")
        print("  Free some space, then retry. Quick wins:")
        print("    .\\cleanup.ps1 -Force          removes superseded models and caches")
        print("    python scripts/download_models.py --llm-size 0.6b   (~490 MB instead)")
        raise SystemExit(1)

    print(f"  [get ] {filename}  <-  {repo}  ({_human(needed)})")
    # local_dir downloads STRAIGHT into models/. Without it huggingface_hub
    # stores a blob in ~/.cache/huggingface and we then copy it, needing twice
    # the free space for no benefit.
    hf_hub_download(repo_id=repo, filename=filename, local_dir=str(models_dir))

    if not dest.exists():
        # Some hub versions nest the file under its repo path inside local_dir.
        found = sorted(models_dir.rglob(filename))
        if found and found[0] != dest:
            found[0].replace(dest)
    if dest.exists():
        print(f"  [ok  ] {dest.name} ({_human(dest.stat().st_size)})")
    else:
        print(f"  WARNING: {filename} not found under {models_dir} after download")
    if size:
        print(f"  NOTE: --llm-size overrode config.yaml. Set llm.model_file to")
        print(f"        {filename} (and llm.hf_repo to {repo}) to make it stick.")


def fetch_piper(models_dir: Path) -> None:
    print("\n-- Piper TTS voice --")
    for name in PIPER_FILES:
        download(f"{PIPER_BASE}/{name}", models_dir / name)


def fetch_llamacpp(models_dir: Path, tag: str | None = None) -> None:
    """Fetch the official prebuilt llama.cpp CPU build and extract llama-server."""
    print("\n-- llama.cpp prebuilt CPU binaries --")
    dest_dir = models_dir / "llama.cpp"
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    if (dest_dir / exe).exists() or any(dest_dir.rglob(exe)):
        print(f"  [skip] {exe} already present")
        return

    system, arch = platform.system(), _arch()
    patterns = LLAMACPP_ASSET_PATTERNS.get((system, arch))
    if not patterns:
        print(f"  Unsupported platform {system}/{arch}. Download llama.cpp manually and point")
        print("  llm.server.binary at the llama-server executable.")
        return

    def _get(url: str):
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "voice-agent-setup"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)

    # Windows assets are .zip; Linux/macOS are .tar.gz.
    archive_suffixes = (".zip", ".tar.gz", ".tgz")

    def _pick(release):
        for pattern in patterns:
            for asset in release.get("assets", []):
                name = asset["name"]
                if pattern in name and name.endswith(archive_suffixes):
                    return asset
        return None

    try:
        if tag is not None:
            candidates = [_get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")]
        else:
            # llama.cpp tags a release before its assets finish uploading, so
            # "latest" is sometimes empty or partial. Walk back a few releases
            # rather than failing on a transient state.
            candidates = _get(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8"
            )
    except Exception as exc:
        print(f"  Could not query GitHub releases: {exc}")
        print("  Download a CPU build manually from https://github.com/ggml-org/llama.cpp/releases")
        print(f"  and extract {exe} to {dest_dir}")
        return

    release = chosen = None
    for candidate in candidates:
        if candidate.get("draft"):
            continue
        asset = _pick(candidate)
        if asset is not None:
            release, chosen = candidate, asset
            break
        print(f"  (no {patterns[0]} asset in {candidate.get('tag_name')} yet, trying older)")

    if chosen is None:
        newest = candidates[0] if candidates else {}
        print(f"  No CPU asset matching {patterns} in the last {len(candidates)} release(s).")
        print("  Newest had:", ", ".join(a["name"] for a in newest.get("assets", [])[:10]) or "(none)")
        print("  Download a CPU build manually from https://github.com/ggml-org/llama.cpp/releases")
        print(f"  and extract {exe} to {dest_dir}")
        return

    print(f"  release {release.get('tag_name')}  asset {chosen['name']} "
          f"({_human(chosen.get('size', 0))})")
    archive = models_dir / chosen["name"]
    download(chosen["browser_download_url"], archive)

    # Extract preserving structure. Do NOT flatten: the Linux/macOS tarballs use
    # symlinks (libllama-common.so.0 -> ...so.0.18.1) and nest binaries under
    # build/bin/, so flattening silently breaks the dynamic loader. The Windows
    # zip is already flat, which is why the default config path just works there.
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()

    def _safe(name: str) -> bool:
        try:
            return root == (root / name).resolve() or root in (root / name).resolve().parents
        except (OSError, ValueError):
            return False

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir, members=[n for n in zf.namelist() if _safe(n)])
    else:
        with tarfile.open(archive, "r:gz") as tf:
            members = [m for m in tf.getmembers() if _safe(m.name)]
            try:
                tf.extractall(dest_dir, members=members, filter="data")
            except TypeError:      # Python < 3.12 has no `filter`
                tf.extractall(dest_dir, members=members)

    archive.unlink(missing_ok=True)

    # Locate the executable wherever the archive put it.
    target = dest_dir / exe
    if not target.exists():
        found = sorted(dest_dir.rglob(exe))
        if found:
            target = found[0]

    if not target.exists():
        listing = sorted(p.name for p in dest_dir.rglob("*") if p.is_file())
        print(f"  WARNING: {exe} not found after extraction. Extracted: {listing[:15]}")
        return

    if platform.system() != "Windows":
        for p in target.parent.rglob("*"):
            if p.is_file() and not p.is_symlink():
                if p.suffix in ("", ".so", ".dylib") or ".so." in p.name:
                    p.chmod(0o755)

    print(f"  [ok  ] {target}")
    rel = target.relative_to(models_dir)
    if rel != Path("llama.cpp") / exe:
        print(f"  NOTE: set llm.server.binary to  {rel.as_posix()}  in config.yaml")


def fetch_vosk(models_dir: Path, name: str = "small-en-us") -> None:
    """Download and unzip a Vosk model directory."""
    folder = VOSK_MODELS[name]
    print(f"\n-- Vosk model ({folder}) --")
    dest = models_dir / folder
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {folder} already present")
        return

    archive = models_dir / f"{folder}.zip"
    download(f"{VOSK_BASE}/{folder}.zip", archive)

    root = models_dir.resolve()

    def _safe(member: str) -> bool:
        try:
            target = (root / member).resolve()
            return root == target or root in target.parents
        except (OSError, ValueError):
            return False

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(models_dir, members=[n for n in zf.namelist() if _safe(n)])
    archive.unlink(missing_ok=True)

    if dest.exists():
        size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        print(f"  [ok  ] {dest} ({_human(size)})")
    else:
        found = [p.name for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("vosk")]
        print(f"  WARNING: expected {folder}. Extracted directories: {found}")


def fetch_parakeet(quantization: str = "int8") -> None:
    """Pull the Parakeet ONNX weights into the HF cache (~670 MB for int8)."""
    print(f"\n-- Parakeet TDT 0.6B v3 ONNX ({quantization}) --")
    try:
        import onnx_asr
    except ImportError:
        print('  onnx-asr is not installed. Run: pip install "onnx-asr[cpu,hub]"')
        return
    print("  [get ] fetching weights into the Hugging Face cache...")
    onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization=quantization)
    print("  [ok  ] Parakeet cached")


NEMOTRON_REPO = "nvidia/nemotron-speech-streaming-en-0.6b"
NEMOTRON_GGUF = "nemotron-speech-streaming-en-0.6b.q8_0.gguf"

KOKORO_FILES = {
    # onnx-community mirrors the int8 build; ~88 MB model + ~26 MB voices.
    "kokoro-v1.0.int8.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def fetch_nemotron(models_dir: Path, quantization: str = "int8") -> None:
    """NVIDIA Nemotron cache-aware streaming ASR 0.6B (~630 MB).

    Tries the Hugging Face cache first (what onnx-asr and transformers both
    read), then falls back to pulling the q8_0 GGUF for NeMo-Speech.cpp, which
    is the lightest CPU runtime NVIDIA ship for it.
    """
    print(f"\n-- Nemotron streaming ASR 0.6B ({quantization}) --")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub is not installed. Run: pip install huggingface_hub")
        return

    dest = models_dir / "nemotron-speech-streaming-en-0.6b"
    try:
        print(f"  [get ] {NEMOTRON_REPO} -> {dest}")
        snapshot_download(
            repo_id=NEMOTRON_REPO,
            local_dir=str(dest),
            allow_patterns=["*.onnx", "*.json", "*.txt", "*.yaml", "*.model", "*.gguf"],
        )
        size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        print(f"  [ok  ] {dest} ({_human(size)})")
    except Exception as exc:
        print(f"  WARNING: could not fetch {NEMOTRON_REPO}: {exc}")
        print("  The model is gated by the NVIDIA Open Model License -- accept it at")
        print(f"    https://huggingface.co/{NEMOTRON_REPO}")
        print("  then run:  hf auth login")


def fetch_kokoro(models_dir: Path) -> None:
    """Kokoro-82M int8 ONNX plus its voice pack (~115 MB total)."""
    print("\n-- Kokoro TTS (int8 ONNX) --")
    for name, url in KOKORO_FILES.items():
        dest = models_dir / name
        if dest.exists():
            print(f"  [skip] {name} ({_human(dest.stat().st_size)})")
            continue
        try:
            download(url, dest)
        except Exception as exc:
            print(f"  WARNING: could not fetch {name}: {exc}")


def warm_stt(model_name: str, compute_type: str = "int8") -> None:
    """Pull the CTranslate2 weights into the HF cache now, not on the first call."""
    print(f"\n-- distil-Whisper ({model_name}, {compute_type}) --")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  faster-whisper is not installed. Run: pip install faster-whisper")
        return
    print("  [get ] fetching weights into the Hugging Face cache...")
    WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=1)
    print("  [ok  ] distil-Whisper cached")


def _active_names(cfg: dict) -> set[str]:
    """Files and directories the CURRENT config actually loads."""
    stt, llm, tts, vad = (cfg.get(k, {}) or {} for k in ("stt", "llm", "tts", "vad"))
    names = {
        vad.get("model_file"),
        llm.get("model_file"),
        tts.get("model_file"),          # Piper is always the fallback engine
        tts.get("voices_file"),
        (tts.get("kokoro", {}) or {}).get("model_file"),
        (tts.get("kokoro", {}) or {}).get("voices_file"),
        (stt.get("nemotron", {}) or {}).get("model_dir"),
        "llama.cpp",                    # runtime, not a model, but required
    }
    engine = str(stt.get("engine", "")).lower()
    if engine == "vosk":
        names.add((stt.get("vosk", {}) or {}).get("model_dir"))
    return {n for n in names if n}


def report_footprint(models_dir: Path) -> None:
    """Report ACTIVE footprint separately from leftovers.

    Totalling the whole directory made the verdict meaningless: three superseded
    GGUFs from earlier experiments were still on disk, so a 2.8 GB stack reported
    4.9 GB and "OVER" no matter what. What matters is what the current config
    loads; everything else is just disk you can reclaim.
    """
    cfg = load_project_config()
    active = _active_names(cfg)

    entries: list[tuple[str, int, bool]] = []
    for path in sorted(models_dir.glob("*")):
        if path.name.endswith(".part") or path.name.endswith(".log"):
            continue
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        else:
            continue
        entries.append((path.name, size, path.name in active))

    print("\n-- On-disk footprint --")
    used = sum(s for _, s, keep in entries if keep)
    spare = sum(s for _, s, keep in entries if not keep)

    for name, size, keep in entries:
        if keep:
            note = " (runtime)" if name == "llama.cpp" else ""
            print(f"  {name + note:<44} {_human(size):>10}")
    print(f"  {'ACTIVE TOTAL (what this config loads)':<44} {_human(used):>10}")

    limit = 2.5 * 1024 ** 3
    print(f"  Footprint limit 2.5 GB: {'PASS' if used <= limit else 'OVER'}")

    if spare:
        print("\n  Not used by the current config:")
        for name, size, keep in entries:
            if not keep:
                print(f"    {name:<42} {_human(size):>10}")
        print(f"    {'RECLAIMABLE':<42} {_human(spare):>10}")
        print("    Delete them with:  .\\cleanup.ps1 -Force")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--only", default="vad,llm,piper,llamacpp,nemotron,kokoro",
                        help="comma list: vad,llm,piper,llamacpp,nemotron,kokoro,parakeet,vosk")
    parser.add_argument("--vosk-model", choices=list(VOSK_MODELS), default="small-en-us",
                        help="which Vosk model to fetch")
    parser.add_argument("--llm-size", choices=list(QWEN_SIZES), default=None,
                        help="override config.yaml and fetch one of the bundled Qwen "
                             "sizes instead (smaller is faster on few cores)")
    parser.add_argument("--llamacpp-tag", default=None,
                        help="pin a llama.cpp release tag instead of latest, e.g. b6234")
    parser.add_argument("--warm-stt", action="store_true",
                        help="download the distil-Whisper weights now")
    parser.add_argument("--stt-model", default="Systran/faster-whisper-base.en")
    parser.add_argument("--parakeet-quantization", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--stt-compute-type", default="int8")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"Model directory: {models_dir}")
    if "vad" in wanted:
        fetch_vad(models_dir)
    if "llamacpp" in wanted:
        fetch_llamacpp(models_dir, args.llamacpp_tag)
    if "piper" in wanted:
        fetch_piper(models_dir)
    if "vosk" in wanted:
        fetch_vosk(models_dir, args.vosk_model)
    if "nemotron" in wanted:
        fetch_nemotron(models_dir, args.parakeet_quantization)
    if "kokoro" in wanted:
        fetch_kokoro(models_dir)
    if "parakeet" in wanted:
        fetch_parakeet(args.parakeet_quantization)
    if "llm" in wanted:
        fetch_llm(models_dir, args.llm_size)
    if args.warm_stt:
        warm_stt(args.stt_model, args.stt_compute_type)

    report_footprint(models_dir)
    print("\nDone. Next: python run_agent.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
