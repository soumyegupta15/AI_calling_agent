"""Static checks on the CLI scripts.

Regression: an editing pass deleted `fetch_llamacpp` while removing the Kokoro
downloader that sat next to it. `py_compile` was happy -- a missing global is a
NameError at call time, not a syntax error -- so it only surfaced when a user ran
the downloader. These tests resolve every name each script's entry point calls,
without executing anything or touching the network.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRIPTS = {
    "download_models": ROOT / "scripts" / "download_models.py",
    "run_agent": ROOT / "run_agent.py",
    "diagnose": ROOT / "diagnose.py",
    "benchmark": ROOT / "benchmark.py",
}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _called_names(path: Path) -> set[str]:
    """Every bare function name called anywhere in the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _defined_names(module) -> set[str]:
    return set(dir(module))


@pytest.mark.parametrize("name,path", SCRIPTS.items())
def test_script_imports_cleanly(name, path):
    assert path.exists(), f"{path} is missing"
    _load(path)


@pytest.mark.parametrize("name,path", SCRIPTS.items())
def test_every_function_called_actually_exists(name, path):
    import builtins

    module = _load(path)
    available = _defined_names(module) | set(dir(builtins))

    missing = set()
    for called in _called_names(path):
        if called in available:
            continue
        # Locally-bound names (comprehension vars, params, nested defs) are fine;
        # only flag things that look like module-level helpers.
        if called.islower() and "_" in called or called.startswith("fetch_"):
            missing.add(called)

    # Names bound locally inside functions are not module attributes, so filter
    # against everything assigned, defined, or imported anywhere in the file.
    # Function-local imports (`from voice_agent.llm import build_llm` inside a
    # function body) are the common case in these scripts.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            for arg in node.args.args + node.args.kwonlyargs:
                bound.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
    missing -= bound

    assert not missing, f"{path.name} calls undefined function(s): {sorted(missing)}"


def test_downloader_defines_every_fetch_its_main_dispatches():
    """The --only dispatch table must line up with the functions that exist."""
    module = _load(SCRIPTS["download_models"])
    for fn in ("fetch_vad", "fetch_llm", "fetch_piper", "fetch_llamacpp",
               "fetch_vosk", "fetch_parakeet", "fetch_nemotron", "fetch_kokoro",
               "warm_stt"):
        assert hasattr(module, fn), f"download_models.py is missing {fn}()"

    # The default --only list must only name stages main() can dispatch.
    source = SCRIPTS["download_models"].read_text(encoding="utf-8")
    assert 'default="vad,llm,piper,llamacpp,nemotron,kokoro"' in source
    for stage in ("vad", "llm", "piper", "llamacpp", "vosk", "parakeet",
                  "nemotron", "kokoro"):
        assert f'if "{stage}" in wanted:' in source, f"--only {stage} is never handled"


def test_downloader_has_no_dead_engines_left():
    """Kokoro is back on purpose; these others are gone for good."""
    source = SCRIPTS["download_models"].read_text(encoding="utf-8").lower()
    for dead in ("moonshine", "fetch_tts", "tts_precision"):
        assert dead not in source, f"stale reference to {dead}"
    # Kokoro and Nemotron are current, so they MUST be fetchable.
    for live in ("kokoro", "nemotron"):
        assert live in source, f"{live} is configured but not downloadable"


def test_run_agent_preflight_checks_the_shipped_stack():
    module = _load(SCRIPTS["run_agent"])
    from voice_agent.config import load_config

    problems = module.preflight(load_config())
    # We cannot assert it passes (models may be absent in CI), but every message
    # must be a string and none may mention a removed component.
    joined = " ".join(problems).lower()
    for dead in ("kokoro", "moonshine", "llama_cpp"):
        assert dead not in joined, f"preflight still checks {dead}"


# --------------------------------------------------------------------------
#  The downloader must agree with config.yaml
# --------------------------------------------------------------------------

def test_downloader_fetches_the_model_the_config_asks_for():
    """One source of truth for which LLM to download.

    The downloader used to carry its own hardcoded Qwen table plus an
    --llm-size default, so pointing llm.model_file at Llama-3.2-3B fetched
    Qwen3-1.7B anyway and preflight then failed with "missing GGUF model" --
    with the download having appeared to succeed.
    """
    module = _load(SCRIPTS["download_models"])
    cfg = module.load_project_config()
    assert cfg, "download_models.py cannot read config.yaml"

    import yaml

    shipped = yaml.safe_load(
        (SCRIPTS["download_models"].parent.parent / "config.yaml").read_text(encoding="utf-8")
    )
    assert cfg["llm"]["model_file"] == shipped["llm"]["model_file"]
    assert cfg["llm"]["hf_repo"] == shipped["llm"]["hf_repo"]

    # --llm-size must be an explicit override, never a silent default.
    source = SCRIPTS["download_models"].read_text(encoding="utf-8")
    assert '"--llm-size", choices=list(QWEN_SIZES), default=None' in source


def test_footprint_counts_only_what_the_config_loads(tmp_path):
    """Leftover models must not decide the pass/fail verdict.

    Three superseded GGUFs from earlier experiments made a 2.8 GB stack report
    4.9 GB and "OVER" regardless of what was configured.
    """
    module = _load(SCRIPTS["download_models"])
    cfg = module.load_project_config()
    active = module._active_names(cfg)

    assert cfg["llm"]["model_file"] in active
    assert cfg["vad"]["model_file"] in active
    assert cfg["tts"]["model_file"] in active, "Piper is the fallback; always needed"
    assert cfg["tts"]["kokoro"]["model_file"] in active
    assert cfg["stt"]["nemotron"]["model_dir"] in active
    assert "llama.cpp" in active

    # A superseded model is NOT active and must be reported as reclaimable.
    assert "Qwen_Qwen3-1.7B-Q4_K_M.gguf" not in active
    assert "Qwen2.5-3B-Instruct-Q4_K_M.gguf" not in active


def test_downloader_has_no_invalid_escape_sequences():
    """`.\\cleanup.ps1` in a non-raw docstring is a SyntaxWarning on 3.12+."""
    import warnings

    source = SCRIPTS["download_models"].read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source, "download_models.py", "exec")


def test_cleanup_never_deletes_a_file_the_config_needs():
    """cleanup.ps1 had Kokoro on its superseded list.

    Kokoro was dropped for Piper once, then brought back into the stack. The
    delete list was never updated, so running cleanup on a freshly downloaded
    install would have removed kokoro-v1.0.int8.onnx and voices-v1.0.bin and
    broken it. This asserts the two lists cannot drift apart again.
    """
    import re

    import yaml

    root = SCRIPTS["download_models"].parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    active = {
        cfg["llm"]["model_file"],
        cfg["vad"]["model_file"],
        cfg["tts"]["model_file"],
        cfg["tts"]["voices_file"],
        cfg["tts"]["kokoro"]["model_file"],
        cfg["tts"]["kokoro"]["voices_file"],
        cfg["stt"]["nemotron"]["model_dir"],
    }

    source = (root / "cleanup.ps1").read_text(encoding="utf-8")
    targets = set(re.findall(r'"models\\([^"]+)"', source))
    assert targets, "found no delete targets - this check would be vacuous"

    overlap = active & targets
    assert not overlap, f"cleanup.ps1 would delete files the config loads: {overlap}"


def test_cleanup_reclaims_the_superseded_llm():
    """The model we just moved away from should actually be reclaimable."""
    root = SCRIPTS["download_models"].parent.parent
    source = (root / "cleanup.ps1").read_text(encoding="utf-8")
    assert "Qwen_Qwen3-1.7B-Q4_K_M.gguf" in source
