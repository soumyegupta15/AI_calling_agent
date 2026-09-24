# Removes everything the current pipeline no longer uses.
# Safe to run repeatedly.
#
#   .\cleanup.ps1            show what would be deleted, and free space
#   .\cleanup.ps1 -Force     actually delete
#
# Current stack: Silero VAD -> Nemotron streaming ASR 0.6B -> Llama-3.2-3B
#                -> Kokoro (>=4 cores) / Piper (fallback)

param([switch]$Force)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot

function Show-Space {
    $drive = (Get-Item $root).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    Write-Host ("Free space on {0}: {1} GB" -f $drive.Name, $freeGB) -ForegroundColor Cyan
}

Show-Space

$targets = @(
    # --- superseded LLMs ------------------------------------------------------
    # All replaced by Llama-3.2-3B-Instruct-Q4_K_M.
    "models\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    "models\Qwen_Qwen3-1.7B-Q4_K_M.gguf",
    "models\Qwen_Qwen3-0.6B-Q4_K_M.gguf",
    "models\Qwen_Qwen3-4B-Q4_K_M.gguf",

    # --- leftover archives ----------------------------------------------------
    # The llama.cpp release tarball/zip is not needed once it is extracted.
    "models\llama-b10293-bin-ubuntu-x64.tar.gz",

    # NOTE: kokoro-v1.0.int8.onnx and voices-v1.0.bin used to be listed here,
    # back when Kokoro had been dropped for Piper. Kokoro is IN the stack again
    # (auto-selected on >= 4 physical cores), so deleting them would break a
    # working install. Do not re-add them.

    # --- superseded config / docs --------------------------------------------
    "config.lowspec.yaml",
    "MIGRATION.md",

    # --- caches and leftovers -------------------------------------------------
    "__pycache__",
    ".pytest_cache",
    "voice_agent\__pycache__",
    "tests\__pycache__",
    "scripts\__pycache__"
)

Write-Host "`nProject files" -ForegroundColor Cyan
$total = 0
foreach ($rel in $targets) {
    $path = Join-Path $root $rel
    if (-not (Test-Path $path)) { continue }
    $size = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
    if (-not $size) { $size = 0 }
    $total += $size
    $mb = [math]::Round($size / 1MB, 1)
    if ($Force) {
        Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host ("  removed  {0,-46} {1,9} MB" -f $rel, $mb) -ForegroundColor Green
    } else {
        Write-Host ("  would remove  {0,-41} {1,9} MB" -f $rel, $mb) -ForegroundColor Yellow
    }
}

# --- leftover download archives and non-Windows llama.cpp binaries -----------
$modelsDir = Join-Path $root "models"
if (Test-Path $modelsDir) {
    Write-Host "`nLeftover archives and stray binaries" -ForegroundColor Cyan
    $stray = @()
    $stray += Get-ChildItem $modelsDir -File -ErrorAction SilentlyContinue |
              Where-Object { $_.Extension -in ".zip", ".gz", ".tgz", ".part", ".log" }
    $llamaDir = Join-Path $modelsDir "llama.cpp"
    if (Test-Path $llamaDir) {
        # The Windows build is llama-server.exe; extensionless files and .so
        # libraries came from a Linux archive and cannot run here.
        $stray += Get-ChildItem $llamaDir -Recurse -File | Where-Object {
            $_.Extension -eq "" -or $_.Name -like "*.so" -or $_.Name -like "*.so.*"
        }
    }
    foreach ($f in $stray) {
        $mb = [math]::Round($f.Length / 1MB, 1)
        $total += $f.Length
        if ($Force) {
            Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
            Write-Host ("  removed  {0,-46} {1,9} MB" -f $f.Name, $mb) -ForegroundColor Green
        } else {
            Write-Host ("  would remove  {0,-41} {1,9} MB" -f $f.Name, $mb) -ForegroundColor Yellow
        }
    }
    if (-not $stray) { Write-Host "  none" -ForegroundColor DarkGray }
}

# --- Hugging Face caches for models we no longer load ------------------------
# These are the big ones. The GGUF now downloads straight into models\, but
# older runs left a duplicate blob in the cache.
Write-Host "`nHugging Face cache" -ForegroundColor Cyan
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$deadCaches = @(
    "models--bartowski--Qwen2.5-1.5B-Instruct-GGUF",
    "models--bartowski--Qwen2.5-3B-Instruct-GGUF",
    "models--bartowski--Qwen2.5-0.5B-Instruct-GGUF",
    "models--bartowski--Qwen_Qwen3-1.7B-GGUF",
    "models--bartowski--Llama-3.2-3B-Instruct-GGUF", # duplicate of models\ copy
    "models--Systran--faster-distil-whisper-small.en",
    "models--Systran--faster-whisper-tiny.en",
    "models--Systran--faster-whisper-base.en",
    "models--UsefulSensors--moonshine-base",
    "models--UsefulSensors--moonshine-tiny",
    "models--usefulsensors--moonshine-base",
    "models--usefulsensors--moonshine-tiny"
)
if (Test-Path $hub) {
    foreach ($name in $deadCaches) {
        $path = Join-Path $hub $name
        if (-not (Test-Path $path)) { continue }
        $size = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum).Sum
        if (-not $size) { $size = 0 }
        $mb = [math]::Round($size / 1MB, 1)
        $total += $size
        if ($Force) {
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host ("  removed  {0,-46} {1,9} MB" -f $name, $mb) -ForegroundColor Green
        } else {
            Write-Host ("  would remove  {0,-41} {1,9} MB" -f $name, $mb) -ForegroundColor Yellow
        }
    }

    # Any half-finished downloads left by a crash.
    $incomplete = Get-ChildItem $hub -Recurse -File -Filter "*.incomplete" -ErrorAction SilentlyContinue
    foreach ($f in $incomplete) {
        $mb = [math]::Round($f.Length / 1MB, 1)
        $total += $f.Length
        if ($Force) {
            Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
            Write-Host ("  removed  {0,-46} {1,9} MB" -f "(incomplete) $($f.Name)", $mb) -ForegroundColor Green
        } else {
            Write-Host ("  would remove  {0,-41} {1,9} MB" -f "(incomplete) $($f.Name)", $mb) -ForegroundColor Yellow
        }
    }
}

Write-Host ("`nTotal: {0} MB ({1} GB)" -f [math]::Round($total / 1MB, 1), [math]::Round($total / 1GB, 2)) -ForegroundColor Cyan
if (-not $Force) {
    Write-Host "Dry run. Re-run with -Force to delete.`n" -ForegroundColor Yellow
} else {
    Write-Host "Done." -ForegroundColor Green
    Show-Space
    Write-Host "`nAlso uninstall the packages we no longer import:" -ForegroundColor Cyan
    Write-Host "  pip uninstall -y useful-moonshine-onnx kokoro-onnx llama-cpp-python`n"
}
