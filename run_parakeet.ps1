# Run the agent with NVIDIA Parakeet TDT 0.6B v3 (int8 ONNX) — the accuracy pick.
#
# Parakeet tops the Open ASR leaderboard: lower word error rate than Whisper
# large-v3 at a quarter of the size, several times faster than Whisper on CPU,
# and it does not hallucinate text out of silence the way Whisper does.
#
# Cost scales with how long you actually spoke, unlike Whisper which pads every
# clip to a fixed 30-second window.
#
#   .\run_parakeet.ps1              live call   (this is the config default)
#   .\run_parakeet.ps1 -Text        typed input, spoken output
#   .\run_parakeet.ps1 -Verbose     show VAD events and stage timings

param(
    [switch]$Text,
    [switch]$Verbose,
    [switch]$NoGreeting
)

$args = @("--stt-engine", "parakeet")
if ($Text)       { $args += "--text" }
if ($Verbose)    { $args += "--verbose" }
if ($NoGreeting) { $args += "--no-greeting" }

Write-Host "STT engine: parakeet-tdt-0.6b-v3 (int8)" -ForegroundColor Cyan
python run_agent.py @args
