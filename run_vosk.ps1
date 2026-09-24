# Run the agent with the Vosk streaming recognizer.
#
# Vosk decodes WHILE you speak, so when you stop talking the transcript is
# essentially already done. That is the structural fix for turn latency --
# Whisper cannot start until you finish, so its whole cost lands on the
# critical path.
#
# Trade-off: less accurate than Whisper on hard audio, accents, or noise.
#
#   .\run_vosk.ps1              live call
#   .\run_vosk.ps1 -Text        typed input, spoken output
#   .\run_vosk.ps1 -Verbose     show VAD events and stage timings

param(
    [switch]$Text,
    [switch]$Verbose,
    [switch]$NoGreeting
)

$args = @("--stt-engine", "vosk")
if ($Text)       { $args += "--text" }
if ($Verbose)    { $args += "--verbose" }
if ($NoGreeting) { $args += "--no-greeting" }

Write-Host "STT engine: vosk (streaming)" -ForegroundColor Cyan
python run_agent.py @args
