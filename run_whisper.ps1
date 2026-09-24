# Run the agent with faster-whisper.
#
# More accurate than Vosk, but Whisper cannot begin until you stop talking, and
# it pads every clip to a fixed mel window -- so a one-second utterance costs
# about the same as a ten-second one. That cost lands entirely on the critical
# path and cannot be overlapped with anything.
#
#   .\run_whisper.ps1                       config default (base.en)
#   .\run_whisper.ps1 -Model tiny.en        faster, noticeably more errors
#   .\run_whisper.ps1 -Model small.en       more accurate, ~3x slower
#   .\run_whisper.ps1 -Text                 typed input, spoken output

param(
    [ValidateSet("tiny.en", "base.en", "small.en", "distil-small.en")]
    [string]$Model,
    [switch]$Text,
    [switch]$Verbose,
    [switch]$NoGreeting
)

$args = @("--stt-engine", "whisper")

if ($Model) {
    $repo = if ($Model -eq "distil-small.en") {
        "Systran/faster-distil-whisper-small.en"
    } else {
        "Systran/faster-whisper-$Model"
    }
    $args += @("--stt", $repo)
    Write-Host "STT engine: whisper ($repo)" -ForegroundColor Cyan
} else {
    Write-Host "STT engine: whisper (config default)" -ForegroundColor Cyan
}

if ($Text)       { $args += "--text" }
if ($Verbose)    { $args += "--verbose" }
if ($NoGreeting) { $args += "--no-greeting" }

python run_agent.py @args
