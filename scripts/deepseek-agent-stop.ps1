param()
$ErrorActionPreference = "SilentlyContinue"
Get-Process rustmoe-agent | Stop-Process -Force
$homeDir = if ($env:STRATA_HOME) { $env:STRATA_HOME } else { "C:\Strata" }
$py = Join-Path $homeDir ".venv\Scripts\python.exe"
$script = Join-Path $homeDir "strata.py"
if (Test-Path $py) { & $py $script launch --stop --port 8080 }
Write-Host "Stopped rustmoe-agent and requested Strata stop."
