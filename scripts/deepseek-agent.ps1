param(
    [switch]$Check,
    [switch]$NoAgent,
    [int]$TimeoutSec = 1200
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StrataHome = if ($env:STRATA_HOME) { $env:STRATA_HOME } else { "C:\Strata" }
$Addr = if ($env:RUSTMOE_ADDR) { $env:RUSTMOE_ADDR } else { "127.0.0.1:8080" }
$AgentAddr = "http://127.0.0.1:8081"
$Model = "deepseek-v4.1-flash"
$StrataModel = "IST-DAS-2.00bpw"
$AgentExe = Join-Path $Root "target\release\rustmoe-agent.exe"
$StrataPython = Join-Path $StrataHome ".venv\Scripts\python.exe"
$StrataScript = Join-Path $StrataHome "strata.py"
$LogDir = Join-Path $Root ".models"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-StrataHealth {
    try { return Invoke-RestMethod -Uri "http://$Addr/health" -TimeoutSec 3 } catch { return $null }
}

function Get-AgentHealth {
    try { return Invoke-RestMethod -Uri "$AgentAddr/health" -TimeoutSec 3 } catch { return $null }
}

function Wait-Strata([int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $h = Get-StrataHealth
        if ($h -and $h.model -eq $Model -and $h.loaded) { return $h }
        Start-Sleep -Seconds 5
    }
    return $null
}

function Wait-Agent([int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $a = Get-AgentHealth
        if ($a -and $a.service -eq "rustmoe-fast-agent") { return $a }
        Start-Sleep -Seconds 2
    }
    return $null
}

if (-not (Test-Path $AgentExe)) {
    throw "rustmoe-agent.exe is missing: $AgentExe. Build: cargo build --release -p rustmoe-server --bin rustmoe-agent"
}
if (-not (Test-Path $StrataPython) -or -not (Test-Path $StrataScript)) {
    throw "Strata terminal not found at $StrataHome"
}

$h = Get-StrataHealth
$needStrata = -not ($h -and $h.model -eq $Model -and $h.loaded)

if ($Check) {
    Write-Host "[check] agent   $AgentExe"
    Write-Host "[check] strata  $StrataHome"
    if ($h) { Write-Host "[check] 8080    $($h.model) loaded=$($h.loaded)" } else { Write-Host "[check] 8080    down" }
    if ($needStrata) {
        Write-Host "[check] would start DeepSeek: $StrataModel, 8K, 5 experts"
        Write-Host "[check] would wait up to $TimeoutSec s for model load"
    } else {
        Write-Host "[check] DeepSeek already running"
    }
    if (-not $NoAgent) { Write-Host "[check] would start agent on 8081 upstream http://$Addr" }
    return
}

if ($needStrata) {
    if ($h) {
        Write-Host "[deepseek] stopping current engine: $($h.model)"
    } else {
        Write-Host "[deepseek] starting Strata DeepSeek"
    }
    Write-Host "[deepseek] model $StrataModel, 8K, 5 experts; first load can take 8 minutes"
    $strataArgs = @(
        $StrataScript, "launch",
        "--family", "deepseek41",
        "--model", $StrataModel,
        "--context", "8k",
        "--speed", "faster",
        "--access", "local",
        "--port", "8080",
        "--restart",
        "--no-tui",
        "--yes"
    )
    $strataProc = @{
        FilePath = $StrataPython
        ArgumentList = $strataArgs
        WorkingDirectory = $StrataHome
        RedirectStandardOutput = (Join-Path $LogDir "deepseek-agent-strata.out.log")
        RedirectStandardError = (Join-Path $LogDir "deepseek-agent-strata.err.log")
        WindowStyle = "Hidden"
    }
    Start-Process @strataProc | Out-Null

    $up = Wait-Strata $TimeoutSec
    if (-not $up) {
        throw "DeepSeek did not become ready within $TimeoutSec s. See $LogDir\deepseek-agent-strata.out.log"
    }
    Write-Host "[deepseek] ready: http://$Addr ($($up.max_context) tokens)"
} else {
    Write-Host "[deepseek] already running"
}

if ($NoAgent) {
    Write-Host "[agent] skipped (-NoAgent)"
    return
}

Get-Process rustmoe-agent -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
$agentArgs = @(
    "--addr", "127.0.0.1:8081",
    "--upstream", "http://$Addr",
    "--model", $Model,
    "--agent-root", (Join-Path $Root "workspace"),
    "--skills-dir", (Join-Path $Root "skills"),
    "--agent-allow-shell",
    "--agent-max-steps", "24"
)
$agentProc = @{
    FilePath = $AgentExe
    ArgumentList = $agentArgs
    WorkingDirectory = $Root
    WindowStyle = "Hidden"
}
Start-Process @agentProc | Out-Null
$agent = Wait-Agent 60
if (-not $agent) {
    throw "rustmoe-agent did not start on 8081"
}
Write-Host "[agent] ready: $AgentAddr/agent"
Write-Host ""
Write-Host "  DeepSeek API : http://$Addr/v1"
Write-Host "  DeepSeek UI  : http://$Addr"
Write-Host "  Fast Agent   : $AgentAddr/agent"
Write-Host "  Model        : $Model"
