param(
    [ValidateSet("status", "fast", "deep")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StrataHome = if ($env:STRATA_HOME) { $env:STRATA_HOME } else { "C:\Strata" }
$Addr = if ($env:RUSTMOE_ADDR) { $env:RUSTMOE_ADDR } else { "127.0.0.1:8080" }
$HealthUrl = "http://$Addr/health"
$AgentAddr = "http://127.0.0.1:8081"
$AgentHealth = "$AgentAddr/health"

function Get-Health {
    try {
        return Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
    } catch {
        return $null
    }
}

function Get-AgentHealth {
    try {
        return Invoke-RestMethod -Uri $AgentHealth -TimeoutSec 3
    } catch {
        return $null
    }
}

function Wait-Health([scriptblock]$Match, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $h = Get-Health
        if ($h -and (& $Match $h)) { return $h }
        Start-Sleep -Seconds 2
    }
    return $null
}

function Stop-EngineProcesses {
    Get-Process -Name rustmoe,rustmoe-agent,strata,strata-vision -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Show-Status {
    $h = Get-Health
    $gpu = nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>$null
    if (-not $h) {
        Write-Host "[engine] no service on $Addr"
    } elseif ($h.service -eq "strata") {
        Write-Host "[engine] FAST  Strata"
        Write-Host "  model:  $($h.model)"
        Write-Host "  ctx:    $($h.max_context)"
        Write-Host "  vision: $($h.images)"
    } elseif ($h.backend) {
        Write-Host "[engine] DEEP  RustMoE"
        Write-Host "  model:  $($h.model)"
        Write-Host "  backend:$($h.backend)"
        Write-Host "  sim:    $($h.simulation)"
    } else {
        Write-Host "[engine] unknown service on $Addr"
        $h | ConvertTo-Json -Depth 5
    }
    $a = Get-AgentHealth
    if ($a) {
        Write-Host "[agent] FAST  http://127.0.0.1:8081/agent"
        Write-Host "  model: $($a.model)  upstream: $($a.upstream)"
    } else {
        Write-Host "[agent] not running on 8081"
    }
    if ($gpu) {
        Write-Host "[gpu]"
        Write-Host $gpu
    }
}

function Start-FastAgent {
    $a = Get-AgentHealth
    if ($a -and $a.service -eq "rustmoe-fast-agent") {
        Write-Host "[fast] agent already running on 8081"
        return
    }
    $exe = Join-Path $Root "target\release\rustmoe-agent.exe"
    if (-not (Test-Path $exe)) {
        throw "rustmoe-agent.exe not found; run: cargo build --release -p rustmoe-server --bin rustmoe-agent"
    }
    $upstream = "http://$Addr"
    $model = if ($env:RUSTMOE_FAST_MODEL) { $env:RUSTMOE_FAST_MODEL } else { "qwen3.8-flash-next-iq3_s" }
    $agentRoot = if ($env:RUSTMOE_AGENT_ROOT) { $env:RUSTMOE_AGENT_ROOT } else { Join-Path $Root "workspace" }
    $skillsDir = if ($env:RUSTMOE_SKILLS_DIR) { $env:RUSTMOE_SKILLS_DIR } else { Join-Path $Root "skills" }
    Write-Host "[fast] starting RustMoE fast agent on 8081 ..."
    Start-Process -FilePath $exe -ArgumentList @(
        "--addr", "127.0.0.1:8081",
        "--upstream", $upstream,
        "--model", $model,
        "--agent-root", $agentRoot,
        "--skills-dir", $skillsDir,
        "--agent-allow-shell",
        "--agent-max-steps", "24"
    ) -WorkingDirectory $Root -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        $a = Get-AgentHealth
        if ($a -and $a.service -eq "rustmoe-fast-agent") {
            Write-Host "[fast] agent ready: http://127.0.0.1:8081/agent"
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "fast agent did not start on 8081"
}

function Start-Fast {
    $h = Get-Health
    if (-not ($h -and $h.service -eq "strata")) {
        if (Get-Process -Name rustmoe -ErrorAction SilentlyContinue) {
            Write-Host "[fast] stopping rustmoe to free VRAM"
            Stop-EngineProcesses
        }
        if (-not (Test-Path (Join-Path $StrataHome "START-HERE.bat"))) {
            throw "Strata launcher not found: $StrataHome\START-HERE.bat"
        }
        Write-Host "[fast] starting Strata from $StrataHome ..."
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "START-HERE.bat" -WorkingDirectory $StrataHome -WindowStyle Hidden
        $up = Wait-Health { param($x) $x.service -eq "strata" } 300
        if (-not $up) { throw "Strata did not become ready on $Addr within 300s" }
        Write-Host "[fast] ready: $($up.model)"
    } else {
        Write-Host "[fast] Strata is already running on $Addr"
    }
    Start-FastAgent
}

function Start-Deep {
    $h = Get-Health
    if ($h -and $h.backend -and $h.model -like "deepseek*") {
        Write-Host "[deep] RustMoE is already running on $Addr"
        return
    }
    if (Get-Process -Name rustmoe-agent -ErrorAction SilentlyContinue) {
        Write-Host "[deep] stopping fast agent"
        Get-Process -Name rustmoe-agent -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    if (Get-Process -Name strata,strata-vision -ErrorAction SilentlyContinue) {
        Write-Host "[deep] stopping Strata to free VRAM"
        Stop-EngineProcesses
    }
    if (Get-Process -Name rustmoe -ErrorAction SilentlyContinue) {
        Write-Host "[deep] stopping previous rustmoe"
        Get-Process -Name rustmoe -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    if (-not (Test-Path (Join-Path $Root "startdeepseek.bat"))) {
        throw "startdeepseek.bat not found in $Root"
    }
    Write-Host "[deep] starting RustMoE (model load may take minutes) ..."
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "startdeepseek.bat" -WorkingDirectory $Root -WindowStyle Hidden
    $up = Wait-Health { param($x) $x.backend -and $x.model -like "deepseek*" } 900
    if (-not $up) { throw "RustMoE did not become ready on $Addr within 900s; check the loader window/log" }
    Write-Host "[deep] ready: $($up.model) backend=$($up.backend)"
}

Show-Status
switch ($Action) {
    "fast" { Start-Fast; Show-Status }
    "deep" { Start-Deep; Show-Status }
    default { }
}
