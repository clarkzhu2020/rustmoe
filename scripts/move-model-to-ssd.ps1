param(
    [string]$Source = "",
    [string]$TargetRoot = "C:\rustmoe-models",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = Join-Path $root ".models\UD-IQ1_S" }
$Source = (Resolve-Path -LiteralPath $Source).Path
$name = Split-Path -Leaf $Source
$target = Join-Path $TargetRoot $name
$modelName = "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"

$running = Get-Process rustmoe -ErrorAction SilentlyContinue
if ($running) {
    throw "rustmoe.exe is running (PID $($running.Id)); stop it before copying the model"
}

$files = Get-ChildItem -LiteralPath $Source -Filter *.gguf -File
if ($files.Count -ne 5) { throw "expected 5 GGUF shards in $Source, found $($files.Count)" }
$bytes = ($files | Measure-Object -Property Length -Sum).Sum
$gib = [math]::Round($bytes / 1GB, 1)

$targetDrive = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath (Split-Path -Parent $TargetRoot)).Path).TrimEnd('\').Substring(0, 1)
$volume = Get-Volume -DriveLetter $targetDrive
if ($volume.SizeRemaining -lt ($bytes + 20GB)) {
    throw ("target {0}: has {1:N1} GiB free, need at least {2:N1} GiB" -f $targetDrive, ($volume.SizeRemaining / 1GB), (($bytes + 20GB) / 1GB))
}

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
if ((Test-Path -LiteralPath $target) -and -not $Force) {
    $existing = Get-ChildItem -LiteralPath $target -Filter *.gguf -File
    $existingBytes = ($existing | Measure-Object -Property Length -Sum).Sum
    if ($existing.Count -eq 5 -and $existingBytes -eq $bytes) {
        Write-Host "target already has a complete copy: $target"
    } else {
        throw "target exists but is incomplete: $target; rerun with -Force"
    }
} else {
    Write-Host ("copying {0} GiB from {1} to {2}" -f $gib, $Source, $target)
    Write-Host "this is a one-time copy and can take 20-40 minutes on a hard disk"
    $rc = robocopy $Source $target /J /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NP
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
}

$missing = @()
foreach ($f in $files) {
    $dst = Join-Path $target $f.Name
    if (-not (Test-Path -LiteralPath $dst)) { $missing += $f.Name; continue }
    $size = (Get-Item -LiteralPath $dst).Length
    if ($size -ne $f.Length) { $missing += "$($f.Name) size $size != $($f.Length)" }
}
if ($missing.Count -gt 0) { throw ("copy verification failed: " + ($missing -join "; ")) }

$model = Join-Path $target $modelName
$pointer = Join-Path $root ".models\model-path.txt"
[System.IO.File]::WriteAllText($pointer, $model, [System.Text.UTF8Encoding]::new($false))
Write-Host "copy verified: $target"
Write-Host "wrote model pointer: $pointer"
Write-Host "now run: startdeepseek.bat"
