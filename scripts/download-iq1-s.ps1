param(
    [string]$Quant = "UD-IQ1_S",
    [string]$Destination = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Destination) { $Destination = Join-Path $root ".models\$Quant" }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$base = "https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main/$Quant"
$files = @(
    "DeepSeek-V4.1-Flash-1017-$Quant-00001-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-$Quant-00002-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-$Quant-00003-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-$Quant-00004-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-$Quant-00005-of-00005.gguf"
)
foreach ($file in $files) {
    $out = Join-Path $Destination $file
    Write-Host "download $file"
    curl.exe -L --fail --retry 10 --retry-delay 5 -C - -o $out "$base/$file"
    if ($LASTEXITCODE -ne 0) { throw "download failed: $file" }
}
Write-Host "model ready in $Destination"
