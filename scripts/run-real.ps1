param(
    [string]$Model = "",
    [int]$Context = 0,
    [string]$Addr = "127.0.0.1:8080",
    [int]$Gpu = 1
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Model) {
    $Model = Join-Path $root ".models\UD-IQ1_S\DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"
}
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$exe = Join-Path $root "target\release\rustmoe.exe"
if (-not (Test-Path $exe)) { throw "build first: cargo build --release -p rustmoe-server --features llama" }
$serveArgs = @("serve", $Model, "--backend", "llama", "--addr", $Addr,
    "--context", "$Context", "--llama-batch", "512", "--threads", "16",
    "--load-mode", "1", "--lazy-mode", "on")
& $exe @serveArgs
