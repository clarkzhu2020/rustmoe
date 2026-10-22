param(
    [string]$Branch = "dsv41-all",
    [string]$Architectures = "86;89",
    [string]$Generator = "Visual Studio 18 2026"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root ".models"
$src = Join-Path $models "llama.cpp"
$build = Join-Path $models "llama-build"
New-Item -ItemType Directory -Force -Path $models | Out-Null
if (-not (Test-Path (Join-Path $src ".git"))) {
    git clone --depth 1 --branch $Branch https://github.com/unslothai/llama.cpp.git $src
}
$pinned = "d1781c3ea6ab7c75c635a8fea60d2b3f4937c645"
git -C $src fetch --depth 1 origin $pinned
if ($LASTEXITCODE -eq 0) { git -C $src checkout --detach $pinned }
$patch = Join-Path $root "patches\llama-moe-cache-stats.patch"
if (Test-Path $patch) {
    git -C $src apply --check $patch
    if ($LASTEXITCODE -eq 0) { git -C $src apply $patch }
}
$cfg = @("-S", $src, "-B", $build, "-G", $Generator, "-A", "x64",
    "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=$Architectures",
    "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF")
cmake @cfg
cmake --build $build --config Release --target llama llama-cli llama-server -- /m:16
Write-Host "llama.cpp reference built in $build"
Write-Host "Now run: cargo build --release -p rustmoe-server --features llama"
