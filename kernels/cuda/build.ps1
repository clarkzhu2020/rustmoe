$ErrorActionPreference = "Stop"
$nvcc = Get-Command nvcc -ErrorAction Stop
& $nvcc.Source -O3 -arch=sm_89 -cubin "$PSScriptRoot\mxfp4_gemv.cu" -o "$PSScriptRoot\mxfp4_gemv.cubin"
Write-Host "built $PSScriptRoot\mxfp4_gemv.cubin"
