param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [int]$MaxTokens = 1
)
$ErrorActionPreference = "Stop"
$BaseUrl = $BaseUrl.TrimEnd('/')
if ($BaseUrl.EndsWith('/v1')) { $BaseUrl = $BaseUrl.Substring(0, $BaseUrl.Length - 3) }
$chatUrl = "$BaseUrl/v1/chat/completions"

Write-Host "base:  $BaseUrl"
Write-Host "chat:  $chatUrl"
Write-Host "note:  the first real request can take a minute (prompt prefill)"
Write-Host ""

try {
    $health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 10
    Write-Host "health: 200"
    $health | ConvertTo-Json -Depth 5
} catch {
    Write-Host "health: FAILED"
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails) { Write-Host $_.ErrorDetails.Message }
    exit 1
}

try {
    $models = Invoke-RestMethod -Method Get -Uri "$BaseUrl/v1/models" -TimeoutSec 10
    Write-Host "models: 200"
    $models | ConvertTo-Json -Depth 5
} catch {
    Write-Host "models: FAILED"
    Write-Host $_.Exception.Message
    exit 1
}

$modelId = "deepseek-v4.1-flash-1017"
if ($models.data -and $models.data.Count -gt 0 -and $models.data[0].id) {
    $modelId = $models.data[0].id
}

$body = @{
    model = $modelId
    messages = @(@{ role = "user"; content = "Hello" })
    max_tokens = $MaxTokens
    temperature = 0
} | ConvertTo-Json -Depth 8

try {
    $response = Invoke-WebRequest -UseBasicParsing -Method Post -Uri $chatUrl -ContentType "application/json" -Body $body -TimeoutSec 300
    Write-Host "chat: $($response.StatusCode)"
    Write-Host $response.Content
    try {
        $obj = $response.Content | ConvertFrom-Json
        $prefill = [double]$obj.rustmoe.prefill_ms / 1000.0
        $decode = [double]$obj.rustmoe.decode_ms / 1000.0
        $completion = [int]$obj.usage.completion_tokens
        $overall = if (($prefill + $decode) -gt 0) { $completion / ($prefill + $decode) } else { 0 }
        $decodeTps = if ($decode -gt 0) { $completion / $decode } else { 0 }
        Write-Host ("timing: prefill {0:N1}s, decode {1:N1}s, overall {2:N2} tok/s, decode {3:N2} tok/s" -f $prefill, $decode, $overall, $decodeTps)
        if ($prefill -gt 30) {
            Write-Host "[warn] slow prefill: the model is probably on an HDD; run move-model-to-ssd.bat"
        }
    } catch {}
    Write-Host "[ok] OpenAI-compatible chat endpoint works"
} catch {
    Write-Host "chat: FAILED"
    if ($_.Exception.Response) {
        Write-Host "status: $($_.Exception.Response.StatusCode.value__)"
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        Write-Host $reader.ReadToEnd()
    } else {
        Write-Host $_.Exception.Message
    }
    exit 1
}
