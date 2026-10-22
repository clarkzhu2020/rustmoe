param(
    [Parameter(Mandatory = $true)][string]$Path
)
$ErrorActionPreference = "Stop"
$full = (Resolve-Path -LiteralPath $Path).Path
$letter = [System.IO.Path]::GetPathRoot($full).TrimEnd('\').Substring(0, 1)
$dir = Split-Path -Parent $full
$bytes = (Get-ChildItem -LiteralPath $dir -Filter *.gguf -File | Measure-Object -Property Length -Sum).Sum
$gib = if ($bytes) { [math]::Round($bytes / 1GB, 1) } else { 0 }
$media = "unknown"
$bus = "unknown"
$name = "unknown"
try {
    $part = Get-Partition -DriveLetter $letter | Select-Object -First 1
    $disk = Get-PhysicalDisk -DeviceNumber $part.DiskNumber
    $media = [string]$disk.MediaType
    $bus = [string]$disk.BusType
    $name = [string]$disk.FriendlyName
} catch {
    $media = "unknown"
}
$speed = switch ($bus) {
    "NVMe" { 3000 }
    "SATA" { if ($media -eq "HDD") { 200 } else { 550 } }
    default { 200 }
}
$minutes = if ($bytes) { [math]::Round(($bytes / 1MB) / $speed / 60, 1) } else { 0 }
"{0}: {1}/{2} ({3}); model {4} GiB; worst-case full read ~{5} min at {6} MB/s" -f $letter, $media, $bus, $name, $gib, $minutes, $speed
