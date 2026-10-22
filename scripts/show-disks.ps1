Get-Partition | Where-Object { $_.DriveLetter } | ForEach-Object {
    $part = $_
    $disk = Get-PhysicalDisk -DeviceNumber $part.DiskNumber
    [pscustomobject]@{
        Drive  = [string]$part.DriveLetter
        Disk   = $part.DiskNumber
        Model  = $disk.FriendlyName
        Media  = $disk.MediaType
        Bus    = $disk.BusType
        DiskGB = [math]::Round($disk.Size / 1GB, 0)
        PartGB = [math]::Round($part.Size / 1GB, 0)
        FreeGB = [math]::Round((Get-Volume -DriveLetter $part.DriveLetter).SizeRemaining / 1GB, 0)
    }
} | Sort-Object Drive | Format-Table -AutoSize
