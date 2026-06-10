$nodeUrl = "https://nodejs.org/dist/v20.11.1/node-v20.11.1-win-x64.zip"
$zipPath = Join-Path $pwd "node-portable.zip"
$destPath = Join-Path $pwd ".node"

if (Test-Path $destPath) {
    Write-Host "Portable Node directory already exists at $destPath."
} else {
    Write-Host "Creating portable Node directory at $destPath..."
    New-Item -ItemType Directory -Path $destPath -Force | Out-Null
    
    Write-Host "Downloading Node.js v20.11.1 from $nodeUrl..."
    Invoke-WebRequest -Uri $nodeUrl -OutFile $zipPath
    
    Write-Host "Extracting zip archive..."
    Expand-Archive -Path $zipPath -DestinationPath $destPath
    
    Write-Host "Cleaning up files..."
    $extractedFolder = Get-ChildItem -Path $destPath -Directory | Select-Object -First 1
    if ($extractedFolder) {
        # Move all files up to the root of .node
        Get-ChildItem -Path $extractedFolder.FullName | Move-Item -Destination $destPath -Force
        Remove-Item -Path $extractedFolder.FullName -Recurse -Force
    }
    Remove-Item -Path $zipPath -Force
    Write-Host "Portable Node.js successfully set up!"
}

# Print versions to verify
$nodeExe = Join-Path $destPath "node.exe"
$npmCmd = Join-Path $destPath "npm.cmd"
if (Test-Path $nodeExe) {
    $nodeVer = & $nodeExe -v
    $npmVer = & $npmCmd -v
    Write-Host "Node version: $nodeVer"
    Write-Host "NPM version: $npmVer"
} else {
    Write-Error "node.exe was not found at $nodeExe!"
}
