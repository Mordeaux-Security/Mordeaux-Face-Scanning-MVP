# PowerShell script to download thumbnails from MinIO
# Usage: .\scripts\download-thumbs.ps1

Write-Host "=== Downloading Thumbnails from MinIO ===" -ForegroundColor Green
Write-Host ""

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Working directory: $projectRoot" -ForegroundColor Cyan
Write-Host ""

# Run the Python script
try {
    python scripts/download_thumbs.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Script failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Done!" -ForegroundColor Green
