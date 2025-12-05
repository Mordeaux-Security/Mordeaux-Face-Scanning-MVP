# Simple crawler runner with log capture
# Usage: .\run-crawler.ps1 [--sites-file sites.txt] [--sites url1 url2 ...]

param(
    [string]$SitesFile = "sites.txt",
    [string[]]$Sites = @(),
    [string]$LogFile = "debugamd.txt"
)

# Change to script directory
Set-Location $PSScriptRoot

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Virtual environment not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "src"

# Build command
$cmd = "python -m diabetes_crawler.main"

if ($SitesFile -and (Test-Path $SitesFile)) {
    $cmd += " --sites-file $SitesFile"
} elseif ($Sites.Count -gt 0) {
    $sitesArg = $Sites -join " "
    $cmd += " --sites $sitesArg"
} else {
    Write-Host "Error: No sites provided. Use -SitesFile or -Sites" -ForegroundColor Red
    exit 1
}

Write-Host "Running: $cmd" -ForegroundColor Green
Write-Host "Logging to: $LogFile" -ForegroundColor Green
Write-Host ""

# Run command and capture all output (stdout + stderr) to both console and file
& Invoke-Expression $cmd *>&1 | Tee-Object -FilePath $LogFile


