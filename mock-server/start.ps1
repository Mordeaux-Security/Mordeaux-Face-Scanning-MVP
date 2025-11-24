# Start Mock Server - Phase 3 (Windows)
# =====================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🎭 Mock Server - Phase 3" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt

# Print fixture summary
Write-Host ""
Write-Host "Generating fixtures..." -ForegroundColor Yellow
python fixtures.py

# Start server
Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
Write-Host "API docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Mock config: http://localhost:8000/mock/fixtures" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

python app.py

