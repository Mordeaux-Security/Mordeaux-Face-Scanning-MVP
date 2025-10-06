# PowerShell script to start Mordeaux Face Scanning MVP locally

Write-Host "Starting Mordeaux Face Scanning MVP locally..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Create data directory if it doesn't exist
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
    Write-Host "Created data directory" -ForegroundColor Yellow
}

# Start the services
Write-Host "Starting Docker services..." -ForegroundColor Yellow
docker-compose up --build -d

Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service health
Write-Host "Checking service health..." -ForegroundColor Yellow
docker-compose ps

Write-Host ""
Write-Host "Services should be available at:" -ForegroundColor Green
Write-Host "- Frontend: http://localhost" -ForegroundColor White
Write-Host "- Backend API: http://localhost/api" -ForegroundColor White
Write-Host "- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "- Qdrant: http://localhost:6333" -ForegroundColor White

Write-Host ""
Write-Host "To stop services, run: docker-compose down" -ForegroundColor Cyan
Write-Host "To view logs, run: docker-compose logs -f" -ForegroundColor Cyan
