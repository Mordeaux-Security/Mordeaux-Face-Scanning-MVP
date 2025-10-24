# PowerShell script to start Mordeaux Face Scanning MVP with GPU acceleration (Simplified)

Write-Host "Starting Mordeaux Face Scanning MVP with GPU acceleration (Simplified)..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check for NVIDIA GPU support
Write-Host "Checking for NVIDIA GPU support..." -ForegroundColor Yellow
try {
    nvidia-smi | Out-Null
    Write-Host "✅ NVIDIA GPU detected!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Warning: NVIDIA GPU not detected. GPU acceleration may not work." -ForegroundColor Yellow
}

# Stop existing containers
Write-Host "Stopping existing containers..." -ForegroundColor Yellow
docker-compose down

# Create data directory if it doesn't exist
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
    Write-Host "Created data directory" -ForegroundColor Yellow
}

# Start the GPU-optimized services using simplified approach
Write-Host "Starting GPU-optimized Docker services (simplified approach)..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu-simple.yml up --build -d

Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Check service health
Write-Host "Checking service health..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu-simple.yml ps

Write-Host ""
Write-Host "🚀 GPU-Optimized Services are now running!" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "- Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "- Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "- Face Pipeline: http://localhost:8001" -ForegroundColor White
Write-Host "- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "- Qdrant: http://localhost:6333" -ForegroundColor White
Write-Host ""
Write-Host "GPU Configuration:" -ForegroundColor Cyan
Write-Host "- CUDA_VISIBLE_DEVICES: 0" -ForegroundColor White
Write-Host "- ONNX_PROVIDERS: CUDAExecutionProvider,CPUExecutionProvider" -ForegroundColor White
Write-Host ""
Write-Host "GPU Status:" -ForegroundColor Cyan
try {
    nvidia-smi
} catch {
    Write-Host "Could not check GPU status" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To stop services, run: docker-compose -f docker-compose.gpu-simple.yml down" -ForegroundColor Cyan
Write-Host "To view logs, run: docker-compose -f docker-compose.gpu-simple.yml logs -f" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: This simplified approach uses CPU containers with GPU runtime access." -ForegroundColor Yellow
Write-Host "GPU acceleration will be handled by ONNX Runtime and InsightFace at runtime." -ForegroundColor Yellow
