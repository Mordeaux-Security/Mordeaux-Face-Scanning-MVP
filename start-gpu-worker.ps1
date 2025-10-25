# Windows GPU Worker Startup Script
# Starts the GPU worker service and Docker Compose services

param(
    [switch]$SkipWorker,
    [switch]$SkipDocker,
    [string]$WorkerUrl = "http://localhost:8765"
)

Write-Host "=== Mordeaux GPU Worker Startup Script ===" -ForegroundColor Green

# Check if we're running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning "This script should be run as Administrator for best results"
}

# Function to check if a port is available
function Test-Port {
    param([int]$Port)
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    }
    catch {
        return $false
    }
}

# Function to wait for a service to be ready
function Wait-ForService {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30,
        [int]$IntervalSeconds = 2
    )
    
    $startTime = Get-Date
    $timeout = $startTime.AddSeconds($TimeoutSeconds)
    
    while ((Get-Date) -lt $timeout) {
        try {
            $response = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "✓ Service is ready at $Url" -ForegroundColor Green
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Start-Sleep -Seconds $IntervalSeconds
    }
    
    Write-Warning "Service at $Url did not become ready within $TimeoutSeconds seconds"
    return $false
}

# Function to start GPU worker
function Start-GPUWorker {
    Write-Host "Starting GPU Worker Service..." -ForegroundColor Yellow
    
    # Check if port 8765 is available
    if (-not (Test-Port -Port 8765)) {
        Write-Warning "Port 8765 is not available. Checking if GPU worker is already running..."
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8765/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "✓ GPU worker is already running" -ForegroundColor Green
                return $true
            }
        }
        catch {
            Write-Error "Port 8765 is occupied but GPU worker is not responding. Please stop the conflicting service."
            return $false
        }
    }
    
    # Check if GPU worker directory exists
    $workerDir = "backend\gpu_worker"
    if (-not (Test-Path $workerDir)) {
        Write-Error "GPU worker directory not found: $workerDir"
        return $false
    }
    
    # Start GPU worker in background
    try {
        $workerScript = Join-Path $workerDir "launch.py"
        if (-not (Test-Path $workerScript)) {
            Write-Error "GPU worker launcher not found: $workerScript"
            return $false
        }
        
        Write-Host "Starting GPU worker launcher..." -ForegroundColor Yellow
        $workerProcess = Start-Process -FilePath "python" -ArgumentList $workerScript -WorkingDirectory $workerDir -WindowStyle Hidden -PassThru
        
        # Wait for GPU worker to be ready
        if (Wait-ForService -Url "http://localhost:8765/health" -TimeoutSeconds 30) {
            Write-Host "✓ GPU worker started successfully (PID: $($workerProcess.Id))" -ForegroundColor Green
            return $true
        } else {
            Write-Error "GPU worker failed to start or become ready"
            $workerProcess.Kill()
            return $false
        }
    }
    catch {
        Write-Error "Failed to start GPU worker: $_"
        return $false
    }
}

# Function to start Docker services
function Start-DockerServices {
    Write-Host "Starting Docker services..." -ForegroundColor Yellow
    
    # Check if Docker is running
    try {
        docker version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is not running. Please start Docker Desktop first."
            return $false
        }
    }
    catch {
        Write-Error "Docker is not available. Please install and start Docker Desktop."
        return $false
    }
    
    # Check if docker-compose.yml exists
    if (-not (Test-Path "docker-compose.yml")) {
        Write-Error "docker-compose.yml not found in current directory"
        return $false
    }
    
    # Start Docker services
    try {
        Write-Host "Starting Docker Compose services..." -ForegroundColor Yellow
        docker-compose up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Docker services started successfully" -ForegroundColor Green
            
            # Wait for backend service to be ready
            if (Wait-ForService -Url "http://localhost:8000/health" -TimeoutSeconds 60) {
                Write-Host "✓ Backend service is ready" -ForegroundColor Green
                return $true
            } else {
                Write-Warning "Backend service may not be fully ready yet"
                return $true
            }
        } else {
            Write-Error "Failed to start Docker services"
            return $false
        }
    }
    catch {
        Write-Error "Error starting Docker services: $_"
        return $false
    }
}

# Main execution
try {
    $success = $true
    
    # Start GPU worker if not skipped
    if (-not $SkipWorker) {
        if (-not (Start-GPUWorker)) {
            $success = $false
        }
    } else {
        Write-Host "Skipping GPU worker startup" -ForegroundColor Yellow
    }
    
    # Start Docker services if not skipped
    if (-not $SkipDocker) {
        if (-not (Start-DockerServices)) {
            $success = $false
        }
    } else {
        Write-Host "Skipping Docker services startup" -ForegroundColor Yellow
    }
    
    if ($success) {
        Write-Host "`n=== Startup Complete ===" -ForegroundColor Green
        Write-Host "GPU Worker: http://localhost:8765" -ForegroundColor Cyan
        Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
        Write-Host "`nTo stop services, run: docker-compose down" -ForegroundColor Yellow
    } else {
        Write-Host "`n=== Startup Failed ===" -ForegroundColor Red
        Write-Host "Please check the error messages above and try again." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Error "Unexpected error during startup: $_"
    exit 1
}
