# Mordeaux Face Scanning MVP - Docker Build Script (PowerShell)
# This script builds and starts the complete Docker environment

param(
    [string]$Command = "build"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if .env file exists
function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Warning ".env file not found. Creating from .env.example..."
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Success ".env file created from .env.example"
            Write-Warning "Please edit .env file with your actual configuration values"
        } else {
            Write-Error ".env.example file not found. Cannot create .env file."
            exit 1
        }
    } else {
        Write-Success ".env file found"
    }
}

# Function to check Docker and Docker Compose
function Test-Docker {
    Write-Status "Checking Docker installation..."
    
    try {
        $null = Get-Command docker -ErrorAction Stop
    } catch {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    try {
        $null = Get-Command docker-compose -ErrorAction Stop
        $ComposeCmd = "docker-compose"
    } catch {
        try {
            $null = docker compose version 2>$null
            $ComposeCmd = "docker compose"
        } catch {
            Write-Error "Docker Compose is not available. Please install Docker Desktop with Compose support."
            exit 1
        }
    }
    
    Write-Success "Docker and Docker Compose are available"
    return $ComposeCmd
}

# Function to build and start services
function Start-Services {
    param([string]$ComposeCmd)
    
    Write-Status "Building and starting Docker services..."
    
    # Build and start all services
    Invoke-Expression "$ComposeCmd up --build -d"
    
    Write-Success "All services started successfully!"
}

# Function to show service status
function Show-Status {
    param([string]$ComposeCmd)
    
    Write-Status "Checking service status..."
    Invoke-Expression "$ComposeCmd ps"
}

# Function to show logs
function Show-Logs {
    param([string]$ComposeCmd)
    
    Write-Status "Showing recent logs..."
    Invoke-Expression "$ComposeCmd logs --tail=50"
}

# Function to show service URLs
function Show-Urls {
    Write-Success "Service URLs:"
    Write-Host "  🌐 Frontend:        http://localhost:3000" -ForegroundColor Cyan
    Write-Host "  🔧 Backend API:     http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  🧠 Face Pipeline:   http://localhost:8001" -ForegroundColor Cyan
    Write-Host "  🗄️  MinIO Console:   http://localhost:9001" -ForegroundColor Cyan
    Write-Host "  📊 pgAdmin:         http://localhost:5050" -ForegroundColor Cyan
    Write-Host "  🔍 Qdrant:          http://localhost:6333" -ForegroundColor Cyan
    Write-Host "  🌍 Nginx (Main):    http://localhost:80" -ForegroundColor Cyan
    Write-Host ""
    Write-Status "Default credentials:"
    Write-Host "  MinIO: minioadmin / minioadmin" -ForegroundColor Yellow
    Write-Host "  pgAdmin: admin@admin.com / admin" -ForegroundColor Yellow
}

# Function to stop services
function Stop-Services {
    param([string]$ComposeCmd)
    
    Write-Status "Stopping all services..."
    Invoke-Expression "$ComposeCmd down"
    Write-Success "All services stopped"
}

# Function to clean up
function Remove-All {
    param([string]$ComposeCmd)
    
    Write-Status "Cleaning up Docker resources..."
    Invoke-Expression "$ComposeCmd down -v --remove-orphans"
    docker system prune -f
    Write-Success "Cleanup completed"
}

# Main script logic
function Main {
    param([string]$Cmd)
    
    Write-Host "🐳 Mordeaux Face Scanning MVP - Docker Build Script" -ForegroundColor Magenta
    Write-Host "==================================================" -ForegroundColor Magenta
    Write-Host ""
    
    $ComposeCmd = Test-Docker
    
    switch ($Cmd.ToLower()) {
        "build" {
            Test-EnvFile
            Start-Services -ComposeCmd $ComposeCmd
            Show-Status -ComposeCmd $ComposeCmd
            Show-Urls
        }
        "start" {
            Test-EnvFile
            Start-Services -ComposeCmd $ComposeCmd
            Show-Status -ComposeCmd $ComposeCmd
            Show-Urls
        }
        "stop" {
            Stop-Services -ComposeCmd $ComposeCmd
        }
        "restart" {
            Stop-Services -ComposeCmd $ComposeCmd
            Start-Sleep -Seconds 2
            Test-EnvFile
            Start-Services -ComposeCmd $ComposeCmd
            Show-Status -ComposeCmd $ComposeCmd
            Show-Urls
        }
        "status" {
            Show-Status -ComposeCmd $ComposeCmd
        }
        "logs" {
            Show-Logs -ComposeCmd $ComposeCmd
        }
        "cleanup" {
            Remove-All -ComposeCmd $ComposeCmd
        }
        "help" {
            Write-Host "Usage: .\build-docker.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  build, start  - Build and start all services (default)"
            Write-Host "  stop          - Stop all services"
            Write-Host "  restart       - Restart all services"
            Write-Host "  status        - Show service status"
            Write-Host "  logs          - Show recent logs"
            Write-Host "  cleanup       - Stop services and clean up resources"
            Write-Host "  help          - Show this help message"
        }
        default {
            Write-Error "Unknown command: $Cmd"
            Write-Host "Use '.\build-docker.ps1 help' for available commands"
            exit 1
        }
    }
}

# Run main function
Main -Cmd $Command
