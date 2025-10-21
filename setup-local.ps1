# Mordeaux Face Scanning MVP - Local Setup Script
# This script sets up the complete local development environment

param(
    [switch]$SkipModels = $false,
    [switch]$SkipDocker = $false
)

# Colors for output
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

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to create virtual environment
function New-VirtualEnvironment {
    param([string]$Path, [string]$Name)
    
    Write-Status "Creating virtual environment: $Name"
    
    if (Test-Path "$Path\$Name") {
        Write-Warning "Virtual environment $Name already exists. Removing..."
        Remove-Item "$Path\$Name" -Recurse -Force
    }
    
    python -m venv "$Path\$Name"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Virtual environment created: $Name"
    } else {
        Write-Error "Failed to create virtual environment: $Name"
        exit 1
    }
}

# Function to activate virtual environment and install requirements
function Install-Requirements {
    param([string]$VenvPath, [string]$RequirementsFile, [string]$Name)
    
    Write-Status "Installing requirements for $Name..."
    
    $activateScript = if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "$VenvPath\Scripts\Activate.ps1"
    } else {
        "$VenvPath/bin/activate"
    }
    
    if (Test-Path $activateScript) {
        & $activateScript
        pip install --upgrade pip
        pip install -r $RequirementsFile
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Requirements installed for $Name"
        } else {
            Write-Error "Failed to install requirements for $Name"
            exit 1
        }
    } else {
        Write-Error "Virtual environment activation script not found: $activateScript"
        exit 1
    }
}

# Function to download InsightFace models
function Download-InsightFaceModels {
    Write-Status "Downloading InsightFace models..."
    
    $pythonScript = @"
import insightface
import os

try:
    print("Downloading InsightFace buffalo_l model...")
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    print("Model downloaded successfully!")
    
    # Check model location
    model_path = os.path.expanduser('~/.insightface')
    if os.path.exists(model_path):
        print(f"Models stored at: {model_path}")
        for root, dirs, files in os.walk(model_path):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
    else:
        print("Model path not found")
        
except Exception as e:
    print(f"Error downloading models: {e}")
    exit(1)
"@
    
    $pythonScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "InsightFace models downloaded successfully"
    } else {
        Write-Warning "Failed to download InsightFace models. They will be downloaded on first use."
    }
}

# Function to setup frontend dependencies
function Setup-Frontend {
    Write-Status "Setting up frontend dependencies..."
    
    if (Test-Path "frontend/package.json") {
        Set-Location frontend
        
        if (Test-Command "npm") {
            npm install
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Frontend dependencies installed"
            } else {
                Write-Warning "Failed to install frontend dependencies"
            }
        } else {
            Write-Warning "npm not found. Please install Node.js to build the frontend."
        }
        
        Set-Location ..
    } else {
        Write-Warning "Frontend package.json not found"
    }
}

# Function to verify Docker setup
function Test-DockerSetup {
    Write-Status "Verifying Docker setup..."
    
    if (Test-Command "docker") {
        if (Test-Command "docker-compose") {
            $composeCmd = "docker-compose"
        } elseif (docker compose version 2>$null) {
            $composeCmd = "docker compose"
        } else {
            Write-Error "Docker Compose not found. Please install Docker Desktop."
            return $false
        }
        
        # Test docker-compose configuration
        & $composeCmd config --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker configuration is valid"
            return $true
        } else {
            Write-Error "Docker configuration is invalid"
            return $false
        }
    } else {
        Write-Error "Docker not found. Please install Docker Desktop."
        return $false
    }
}

# Function to create data directories
function New-DataDirectories {
    Write-Status "Creating data directories..."
    
    $directories = @(
        "data/images",
        "data/models", 
        "data/cache",
        "data/logs",
        "data/backups"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        } else {
            Write-Status "Directory already exists: $dir"
        }
    }
}

# Function to test local services
function Test-LocalServices {
    Write-Status "Testing local services..."
    
    # Test face-pipeline
    Set-Location face-pipeline
    try {
        & "..\venv\Scripts\python.exe" -c "import fastapi, uvicorn; print('Face-pipeline dependencies OK')"
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Face-pipeline dependencies verified"
        } else {
            Write-Warning "Face-pipeline dependencies not working"
        }
    } catch {
        Write-Warning "Could not test face-pipeline dependencies"
    }
    Set-Location ..
    
    # Test backend
    Set-Location backend
    try {
        & "..\venv\Scripts\python.exe" -c "import fastapi, celery; print('Backend dependencies OK')"
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Backend dependencies verified"
        } else {
            Write-Warning "Backend dependencies not working"
        }
    } catch {
        Write-Warning "Could not test backend dependencies"
    }
    Set-Location ..
}

# Main setup function
function Main {
    Write-Host "🚀 Mordeaux Face Scanning MVP - Local Setup" -ForegroundColor Magenta
    Write-Host "===========================================" -ForegroundColor Magenta
    Write-Host ""
    
    # Check prerequisites
    Write-Status "Checking prerequisites..."
    
    if (-not (Test-Command "python")) {
        Write-Error "Python not found. Please install Python 3.11+ first."
        exit 1
    }
    
    $pythonVersion = python --version
    Write-Success "Found: $pythonVersion"
    
    # Create virtual environment
    New-VirtualEnvironment -Path "." -Name "venv"
    
    # Install backend requirements
    Install-Requirements -VenvPath "venv" -RequirementsFile "backend/requirements.txt" -Name "Backend"
    
    # Install face-pipeline requirements
    Install-Requirements -VenvPath "venv" -RequirementsFile "face-pipeline/requirements.txt" -Name "Face-Pipeline"
    
    # Install worker requirements
    Install-Requirements -VenvPath "venv" -RequirementsFile "worker/requirements.txt" -Name "Worker"
    
    # Download models if not skipped
    if (-not $SkipModels) {
        Download-InsightFaceModels
    } else {
        Write-Warning "Skipping model download (use -SkipModels:$false to download)"
    }
    
    # Setup frontend
    Setup-Frontend
    
    # Create data directories
    New-DataDirectories
    
    # Verify Docker setup if not skipped
    if (-not $SkipDocker) {
        if (Test-DockerSetup) {
            Write-Success "Docker setup verified"
        } else {
            Write-Warning "Docker setup has issues. You can still run services locally."
        }
    } else {
        Write-Warning "Skipping Docker verification"
    }
    
    # Test local services
    Test-LocalServices
    
    Write-Host ""
    Write-Success "🎉 Local setup completed successfully!"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "2. Run face-pipeline: cd face-pipeline && python main.py" -ForegroundColor Cyan
    Write-Host "3. Run backend: cd backend && uvicorn app.main:app --reload" -ForegroundColor Cyan
    Write-Host "4. Or use Docker: .\build-docker.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Service URLs (when running):" -ForegroundColor Yellow
    Write-Host "  - Face Pipeline: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  - Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  - Frontend: http://localhost:3000" -ForegroundColor Cyan
}

# Run main function
Main
