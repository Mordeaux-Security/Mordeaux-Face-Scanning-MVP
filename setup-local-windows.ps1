# Mordeaux Face Scanning MVP - Windows Local Setup Script
# This script sets up the complete local development environment for Windows

param(
    [switch]$SkipModels = $false,
    [switch]$SkipDocker = $false,
    [switch]$UseDockerOnly = $false
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

# Function to install Visual Studio Build Tools
function Install-BuildTools {
    Write-Status "Checking for Visual Studio Build Tools..."
    
    # Check if Visual Studio Build Tools are already installed
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $buildTools = & $vsWhere -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($buildTools) {
            Write-Success "Visual Studio Build Tools found: $buildTools"
            return $true
        }
    }
    
    Write-Warning "Visual Studio Build Tools not found."
    Write-Host ""
    Write-Host "To install InsightFace locally, you need Microsoft Visual C++ Build Tools." -ForegroundColor Yellow
    Write-Host "You have several options:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Install Build Tools (Recommended for local development)" -ForegroundColor Cyan
    Write-Host "  1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor White
    Write-Host "  2. Install with 'C++ build tools' workload" -ForegroundColor White
    Write-Host "  3. Restart this script" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2: Use Docker only (Recommended for production)" -ForegroundColor Cyan
    Write-Host "  Run: .\setup-local-windows.ps1 -UseDockerOnly" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 3: Use pre-compiled InsightFace" -ForegroundColor Cyan
    Write-Host "  We'll try to install a pre-compiled version" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "Choose option (1/2/3) or press Enter to continue with Docker only"
    
    switch ($choice) {
        "1" {
            Write-Status "Please install Visual Studio Build Tools and restart this script."
            Write-Host "Download link: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Cyan
            exit 0
        }
        "2" {
            return $false
        }
        "3" {
            return $false
        }
        default {
            return $false
        }
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

# Function to install requirements without InsightFace
function Install-Requirements-NoInsightFace {
    param([string]$VenvPath, [string]$RequirementsFile, [string]$Name)
    
    Write-Status "Installing requirements for $Name (without InsightFace)..."
    
    $activateScript = "$VenvPath\Scripts\Activate.ps1"
    
    if (Test-Path $activateScript) {
        & $activateScript
        pip install --upgrade pip
        
        # Create a temporary requirements file without InsightFace
        $tempReqFile = "temp_requirements.txt"
        $content = Get-Content $RequirementsFile | Where-Object { $_ -notmatch "insightface" }
        $content | Out-File -FilePath $tempReqFile -Encoding UTF8
        
        pip install -r $tempReqFile
        
        # Clean up temp file
        Remove-Item $tempReqFile -Force
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Requirements installed for $Name (without InsightFace)"
        } else {
            Write-Error "Failed to install requirements for $Name"
            exit 1
        }
    } else {
        Write-Error "Virtual environment activation script not found: $activateScript"
        exit 1
    }
}

# Function to install requirements with InsightFace
function Install-Requirements-WithInsightFace {
    param([string]$VenvPath, [string]$RequirementsFile, [string]$Name)
    
    Write-Status "Installing requirements for $Name (with InsightFace)..."
    
    $activateScript = "$VenvPath\Scripts\Activate.ps1"
    
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

# Function to try installing pre-compiled InsightFace
function Install-InsightFacePrecompiled {
    Write-Status "Attempting to install pre-compiled InsightFace..."
    
    $activateScript = "venv\Scripts\Activate.ps1"
    
    if (Test-Path $activateScript) {
        & $activateScript
        
        # Try different approaches to install InsightFace
        Write-Status "Trying pip install insightface (pre-compiled)..."
        pip install insightface --only-binary=all
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Pre-compiled InsightFace installed successfully"
            return $true
        }
        
        Write-Status "Trying conda-forge..."
        if (Test-Command "conda") {
            conda install -c conda-forge insightface -y
            if ($LASTEXITCODE -eq 0) {
                Write-Success "InsightFace installed via conda"
                return $true
            }
        }
        
        Write-Warning "Could not install pre-compiled InsightFace"
        return $false
    }
    
    return $false
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
    Write-Host "🚀 Mordeaux Face Scanning MVP - Windows Local Setup" -ForegroundColor Magenta
    Write-Host "===================================================" -ForegroundColor Magenta
    Write-Host ""
    
    # Check prerequisites
    Write-Status "Checking prerequisites..."
    
    if (-not (Test-Command "python")) {
        Write-Error "Python not found. Please install Python 3.11+ first."
        exit 1
    }
    
    $pythonVersion = python --version
    Write-Success "Found: $pythonVersion"
    
    # Handle InsightFace installation
    $hasBuildTools = $false
    if (-not $UseDockerOnly) {
        $hasBuildTools = Install-BuildTools
    }
    
    # Create virtual environment
    New-VirtualEnvironment -Path "." -Name "venv"
    
    if ($hasBuildTools) {
        # Install with InsightFace
        Install-Requirements-WithInsightFace -VenvPath "venv" -RequirementsFile "backend/requirements.txt" -Name "Backend"
        Install-Requirements-WithInsightFace -VenvPath "venv" -RequirementsFile "face-pipeline/requirements.txt" -Name "Face-Pipeline"
        Install-Requirements-WithInsightFace -VenvPath "venv" -RequirementsFile "worker/requirements.txt" -Name "Worker"
    } else {
        # Install without InsightFace
        Install-Requirements-NoInsightFace -VenvPath "venv" -RequirementsFile "backend/requirements.txt" -Name "Backend"
        Install-Requirements-NoInsightFace -VenvPath "venv" -RequirementsFile "face-pipeline/requirements.txt" -Name "Face-Pipeline"
        Install-Requirements-NoInsightFace -VenvPath "venv" -RequirementsFile "worker/requirements.txt" -Name "Worker"
        
        # Try to install pre-compiled InsightFace
        if (-not $UseDockerOnly) {
            Install-InsightFacePrecompiled
        }
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
    
    if ($hasBuildTools) {
        Write-Host "✅ Full local development environment ready!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
        Write-Host "2. Run face-pipeline: cd face-pipeline && python main.py" -ForegroundColor Cyan
        Write-Host "3. Run backend: cd backend && uvicorn app.main:app --reload" -ForegroundColor Cyan
        Write-Host "4. Or use Docker: .\build-docker.ps1" -ForegroundColor Cyan
    } else {
        Write-Host "⚠️  Local setup completed (without InsightFace)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "For full functionality, use Docker:" -ForegroundColor Yellow
        Write-Host "1. Run: .\build-docker.ps1" -ForegroundColor Cyan
        Write-Host "2. Or install Visual Studio Build Tools and restart this script" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Limited local development:" -ForegroundColor Yellow
        Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
        Write-Host "2. Run face-pipeline (without face detection): cd face-pipeline && python main.py" -ForegroundColor Cyan
        Write-Host "3. Run backend (without face processing): cd backend && uvicorn app.main:app --reload" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "Service URLs (when running):" -ForegroundColor Yellow
    Write-Host "  - Face Pipeline: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  - Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  - Frontend: http://localhost:3000" -ForegroundColor Cyan
}

# Run main function
Main
