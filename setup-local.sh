#!/bin/bash

# Mordeaux Face Scanning MVP - Local Setup Script
# This script sets up the complete local development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create virtual environment
create_virtual_environment() {
    local venv_name="$1"
    
    print_status "Creating virtual environment: $venv_name"
    
    if [ -d "$venv_name" ]; then
        print_warning "Virtual environment $venv_name already exists. Removing..."
        rm -rf "$venv_name"
    fi
    
    python3 -m venv "$venv_name"
    
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created: $venv_name"
    else
        print_error "Failed to create virtual environment: $venv_name"
        exit 1
    fi
}

# Function to install requirements
install_requirements() {
    local venv_path="$1"
    local requirements_file="$2"
    local name="$3"
    
    print_status "Installing requirements for $name..."
    
    source "$venv_path/bin/activate"
    pip install --upgrade pip
    pip install -r "$requirements_file"
    
    if [ $? -eq 0 ]; then
        print_success "Requirements installed for $name"
    else
        print_error "Failed to install requirements for $name"
        exit 1
    fi
    
    deactivate
}

# Function to download InsightFace models
download_insightface_models() {
    print_status "Downloading InsightFace models..."
    
    source venv/bin/activate
    
    python3 -c "
import insightface
import os

try:
    print('Downloading InsightFace buffalo_l model...')
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    print('Model downloaded successfully!')
    
    # Check model location
    model_path = os.path.expanduser('~/.insightface')
    if os.path.exists(model_path):
        print(f'Models stored at: {model_path}')
        for root, dirs, files in os.walk(model_path):
            for file in files:
                print(f'  - {os.path.join(root, file)}')
    else:
        print('Model path not found')
        
except Exception as e:
    print(f'Error downloading models: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "InsightFace models downloaded successfully"
    else
        print_warning "Failed to download InsightFace models. They will be downloaded on first use."
    fi
    
    deactivate
}

# Function to setup frontend dependencies
setup_frontend() {
    print_status "Setting up frontend dependencies..."
    
    if [ -f "frontend/package.json" ]; then
        cd frontend
        
        if command_exists npm; then
            npm install
            if [ $? -eq 0 ]; then
                print_success "Frontend dependencies installed"
            else
                print_warning "Failed to install frontend dependencies"
            fi
        else
            print_warning "npm not found. Please install Node.js to build the frontend."
        fi
        
        cd ..
    else
        print_warning "Frontend package.json not found"
    fi
}

# Function to verify Docker setup
test_docker_setup() {
    print_status "Verifying Docker setup..."
    
    if command_exists docker; then
        if command_exists docker-compose; then
            compose_cmd="docker-compose"
        elif docker compose version >/dev/null 2>&1; then
            compose_cmd="docker compose"
        else
            print_error "Docker Compose not found. Please install Docker Desktop."
            return 1
        fi
        
        # Test docker-compose configuration
        $compose_cmd config --quiet
        if [ $? -eq 0 ]; then
            print_success "Docker configuration is valid"
            return 0
        else
            print_error "Docker configuration is invalid"
            return 1
        fi
    else
        print_error "Docker not found. Please install Docker Desktop."
        return 1
    fi
}

# Function to create data directories
create_data_directories() {
    print_status "Creating data directories..."
    
    directories=(
        "data/images"
        "data/models"
        "data/cache"
        "data/logs"
        "data/backups"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
}

# Function to test local services
test_local_services() {
    print_status "Testing local services..."
    
    # Test face-pipeline
    cd face-pipeline
    source ../venv/bin/activate
    python3 -c "import fastapi, uvicorn; print('Face-pipeline dependencies OK')" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Face-pipeline dependencies verified"
    else
        print_warning "Face-pipeline dependencies not working"
    fi
    deactivate
    cd ..
    
    # Test backend
    cd backend
    source ../venv/bin/activate
    python3 -c "import fastapi, celery; print('Backend dependencies OK')" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Backend dependencies verified"
    else
        print_warning "Backend dependencies not working"
    fi
    deactivate
    cd ..
}

# Parse command line arguments
SKIP_MODELS=false
SKIP_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-models    Skip downloading AI models"
            echo "  --skip-docker    Skip Docker verification"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main setup function
main() {
    echo "🚀 Mordeaux Face Scanning MVP - Local Setup"
    echo "==========================================="
    echo ""
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists python3; then
        print_error "Python 3 not found. Please install Python 3.11+ first."
        exit 1
    fi
    
    python_version=$(python3 --version)
    print_success "Found: $python_version"
    
    # Create virtual environment
    create_virtual_environment "venv"
    
    # Install backend requirements
    install_requirements "venv" "backend/requirements.txt" "Backend"
    
    # Install face-pipeline requirements
    install_requirements "venv" "face-pipeline/requirements.txt" "Face-Pipeline"
    
    # Install worker requirements
    install_requirements "venv" "worker/requirements.txt" "Worker"
    
    # Download models if not skipped
    if [ "$SKIP_MODELS" = false ]; then
        download_insightface_models
    else
        print_warning "Skipping model download (use --skip-models=false to download)"
    fi
    
    # Setup frontend
    setup_frontend
    
    # Create data directories
    create_data_directories
    
    # Verify Docker setup if not skipped
    if [ "$SKIP_DOCKER" = false ]; then
        if test_docker_setup; then
            print_success "Docker setup verified"
        else
            print_warning "Docker setup has issues. You can still run services locally."
        fi
    else
        print_warning "Skipping Docker verification"
    fi
    
    # Test local services
    test_local_services
    
    echo ""
    print_success "🎉 Local setup completed successfully!"
    echo ""
    echo "Next steps:" -e "${YELLOW}"
    echo "1. Activate virtual environment: source venv/bin/activate" -e "${BLUE}"
    echo "2. Run face-pipeline: cd face-pipeline && python main.py" -e "${BLUE}"
    echo "3. Run backend: cd backend && uvicorn app.main:app --reload" -e "${BLUE}"
    echo "4. Or use Docker: ./build-docker.sh" -e "${BLUE}"
    echo ""
    echo "Service URLs (when running):" -e "${YELLOW}"
    echo "  - Face Pipeline: http://localhost:8000" -e "${BLUE}"
    echo "  - Backend API: http://localhost:8000" -e "${BLUE}"
    echo "  - Frontend: http://localhost:3000" -e "${BLUE}"
}

# Run main function
main
