#!/bin/bash

# Mordeaux Face Scanning MVP - Docker Build Script
# This script builds and starts the complete Docker environment

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

# Function to check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success ".env file created from .env.example"
            print_warning "Please edit .env file with your actual configuration values"
        else
            print_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Function to check Docker and Docker Compose
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Function to build and start services
build_and_start() {
    print_status "Building and starting Docker services..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Build and start all services
    $COMPOSE_CMD up --build -d
    
    print_success "All services started successfully!"
}

# Function to show service status
show_status() {
    print_status "Checking service status..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD ps
}

# Function to show logs
show_logs() {
    print_status "Showing recent logs..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD logs --tail=50
}

# Function to show service URLs
show_urls() {
    print_success "Service URLs:"
    echo "  🌐 Frontend:        http://localhost:3000"
    echo "  🔧 Backend API:     http://localhost:8000"
    echo "  🧠 Face Pipeline:   http://localhost:8001"
    echo "  🗄️  MinIO Console:   http://localhost:9001"
    echo "  📊 pgAdmin:         http://localhost:5050"
    echo "  🔍 Qdrant:          http://localhost:6333"
    echo "  🌍 Nginx (Main):    http://localhost:80"
    echo ""
    print_status "Default credentials:"
    echo "  MinIO: minioadmin / minioadmin"
    echo "  pgAdmin: admin@admin.com / admin"
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD down
    print_success "All services stopped"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Main script logic
main() {
    echo "🐳 Mordeaux Face Scanning MVP - Docker Build Script"
    echo "=================================================="
    echo ""
    
    case "${1:-build}" in
        "build"|"start")
            check_docker
            check_env_file
            build_and_start
            show_status
            show_urls
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            check_docker
            check_env_file
            build_and_start
            show_status
            show_urls
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  build, start  - Build and start all services (default)"
            echo "  stop          - Stop all services"
            echo "  restart       - Restart all services"
            echo "  status        - Show service status"
            echo "  logs          - Show recent logs"
            echo "  cleanup       - Stop services and clean up resources"
            echo "  help          - Show this help message"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
