#!/bin/bash

echo "Starting Mordeaux Face Scanning MVP locally..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Start the services
echo "Starting Docker services..."
docker-compose up --build -d

echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose ps

echo ""
echo "Services should be available at:"
echo "- Frontend: http://localhost"
echo "- Backend API: http://localhost/api"
echo "- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "- Qdrant: http://localhost:6333"

echo ""
echo "To stop services, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
