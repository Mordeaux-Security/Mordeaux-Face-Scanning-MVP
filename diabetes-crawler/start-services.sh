#!/bin/bash
# Start Redis and MinIO services for diabetes-crawler

cd "$(dirname "$0")"

echo "Starting Redis and MinIO services..."
docker-compose up -d

echo ""
echo "Services status:"
docker-compose ps

echo ""
echo "✓ Services started!"
echo ""
echo "Redis:     redis://localhost:6379"
echo "MinIO:     http://localhost:9000 (Console: http://localhost:9001)"
echo "           Username: MINIOADMIN"
echo "           Password: MINIOADMIN"
echo ""
echo "To stop services: ./stop-services.sh"
echo "To view logs:     docker-compose logs -f"

