#!/bin/bash
# Stop Redis and MinIO services for diabetes-crawler

cd "$(dirname "$0")"

echo "Stopping Redis and MinIO services..."
docker-compose down

echo ""
echo "✓ Services stopped!"

