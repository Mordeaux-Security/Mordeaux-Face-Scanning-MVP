#!/bin/bash

echo "🚀 Setting up Mordeaux Face Scanning MVP..."

# Stop any existing containers
echo "📦 Stopping existing containers..."
docker compose down

# Build and start all services
echo "🔨 Building and starting services..."
docker compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Seed the database and create buckets
echo "🌱 Seeding database and creating buckets..."
docker compose exec backend-cpu python /app/seed_demo.py

# Show status
echo "✅ Setup complete! Service status:"
docker compose ps

echo ""
echo "🌐 Access points:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - MinIO Console: http://localhost:9001"
echo "  - API Health: http://localhost:8000/healthz"
