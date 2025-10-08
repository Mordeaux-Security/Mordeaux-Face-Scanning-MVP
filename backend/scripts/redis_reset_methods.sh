#!/bin/bash

# Redis Cache Reset Methods
# This script provides multiple ways to reset Redis cache depending on your environment

set -e

echo "🔧 Redis Cache Reset Methods"
echo "============================"

# Method 1: Direct Docker command (always works if Redis container is running)
echo "Method 1: Direct Docker Redis CLI"
echo "Command: docker compose exec redis redis-cli FLUSHDB"
docker compose exec redis redis-cli FLUSHDB
echo "✅ Redis cache cleared via Docker"
echo ""

# Method 2: Via API (requires running backend)
echo "Method 2: Via API endpoint"
echo "Command: curl -X DELETE http://localhost:8000/cache/all"
if curl -s -X DELETE http://localhost:8000/cache/all > /dev/null 2>&1; then
    echo "✅ Redis cache cleared via API"
else
    echo "❌ API method failed (backend might not be running)"
fi
echo ""

# Method 3: Python script (requires dependencies)
echo "Method 3: Python script"
echo "Command: python backend/scripts/reset_redis_cache.py --all"
if python backend/scripts/reset_redis_cache.py --all 2>/dev/null; then
    echo "✅ Redis cache cleared via Python script"
else
    echo "❌ Python script method failed (dependencies might be missing)"
fi
echo ""

# Method 4: Multiple database clearing
echo "Method 4: Clear multiple Redis databases"
echo "Clearing databases 0, 1, 2, 14, 15..."

for db in 0 1 2 14 15; do
    echo "  Clearing database $db..."
    docker compose exec redis redis-cli -n $db FLUSHDB > /dev/null 2>&1 || echo "    Database $db not accessible"
done

echo "✅ Multiple databases cleared"
echo ""

echo "🎉 Redis cache reset complete!"
echo ""
echo "Alternative commands for future use:"
echo "  make reset-redis-docker    # Clear Redis via Docker"
echo "  make reset-redis           # Clear Redis via Python script"
echo "  make reset-both           # Clear cache, Redis, and MinIO"
echo ""
