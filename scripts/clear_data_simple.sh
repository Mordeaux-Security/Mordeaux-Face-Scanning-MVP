#!/bin/bash
# Clear all data from MinIO buckets and Qdrant collections
# Preserves buckets and collections, only deletes content

set -e

echo "============================================================"
echo "CLEARING MINIO BUCKETS"
echo "============================================================"

BUCKETS=("raw-images" "face-crops" "thumbnails" "face-metadata")

for bucket in "${BUCKETS[@]}"; do
    echo "Clearing bucket: $bucket"
    # Remove all objects but keep the bucket
    docker-compose exec -T minio mc rm --recursive --force myminio/$bucket/ 2>&1 | grep -v "does not exist" || echo "  Bucket '$bucket' is empty or does not exist"
done

echo ""
echo "============================================================"
echo "CLEARING QDRANT COLLECTIONS"
echo "============================================================"

COLLECTIONS=("faces_v1" "identities_v1")

for collection in "${COLLECTIONS[@]}"; do
    echo "Clearing collection: $collection"
    
    # Check if collection exists and get point count
    POINT_COUNT=$(curl -s "http://localhost:6333/collections/$collection" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', {}).get('points_count', 0))" 2>/dev/null || echo "0")
    
    if [ "$POINT_COUNT" = "0" ] || [ -z "$POINT_COUNT" ]; then
        echo "  Collection '$collection' is empty or does not exist"
        continue
    fi
    
    echo "  Found $POINT_COUNT points, deleting..."
    
    # Delete all points using filter that matches everything
    curl -s -X POST "http://localhost:6333/collections/$collection/points/delete" \
        -H "Content-Type: application/json" \
        -d '{
            "filter": {
                "must": []
            },
            "points": null
        }' | python3 -c "import sys, json; data=json.load(sys.stdin); print('  ✅ Deleted all points' if data.get('status') == 'ok' or data.get('result') else '  ❌ Error:', data)" 2>/dev/null || echo "  ⚠️  Collection may not exist"
done

echo ""
echo "============================================================"
echo "✅ ALL DATA CLEARED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Buckets and collections are preserved and ready for new data."

