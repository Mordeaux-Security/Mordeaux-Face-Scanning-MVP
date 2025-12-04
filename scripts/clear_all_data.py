#!/usr/bin/env python3
"""
Clear all data from MinIO buckets and Qdrant collections.

This script:
- Empties MinIO buckets (deletes all objects but keeps buckets)
- Clears Qdrant collections (deletes all points but keeps collections)

Usage:
    python scripts/clear_all_data.py [--confirm]

Options:
    --confirm: Skip confirmation prompt (use with caution!)
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

def clear_minio_buckets(confirm: bool = False):
    """Clear all objects from MinIO buckets while keeping the buckets."""
    from minio import Minio
    from minio.error import S3Error
    
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    buckets = [
        os.getenv("MINIO_BUCKET_RAW", "raw-images"),
        os.getenv("MINIO_BUCKET_CROPS", "face-crops"),
        os.getenv("MINIO_BUCKET_THUMBS", "thumbnails"),
        os.getenv("MINIO_BUCKET_METADATA", "face-metadata"),
    ]
    
    try:
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure
        )
        print(f"✅ Connected to MinIO: {minio_endpoint}")
    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")
        return False
    
    total_deleted = 0
    for bucket_name in buckets:
        try:
            # Check if bucket exists
            if not client.bucket_exists(bucket_name):
                print(f"⚠️  Bucket '{bucket_name}' does not exist, skipping")
                continue
            
            # List all objects in bucket
            objects = list(client.list_objects(bucket_name, recursive=True))
            object_count = len(objects)
            
            if object_count == 0:
                print(f"ℹ️  Bucket '{bucket_name}' is already empty")
                continue
            
            if not confirm:
                response = input(f"⚠️  Delete {object_count} objects from bucket '{bucket_name}'? (yes/no): ")
                if response.lower() != "yes":
                    print(f"⏭️  Skipping bucket '{bucket_name}'")
                    continue
            
            # Delete all objects
            deleted = 0
            for obj in objects:
                try:
                    client.remove_object(bucket_name, obj.object_name)
                    deleted += 1
                except Exception as e:
                    print(f"⚠️  Failed to delete {obj.object_name}: {e}")
            
            total_deleted += deleted
            print(f"✅ Deleted {deleted} objects from bucket '{bucket_name}'")
            
        except S3Error as e:
            print(f"❌ Error accessing bucket '{bucket_name}': {e}")
        except Exception as e:
            print(f"❌ Error processing bucket '{bucket_name}': {e}")
    
    print(f"🎉 MinIO: Deleted {total_deleted} total objects (buckets preserved)")
    return True

def clear_qdrant_collections(confirm: bool = False):
    """Clear all points from Qdrant collections while keeping the collections."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
    except ImportError:
        print("❌ qdrant-client not installed. Install with: pip install qdrant-client")
        return False
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    
    collections = [
        os.getenv("QDRANT_COLLECTION", "faces_v1"),
        os.getenv("IDENTITY_COLLECTION", "identities_v1"),
    ]
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print(f"✅ Connected to Qdrant: {qdrant_url}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False
    
    for collection_name in collections:
        try:
            # Check if collection exists
            try:
                collection_info = client.get_collection(collection_name)
                point_count = collection_info.points_count
            except Exception:
                print(f"⚠️  Collection '{collection_name}' does not exist, skipping")
                continue
            
            if point_count == 0:
                print(f"ℹ️  Collection '{collection_name}' is already empty")
                continue
            
            if not confirm:
                response = input(f"⚠️  Delete {point_count} points from collection '{collection_name}'? (yes/no): ")
                if response.lower() != "yes":
                    print(f"⏭️  Skipping collection '{collection_name}'")
                    continue
            
            # Delete all points by using a filter that matches everything
            # We'll use scroll to get all IDs and delete them
            print(f"🔄 Fetching all point IDs from '{collection_name}'...")
            all_ids = []
            offset = None
            while True:
                result, offset = client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                all_ids.extend([point.id for point in result])
                if offset is None:
                    break
            
            if all_ids:
                print(f"🔄 Deleting {len(all_ids)} points from '{collection_name}'...")
                # Delete in batches of 1000
                batch_size = 1000
                deleted = 0
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i:i + batch_size]
                    client.delete(
                        collection_name=collection_name,
                        points_selector=qm.PointIdsList(
                            points=batch
                        )
                    )
                    deleted += len(batch)
                    print(f"  Deleted {deleted}/{len(all_ids)} points...", end='\r')
                print(f"\n✅ Deleted {deleted} points from collection '{collection_name}'")
            else:
                print(f"ℹ️  No points found in collection '{collection_name}'")
                
        except Exception as e:
            print(f"❌ Error processing collection '{collection_name}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"🎉 Qdrant: All collections cleared (collections preserved)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Clear all data from MinIO and Qdrant")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompts (use with caution!)"
    )
    parser.add_argument(
        "--minio-only",
        action="store_true",
        help="Only clear MinIO buckets"
    )
    parser.add_argument(
        "--qdrant-only",
        action="store_true",
        help="Only clear Qdrant collections"
    )
    args = parser.parse_args()
    
    if not args.confirm:
        print("⚠️  WARNING: This will delete ALL data from MinIO buckets and Qdrant collections!")
        print("⚠️  Buckets and collections will be preserved, but all objects/points will be deleted.")
        response = input("\nAre you sure you want to continue? (type 'yes' to confirm): ")
        if response.lower() != "yes":
            print("❌ Aborted")
            return
    
    success = True
    
    if not args.qdrant_only:
        print("\n" + "=" * 60)
        print("CLEARING MINIO BUCKETS")
        print("=" * 60)
        success = clear_minio_buckets(confirm=args.confirm) and success
    
    if not args.minio_only:
        print("\n" + "=" * 60)
        print("CLEARING QDRANT COLLECTIONS")
        print("=" * 60)
        success = clear_qdrant_collections(confirm=args.confirm) and success
    
    if success:
        print("\n" + "=" * 60)
        print("✅ ALL DATA CLEARED SUCCESSFULLY")
        print("=" * 60)
        print("\nBuckets and collections are preserved and ready for new data.")
    else:
        print("\n" + "=" * 60)
        print("❌ SOME OPERATIONS FAILED")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()

