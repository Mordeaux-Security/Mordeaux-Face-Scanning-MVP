#!/usr/bin/env python3
"""
Script to trigger vectorization for all images in MinIO buckets.
Especially focuses on the 'sample' folder.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from app.services.storage import list_objects
from app.core.config import get_settings

def main():
    settings = get_settings()
    tenant_id = os.getenv("TENANT_ID", "demo-tenant")
    
    # Buckets to check
    buckets = ["raw-images", "thumbnails"]
    
    print(f"🔍 Scanning MinIO buckets for images...")
    print(f"Tenant ID: {tenant_id}\n")
    
    all_images = []
    
    for bucket in buckets:
        print(f"📦 Checking bucket: {bucket}")
        try:
            # List all objects
            objects = list_objects(bucket, prefix="")
            
            # Filter for image files and prioritize sample folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
            sample_images = []
            other_images = []
            
            for obj_key in objects:
                if any(obj_key.lower().endswith(ext) for ext in image_extensions):
                    if 'sample' in obj_key.lower():
                        sample_images.append((bucket, obj_key))
                    else:
                        other_images.append((bucket, obj_key))
            
            # Prioritize sample folder images
            all_images.extend(sample_images)
            all_images.extend(other_images)
            
            print(f"  Found {len(sample_images)} images in sample folder")
            print(f"  Found {len(other_images)} other images")
            print(f"  Total: {len(sample_images) + len(other_images)} images\n")
            
        except Exception as e:
            print(f"  ❌ Error listing bucket {bucket}: {e}\n")
    
    if not all_images:
        print("❌ No images found in MinIO buckets")
        return
    
    print(f"📊 Total images to vectorize: {len(all_images)}")
    print(f"   - Sample folder: {sum(1 for b, k in all_images if 'sample' in k.lower())}")
    print(f"   - Other: {sum(1 for b, k in all_images if 'sample' not in k.lower())}\n")
    
    # Prepare batch ingest requests
    batch_size = 100
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    print(f"🚀 Starting vectorization via {api_url}/api/v1/ingest/batch\n")
    
    total_processed = 0
    total_errors = 0
    
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(all_images) + batch_size - 1) // batch_size
        
        print(f"📤 Batch {batch_num}/{total_batches}: Processing {len(batch)} images...")
        
        # Build batch request
        items = []
        for bucket, key in batch:
            items.append({
                "tenant_id": tenant_id,
                "bucket": bucket,
                "key": key,
                "site": "minio-batch-import"
            })
        
        try:
            response = httpx.post(
                f"{api_url}/api/v1/ingest/batch",
                json={"items": items},
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                batch_ok = result.get("ok", False)
                results = result.get("results", [])
                
                ok_count = sum(1 for r in results if r.get("ok", False))
                error_count = len(results) - ok_count
                
                total_processed += ok_count
                total_errors += error_count
                
                print(f"  ✅ {ok_count} enqueued successfully")
                if error_count > 0:
                    print(f"  ⚠️  {error_count} errors")
            else:
                print(f"  ❌ HTTP {response.status_code}: {response.text}")
                total_errors += len(batch)
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            total_errors += len(batch)
    
    print(f"\n✅ Vectorization trigger complete!")
    print(f"   - Successfully enqueued: {total_processed}")
    print(f"   - Errors: {total_errors}")
    print(f"\n💡 Images are now queued in Redis. The face-pipeline worker will process them.")
    print(f"   Check face-pipeline logs to see processing progress.")

if __name__ == "__main__":
    main()

