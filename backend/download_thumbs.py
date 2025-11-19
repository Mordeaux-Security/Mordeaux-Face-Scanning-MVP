#!/usr/bin/env python3
"""
Download all thumbnails from MinIO to local 'thumbs' folder.
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)

# Try importing settings, if that fails, load from environment
try:
    from backend.app.core.config import Settings
    use_settings = True
except ImportError:
    # Fall back to environment variables
    use_settings = False

from minio import Minio
from minio.error import S3Error

def download_thumbnails():
    """Download all thumbnails from MinIO to local thumbs folder."""
    
    # Load settings
    if use_settings:
        try:
            settings = Settings()
            s3_endpoint = settings.s3_endpoint
            s3_access_key = settings.s3_access_key
            s3_secret_key = settings.s3_secret_key
            s3_use_ssl = settings.s3_use_ssl
            bucket_name = settings.s3_bucket_thumbs
        except Exception as e:
            print(f"Error loading settings: {e}")
            print("Falling back to environment variables...")
            settings = None
    else:
        settings = None
    
    # Use environment variables if settings not available
    if settings is None:
        s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:9000")
        s3_access_key = os.getenv("S3_ACCESS_KEY", "minioadmin")
        s3_secret_key = os.getenv("S3_SECRET_KEY", "minioadmin")
        s3_use_ssl = os.getenv("S3_USE_SSL", "false").lower() == "true"
        bucket_name = os.getenv("S3_BUCKET_THUMBS", "thumbnails")
    
    # Check required settings
    if not s3_endpoint or not s3_access_key or not s3_secret_key:
        print("Error: Missing MinIO configuration (S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY)")
        sys.exit(1)
    
    print(f"Endpoint: {s3_endpoint}")
    print(f"Bucket: {bucket_name}")
    
    # Create thumbs directory if it doesn't exist
    thumbs_dir = Path("thumbs")
    thumbs_dir.mkdir(exist_ok=True)
    print(f"Created/using directory: {thumbs_dir.absolute()}")
    
    # Initialize MinIO client
    try:
        # Parse endpoint (remove http:// or https://)
        endpoint = s3_endpoint.replace("https://", "").replace("http://", "")
        
        client = Minio(
            endpoint,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            secure=s3_use_ssl
        )
        
        print(f"Connected to MinIO at {endpoint}")
        
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        sys.exit(1)
    
    # Download all objects from thumbnails bucket
    try:
        objects = client.list_objects(bucket_name, recursive=True)
        
        count = 0
        for obj in objects:
            try:
                # Get object data
                response = client.get_object(bucket_name, obj.object_name)
                
                # Create local file path
                # Preserve folder structure but use last part as filename
                local_path = thumbs_dir / os.path.basename(obj.object_name)
                
                # Skip if the file already exists
                if local_path.exists():
                    print(f"  Skipping existing: {local_path.name}")
                    response.close()
                    response.release_conn()
                    continue
                
                # Write to file
                with open(local_path, 'wb') as f:
                    for chunk in response.stream(32*1024):  # 32KB chunks
                        f.write(chunk)
                
                response.close()
                response.release_conn()
                
                count += 1
                if count % 10 == 0:
                    print(f"  Downloaded {count} images...")
                    
            except Exception as e:
                print(f"  Error downloading {obj.object_name}: {e}")
                continue
        
        print(f"\n✓ Downloaded {count} images to {thumbs_dir.absolute()}")
        
    except S3Error as e:
        print(f"Error accessing bucket: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Downloading Thumbnails from MinIO ===")
    print()
    download_thumbnails()
