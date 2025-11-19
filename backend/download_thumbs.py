#!/usr/bin/env python3
"""
Download all thumbnails from MinIO to local 'thumbs' folder.

Usage:
    python backend/download_thumbs.py [--preserve-structure]
    
Options:
    --preserve-structure    Preserve tenant folder structure (default: flatten to filename only)
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend to path for imports
script_dir = Path(__file__).parent
backend_app_path = script_dir / "app"
if backend_app_path.exists():
    sys.path.insert(0, str(script_dir))

# Try importing settings, if that fails, load from environment
try:
    from app.core.config import Settings
    use_settings = True
except ImportError:
    try:
        # Try alternative path
        backend_path = script_dir.parent
        sys.path.insert(0, str(backend_path))
        from backend.app.core.config import Settings
        use_settings = True
    except ImportError:
        # Fall back to environment variables
        use_settings = False

from minio import Minio
from minio.error import S3Error

def download_thumbnails(preserve_structure: bool = False):
    """Download all thumbnails from MinIO to local thumbs folder.
    
    Args:
        preserve_structure: If True, preserve tenant folder structure. If False, flatten to filename only.
    """
    
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
    print(f"Preserve structure: {preserve_structure}")
    
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
        
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            print(f"Error: Bucket '{bucket_name}' does not exist")
            sys.exit(1)
        
        print(f"Connected to MinIO at {endpoint}")
        
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        sys.exit(1)
    
    # Download all objects from thumbnails bucket
    try:
        objects = client.list_objects(bucket_name, recursive=True)
        
        count = 0
        skipped = 0
        errors = 0
        
        for obj in objects:
            try:
                object_name = obj.object_name
                
                # Skip non-image files
                if not any(object_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    continue
                
                # Create local file path
                if preserve_structure:
                    # Preserve folder structure
                    local_path = thumbs_dir / object_name
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Flatten to filename only
                    local_path = thumbs_dir / os.path.basename(object_name)
                
                # Skip if the file already exists
                if local_path.exists():
                    skipped += 1
                    if skipped % 100 == 0:
                        print(f"  Skipped {skipped} existing files...")
                    continue
                
                # Download using fget_object (simpler and more efficient)
                client.fget_object(bucket_name, object_name, str(local_path))
                
                count += 1
                if count % 10 == 0:
                    print(f"  Downloaded {count} images...")
                    
            except Exception as e:
                errors += 1
                print(f"  Error downloading {obj.object_name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"  Downloaded: {count} images")
        print(f"  Skipped (existing): {skipped} files")
        print(f"  Errors: {errors} files")
        print(f"  Output directory: {thumbs_dir.absolute()}")
        print(f"{'='*60}")
        
    except S3Error as e:
        print(f"Error accessing bucket: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download thumbnails from MinIO bucket")
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve tenant folder structure (default: flatten to filename only)"
    )
    args = parser.parse_args()
    
    print("=== Downloading Thumbnails from MinIO ===")
    print()
    download_thumbnails(preserve_structure=args.preserve_structure)
