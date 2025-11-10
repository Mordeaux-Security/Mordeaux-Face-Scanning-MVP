#!/usr/bin/env python3
"""
Publish a test message to Redis Stream for worker consumption.

Usage:
    python face-pipeline/publish_test_message.py \
      --tenant demo \
      --bucket raw-images \
      --key samples/person4.jpg \
      --url file:///face-pipeline/samples/person4.jpg
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import redis
    from PIL import Image
    from minio import Minio
    try:
        import imagehash
        HAS_IMAGEHASH = True
    except ImportError:
        HAS_IMAGEHASH = False
except ImportError as e:
    print(f"Error: Missing dependency. Install with: pip install redis pillow minio")
    sys.exit(1)


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_phash(file_path: str) -> str:
    """Compute perceptual hash (pHash) of image."""
    if not HAS_IMAGEHASH:
        return "0" * 16
    try:
        img = Image.open(file_path)
        phash = imagehash.phash(img)
        return str(phash)
    except Exception as e:
        print(f"Warning: Could not compute pHash: {e}, using placeholder")
        return "0" * 16


def upload_to_minio(file_path: str, bucket: str, key: str):
    """Upload file to MinIO."""
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_access = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    client = Minio(minio_endpoint, access_key=minio_access, secret_key=minio_secret, secure=minio_secure)
    
    # Ensure bucket exists
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Created bucket: {bucket}")
    
    # Upload file
    client.fput_object(bucket, key, file_path)
    print(f"Uploaded {file_path} to MinIO: {bucket}/{key}")


def publish_message(tenant: str, bucket: str, key: str, url: str, file_path: str = None):
    """Publish a test message to Redis stream."""
    # Upload to MinIO if file path provided
    if file_path and Path(file_path).exists():
        upload_to_minio(file_path, bucket, key)
        image_sha256 = compute_sha256(file_path)
        image_phash = compute_phash(file_path)
        print(f"Computed SHA256: {image_sha256}")
        print(f"Computed pHash: {image_phash}")
    else:
        # Generate placeholder hashes
        image_sha256 = hashlib.sha256(f"{key}_{tenant}".encode()).hexdigest()
        image_phash = "0" * 16
        print(f"Using placeholder SHA256: {image_sha256}")
    
    # Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    stream_name = os.getenv("REDIS_STREAM_NAME", "face:ingest")
    
    r = redis.Redis.from_url(redis_url)
    
    # Build message payload
    payload = {
        "image_sha256": image_sha256,
        "bucket": bucket,
        "key": key,
        "tenant_id": tenant,
        "site": "test",
        "url": url,
        "image_phash": image_phash,
        "face_hints": None
    }
    
    # Publish to stream
    message_id = r.xadd(stream_name, {"message": json.dumps(payload)})
    
    print(f"✅ Published test message to stream '{stream_name}'")
    print(f"   Message ID: {message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    # Show stream info
    try:
        stream_info = r.xinfo_stream(stream_name)
        print(f"\nStream info:")
        print(f"   Length: {stream_info.get('length', 'N/A')}")
    except Exception as e:
        print(f"\nCould not get stream info: {e}")
    
    print(f"\nTo consume this message, run:")
    print(f"   make worker-once")


def main():
    parser = argparse.ArgumentParser(
        description="Publish test message to Redis Stream",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--tenant", required=True, help="Tenant ID")
    parser.add_argument("--bucket", required=True, help="MinIO bucket name")
    parser.add_argument("--key", required=True, help="Object key in bucket")
    parser.add_argument("--url", required=True, help="Source URL (file:// or http://)")
    parser.add_argument("--file", help="Local file path (for computing hashes)")
    
    args = parser.parse_args()
    
    # If file:// URL, extract path
    file_path = args.file
    if not file_path and args.url.startswith("file://"):
        file_path = args.url.replace("file://", "")
    
    try:
        publish_message(args.tenant, args.bucket, args.key, args.url, file_path)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

