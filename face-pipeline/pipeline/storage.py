from __future__ import annotations
from typing import Optional
from io import BytesIO
from datetime import timedelta
import threading
import re

from minio import Minio
from minio.error import S3Error

from config.settings import settings

_client: Minio | None = None
_client_lock = threading.Lock()

def get_client() -> Minio:
    """Get MinIO client for internal operations (within Docker network)."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        return _client

def ensure_buckets() -> None:
    """Create required buckets and set public read policy on crops/thumbnails."""
    import json
    c = get_client()
    
    # Public read policy for face crops (allows browser access without signatures)
    public_read_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::{bucket}/*"]
        }]
    }
    
    for b in [
        settings.MINIO_BUCKET_RAW,
        settings.MINIO_BUCKET_CROPS,
        settings.MINIO_BUCKET_THUMBS,
        settings.MINIO_BUCKET_METADATA,
    ]:
        if not c.bucket_exists(b):
            c.make_bucket(b)
        
        # Set public read on crops, thumbnails, and raw images buckets
        if b in [settings.MINIO_BUCKET_CROPS, settings.MINIO_BUCKET_THUMBS, settings.MINIO_BUCKET_RAW]:
            try:
                policy = json.dumps(public_read_policy).replace("{bucket}", b)
                c.set_bucket_policy(b, policy)
            except Exception:
                pass  # Policy may already exist

def exists(bucket: str, key: str) -> bool:
    c = get_client()
    try:
        c.stat_object(bucket, key)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        # If bucket not found or other errors, surface up
        raise

def get_bytes(bucket: str, key: str) -> bytes:
    c = get_client()
    resp = c.get_object(bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()

def put_bytes(bucket: str, key: str, data: bytes, content_type: str) -> None:
    c = get_client()
    bio = BytesIO(data)
    c.put_object(
        bucket,
        key,
        data=bio,
        length=len(data),
        content_type=content_type,
    )

def presign(bucket: str, key: str, ttl_sec: int | None = None) -> str:
    """Generate a URL for external access to an object.
    
    Uses public URL (no signature) if external endpoint is configured,
    since the bucket should have public read policy.
    Falls back to signed URL for internal access.
    """
    external = settings.MINIO_EXTERNAL_ENDPOINT
    if external:
        # Use simple public URL (bucket has public read policy)
        scheme = "https" if settings.MINIO_SECURE else "http"
        return f"{scheme}://{external}/{bucket}/{key}"
    
    # Fall back to signed URL for internal/legacy access
    c = get_client()
    expires_sec = ttl_sec if ttl_sec is not None else settings.PRESIGN_TTL_SEC
    return c.presigned_get_object(
        bucket,
        key,
        expires=timedelta(seconds=expires_sec)
    )