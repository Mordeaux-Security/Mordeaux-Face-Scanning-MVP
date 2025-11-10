from __future__ import annotations
from typing import Optional
from io import BytesIO
from datetime import timedelta
import threading

from minio import Minio
from minio.error import S3Error

from config.settings import settings

_client: Minio | None = None
_client_lock = threading.Lock()

def get_client() -> Minio:
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
    c = get_client()
    for b in [
        settings.MINIO_BUCKET_RAW,
        settings.MINIO_BUCKET_CROPS,
        settings.MINIO_BUCKET_THUMBS,
        settings.MINIO_BUCKET_METADATA,
    ]:
        if not c.bucket_exists(b):
            c.make_bucket(b)

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
    c = get_client()
    expires_sec = ttl_sec if ttl_sec is not None else settings.PRESIGN_TTL_SEC
    return c.presigned_get_object(
        bucket,
        key,
        expires=timedelta(seconds=expires_sec)
    )