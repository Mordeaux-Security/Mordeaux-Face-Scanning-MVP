import io
import os
import uuid
import time
from typing import Tuple, Optional, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import urllib3
from ..core.config import get_settings

# Lazy singletons
_minio_client = None
_minio_http = None
_boto3_client = None
_thread_pool = None

def _get_thread_pool() -> ThreadPoolExecutor:
    """Get thread pool for CPU-intensive operations."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="storage_processing")
    return _thread_pool


def _minio():
    """Return a MinIO client if S3_ENDPOINT is defined (local dev)."""
    from minio import Minio  # lazy import for prod lightness
    global _minio_client
    global _minio_http
    if _minio_client is None:
        settings = get_settings()
        endpoint = settings.s3_endpoint.replace("https://", "").replace("http://", "")
        # Increase underlying HTTP connection pool size to reduce 'pool is full' warnings
        if _minio_http is None:
            # Tune pool sizes; retries kept default/minimal since MinIO ops are retried upstream
            _minio_http = urllib3.PoolManager(
                num_pools=64,
                maxsize=64,
                timeout=urllib3.util.Timeout(connect=5.0, read=30.0),
            )
        _minio_client = Minio(
            endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_use_ssl,
            http_client=_minio_http,
        )
    return _minio_client




def _boto3_s3():
    """Return a boto3 S3 client (no endpoint = real AWS)."""
    import boto3  # lazy import
    global _boto3_client
    if _boto3_client is None:
        settings = get_settings()
        # Prefer role/instance profile; fall back to explicit keys only if provided
        if settings.s3_access_key and settings.s3_secret_key:
            _boto3_client = boto3.client(
                "s3",
                region_name=settings.s3_region,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
            )
        else:
            _boto3_client = boto3.client("s3", region_name=settings.s3_region)
    return _boto3_client


def put_object(bucket: str, key: str, data: bytes, content_type: str) -> None:
    """Upload bytes to object storage."""
    settings = get_settings()
    if settings.using_minio:
        # MinIO path
        cli = _minio()
        # Ensure bucket exists on local dev
        from minio.error import S3Error  # type: ignore
        try:
            if not cli.bucket_exists(bucket):
                cli.make_bucket(bucket)
        except S3Error:
            pass
        cli.put_object(
            bucket_name=bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
    else:
        # AWS S3 path
        s3 = _boto3_s3()
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def get_presigned_url(bucket: str, key: str, method: str = "GET", expires: Optional[int] = None) -> Optional[str]:
    """Return a signed URL (GET/PUT) if supported."""
    settings = get_settings()
    if expires is None:
        expires = settings.presigned_url_ttl
    
    if settings.using_minio:
        # For local development, return a URL that goes through our backend
        # This avoids the localhost connection issue from inside containers
        if method.upper() == "GET":
            # Return a URL that will be handled by our backend proxy endpoint
            return f"/api/images/{bucket}/{key}"
        else:
            # For PUT operations, still use presigned URLs but with internal endpoint
            from datetime import timedelta
            cli = _minio()
            expires_delta = timedelta(seconds=expires)
            return cli.presigned_put_object(bucket, key, expires=expires_delta)
    else:
        # AWS S3 signed URL
        s3 = _boto3_s3()
        http_method = "get_object" if method.upper() == "GET" else "put_object"
        return s3.generate_presigned_url(
            ClientMethod=http_method,
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )


def get_object_from_storage(bucket: str, key: str) -> bytes:
    """Get object data from storage."""
    settings = get_settings()
    if settings.using_minio:
        # MinIO get object
        cli = _minio()
        response = cli.get_object(bucket, key)
        return response.read()
    else:
        # AWS S3 get object
        s3 = _boto3_s3()
        response = s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()


def list_objects(bucket: str, prefix: str = "") -> List[str]:
    """List objects in a bucket."""
    settings = get_settings()
    if settings.using_minio:
        # MinIO list objects
        cli = _minio()
        objects = cli.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]
    else:
        # AWS S3 list objects
        s3 = _boto3_s3()
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]


def _make_thumbnail(jpeg_bytes: bytes, max_w: int = 256) -> bytes:
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h = im.size
    if w > max_w:
        new_h = int(h * (max_w / float(w)))
        im = im.resize((max_w, new_h), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()


def save_raw_image_only(image_bytes: bytes, tenant_id: str, key_prefix: str = "") -> Tuple[str, str]:
    """
    Store raw image to BUCKET_RAW/<tenant_id>/<prefix><uuid>.jpg only
    Returns (raw_key, raw_url)
    """
    settings = get_settings()
    img_id = str(uuid.uuid4()).replace("-", "")
    raw_key = f"{tenant_id}/{key_prefix}{img_id}.jpg"

    put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg")

    raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
    return raw_key, raw_url






async def save_raw_and_thumb_async(image_bytes: bytes, tenant_id: str, key_prefix: str = "") -> Tuple[str, str, str, str]:
    """
    Async version of save_raw_and_thumb_with_precreated_thumb for better performance.
    Creates a thumbnail from the raw image.
    """
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()
    
    try:
        # Create thumbnail from raw image
        thumbnail_bytes = _make_thumbnail(image_bytes)
        result = await loop.run_in_executor(thread_pool, save_raw_and_thumb_with_precreated_thumb, image_bytes, thumbnail_bytes, tenant_id, key_prefix)
        return result
    except Exception as e:
        # Fallback to sync version if async fails
        thumbnail_bytes = _make_thumbnail(image_bytes)
        return save_raw_and_thumb_with_precreated_thumb(image_bytes, thumbnail_bytes, tenant_id, key_prefix)


def save_raw_and_thumb_with_precreated_thumb(image_bytes: bytes, thumbnail_bytes: bytes, tenant_id: str, key_prefix: str = "") -> Tuple[str, str, str, str]:
    """
    Store raw JPG to BUCKET_RAW/<tenant_id>/<prefix><uuid>.jpg and pre-created thumbnail to BUCKET_THUMBS/<tenant_id>/<prefix><uuid>.jpg
    Returns (raw_key, raw_url, thumb_key, thumb_url)
    
    This function combines the efficiency of basic_crawler1.1 (pre-created thumbnails) 
    with the multi-tenancy support of main branch.
    """
    settings = get_settings()
    img_id = str(uuid.uuid4()).replace("-", "")
    raw_key = f"{tenant_id}/{key_prefix}{img_id}.jpg"
    thumb_key = f"{tenant_id}/{key_prefix}{img_id}_thumb.jpg"

    put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg")
    put_object(settings.s3_bucket_thumbs, thumb_key, thumbnail_bytes, "image/jpeg")

    raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
    thumb_url = get_presigned_url(settings.s3_bucket_thumbs, thumb_key, "GET") or ""
    return raw_key, raw_url, thumb_key, thumb_url


