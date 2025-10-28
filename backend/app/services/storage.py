import io
import uuid
import logging
import gc
from typing import Tuple, Optional, List, Dict, Any
import asyncio

from PIL import Image, ImageFile
import urllib3
import blake3
from minio import Minio  # lazy import for prod lightness
import boto3  # lazy import
from datetime import timedelta

from concurrent.futures import ThreadPoolExecutor
from ..core.config import get_settings
from minio.error import S3Error  # type: ignore

logger = logging.getLogger(__name__)

# Image safety configuration - set once on import
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def put_object(bucket: str, key: str, data: bytes, content_type: str, tags: Optional[Dict[str, str]] = None) -> None:
    """Upload bytes to object storage with optional tags."""
    settings = get_settings()
    if settings.using_minio:
        # MinIO path
        cli = _minio()
        # Ensure bucket exists on local dev
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
            tags=tags,
        )
    else:
        # AWS S3 path
        s3 = _boto3_s3()
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type, Tagging=tags)


def get_presigned_url(bucket: str, key: str, method: str = "GET", expires: Optional[int] = None) -> Optional[str]:
    """Return a presigned URL with TTL = 10 minutes (max)."""
    settings = get_settings()
    if expires is None:
        expires = settings.presigned_url_ttl

    # Ensure TTL never exceeds 10 minutes (600 seconds)
    expires = min(expires, 600)

    if settings.using_minio:
        # For local development, return a URL that goes through our backend
        # This avoids the localhost connection issue from inside containers
        if method.upper() == "GET":
            # Return a URL that will be handled by our backend proxy endpoint
            return f"/api/images/{bucket}/{key}"
        else:
            # For PUT operations, still use presigned URLs but with internal endpoint
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


def _compute_content_hash(content: bytes) -> str:
    """Compute content-addressed hash using blake3."""
    return blake3.blake3(content).hexdigest()


def _generate_content_addressed_key(content: bytes, tenant_id: str, extension: str = ".jpg") -> str:
    """
    Generate content-addressed storage key.
    Format: tenant/{hash[:2]}/{hash}{extension}
    """
    content_hash = _compute_content_hash(content)
    return f"{tenant_id}/{content_hash[:2]}/{content_hash}{extension}"


def _get_content_metadata(content: bytes, raw_key: str, thumb_key: Optional[str] = None, source_url: Optional[str] = None) -> Dict[str, Any]:
    """Extract metadata from content for caching."""
    metadata = {
        "hash": _compute_content_hash(content),
        "length": len(content),
        "mime": "image/jpeg",  # We standardize to JPEG for storage
        "raw_key": raw_key,
        "thumb_key": thumb_key
    }

    # Add source URL if provided
    if source_url:
        metadata["source_url"] = source_url

    return metadata


def save_raw_image_content_addressed(image_bytes: bytes, tenant_id: str, source_url: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Store raw image using content-addressed keys.
    Returns (raw_key, raw_url, metadata)
    """
    settings = get_settings()

    # Generate content-addressed key
    raw_key = _generate_content_addressed_key(image_bytes, tenant_id, ".jpg")

    # Check if object already exists (deduplication)
    try:
        if settings.using_minio:
            cli = _minio()
            try:
                # Try to get object metadata to check if it exists
                cli.stat_object(settings.s3_bucket_raw, raw_key)
                logger.info(f"Content-addressed object already exists: {raw_key}")
                # Object exists, return existing key and URL
                raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
                metadata = _get_content_metadata(image_bytes, raw_key, source_url=source_url)
                return raw_key, raw_url, metadata
            except Exception:
                # Object doesn't exist, proceed to upload
                pass
        else:
            s3 = _boto3_s3()
            try:
                # Try to get object metadata to check if it exists
                s3.head_object(Bucket=settings.s3_bucket_raw, Key=raw_key)
                logger.info(f"Content-addressed object already exists: {raw_key}")
                # Object exists, return existing key and URL
                raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
                metadata = _get_content_metadata(image_bytes, raw_key, source_url=source_url)
                return raw_key, raw_url, metadata
            except Exception:
                # Object doesn't exist, proceed to upload
                pass
    except Exception as e:
        logger.warning(f"Error checking for existing object {raw_key}: {e}")

    # Upload the object with source URL tag
    tags = {"source_url": source_url} if source_url else None
    put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg", tags)

    raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
    metadata = _get_content_metadata(image_bytes, raw_key, source_url=source_url)

    return raw_key, raw_url, metadata


def save_raw_and_thumb_content_addressed(image_bytes: bytes, thumbnail_bytes: bytes, tenant_id: str, source_url: Optional[str] = None) -> Tuple[str, str, str, str, Dict[str, Any]]:
    """
    Store raw image and thumbnail using content-addressed keys.
    Returns (raw_key, raw_url, thumb_key, thumb_url, metadata)
    """
    settings = get_settings()

    # Generate content-addressed keys
    raw_key = _generate_content_addressed_key(image_bytes, tenant_id, ".jpg")
    thumb_key = _generate_content_addressed_key(thumbnail_bytes, tenant_id, "_thumb.jpg")

    # Check if objects already exist (deduplication)
    raw_exists = False
    thumb_exists = False

    try:
        if settings.using_minio:
            cli = _minio()
            try:
                cli.stat_object(settings.s3_bucket_raw, raw_key)
                raw_exists = True
                logger.info(f"Content-addressed raw object already exists: {raw_key}")
            except Exception:
                pass

            try:
                cli.stat_object(settings.s3_bucket_thumbs, thumb_key)
                thumb_exists = True
                logger.info(f"Content-addressed thumb object already exists: {thumb_key}")
            except Exception:
                pass
        else:
            s3 = _boto3_s3()
            try:
                s3.head_object(Bucket=settings.s3_bucket_raw, Key=raw_key)
                raw_exists = True
                logger.info(f"Content-addressed raw object already exists: {raw_key}")
            except Exception:
                pass

            try:
                s3.head_object(Bucket=settings.s3_bucket_thumbs, Key=thumb_key)
                thumb_exists = True
                logger.info(f"Content-addressed thumb object already exists: {thumb_key}")
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error checking for existing objects: {e}")

    # Upload objects only if they don't exist
    tags = {"source_url": source_url} if source_url else None
    if not raw_exists:
        put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg", tags)

    if not thumb_exists:
        put_object(settings.s3_bucket_thumbs, thumb_key, thumbnail_bytes, "image/jpeg", tags)

    raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
    thumb_url = get_presigned_url(settings.s3_bucket_thumbs, thumb_key, "GET") or ""
    metadata = _get_content_metadata(image_bytes, raw_key, thumb_key, source_url=source_url)

    return raw_key, raw_url, thumb_key, thumb_url, metadata


async def save_raw_and_thumb_content_addressed_async(image_bytes: bytes, thumbnail_bytes: bytes, tenant_id: str, source_url: Optional[str] = None) -> Tuple[str, str, str, str, Dict[str, Any]]:
    """
    Async version of save_raw_and_thumb_content_addressed for better performance.
    """
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()

    try:
        result = await loop.run_in_executor(
            thread_pool,
            save_raw_and_thumb_content_addressed,
            image_bytes,
            thumbnail_bytes,
            tenant_id,
            source_url
        )
        return result
    except Exception as e:
        # Fallback to sync version if async fails
        logger.warning(f"Async storage failed, falling back to sync: {e}")
        return save_raw_and_thumb_content_addressed(image_bytes, thumbnail_bytes, tenant_id, source_url)


def save_raw_image_only(image_bytes: bytes, tenant_id: str, key_prefix: str = "", source_url: Optional[str] = None) -> Tuple[str, str]:
    """
    Store raw image to BUCKET_RAW/<tenant_id>/<prefix><uuid>.jpg only
    Returns (raw_key, raw_url)
    """
    settings = get_settings()
    img_id = str(uuid.uuid4()).replace("-", "")
    raw_key = f"{tenant_id}/{key_prefix}{img_id}.jpg"

    tags = {"source_url": source_url} if source_url else None
    put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg", tags)

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
    except Exception:
        # Fallback to sync version if async fails
        thumbnail_bytes = _make_thumbnail(image_bytes)
        return save_raw_and_thumb_with_precreated_thumb(image_bytes, thumbnail_bytes, tenant_id, key_prefix)


def save_raw_and_thumb_with_precreated_thumb(image_bytes: bytes, thumbnail_bytes: bytes, tenant_id: str, key_prefix: str = "", source_url: Optional[str] = None) -> Tuple[str, str, str, str]:
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

    tags = {"source_url": source_url} if source_url else None
    put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg", tags)
    put_object(settings.s3_bucket_thumbs, thumb_key, thumbnail_bytes, "image/jpeg", tags)

    raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
    thumb_url = get_presigned_url(settings.s3_bucket_thumbs, thumb_key, "GET") or ""
    return raw_key, raw_url, thumb_key, thumb_url


def close_storage_resources():
    """
    Clean shutdown of storage service resources.

    This function:
    1. Shuts down thread pools with wait=True
    2. Closes HTTP connection pools
    3. Clears client references to free memory
    4. Resets global variables to None
    5. Forces garbage collection
    """
    global _minio_client, _minio_http, _boto3_client, _thread_pool

    logger.info("Closing storage service resources...")

    try:
        # Shutdown thread pool if it exists
        if _thread_pool is not None:
            logger.info("Shutting down storage service thread pool...")
            _thread_pool.shutdown(wait=True)
            logger.info("Storage service thread pool shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down storage service thread pool: {e}")

    try:
        # Close MinIO HTTP connection pool if it exists
        if _minio_http is not None:
            logger.info("Closing MinIO HTTP connection pool...")
            _minio_http.clear()
            logger.info("MinIO HTTP connection pool closed")
    except Exception as e:
        logger.warning(f"Error closing MinIO HTTP connection pool: {e}")

    try:
        # Clear client references
        if _minio_client is not None:
            logger.info("Clearing MinIO client reference...")
            del _minio_client
            logger.info("MinIO client reference cleared")
    except Exception as e:
        logger.warning(f"Error clearing MinIO client: {e}")

    try:
        if _boto3_client is not None:
            logger.info("Clearing boto3 S3 client reference...")
            del _boto3_client
            logger.info("boto3 S3 client reference cleared")
    except Exception as e:
        logger.warning(f"Error clearing boto3 S3 client: {e}")

    try:
        # Reset global variables
        _minio_client = None
        _minio_http = None
        _boto3_client = None
        _thread_pool = None
        logger.info("Storage service global variables reset")
    except Exception as e:
        logger.warning(f"Error resetting storage service globals: {e}")

    try:
        # Force garbage collection to free memory
        gc.collect()
        logger.info("Storage service cleanup complete - garbage collection triggered")
    except Exception as e:
        logger.warning(f"Error during storage service garbage collection: {e}")


# Expose the cleanup function for external use
def get_storage_cleanup_function():
    """Get the storage cleanup function for use by other modules."""
    return close_storage_resources
