import io
import os
import uuid
import time
import hashlib
import logging
import gc
import base64
import zlib
import gzip
import bz2
import lzma
import json
from typing import Tuple, Optional, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from PIL import Image, ImageFile
import urllib3
import blake3
from ..core.config import get_settings

logger = logging.getLogger(__name__)

# --- URL metadata constants (S3/MinIO user metadata) ---
VIDEO_URL_ENC_KEY = "video-url-enc"
VIDEO_URL_ALG_KEY = "video-url-enc-alg"
VIDEO_URL_SHA_KEY = "video-url-sha256"
VIDEO_URL_PREVIEW_KEY = "video-url-head"  # optional, human preview

# Keep total custom metadata under ~2KB to be safe.
USER_META_BUDGET = 2000

# --- Sidecar layout & schema ---
SIDECAR_FILENAME = "meta.json"       # sidecar file name next to image
SCHEMA_VERSION = 1

@dataclass
class ImageSidecar:
    schema: int
    doc_id: str
    site: str | None
    page_url: str | None
    source_video_url: str | None
    source_image_url: str | None
    crawl_ts: str                    # ISO8601
    sha256: str                      # of image bytes

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_object_paths(image_hash: str, filename: str) -> tuple[str, str]:
    """
    Returns (image_key, sidecar_key).
    Example: image_hash='a6050bd523f83e73a5f2d332dda384c79b03cc88eeca4b9a903c125bd7e2895b', filename='image.jpg'
             => 'default/a6/a6050bd523f83e73a5f2d332dda384c79b03cc88eeca4b9a903c125bd7e2895b.jpg', 
                'default/a6/a6050bd523f83e73a5f2d332dda384c79b03cc88eeca4b9a903c125bd7e2895b.json'
    """
    first2 = image_hash[:2]
    # Use hash as filename with appropriate extension
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image_filename = f"{image_hash}.jpg"
    elif filename.endswith('.png'):
        image_filename = f"{image_hash}.png"
    else:
        image_filename = f"{image_hash}.jpg"  # default to jpg
    
    sidecar_filename = f"{image_hash}.json"
    
    return f"default/{first2}/{image_filename}", f"default/{first2}/{sidecar_filename}"


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def compress_url(url: str) -> tuple[str, str]:
    """Return (encoded, alg). Try multiple compression algorithms and pick the best."""
    raw = url.encode("utf-8")
    
    # Try different compression algorithms and levels
    candidates = []
    
    # zlib deflate with different levels
    for level in range(1, 10):
        try:
            compressed = zlib.compress(raw, level=level)
            candidates.append((compressed, f"zlib-l{level}"))
        except:
            continue
    
    # gzip compression
    try:
        compressed = gzip.compress(raw, compresslevel=9)
        candidates.append((compressed, "gzip-9"))
    except:
        pass
    
    # bz2 compression
    try:
        compressed = bz2.compress(raw, compresslevel=9)
        candidates.append((compressed, "bz2-9"))
    except:
        pass
    
    # lzma compression (most aggressive)
    try:
        compressed = lzma.compress(raw, preset=9)
        candidates.append((compressed, "lzma-9"))
    except:
        pass
    
    # Pick the smallest result
    if not candidates:
        # Fallback to zlib level 9
        compressed = zlib.compress(raw, level=9)
        encoded = _b64url_encode(compressed)
        return encoded, "zlib-l9"
    
    best_compressed, best_alg = min(candidates, key=lambda x: len(x[0]))
    encoded = _b64url_encode(best_compressed)
    return encoded, best_alg


def decompress_url(encoded: str, alg: str) -> str:
    """Decompress URL using the specified algorithm."""
    raw = _b64url_decode(encoded)
    
    # Handle legacy format
    if alg.startswith("deflate-b64url-v1"):
        return zlib.decompress(raw).decode("utf-8")
    
    # Handle new algorithm formats
    if alg.startswith("zlib-"):
        return zlib.decompress(raw).decode("utf-8")
    elif alg.startswith("gzip-"):
        return gzip.decompress(raw).decode("utf-8")
    elif alg.startswith("bz2-"):
        return bz2.decompress(raw).decode("utf-8")
    elif alg.startswith("lzma-"):
        return lzma.decompress(raw).decode("utf-8")
    else:
        raise ValueError(f"unsupported algorithm: {alg}")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fits_user_metadata(pairs: dict[str, str], budget: int = USER_META_BUDGET) -> bool:
    size = 0
    for k, v in pairs.items():
        size += len(k.encode("utf-8")) + len(v.encode("utf-8"))
    return size < budget


def preview_head(url: str, max_chars: int = 512) -> str:
    # ASCII-only preview for headers; strip newlines just in case
    pv = url[:max_chars].replace("\r", " ").replace("\n", " ")
    try:
        pv.encode("ascii")
        return pv
    except UnicodeEncodeError:
        return pv.encode("utf-8", "ignore").decode("ascii", "ignore")


async def save_image(
    *,
    image_bytes: bytes,
    mime: str,
    filename: str,                  # e.g., 'image.jpg'
    bucket: str,
    client,                         # your MinIO/S3 client
    site: str,
    page_url: str | None = None,
    source_video_url: str | None = None,
    source_image_url: str | None = None,
) -> dict:
    """
    Uploads the image and a JSON sidecar. Sets small user metadata:
    - x-amz-meta-doc-id
    - x-amz-meta-video-url-sha256
    - x-amz-meta-video-url-head (short preview, optional)
    Returns a dict with doc_id, keys, and hashes.
    """
    # 1) ids + hashes
    doc_id = str(uuid.uuid4())
    sha_img = _sha256_hex(image_bytes)
    vid_sha = hashlib.sha256((source_video_url or "").encode("utf-8")).hexdigest() if source_video_url else None
    preview = None
    if source_video_url:
        pv = source_video_url[:512].replace("\r"," ").replace("\n"," ")
        try:
            pv.encode("ascii")
            preview = pv
        except UnicodeEncodeError:
            preview = pv.encode("utf-8","ignore").decode("ascii","ignore")

    # 2) keys - use image hash for path structure
    image_key, sidecar_key = make_object_paths(sha_img, filename)

    # 3) user metadata (keep tiny)
    user_meta = {"doc-id": doc_id}
    if vid_sha:
        user_meta["video-url-sha256"] = vid_sha
    if preview:
        user_meta["video-url-head"] = preview

    # 4) upload image
    settings = get_settings()
    if settings.using_minio:
        # MinIO path
        # Ensure bucket exists on local dev
        from minio.error import S3Error  # type: ignore
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
        except S3Error:
            pass
        
        try:
            client.put_object(bucket, image_key, io.BytesIO(image_bytes), len(image_bytes),
                              content_type=mime, metadata=user_meta)
            logger.info(f"Successfully uploaded image to {bucket}/{image_key}")
        except Exception as e:
            logger.error(f"Failed to upload image to {bucket}/{image_key}: {e}")
            raise
    else:
        # AWS S3 path
        client.put_object(Bucket=bucket, Key=image_key, Body=image_bytes,
                          ContentType=mime, Metadata=user_meta)

    # 5) build sidecar and upload
    sidecar = ImageSidecar(
        schema=SCHEMA_VERSION,
        doc_id=doc_id,
        site=site,
        page_url=page_url,
        source_video_url=source_video_url,
        source_image_url=source_image_url,
        crawl_ts=_now_iso(),
        sha256=sha_img,
    )
    sidecar_bytes = json.dumps(asdict(sidecar), ensure_ascii=False, separators=(",",":")).encode("utf-8")

    if settings.using_minio:
        # MinIO path
        try:
            client.put_object(bucket, sidecar_key, io.BytesIO(sidecar_bytes), len(sidecar_bytes),
                              content_type="application/json")
            logger.info(f"Successfully uploaded sidecar to {bucket}/{sidecar_key}")
        except Exception as e:
            logger.error(f"Failed to upload sidecar to {bucket}/{sidecar_key}: {e}")
            raise
    else:
        # AWS S3 path
        client.put_object(Bucket=bucket, Key=sidecar_key, Body=sidecar_bytes,
                          ContentType="application/json")

    # 6) Store in crawl cache for duplicate prevention
    try:
        from .cache import get_cache_service
        cache_service = get_cache_service()
        
        # Store cache entry with the new hash-based key structure
        await cache_service.store_crawled_image(
            url=source_image_url or "",
            image_bytes=image_bytes,
            raw_key=image_key,
            thumbnail_key=None,  # We don't store thumbnail keys in this simplified version
            tenant_id="default",
            source_url=page_url
        )
    except Exception as e:
        # Don't fail the entire operation if cache storage fails
        logger.warning(f"Failed to store crawl cache entry: {e}")

    return {
        "doc_id": doc_id,
        "image_key": image_key,
        "sidecar_key": sidecar_key,
        "sha256": sha_img,
        "video_url_sha256": vid_sha,
    }


def head_minio_metadata(client, bucket: str, key: str) -> dict[str,str]:
    """
    Return a normalized dict of user metadata from HEAD/stat: {'x-amz-meta-<k>': v, ...}
    """
    settings = get_settings()
    if settings.using_minio:
        # MinIO path
        stat = client.stat_object(bucket, key)
        md = stat.metadata or {}
        return {f"x-amz-meta-{k}": v for k, v in md.items()}
    else:
        # boto3 path
        resp = client.head_object(Bucket=bucket, Key=key)
        return resp["ResponseMetadata"]["HTTPHeaders"]


def get_sidecar_json(client, bucket: str, sidecar_key: str) -> dict | None:
    """
    Retrieve and parse sidecar JSON metadata.
    Returns None if sidecar doesn't exist or can't be parsed.
    """
    try:
        settings = get_settings()
        if settings.using_minio:
            # MinIO path
            resp = client.get_object(bucket, sidecar_key)
            data = resp.read()
            resp.close()
            resp.release_conn()
        else:
            # boto3 path
            data = client.get_object(Bucket=bucket, Key=sidecar_key)["Body"].read()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def read_video_url_from_head(headers: dict[str, str]) -> tuple[str | None, str | None, str | None]:
    """
    Given HEAD response headers, return (full_url or None, sha256 or None, error or None).
    Headers may provide user metadata as 'x-amz-meta-<key>' or raw '<key>' depending on client.
    """
    # Normalize lookup (some SDKs surface both forms)
    def get_meta(key: str) -> str | None:
        return headers.get(f"x-amz-meta-{key}") or headers.get(key)

    enc = get_meta(VIDEO_URL_ENC_KEY)
    alg = get_meta(VIDEO_URL_ALG_KEY)
    sha = get_meta(VIDEO_URL_SHA_KEY)
    metadata_key = get_meta("video-url-metadata-key")
    
    if enc and alg:
        try:
            url = decompress_url(enc, alg)
            return url, sha, None
        except Exception as e:
            return None, sha, f"decode-error: {e}"
    
    # If URL is stored in separate metadata object, fetch it
    if metadata_key and alg:
        try:
            settings = get_settings()
            if settings.using_minio:
                cli = _minio()
                response = cli.get_object(settings.s3_bucket_raw, metadata_key)
                enc = response.read().decode('utf-8')
                url = decompress_url(enc, alg)
                return url, sha, None
            else:
                s3 = _boto3_s3()
                response = s3.get_object(Bucket=settings.s3_bucket_raw, Key=metadata_key)
                enc = response['Body'].read().decode('utf-8')
                url = decompress_url(enc, alg)
                return url, sha, None
        except Exception as e:
            return None, sha, f"metadata-object-error: {e}"
    
    return None, sha, None


def get_video_url_from_storage(bucket: str, image_key: str) -> tuple[str | None, str | None, str | None]:
    """
    Retrieve video URL from storage for a given image key.
    Returns (full_url or None, sha256 or None, error or None).
    """
    try:
        settings = get_settings()
        if settings.using_minio:
            cli = _minio()
            response = cli.stat_object(bucket, image_key)
            headers = {f"x-amz-meta-{k}": v for k, v in (response.metadata or {}).items()}
        else:
            s3 = _boto3_s3()
            response = s3.head_object(Bucket=bucket, Key=image_key)
            headers = response['ResponseMetadata']['HTTPHeaders']
            # Add metadata from the metadata dict
            if 'Metadata' in response:
                for k, v in response['Metadata'].items():
                    headers[f"x-amz-meta-{k}"] = v
        
        return read_video_url_from_head(headers)
    except Exception as e:
        return None, None, f"storage-error: {e}"


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


def put_object(bucket: str, key: str, data: bytes, content_type: str, tags: Optional[Dict[str, str]] = None) -> None:
    """Upload bytes to object storage with optional tags."""
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
        # Convert tags dict to MinIO Tags object if using MinIO
        minio_tags = None
        if tags:
            from minio.commonconfig import Tags
            minio_tags = Tags()
            for k, v in tags.items():
                # MinIO tags have restrictions: max 256 chars, no special characters
                if isinstance(v, str) and len(v) > 256:
                    # If still too long after compression, use hash + preview approach
                    if k == VIDEO_URL_ENC_KEY:
                        # For compressed video URLs, if they're still too long, 
                        # we'll store just the hash and preview in separate tags
                        continue  # Skip this tag, we'll handle it below
                    else:
                        v = v[:253] + "..."
                # Remove or replace invalid characters for MinIO tags (but preserve compressed data)
                if isinstance(v, str) and k != VIDEO_URL_ENC_KEY:
                    v = v.replace(":", "_").replace("?", "_").replace("&", "_").replace("=", "_")
                minio_tags[k] = v
        
        cli.put_object(
            bucket_name=bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
            tags=minio_tags,
        )
    else:
        # AWS S3 path
        s3 = _boto3_s3()
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type, Tagging=tags)


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


    


    


async def save_raw_and_thumb_content_addressed_async(image_bytes: bytes, thumbnail_bytes: bytes, tenant_id: str, source_url: Optional[str] = None, video_url: Optional[str] = None) -> Tuple[str, str, str, str, Dict[str, Any]]:
    """
    Store raw and thumbnail using content-addressed keys asynchronously.
    """
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()

    def _sync_impl() -> Tuple[str, str, str, str, Dict[str, Any]]:
        settings = get_settings()
        raw_key = _generate_content_addressed_key(image_bytes, tenant_id, ".jpg")
        thumb_key = _generate_content_addressed_key(thumbnail_bytes, tenant_id, "_thumb.jpg")

        raw_exists = False
        thumb_exists = False

        try:
            if settings.using_minio:
                cli = _minio()
                try:
                    cli.stat_object(settings.s3_bucket_raw, raw_key)
                    raw_exists = True
                except Exception:
                    pass
                try:
                    cli.stat_object(settings.s3_bucket_thumbs, thumb_key)
                    thumb_exists = True
                except Exception:
                    pass
            else:
                s3 = _boto3_s3()
                try:
                    s3.head_object(Bucket=settings.s3_bucket_raw, Key=raw_key)
                    raw_exists = True
                except Exception:
                    pass
                try:
                    s3.head_object(Bucket=settings.s3_bucket_thumbs, Key=thumb_key)
                    thumb_exists = True
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Error checking for existing objects: {e}")

        if not raw_exists:
            # Save raw image with metadata
            # Note: save_image is async, but this sync impl is only for run_in_executor fallback; avoid calling it here
            if settings.using_minio:
                cli = _minio()
                cli.put_object(settings.s3_bucket_raw, raw_key, io.BytesIO(image_bytes), len(image_bytes), content_type="image/jpeg")
            else:
                s3 = _boto3_s3()
                s3.put_object(Bucket=settings.s3_bucket_raw, Key=raw_key, Body=image_bytes, ContentType="image/jpeg")

        if not thumb_exists:
            if settings.using_minio:
                cli = _minio()
                cli.put_object(settings.s3_bucket_thumbs, thumb_key, io.BytesIO(thumbnail_bytes), len(thumbnail_bytes), content_type="image/jpeg")
            else:
                s3 = _boto3_s3()
                s3.put_object(Bucket=settings.s3_bucket_thumbs, Key=thumb_key, Body=thumbnail_bytes, ContentType="image/jpeg")

        raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
        thumb_url = get_presigned_url(settings.s3_bucket_thumbs, thumb_key, "GET") or ""
        metadata = _get_content_metadata(image_bytes, raw_key, thumb_key, source_url=source_url)
        return raw_key, raw_url, thumb_key, thumb_url, metadata

    return await loop.run_in_executor(thread_pool, _sync_impl)








async def save_raw_and_thumb_async(image_bytes: bytes, tenant_id: str, key_prefix: str = "") -> Tuple[str, str, str, str]:
    """
    Create a thumbnail and store both raw and thumbnail asynchronously.
    """
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()

    def _sync_impl() -> Tuple[str, str, str, str]:
        settings = get_settings()
        img_id = str(uuid.uuid4()).replace("-", "")
        raw_key = f"{tenant_id}/{key_prefix}{img_id}.jpg"
        thumb_key = f"{tenant_id}/{key_prefix}{img_id}_thumb.jpg"

        thumbnail_bytes = _make_thumbnail(image_bytes)

        tags = None
        put_object(settings.s3_bucket_raw, raw_key, image_bytes, "image/jpeg", tags)
        put_object(settings.s3_bucket_thumbs, thumb_key, thumbnail_bytes, "image/jpeg", tags)

        raw_url = get_presigned_url(settings.s3_bucket_raw, raw_key, "GET") or ""
        thumb_url = get_presigned_url(settings.s3_bucket_thumbs, thumb_key, "GET") or ""
        return raw_key, raw_url, thumb_key, thumb_url

    return await loop.run_in_executor(thread_pool, _sync_impl)


    


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


if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(prog="storage")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_head = sub.add_parser("head", help="HEAD an object and print URL metadata")
    p_head.add_argument("--bucket", required=True)
    p_head.add_argument("--key", required=True)

    p_inspect = sub.add_parser("inspect", help="Print sidecar + header metadata for an object key")
    p_inspect.add_argument("--bucket", required=True)
    p_inspect.add_argument("--key", required=True)

    args = parser.parse_args()
    if args.cmd == "head":
        # Initialize storage clients using existing functions
        settings = get_settings()
        headers = {}
        
        try:
            if settings.using_minio:
                # MinIO path
                cli = _minio()
                try:
                    stat = cli.stat_object(args.bucket, args.key)
                    # Convert MinIO metadata to header format
                    headers = {f"x-amz-meta-{k}": v for k, v in (stat.metadata or {}).items()}
                except Exception as e:
                    print(json.dumps({"error": f"MinIO stat_object failed: {e}"}, indent=2))
                    sys.exit(1)
            else:
                # AWS S3 path
                s3 = _boto3_s3()
                try:
                    resp = s3.head_object(Bucket=args.bucket, Key=args.key)
                    # Extract headers from S3 response
                    headers = resp.get('ResponseMetadata', {}).get('HTTPHeaders', {})
                    # Also check for metadata in the response
                    metadata = resp.get('Metadata', {})
                    if metadata:
                        # Add metadata with x-amz-meta prefix
                        for k, v in metadata.items():
                            headers[f"x-amz-meta-{k}"] = v
                except Exception as e:
                    print(json.dumps({"error": f"S3 head_object failed: {e}"}, indent=2))
                    sys.exit(1)

            # Decode video URL metadata
            url, sha, err = read_video_url_from_head(headers)
            out = {
                "has_encoded": url is not None or err is not None,
                "sha256": sha,
                "error": err,
                "decoded_preview": (url[:200] if url else None),
                "raw_headers": headers  # Include all headers for debugging
            }
            print(json.dumps(out, indent=2))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({"error": f"Unexpected error: {e}"}, indent=2))
            sys.exit(1)
    
    elif args.cmd == "inspect":
        # Initialize storage clients using existing functions
        settings = get_settings()
        
        try:
            if settings.using_minio:
                # MinIO path
                client = _minio()
            else:
                # AWS S3 path
                client = _boto3_s3()

            # 1) HEAD the image for quick fields
            headers = head_minio_metadata(client, args.bucket, args.key)

            # 2) Derive sidecar key from image key
            # image key format: default/<first2>/<hash>.jpg
            parts = args.key.strip("/").split("/")
            if len(parts) < 3:
                print(json.dumps({"error": "key not in expected format"}, indent=2))
                sys.exit(1)
            
            # Extract hash from filename (remove extension)
            filename = parts[-1]
            hash_without_ext = filename.rsplit('.', 1)[0]  # Remove extension
            sidecar_filename = f"{hash_without_ext}.json"
            sidecar_key = "/".join(parts[:-1] + [sidecar_filename])

            # 3) Fetch sidecar JSON
            sidecar = get_sidecar_json(client, args.bucket, sidecar_key)

            out = {
                "image_key": args.key,
                "sidecar_key": sidecar_key,
                "headers": {k: headers[k] for k in sorted(headers.keys()) if "video-url" in k or "doc-id" in k},
                "sidecar": sidecar,
            }
            print(json.dumps(out, indent=2, ensure_ascii=False))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({"error": f"Unexpected error: {e}"}, indent=2))
            sys.exit(1)
    
    parser.print_help()


