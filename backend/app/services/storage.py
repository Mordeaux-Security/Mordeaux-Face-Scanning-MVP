import io
import os
import uuid
import asyncio
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "").strip()
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_USE_SSL = os.getenv("S3_USE_SSL", "false").lower() == "true"
BUCKET_RAW = os.getenv("S3_BUCKET_RAW", "raw-images")
BUCKET_THUMBS = os.getenv("S3_BUCKET_THUMBS", "thumbnails")

# Lazy singletons
_minio_client = None
_boto3_client = None


def _minio():
    """Return a MinIO client if S3_ENDPOINT is defined (local dev)."""
    from minio import Minio  # lazy import for prod lightness
    global _minio_client
    if _minio_client is None:
        endpoint = S3_ENDPOINT.replace("https://", "").replace("http://", "")
        _minio_client = Minio(
            endpoint,
            access_key=S3_ACCESS_KEY,
            secret_key=S3_SECRET_KEY,
            secure=S3_USE_SSL,
        )
    return _minio_client


def _boto3_s3():
    """Return a boto3 S3 client (no endpoint = real AWS)."""
    import boto3  # lazy import
    global _boto3_client
    if _boto3_client is None:
        # Prefer role/instance profile; fall back to explicit keys only if provided
        if S3_ACCESS_KEY and S3_SECRET_KEY:
            _boto3_client = boto3.client(
                "s3",
                region_name=S3_REGION,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
            )
        else:
            _boto3_client = boto3.client("s3", region_name=S3_REGION)
    return _boto3_client


def put_object(bucket: str, key: str, data: bytes, content_type: str) -> None:
    """Upload bytes to object storage."""
    if S3_ENDPOINT:
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


def get_presigned_url(bucket: str, key: str, method: str = "GET", expires: int = 3600) -> Optional[str]:
    """Return a signed URL (GET/PUT) if supported."""
    if S3_ENDPOINT:
        # MinIO signed URL
        from datetime import timedelta
        cli = _minio()
        expires_delta = timedelta(seconds=expires)
        if method.upper() == "GET":
            return cli.presigned_get_object(bucket, key, expires=expires_delta)
        if method.upper() == "PUT":
            return cli.presigned_put_object(bucket, key, expires=expires_delta)
        return None
    else:
        # AWS S3 signed URL
        s3 = _boto3_s3()
        http_method = "get_object" if method.upper() == "GET" else "put_object"
        return s3.generate_presigned_url(
            ClientMethod=http_method,
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )


def list_objects(bucket: str, prefix: str = "") -> List[str]:
    """List objects in a bucket."""
    if S3_ENDPOINT:
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


def save_raw_image_only(image_bytes: bytes, key_prefix: str = "") -> Tuple[str, str]:
    """
    Store raw image to BUCKET_RAW/<prefix><uuid>.jpg only
    Returns (raw_key, raw_url)
    """
    img_id = str(uuid.uuid4()).replace("-", "")
    raw_key = f"{key_prefix}{img_id}.jpg"

    put_object(BUCKET_RAW, raw_key, image_bytes, "image/jpeg")

    raw_url = get_presigned_url(BUCKET_RAW, raw_key, "GET", 3600) or ""
    return raw_key, raw_url


def save_thumbnail_only(image_bytes: bytes, key_prefix: str = "") -> Tuple[str, str]:
    """
    Store thumbnail to BUCKET_THUMBS/<prefix><uuid>.jpg only
    Returns (thumb_key, thumb_url)
    """
    img_id = str(uuid.uuid4()).replace("-", "")
    thumb_jpg = _make_thumbnail(image_bytes)
    thumb_key = f"{key_prefix}{img_id}_thumb.jpg"

    put_object(BUCKET_THUMBS, thumb_key, thumb_jpg, "image/jpeg")

    thumb_url = get_presigned_url(BUCKET_THUMBS, thumb_key, "GET", 3600) or ""
    return thumb_key, thumb_url


def save_raw_and_thumb(image_bytes: bytes, key_prefix: str = "") -> Tuple[str, str, str, str]:
    """
    Store raw JPG to BUCKET_RAW/<prefix><uuid>.jpg and a thumbnail to BUCKET_THUMBS/<prefix><uuid>.jpg
    Returns (raw_key, raw_url, thumb_key, thumb_url)
    """
    img_id = str(uuid.uuid4()).replace("-", "")
    raw_key = f"{key_prefix}{img_id}.jpg"
    thumb_jpg = _make_thumbnail(image_bytes)
    thumb_key = f"{key_prefix}{img_id}_thumb.jpg"

    put_object(BUCKET_RAW, raw_key, image_bytes, "image/jpeg")
    put_object(BUCKET_THUMBS, thumb_key, thumb_jpg, "image/jpeg")

    raw_url = get_presigned_url(BUCKET_RAW, raw_key, "GET", 3600) or ""
    thumb_url = get_presigned_url(BUCKET_THUMBS, thumb_key, "GET", 3600) or ""
    return raw_key, raw_url, thumb_key, thumb_url


def save_raw_and_thumb_with_precreated_thumb(image_bytes: bytes, thumbnail_bytes: bytes, key_prefix: str = "") -> Tuple[str, str, str, str]:
    """
    Store raw JPG to BUCKET_RAW/<prefix><uuid>.jpg and pre-created thumbnail to BUCKET_THUMBS/<prefix><uuid>.jpg
    Returns (raw_key, raw_url, thumb_key, thumb_url)
    """
    img_id = str(uuid.uuid4()).replace("-", "")
    raw_key = f"{key_prefix}{img_id}.jpg"
    thumb_key = f"{key_prefix}{img_id}_thumb.jpg"

    put_object(BUCKET_RAW, raw_key, image_bytes, "image/jpeg")
    put_object(BUCKET_THUMBS, thumb_key, thumbnail_bytes, "image/jpeg")

    raw_url = get_presigned_url(BUCKET_RAW, raw_key, "GET", 3600) or ""
    thumb_url = get_presigned_url(BUCKET_THUMBS, thumb_key, "GET", 3600) or ""
    return raw_key, raw_url, thumb_key, thumb_url




