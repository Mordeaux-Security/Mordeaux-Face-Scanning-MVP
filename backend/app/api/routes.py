import os
import json
import time
import uuid
import hashlib
from typing import Optional, Dict, List, Any
from io import BytesIO

import httpx
import redis
from minio import Minio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator, ValidationError

router = APIRouter()

# ---------- Upstream face-pipeline (search passthrough) ----------
PIPELINE_URL = os.getenv("PIPELINE_URL", "http://face-pipeline:8001")

@router.get("/api/v1/health")
def api_health():
    return {"status": "ok", "service": "backend-router"}

@router.post("/api/v1/search")
async def search_passthrough(payload: dict):
    """
    Passthrough search to face-pipeline so the frontend can call /api/v1/search on this backend.
    """
    url = f"{PIPELINE_URL}/api/v1/search"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
        
        # Check if the face-pipeline returned an error
        if r.status_code != 200:
            error_detail = r.json()
            raise HTTPException(status_code=r.status_code, detail=error_detail)
        
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"pipeline_unreachable: {e}")

# ---------- Identity-safe search proxy ----------
class IdentitySafeSearchReq(BaseModel):
    """Request model for identity-safe search endpoint."""
    tenant_id: str
    identity_id: str
    image_b64: str
    top_k: int = Field(default=50, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

class IdentitySafeSearchResult(BaseModel):
    """Result item for identity-safe search."""
    id: str
    score: float
    image_id: Optional[str] = None
    tenant_id: Optional[str] = None
    identity_id: Optional[str] = None
    payload: Optional[dict] = None

class IdentitySafeSearchResp(BaseModel):
    """Response model for identity-safe search endpoint."""
    verified: bool
    similarity: float
    threshold: float
    reason: Optional[dict] = None
    results: List[IdentitySafeSearchResult] = []
    count: int = Field(default=0)

@router.post("/api/v1/identity_safe_search", response_model=IdentitySafeSearchResp)
async def identity_safe_search_passthrough(req: IdentitySafeSearchReq) -> IdentitySafeSearchResp:
    """
    Proxy identity-safe search to face-pipeline.
    
    Verifies a probe image against an enrolled identity and only returns that identity's faces
    if verification succeeds (strict no-leak rule).
    """
    url = f"{PIPELINE_URL}/api/v1/identity_safe_search"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=req.model_dump())
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "pipeline_unreachable", "message": str(e)}
        )
    
    if r.status_code != 200:
        # Bubble up pipeline errors transparently
        try:
            detail = r.json()
        except Exception:
            detail = {"error": "pipeline_error", "status_code": r.status_code}
        raise HTTPException(status_code=r.status_code, detail=detail)
    
    data = r.json()
    # Ensure count is present (compute from results if missing)
    if "count" not in data:
        data["count"] = len(data.get("results", []))
    # Validate and normalize response with Pydantic
    return IdentitySafeSearchResp(**data)

# ---------- Redis stream ingest (single + batch) ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_STREAM_NAME = os.getenv("REDIS_STREAM_NAME", "face:ingest")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

def _get_redis() -> redis.Redis:
    try:
        return redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"redis_connect_failed: {e}")

def _get_minio() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

def _compute_hashes_for_bucket_key(bucket: str, key: str) -> tuple[str, str]:
    """Compute image_sha256 and image_phash from MinIO object."""
    try:
        client = _get_minio()
        obj_data = client.get_object(bucket, key)
        image_bytes = obj_data.read()
        obj_data.close()
        obj_data.release_conn()
        
        # Compute SHA256
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        
        # Placeholder pHash (worker will recompute during processing)
        image_phash = "0" * 16
        
        return image_sha256, image_phash
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read object from MinIO: {e}"
        )

class IngestRequest(BaseModel):
    """
    Enqueue an image for processing by the face-pipeline worker.

    Provide EITHER (bucket + key) for an object in MinIO/S3, OR a direct URL (file://, http(s)://).
    Optional fields: site, ts (unix seconds), meta (dict), idempotency_key.
    """
    tenant_id: str = Field(..., description="Tenant namespace for payload and search filters")
    bucket: Optional[str] = Field(None, description="MinIO/S3 bucket (if referencing an object)")
    key: Optional[str] = Field(None, description="Object key within the bucket")
    url: Optional[str] = Field(None, description="Direct URL (file:///..., http(s)://...)")
    site: Optional[str] = Field(None, description="Source site/tag for provenance (optional)")
    ts: Optional[int] = Field(None, description="Unix timestamp seconds; defaults to now")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Any extra metadata to attach")
    idempotency_key: Optional[str] = Field(
        default=None, description="Optional caller-supplied key to avoid duplicate enqueue"
    )

    @model_validator(mode="after")
    def _validate_source(self):
        has_obj = bool(self.bucket and self.key)
        has_url = bool(self.url)
        if not (has_obj or has_url):
            raise ValueError("Provide either (bucket AND key) or url")
        return self

def _build_stream_fields(req: IngestRequest) -> Dict[str, str]:
    idem = req.idempotency_key or str(uuid.uuid4())
    
    # Compute hashes if bucket/key are provided
    image_sha256 = None
    image_phash = None
    final_url = req.url
    
    if req.bucket and req.key:
        image_sha256, image_phash = _compute_hashes_for_bucket_key(req.bucket, req.key)
        if not final_url:
            final_url = f"s3://{req.bucket}/{req.key}"
    elif req.url:
        # For URLs, generate placeholder hashes (worker will compute after download)
        image_sha256 = hashlib.sha256(f"{req.url}{req.tenant_id}".encode()).hexdigest()
        image_phash = "0" * 16
    
    payload = {
        "tenant_id": req.tenant_id,
        "bucket": req.bucket,
        "key": req.key,
        "url": final_url or "unknown",
        "site": req.site or "unknown",
        "image_sha256": image_sha256,
        "image_phash": image_phash,
        "ts": int(req.ts or time.time()),
        "meta": req.meta or {},
        "source": "backend-api",
        "idempotency_key": idem,
        "face_hints": None,  # Worker expects this field
    }
    # Worker supports "message" JSON field; include a few indexed fields too
    fields = {
        "message": json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        "idempotency_key": idem,
        "tenant_id": req.tenant_id,
    }
    return fields

@router.post("/api/v1/ingest")
def ingest_now(req: IngestRequest):
    """
    Enqueue ONE item to the Redis stream (default 'face:ingest').
    Returns the Redis message id and the enqueued payload.
    """
    r = _get_redis()
    fields = _build_stream_fields(req)
    try:
        msg_id = r.xadd(REDIS_STREAM_NAME, fields)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"enqueue_failed: {e}")
    return {"ok": True, "stream": REDIS_STREAM_NAME, "message_id": msg_id, "enqueued": json.loads(fields["message"])}

# -------- Batch ingest --------

class IngestBatchRequest(BaseModel):
    """
    Batch version of /ingest. Up to max_items (default 500).
    'items' is an array of IngestRequest-like objects.
    Optional 'dry_run' to validate without enqueuing.
    """
    items: List[IngestRequest]
    dry_run: bool = Field(default=False)
    max_items: int = Field(default=500, ge=1, le=5000)

@router.post("/api/v1/ingest/batch")
def ingest_batch(req: IngestBatchRequest):
    """
    Enqueue MANY items efficiently using a Redis pipeline.
    Returns per-item status: {index, ok, message_id?, error?}.
    """
    # Enforce max batch size
    n = len(req.items)
    if n == 0:
        raise HTTPException(status_code=422, detail="items must be non-empty")
    if n > req.max_items:
        raise HTTPException(status_code=413, detail=f"too_many_items (max {req.max_items})")

    # Dry run? Validate only
    if req.dry_run:
        # Pydantic has already validated each item; echo a summary
        return {
            "ok": True,
            "dry_run": True,
            "count": n,
            "stream": REDIS_STREAM_NAME,
            "results": [{"index": i, "ok": True} for i in range(n)],
        }

    r = _get_redis()
    pipe = r.pipeline(transaction=False)

    fields_list: List[Dict[str, str]] = []
    for item in req.items:
        fields_list.append(_build_stream_fields(item))
        pipe.xadd(REDIS_STREAM_NAME, fields_list[-1])

    results: List[Dict[str, Any]] = []
    try:
        msg_ids = pipe.execute()  # list of message ids or raises
        for i, mid in enumerate(msg_ids):
            if isinstance(mid, Exception):
                results.append({"index": i, "ok": False, "error": str(mid)})
            else:
                results.append({
                    "index": i,
                    "ok": True,
                    "message_id": mid,
                    "idempotency_key": fields_list[i].get("idempotency_key"),
                })
    except Exception as e:
        # If the whole pipeline failed
        raise HTTPException(status_code=500, detail=f"batch_enqueue_failed: {e}")

    return {
        "ok": all(r.get("ok") for r in results),
        "stream": REDIS_STREAM_NAME,
        "count": n,
        "results": results,
    }

# (Optional) tiny debug endpoint to verify env wiring from the container
@router.get("/api/v1/debug/ingest-config")
def debug_ingest_config():
    return {
        "REDIS_URL": REDIS_URL,
        "REDIS_STREAM_NAME": REDIS_STREAM_NAME,
        "PIPELINE_URL": PIPELINE_URL,
    }
