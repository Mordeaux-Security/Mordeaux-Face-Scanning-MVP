from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
import time
import json
import numpy as np
import cv2
from PIL import Image

from pydantic import BaseModel, HttpUrl

"""
Face Pipeline Processor

Orchestrates face detection, quality assessment, embedding generation, and vector indexing.
"""

logger = logging.getLogger(__name__)


def _build_error_response(image_sha256: str, timings: dict, error_msg: str = None) -> dict:
    """Build standard error response structure."""
    resp = {
        "image_sha256": image_sha256,
        "counts": {"faces_total": 0, "accepted": 0, "rejected": 0, "dup_skipped": 0},
        "artifacts": {"crops": [], "thumbs": [], "metadata": []},
        "timings_ms": timings,
    }
    if error_msg:
        resp["error"] = error_msg
    return resp


class PipelineInput(BaseModel):
    """
    Input data contract for the face processing pipeline.

    Represents an image that has been uploaded to object storage and is ready
    for face detection, quality assessment, embedding generation, and vector indexing.
    """

    image_sha256: str
    """SHA-256 hash of the image content for deduplication and integrity verification."""

    bucket: str
    """MinIO bucket name where the raw image is stored."""

    key: str
    """Object key/path within the bucket for the raw image."""

    tenant_id: str
    """Multi-tenant identifier for data isolation and access control."""

    site: str
    """Site identifier (e.g., domain or source) where the image originated."""

    url: HttpUrl
    """Original HTTP(S) URL where the image was sourced from."""

    image_phash: str
    """Perceptual hash (pHash) of the image for near-duplicate detection."""

    face_hints: Optional[List[Dict]]
    """
    Optional hints about face locations/attributes from upstream processing.
    Can be used to optimize detection or validate results.
    Examples: [{"bbox": [x, y, w, h], "confidence": 0.95}]
    """


def process_image(message: dict) -> dict:
    """
    Process a single image through the face detection and embedding pipeline.

    This is the main entrypoint for processing images. It orchestrates face detection,
    quality assessment, cropping, embedding generation, storage, and vector indexing.

    Args:
        message: Raw dictionary containing pipeline input data (validated via PipelineInput)

    Returns:
        Dictionary with processing results containing:
        - image_sha256: Image hash identifier
        - counts: Face processing statistics
        - artifacts: Generated storage artifacts (crops, thumbnails, metadata)
        - timings_ms: Performance timings for each pipeline stage

    Pipeline stages:
        1. Validate input
        2. Download image from MinIO
        3. Decode image to numpy/PIL
        4. Detect faces (use hints if available, otherwise run detector)
        5. Align and crop faces
        6. Quality assessment per face
        7. Compute pHash and prefix
        8. Deduplication precheck
        9. Generate embeddings
        10. Generate artifact paths
        11. Batch upsert to Qdrant
        12. Return summary
    """
    from config.settings import settings
    from pipeline.detector import detect_faces, align_and_crop
    from pipeline.embedder import embed
    from pipeline.quality import evaluate
    from pipeline.storage import put_bytes, presign, ensure_buckets
    from pipeline.indexer import ensure_collection, upsert, make_point
    from pipeline.utils import compute_phash, phash_prefix, now_iso
    from pipeline.dedup import is_duplicate, mark_processed
    from pipeline.stats import increment_processed, increment_rejected, increment_dup_skipped, inc, timer, add_time_ms
    
    # Start total timing for end-to-end processing
    start_time = time.perf_counter()

    # Counters for results
    faces_total = 0
    faces_accepted = 0
    faces_rejected = 0
    faces_dup_skipped = 0

    artifact_crops = []
    artifact_thumbs = []
    artifact_metadata = []

    timings = {}

    # ========================================================================
    # STEP 1: VALIDATE INPUT
    # ========================================================================
    msg = PipelineInput.model_validate(message)
    logger.info(f"Processing image: {msg.image_sha256} from {msg.site}")
    
    # Increment images counter
    inc("images_total", 1)

    # ========================================================================
    # STEP 2: DOWNLOAD IMAGE FROM MINIO
    # ========================================================================
    from pipeline.storage import get_bytes
    
    with timer("download_ms"):
        try:
            image_bytes = get_bytes(bucket=msg.bucket, key=msg.key)
            timings["download_ms"] = 0  # Will be set by timer context manager
            logger.debug(f"Downloaded {len(image_bytes)} bytes from {msg.bucket}/{msg.key}")
        except Exception as e:
            logger.error(f"Failed to download image {msg.image_sha256}: {e}")
            return _build_error_response(msg.image_sha256, timings, f"Download failed: {e}")

    # ========================================================================
    # STEP 3: DECODE IMAGE
    # ========================================================================
    with timer("decode_ms"):
        try:
            # Direct decode to BGR using OpenCV
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            assert img_bgr is not None, "Decode failed"
            timings["decode_ms"] = 0  # Will be set by timer context manager
            logger.debug(f"Decoded image: {img_bgr.shape}")
        except Exception as e:
            logger.error(f"Failed to decode image {msg.image_sha256}: {e}")
            return _build_error_response(msg.image_sha256, timings, f"Decode failed: {e}")

    # ========================================================================
    # STEP 4: DETECT FACES
    # ========================================================================
    with timer("detect_ms"):
        face_detections = []
        
        if msg.face_hints and len(msg.face_hints) > 0:
            logger.debug(f"Using {len(msg.face_hints)} face hints from upstream")
            face_detections = msg.face_hints
        else:
            logger.debug("Running face detector")
            face_detections = detect_faces(img_bgr)
        
        faces_total = len(face_detections)
        timings["detection_ms"] = 0  # Will be set by timer context manager
        logger.info(f"Detected {faces_total} faces")
    
    # Increment faces detected counter
    inc("faces_detected", faces_total)
    
    if faces_total == 0:
        return _build_error_response(msg.image_sha256, timings)

    # ========================================================================
    # PHASE 2 IMPLEMENTATION: REAL PIPELINE PROCESSING
    # ========================================================================
    
    # Ensure buckets and collection exist
    ensure_buckets()
    ensure_collection()

    faces = face_detections
    faces_total = len(faces)

    accepted = 0
    dup_skipped = 0
    rejected = 0

    points = []
    crops_keys: List[str] = []
    thumbs_keys: List[str] = []
    meta_keys: List[str] = []

    for i, fd in enumerate(faces):
        lmk = fd.get("landmarks") or []
        if len(lmk) < 5:
            rejected += 1
            continue

        # Align & crop to 112 BGR
        with timer("align_ms"):
            crop_bgr = align_and_crop(img_bgr, lmk, image_size=settings.IMAGE_SIZE)

        # Quality gate (tweak thresholds later if needed)
        q = evaluate(crop_bgr)
        if not q["pass"]:
            rejected += 1
            continue

        # Embedding (512-d float32)
        with timer("embed_ms"):
            vec = embed(crop_bgr)
            vec_list = vec.astype(np.float32).tolist()

        # Compute pHash on RGB PIL (more stable)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        phex = compute_phash(pil_crop)
        pfx = phash_prefix(phex, bits=16)

        # ========================================================================
        # STEP 7: GLOBAL DEDUPLICATION CHECK
        # ========================================================================
        # Stage 1: Exact-match dedup (existing)
        if settings.enable_global_dedup and is_duplicate(phex):
            logger.debug(f"Face {i} is duplicate (exact, pHash: {phex}), skipping")
            dup_skipped += 1
            continue
        
        # Stage 2: Near-duplicate dedup (new - Hamming distance)
        from pipeline.dedup import should_skip, remember
        if settings.enable_global_dedup and should_skip(msg.tenant_id, pfx, phex, max_dist=settings.dedup_max_hamming):
            logger.debug(f"Face {i} is near-duplicate (pHash: {phex}), skipping")
            dup_skipped += 1
            continue

        # Build keys
        base = f"{msg.image_sha256}_face_{i}"
        crop_key = f"{msg.tenant_id}/{base}.jpg"
        thumb_key = f"{msg.tenant_id}/{base}_thumb.jpg"
        meta_key = f"{msg.tenant_id}/{base}.json"

        # Write JPEG crop/thumbnail to MinIO
        with timer("storage_ms"):
            # Crop
            _, enc = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            put_bytes(settings.MINIO_BUCKET_CROPS, crop_key, enc.tobytes(), "image/jpeg")

            # Thumb (64x64, keep aspect via cv2.resize then center-crop or just direct resize for now)
            thumb = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
            _, enc_t = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            put_bytes(settings.MINIO_BUCKET_THUMBS, thumb_key, enc_t.tobytes(), "image/jpeg")

            # Metadata JSON
            meta = {
                "face_id": f"{msg.image_sha256}:face_{i}",
                "image_sha256": msg.image_sha256,
                "bbox": fd.get("bbox"),
                "landmarks": fd.get("landmarks"),
                "quality": q,
                "tenant_id": msg.tenant_id,
                "site": msg.site,
                "url": str(msg.url),
                "p_hash": phex,
                "p_hash_prefix": pfx,
                "created_at": now_iso(),
            }
            meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            put_bytes(settings.MINIO_BUCKET_METADATA, meta_key, meta_bytes, "application/json")

        # Prepare Qdrant point
        payload = {
            "tenant_id": msg.tenant_id,
            "site": msg.site,
            "url": str(msg.url),
            "ts": meta["created_at"],
            "p_hash": phex,
            "p_hash_prefix": pfx,
            "bbox": fd.get("bbox"),
            "quality": q,
            "image_sha256": msg.image_sha256,
        }
        points.append(make_point(face_id=meta["face_id"], vector=vec_list, payload=payload))

        crops_keys.append(crop_key)
        thumbs_keys.append(thumb_key)
        meta_keys.append(meta_key)

        # Mark as processed for global dedup
        if settings.enable_global_dedup:
            mark_processed(phex)  # Exact-match dedup (existing)
            remember(msg.tenant_id, pfx, phex, max_size=1000, ttl=3600)  # Near-duplicate dedup (new)

        accepted += 1

    # Batch upsert to Qdrant
    if points:
        with timer("upsert_ms"):
            upsert(points)
        inc("upserts_total", len(points))

    # ========================================================================
    # STEP 12: UPDATE STATISTICS
    # ========================================================================
    try:
        if accepted > 0:
            increment_processed(accepted, msg.tenant_id)
            inc("faces_accepted", accepted)
        if rejected > 0:
            increment_rejected(rejected, msg.tenant_id)
            inc("faces_rejected", rejected)
        if dup_skipped > 0:
            increment_dup_skipped(dup_skipped, msg.tenant_id)
            inc("faces_dup_skipped", dup_skipped)
        logger.debug(f"Updated stats: accepted={accepted}, rejected={rejected}, dup_skipped={dup_skipped}")
    except Exception as e:
        logger.error(f"Failed to update statistics: {e}")
        # Don't fail the entire pipeline for stats errors

    # Record total processing time
    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    add_time_ms("total_ms", total_elapsed_ms)

    # Build summary for return
    summary = {
        "image_sha256": msg.image_sha256,
        "counts": {
            "faces_total": faces_total,
            "accepted": accepted,
            "rejected": rejected,
            "dup_skipped": dup_skipped,
        },
        "artifacts": {
            "crops": [f"{settings.MINIO_BUCKET_CROPS}/{k}" for k in crops_keys],
            "thumbs": [f"{settings.MINIO_BUCKET_THUMBS}/{k}" for k in thumbs_keys],
            "metadata": [f"{settings.MINIO_BUCKET_METADATA}/{k}" for k in meta_keys],
        },
        "timings_ms": timings,
    }
    return summary
