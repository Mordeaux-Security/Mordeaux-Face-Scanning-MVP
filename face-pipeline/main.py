import os
from typing import List, Optional, Dict, Any
import hashlib
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, ScoredPoint, PointStruct, SearchParams
)

from pipeline.ensure import ensure_all
from pipeline.insight import get_app
from pipeline.image_utils import decode_image_b64

app = FastAPI(title="face-pipeline")

# ----- Config -----
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
FACES_COLLECTION = os.getenv("QDRANT_COLLECTION", "faces_v1")
IDENTITY_COLLECTION = os.getenv("IDENTITY_COLLECTION", "identities_v1")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))

# ----- Models -----
class SearchReq(BaseModel):
    tenant_id: str
    vector: Optional[List[float]] = None
    image_b64: Optional[str] = None
    top_k: int = Field(default=50, ge=1, le=200)
    threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    mode: str = Field(default="standard")

class EnrollReq(BaseModel):
    tenant_id: str
    identity_id: str
    images_b64: List[str] = Field(..., min_items=2, max_items=10)  # recommend 3–5
    overwrite: bool = Field(default=True)

class VerifyReq(BaseModel):
    tenant_id: str
    identity_id: str
    image_b64: str
    # thresholds: hi = strong accept, lo = borderline accept if corroborated (not used here but exposed)
    hi_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)

# ----- Health -----
@app.get("/api/v1/health")
def health():
    return {"status": "healthy", "service": "face-pipeline-search-api"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "face-pipeline"}

@app.on_event("startup")
def startup():
    ensure_all()

# ----- Embedding -----
def _embed_one_b64(img_b64: str) -> np.ndarray:
    # decode
    import cv2
    img = decode_image_b64(img_b64)
    if img is None:
        raise HTTPException(400, "invalid_image_data")
    # detect & embed with InsightFace pack
    app_ = get_app()
    faces = app_.get(img)
    if not faces:
        raise HTTPException(422, "no_face_detected")
    # choose highest det_score
    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    vec = getattr(best, "normed_embedding", None)
    if vec is None:
        raw = getattr(best, "embedding", None)
        if raw is None:
            raise HTTPException(500, "recognition_model_not_loaded")
        arr = np.array(raw, dtype=np.float32)
        vec = arr / (np.linalg.norm(arr) + 1e-9)
    else:
        vec = np.array(vec, dtype=np.float32)
    if vec.shape[0] != VECTOR_DIM:
        raise HTTPException(500, f"vector_dim_mismatch: got {vec.shape[0]}, expected {VECTOR_DIM}")
    # ensure float32 & unit norm
    vec = vec.astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# ----- Qdrant helpers -----
def _qc() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

def _tenant_filter(tenant_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])

def _tenant_identity_filter(tenant_id: str, identity_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        FieldCondition(key="identity_id", match=MatchValue(value=identity_id)),
    ])

# ----- Public: search (existing) -----
@app.post("/api/v1/search")
def search(req: SearchReq):
    qc = _qc()
    if not req.vector and not req.image_b64:
        raise HTTPException(422, "Provide vector or image_b64")

    if req.image_b64:
        vec = _embed_one_b64(req.image_b64)
    else:
        if len(req.vector) != VECTOR_DIM:
            raise HTTPException(422, f"vector length must be {VECTOR_DIM}")
        arr = np.array(req.vector, dtype=np.float32)
        vec = arr / (np.linalg.norm(arr) + 1e-9)

    hits = qc.search(
        collection_name=FACES_COLLECTION,
        query_vector=vec.tolist(),
        limit=req.top_k,
        score_threshold=req.threshold,
        query_filter=_tenant_filter(req.tenant_id),
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )
    out = [{"id": str(h.id), "score": float(h.score), "payload": h.payload or {}} for h in hits]
    return {
        "query": {"tenant_id": req.tenant_id, "search_mode": "image" if req.image_b64 else "vector",
                  "mode": req.mode, "top_k": req.top_k, "threshold": req.threshold},
        "hits": out, "count": len(out)
    }

# ----- Public: enroll identity -----
@app.post("/api/v1/enroll_identity")
def enroll_identity(req: EnrollReq):
    if len(req.images_b64) < 2:
        raise HTTPException(422, "provide at least 2 images for a stable centroid")

    vecs = []
    for data in req.images_b64:
        vecs.append(_embed_one_b64(data))
    M = np.vstack(vecs).astype(np.float32)
    centroid = np.mean(M, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    qc = _qc()
    # Upsert one point per (tenant_id, identity_id)
    # Generate UUID from tenant_id:identity_id string (for Qdrant compatibility)
    identity_key = f"{req.tenant_id}:{req.identity_id}"
    hash_obj = hashlib.sha256(identity_key.encode())
    hex_dig = hash_obj.hexdigest()
    point_id = uuid.UUID(hex_dig[:32])
    
    payload = {"tenant_id": req.tenant_id, "identity_id": req.identity_id}
    pts = [PointStruct(id=str(point_id), vector=centroid.tolist(), payload=payload)]
    qc.upsert(collection_name=IDENTITY_COLLECTION, points=pts, wait=True)

    return {"ok": True, "identity": {"tenant_id": req.tenant_id, "identity_id": req.identity_id}, "vector_dim": VECTOR_DIM}

# ----- Public: verify probe belongs to identity; return ONLY that identity's faces -----
@app.post("/api/v1/verify")
def verify(req: VerifyReq):
    qc = _qc()
    # 1) fetch identity centroid
    ids = qc.search(
        collection_name=IDENTITY_COLLECTION,
        query_vector=[0.0]*VECTOR_DIM,  # dummy; we use filtering + scroll alternative below if needed
        limit=1,
        score_threshold=0.0,
        query_filter=_tenant_identity_filter(req.tenant_id, req.identity_id),
        with_payload=True,
        with_vectors=True,  # need the vector
        search_params=SearchParams(hnsw_ef=64, exact=True),
    )
    # If search returns nothing (because exact=True with dummy vector), fallback to scroll
    if not ids:
        res = qc.scroll(
            collection_name=IDENTITY_COLLECTION,
            scroll_filter=_tenant_identity_filter(req.tenant_id, req.identity_id),
            with_payload=True, with_vectors=True, limit=1
        )
        id_hits = res[0]
        if not id_hits:
            raise HTTPException(404, "identity_not_enrolled")
        centroid_vec = np.array(id_hits[0].vector, dtype=np.float32)
    else:
        centroid_vec = np.array(ids[0].vector, dtype=np.float32)

    centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-9)

    # 2) embed probe
    probe = _embed_one_b64(req.image_b64)

    # 3) verify cosine
    sim = _cos(probe, centroid_vec)
    passed = bool(sim >= req.hi_threshold)

    # 4) if passed, fetch ONLY this identity's faces from faces_v1
    faces: List[Dict[str, Any]] = []
    if passed:
        hits = qc.search(
            collection_name=FACES_COLLECTION,
            query_vector=probe.tolist(),  # or centroid_vec.tolist(); probe is fine
            limit=req.top_k,
            score_threshold=0.0,  # we'll filter post by identity
            query_filter=_tenant_identity_filter(req.tenant_id, req.identity_id),
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(hnsw_ef=128, exact=False),
        )
        # (optional) threshold again on score if you want
        faces = [{"id": str(h.id), "score": float(h.score), "payload": h.payload or {}} for h in hits]

    return {
        "verified": passed,
        "similarity": sim,
        "threshold": req.hi_threshold,
        "tenant_id": req.tenant_id,
        "identity_id": req.identity_id,
        "results": faces if passed else [],
        "count": len(faces) if passed else 0,
    }
