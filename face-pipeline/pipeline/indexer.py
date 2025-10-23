from __future__ import annotations
from typing import List, Dict, Any, Optional
import time
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, ScoredPoint
)

from config.settings import settings

_client: QdrantClient | None = None

def get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    _client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30,
    )
    return _client

def ensure_collection() -> None:
    qc = get_client()
    coll = settings.QDRANT_COLLECTION
    if coll not in {c.name for c in qc.get_collections().collections}:
        qc.recreate_collection(
            collection_name=coll,
            vectors_config=VectorParams(
                size=settings.VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )
        # Payload index helpers: you can add later if desired

def make_point(face_id: str, vector: list[float], payload: Dict[str, Any]) -> PointStruct:
    # Convert string face_id to a UUID for Qdrant compatibility
    # Use SHA-256 hash of face_id to generate a consistent UUID
    hash_obj = hashlib.sha256(face_id.encode())
    hex_dig = hash_obj.hexdigest()
    # Create a UUID from the first 32 characters of the hash
    point_uuid = uuid.UUID(hex_dig[:32])
    return PointStruct(id=str(point_uuid), vector=vector, payload=payload)

def upsert(points: List[PointStruct]) -> None:
    if not points:
        return
    qc = get_client()
    ensure_collection()
    qc.upsert(collection_name=settings.QDRANT_COLLECTION, points=points, wait=True)

def build_filter(tenant_id: str, site: Optional[str] = None, pfx: Optional[str] = None) -> Filter:
    must = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    if site:
        must.append(FieldCondition(key="site", match=MatchValue(value=site)))
    if pfx:
        must.append(FieldCondition(key="p_hash_prefix", match=MatchValue(value=pfx)))
    return Filter(must=must)

def search(vector: list[float], top_k: int, tenant_id: str,
           threshold: float | None = None, site: Optional[str] = None) -> List[ScoredPoint]:
    qc = get_client()
    ensure_collection()
    flt = build_filter(tenant_id=tenant_id, site=site)
    res = qc.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        score_threshold=threshold or settings.SIMILARITY_THRESHOLD,
        query_filter=flt,
        with_payload=True,
    )
    return res