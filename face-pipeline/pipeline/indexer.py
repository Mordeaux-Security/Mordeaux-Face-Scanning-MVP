
from __future__ import annotations
from typing import List, Dict, Any, Optional
import time
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, ScoredPoint,
    PayloadSchemaType, SearchParams
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


def scoredpoint_to_dict(sp: ScoredPoint) -> dict:
    return {
        "id": sp.id,
        "score": float(sp.score),
        "payload": sp.payload or {},
        "vector": getattr(sp, "vector", None),
    }

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
    
    # Create payload indexes for faster filtering (idempotent)
    fields_to_index = ['tenant_id', 'p_hash_prefix', 'site', 'identity_id']
    
    try:
        collection_info = qc.get_collection(collection_name=coll)
        existing_indexes = collection_info.payload_schema or {}
        
        for field in fields_to_index:
            if field not in existing_indexes:
                try:
                    qc.create_payload_index(
                        collection_name=coll,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    # Handle "index already exists" and other errors gracefully
                    if "already exists" not in str(e).lower():
                        print(f"Warning: Could not create payload index for {field}: {e}")
    except Exception as e:
        print(f"Warning: Could not check/create payload indexes: {e}")

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

def search(
    vector: list[float],
    top_k: int,
    *,
    tenant_id: Optional[str] = None,
    threshold: float | None = None,
    site: Optional[str] = None,
    flt: Optional[Filter] = None,
    hnsw_ef: int | None = None,
) -> List[ScoredPoint]:
    """
    Search for similar face embeddings in Qdrant.
    
    Args:
        vector: Query embedding vector (512-dim, L2-normalized)
        top_k: Maximum number of results to return
        tenant_id: Filter by tenant
        threshold: Minimum similarity score (default from settings.SIMILARITY_THRESHOLD)
        site: Filter by source site
        flt: Custom Qdrant filter
        hnsw_ef: HNSW search parameter (higher = more accurate, slower). 
                 Default from settings.HNSW_EF (256)
    
    Returns:
        List of ScoredPoint results
    """
    qc = get_client()
    ensure_collection()
    # Build a filter if not provided
    if flt is None and tenant_id:
        flt = build_filter(tenant_id=tenant_id, site=site)
    
    # Use configurable HNSW_EF for search accuracy
    # Higher ef = more accurate but slower. Default 256 (was implicit 128)
    effective_hnsw_ef = hnsw_ef if hnsw_ef is not None else settings.HNSW_EF
    
    res = qc.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        score_threshold=threshold or settings.SIMILARITY_THRESHOLD,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
        # Use exact search for maximum accuracy (trades speed for precision)
        # Exact search ensures accurate cosine similarity scores
        search_params=SearchParams(
            exact=True,  # Exact cosine similarity for maximum accuracy
        ),
    )
    return res


def search_dict(
    vector: list[float],
    top_k: int,
    *,
    tenant_id: Optional[str] = None,
    threshold: float | None = None,
    site: Optional[str] = None,
    flt: Optional[Filter] = None,
) -> List[dict]:
    points = search(
        vector=vector,
        top_k=top_k,
        tenant_id=tenant_id,
        threshold=threshold,
        site=site,
        flt=flt,
    )
    return [scoredpoint_to_dict(p) for p in points]