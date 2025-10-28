import uuid
from typing import List, Dict, Any

# Lazy singletons
from qdrant_client import QdrantClient
from pinecone import Pinecone
from qdrant_client.http import models as qm

from ..core.config import get_settings

_qdrant_client = None
_pinecone_index = None


# ---------- Qdrant (local dev) ----------
def _qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
        try:
            _qdrant_client.create_collection(
                collection_name=settings.vector_index,
                vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE),
            )
        except Exception:
            pass
    return _qdrant_client


def _qdrant_upsert(items: List[Dict[str, Any]]) -> None:
    cli = _qdrant()
    settings = get_settings()
    points = []
    for it in items:
        pid = it.get("id") or str(uuid.uuid4())
        points.append(
            qm.PointStruct(
                id=pid,
                vector=it["embedding"],
                payload=it.get("metadata", {}),
            )
        )
    cli.upsert(collection_name=settings.vector_index, points=points)


def _qdrant_search(vector: List[float], tenant_id: str, topk: int = 10) -> List[Dict[str, Any]]:
    cli = _qdrant()
    settings = get_settings()
    # Filter by tenant_id using Qdrant filter
    filter_condition = qm.Filter(
        must=[
            qm.FieldCondition(
                key="tenant_id",
                match=qm.MatchValue(value=tenant_id)
            )
        ]
    )
    res = cli.search(
        collection_name=settings.vector_index,
        query_vector=vector,
        limit=topk,
        with_payload=True,
        query_filter=filter_condition
    )
    return [
        {"id": str(r.id), "score": float(r.score), "metadata": dict(r.payload or {})}
        for r in res
    ]


# ---------- Pinecone (production) ----------
def _pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        # Assume index already created in console as 512-d cosine
        _pinecone_index = pc.Index(settings.pinecone_index)
    return _pinecone_index


def _pinecone_upsert(items: List[Dict[str, Any]]) -> None:
    idx = _pinecone_index()
    vectors = []
    for it in items:
        pid = it.get("id") or str(uuid.uuid4())
        vectors.append({
            "id": pid,
            "values": it["embedding"],
            "metadata": it.get("metadata", {}),
        })
    # Pinecone v3 client: upsert expects list of dicts
    idx.upsert(vectors=vectors)


def _pinecone_search(vector: List[float], tenant_id: str, topk: int = 10) -> List[Dict[str, Any]]:
    idx = _pinecone_index()
    # Filter by tenant_id using Pinecone metadata filter
    filter_condition = {"tenant_id": {"$eq": tenant_id}}
    res = idx.query(
        vector=vector,
        top_k=topk,
        include_metadata=True,
        filter=filter_condition
    )
    return [
        {"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
        for m in res.matches or []
    ]


# ---------- Public API ----------
def using_pinecone() -> bool:
    settings = get_settings()
    return settings.using_pinecone


def upsert_embeddings(items: List[Dict[str, Any]], tenant_id: str) -> None:
    """
    items: [{id?, embedding: List[float], metadata: {...}}]
    tenant_id: Tenant identifier for scoping
    """
    # Add tenant_id to metadata for all items
    for item in items:
        if "metadata" not in item:
            item["metadata"] = {}
        item["metadata"]["tenant_id"] = tenant_id

    if using_pinecone():
        _pinecone_upsert(items)
    else:
        _qdrant_upsert(items)


def search_similar(embedding: List[float], tenant_id: str, topk: int = 10) -> List[Dict[str, Any]]:
    if using_pinecone():
        return _pinecone_search(embedding, tenant_id, topk)
    else:
        return _qdrant_search(embedding, tenant_id, topk)


def get_vector_client():
    class _Vec:
        upsert_embeddings = staticmethod(upsert_embeddings)
        search_similar = staticmethod(search_similar)
        using_pinecone = staticmethod(using_pinecone)
    return _Vec()
