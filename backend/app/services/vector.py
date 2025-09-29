import os
import uuid
from typing import List, Dict, Any

VECTOR_INDEX = os.getenv("VECTOR_INDEX", "faces")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()

# Lazy singletons
_qdrant_client = None
_pinecone_index = None


# ---------- Qdrant (local dev) ----------
def _qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        _qdrant_client = QdrantClient(url=url)
        try:
            _qdrant_client.create_collection(
                collection_name=VECTOR_INDEX,
                vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE),
            )
        except Exception:
            pass
    return _qdrant_client


def _qdrant_upsert(items: List[Dict[str, Any]]) -> None:
    from qdrant_client.http import models as qm
    cli = _qdrant()
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
    cli.upsert(collection_name=VECTOR_INDEX, points=points)


def _qdrant_search(vector: List[float], topk: int = 10) -> List[Dict[str, Any]]:
    cli = _qdrant()
    res = cli.search(collection_name=VECTOR_INDEX, query_vector=vector, limit=topk, with_payload=True)
    return [
        {"id": str(r.id), "score": float(r.score), "metadata": dict(r.payload or {})}
        for r in res
    ]


# ---------- Pinecone (production) ----------
def _pinecone_index():
    from pinecone import Pinecone
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Assume index already created in console as 512-d cosine
        _pinecone_index = pc.Index(VECTOR_INDEX)
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


def _pinecone_search(vector: List[float], topk: int = 10) -> List[Dict[str, Any]]:
    idx = _pinecone_index()
    res = idx.query(vector=vector, top_k=topk, include_metadata=True)
    return [
        {"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
        for m in res.matches or []
    ]


# ---------- Public API ----------
def using_pinecone() -> bool:
    return bool(PINECONE_API_KEY) and ENVIRONMENT.lower() == "production"


def upsert_embeddings(items: List[Dict[str, Any]]) -> None:
    """
    items: [{id?, embedding: List[float], metadata: {...}}]
    """
    if using_pinecone():
        _pinecone_upsert(items)
    else:
        _qdrant_upsert(items)


def search_similar(embedding: List[float], topk: int = 10) -> List[Dict[str, Any]]:
    if using_pinecone():
        return _pinecone_search(embedding, topk)
    else:
        return _qdrant_search(embedding, topk)


def get_vector_client():
    class _Vec:
        upsert_embeddings = staticmethod(upsert_embeddings)
        search_similar = staticmethod(search_similar)
        using_pinecone = staticmethod(using_pinecone)
    return _Vec()