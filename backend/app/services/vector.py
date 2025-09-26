import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

_client = None
_collection = None

def _init():
    global _client, _collection
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        _client = QdrantClient(url=url)
    if _collection is None:
        name = os.getenv("VECTOR_INDEX", "faces")
        _collection = name
        try:
            _client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE),
            )
        except Exception:
            pass
    return _client, _collection

def upsert(points):
    client, col = _init()
    client.upsert(
        collection_name=col,
        points=points,
    )

def search(vector, topk=10):
    client, col = _init()
    r = client.search(collection_name=col, query_vector=vector, limit=topk)
    return r

def get_vector_client():
    class _Vec:
        upsert = staticmethod(upsert)
        search = staticmethod(search)
    return _Vec()
