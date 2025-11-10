import os
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType as PT

def ensure_minio():
    client = Minio(
        os.getenv("MINIO_ENDPOINT","minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY","minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY","minioadmin"),
        secure=os.getenv("MINIO_SECURE","false").lower()=="true"
    )
    for bucket in ["raw-images","thumbnails","crops"]:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

def ensure_qdrant():
    qc = QdrantClient(url=os.getenv("QDRANT_URL","http://qdrant:6333"), api_key=os.getenv("QDRANT_API_KEY",""))
    coll = os.getenv("QDRANT_COLLECTION","faces_v1")
    existing = {c.name for c in qc.get_collections().collections}
    if coll not in existing:
        qc.recreate_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=int(os.getenv("VECTOR_DIM","512")), distance=Distance.COSINE)
        )
        qc.create_payload_index(coll, field_name="tenant_id", field_schema=PT.KEYWORD, wait=True)
        qc.create_payload_index(coll, field_name="p_hash_prefix", field_schema=PT.KEYWORD, wait=True)

def ensure_all():
    try:
        ensure_minio()
    except Exception as e:
        print("minio ensure failed:", e)
    try:
        ensure_qdrant()
    except Exception as e:
        print("qdrant ensure failed:", e)
 

