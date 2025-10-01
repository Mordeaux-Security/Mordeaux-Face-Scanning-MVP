import os
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

s3 = Minio(
  os.getenv("S3_ENDPOINT","minio:9000").replace("http://","").replace("https://",""),
  access_key=os.getenv("S3_ACCESS_KEY"),
  secret_key=os.getenv("S3_SECRET_KEY"),
  secure=os.getenv("S3_USE_SSL","false").lower()=="true",
)
for b in [os.getenv("S3_BUCKET_RAW","raw-images"), os.getenv("S3_BUCKET_THUMBS","thumbnails")]:
  if not s3.bucket_exists(b):
    s3.make_bucket(b)

q = QdrantClient(url=os.getenv("QDRANT_URL","http://qdrant:6333"))
name = os.getenv("VECTOR_INDEX","faces")
try:
  q.create_collection(collection_name=name, vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE))
except Exception:
  pass
print("Seed OK")
