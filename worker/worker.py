import os
from celery import Celery


from insightface.app import FaceAnalysis

app = Celery(
    "mordeaux",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/2"),
)

@app.task(name="faces.embed")
def embed_face():
    # minimal smoke test to ensure models load
    fa = FaceAnalysis(name="buffalo_l")
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    return {"ok": True}
