from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb
from ..services.vector import get_vector_client
import uuid

api_router = APIRouter()

@api_router.post("/search_face")
async def search_face(file: UploadFile = File(...)):
    content = await file.read()
    if not content or (file.content_type or "").lower() not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image")

    svc = get_face_service()
    phash = svc.compute_phash(content)
    faces = svc.detect_and_embed(content)

    raw_key, raw_url, thumb_key, thumb_url = save_raw_and_thumb(content)

    # Upsert embeddings
    vec = get_vector_client()
    items = []
    for f in faces:
        items.append({
            "id": str(uuid.uuid4()),
            "embedding": f["embedding"],
            "metadata": {
                "raw_key": raw_key,
                "thumb_key": thumb_key,
                "raw_url": raw_url,
                "thumb_url": thumb_url,
                "bbox": f["bbox"],
                "det_score": f["det_score"],
                "phash": phash,
            },
        })
    if items:
        vec.upsert_embeddings(items)

    # Query similar for the first face (if any)
    results = []
    if faces:
        results = vec.search_similar(faces[0]["embedding"], topk=10)

    return {
        "phash": phash,
        "faces_found": len(faces),
        "raw_url": raw_url,
        "thumb_url": thumb_url,
        "results": results,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
    }