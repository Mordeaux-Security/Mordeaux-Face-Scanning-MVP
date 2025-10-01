from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb
from ..services.vector import get_vector_client
import uuid

router = APIRouter()

def _require_image(file: UploadFile, content: bytes):
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    ctype = (file.content_type or "").lower()
    if ctype not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image")

@router.post("/index_face")
async def index_face(file: UploadFile = File(...)):
    """Upload image, extract embeddings, and upsert to vector DB (no search)."""
    content = await file.read()
    _require_image(file, content)

    face = get_face_service()
    phash = face.compute_phash(content)
    faces = face.detect_and_embed(content)

    raw_key, raw_url, thumb_key, thumb_url = save_raw_and_thumb(content)

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

    return {
        "indexed": len(items),
        "phash": phash,
        "thumb_url": thumb_url,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
    }

@router.post("/search_face")
async def search_face(file: UploadFile = File(...)):
    """Upload image, embed, and query top matches (also upserts so future queries match)."""
    content = await file.read()
    _require_image(file, content)

    face = get_face_service()
    phash = face.compute_phash(content)
    faces = face.detect_and_embed(content)

    raw_key, raw_url, thumb_key, thumb_url = save_raw_and_thumb(content)

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

    results = []
    if faces:
        results = vec.search_similar(faces[0]["embedding"], topk=10)

    return {
        "faces_found": len(faces),
        "phash": phash,
        "thumb_url": thumb_url,
        "results": results,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
    }