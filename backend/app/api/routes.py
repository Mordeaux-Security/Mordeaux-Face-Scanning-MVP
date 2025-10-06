from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb, get_object_from_storage
from ..services.vector import get_vector_client
import uuid
import io

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


@api_router.post("/compare_face")
async def compare_face(file: UploadFile = File(...)):
    """
    Search-only endpoint: compares uploaded image against existing faces in database
    without uploading the image to storage or database.
    """
    content = await file.read()
    if not content or (file.content_type or "").lower() not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image")

    svc = get_face_service()
    phash = svc.compute_phash(content)
    faces = svc.detect_and_embed(content)

    if not faces:
        return {
            "phash": phash,
            "faces_found": 0,
            "results": [],
            "vector_backend": "pinecone" if get_vector_client().using_pinecone() else "qdrant",
            "message": "No faces detected in the uploaded image"
        }

    # Query similar faces for the first detected face
    vec = get_vector_client()
    results = vec.search_similar(faces[0]["embedding"], topk=10)

    return {
        "phash": phash,
        "faces_found": len(faces),
        "results": results,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
        "message": f"Found {len(faces)} face(s) and {len(results)} similar matches"
    }


@api_router.get("/images/{bucket}/{key:path}")
async def serve_image(bucket: str, key: str):
    """Proxy endpoint to serve images from storage."""
    try:
        image_data = get_object_from_storage(bucket, key)
        return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {str(e)}")