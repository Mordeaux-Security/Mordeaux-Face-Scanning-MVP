from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb, get_object_from_storage
from ..services.vector import get_vector_client
import uuid
import io

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