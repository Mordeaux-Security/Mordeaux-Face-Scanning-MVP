from fastapi import APIRouter, UploadFile, File
from ..services.face import get_face_service

router = APIRouter()

@router.post("/search_face")
async def search_face(file: UploadFile = File(...)):
    content = await file.read()
    svc = get_face_service()
    ph = svc.compute_phash(content)
    faces = svc.detect_and_embed(content)
    return {"phash": ph, "faces_found": len(faces), "dim": 512 if faces else None}
