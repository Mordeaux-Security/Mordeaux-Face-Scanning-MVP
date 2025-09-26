import io
import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from insightface.app import FaceAnalysis

_face_app = None

def _load_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        home = os.path.expanduser("~/.insightface")
        os.makedirs(home, exist_ok=True)
        app = FaceAnalysis(name="buffalo_l", root=home)
        # CPU default (onnxruntime)
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
    return _face_app

def _read_image(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # fallback via PIL if needed
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img

def compute_phash(b: bytes) -> str:
    pil = Image.open(io.BytesIO(b)).convert("RGB")
    return str(imagehash.phash(pil))

def detect_and_embed(content: bytes):
    app = _load_app()
    img = _read_image(content)
    faces = app.get(img)  # returns bbox, kps, det_score, embedding
    out = []
    for f in faces:
        if not hasattr(f, "embedding") or f.embedding is None:
            continue
        emb = f.embedding.astype(np.float32).tolist()
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        out.append({
            "bbox": [x1, y1, x2, y2],
            "embedding": emb,
            "det_score": float(getattr(f, "det_score", 0.0)),
        })
    return out

def get_face_service():
    # simple accessor; in real app you might have a class
    class _FaceSvc:
        detect_and_embed = staticmethod(detect_and_embed)
        compute_phash = staticmethod(compute_phash)
    return _FaceSvc()
