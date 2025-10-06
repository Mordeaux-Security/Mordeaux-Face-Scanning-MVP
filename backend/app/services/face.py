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

def crop_face_from_image(image_bytes: bytes, bbox: list, margin: float = 0.2) -> bytes:
    """
    Crop face region from image with margin around the face.
    
    Args:
        image_bytes: Original image data
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        margin: Margin around face as fraction of face size (default: 0.2 = 20%)
        
    Returns:
        Cropped face image as bytes
    """
    import io
    
    # Load image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_width, img_height = pil_image.size
    
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox
    
    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Calculate margin in pixels
    margin_x = int(face_width * margin)
    margin_y = int(face_height * margin)
    
    # Calculate crop coordinates with margin
    crop_x1 = max(0, int(x1 - margin_x))
    crop_y1 = max(0, int(y1 - margin_y))
    crop_x2 = min(img_width, int(x2 + margin_x))
    crop_y2 = min(img_height, int(y2 + margin_y))
    
    # Crop the image
    cropped_image = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Convert back to bytes
    output = io.BytesIO()
    cropped_image.save(output, format='JPEG', quality=95)
    return output.getvalue()


def get_face_service():
    # simple accessor; in real app you might have a class
    class _FaceSvc:
        detect_and_embed = staticmethod(detect_and_embed)
        compute_phash = staticmethod(compute_phash)
        crop_face_from_image = staticmethod(crop_face_from_image)
    return _FaceSvc()
