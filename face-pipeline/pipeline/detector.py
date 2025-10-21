from __future__ import annotations
from typing import List, Dict, Tuple
import os
import threading

import numpy as np
import cv2

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from config.settings import settings

# Thread-safe singleton loader
_detector_lock = threading.Lock()
_detector_app: FaceAnalysis | None = None

def _parse_det_size(det_size_str: str) -> Tuple[int, int]:
    try:
        w, h = det_size_str.split(",")
        return (int(w), int(h))
    except Exception:
        return (640, 640)

def load_detector() -> FaceAnalysis:
    """Load InsightFace FaceAnalysis (SCRFD-based) once."""
    global _detector_app
    if _detector_app is not None:
        return _detector_app

    with _detector_lock:
        if _detector_app is not None:
            return _detector_app

        det_size = _parse_det_size(settings.DET_SIZE)

        # Providers hint for ONNXRuntime (FaceAnalysis respects env/provider availability)
        providers = [p.strip() for p in settings.ONNX_PROVIDERS_CSV.split(",") if p.strip()]

        # Create app; name 'buffalo_l' packs a detector+recognizer, but we'll
        # still do embeddings in embedder.py for clean separation.
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        # ctx_id: -1=CPU; >=0 GPU id. We'll let providers decide; set ctx_id=0 if CUDA is present.
        # If CUDA provider is first, ctx_id=0 is okay; otherwise -1.
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        app.prepare(ctx_id=ctx_id, det_size=det_size)

        _detector_app = app
        return _detector_app

def detect_faces(img_np_bgr: np.ndarray) -> List[Dict]:
    """
    Run detector -> return list of dicts:
      { "bbox": [x1,y1,x2,y2], "landmarks": [[x,y],...5], "confidence": float }
    """
    app = load_detector()
    # InsightFace expects BGR; we keep that convention throughout.
    faces = app.get(img_np_bgr)
    out: List[Dict] = []
    thresh = settings.DET_SCORE_THRESH
    for f in faces:
        # f.bbox: ndarray [x1,y1,x2,y2], f.kps: (5,2), f.det_score
        score = float(getattr(f, "det_score", 1.0))
        if score < thresh:
            continue
        bbox = [float(v) for v in f.bbox.tolist()]
        kps = f.kps.tolist() if hasattr(f, "kps") else []
        out.append({
            "bbox": [int(round(b)) for b in bbox],
            "landmarks": [[float(x), float(y)] for x, y in kps],
            "confidence": score,
        })
    return out

def align_and_crop(img_np_bgr: np.ndarray, landmarks: List[List[float]], image_size: int | None = None) -> "np.ndarray":
    """
    Align using 5-point landmarks and return a BGR 112x112 (or configured) crop ndarray.
    """
    if not landmarks or len(landmarks) < 5:
        raise ValueError("Need 5-point landmarks for alignment")
    if image_size is None:
        image_size = settings.IMAGE_SIZE
    kps = np.array(landmarks, dtype=np.float32)
    # norm_crop returns an aligned BGR ndarray
    crop = face_align.norm_crop(img_np_bgr, landmark=kps, image_size=image_size)
    return crop

def to_bgr(np_img: np.ndarray | None) -> np.ndarray:
    """
    Ensure BGR np.ndarray. Accepts:
      - already BGR
      - RGB (heuristic channel flip)
      - grayscale
    """
    if np_img is None:
        raise ValueError("Empty image")
    if np_img.ndim == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    if np_img.shape[2] == 3:
        # Assume RGB if mean(B-G) differs more than mean(G-R)? Safer: expose as config later.
        # For now, caller must pass BGR; if they pass RGB, swap here:
        # return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        return np_img
    if np_img.shape[2] == 4:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_BGRA2BGR)
        return bgr
    raise ValueError(f"Unexpected image shape {np_img.shape}")