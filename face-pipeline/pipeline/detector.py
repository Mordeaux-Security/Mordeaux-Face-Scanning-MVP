from __future__ import annotations
from typing import List, Dict

import numpy as np
import cv2

from insightface.utils import face_align

from config.settings import settings
from pipeline.models import load_insightface_app

def detect_faces_raw(img_np_bgr: np.ndarray) -> List:
    """
    Run detector -> return list of raw InsightFace Face objects.
    
    These objects contain attributes like:
      - bbox: ndarray [x1,y1,x2,y2]
      - kps: ndarray (5,2) - 5 facial landmarks
      - det_score: float - detection confidence
      - embedding: ndarray - 512-dim face embedding (if recognition model loaded)
      - normed_embedding: ndarray - L2-normalized embedding
      - pose: ndarray [pitch, yaw, roll] in radians
    
    Used by face_helpers for quality evaluation which needs the raw Face objects.
    """
    app = load_insightface_app()
    # InsightFace expects BGR; we keep that convention throughout.
    faces = app.get(img_np_bgr)
    thresh = settings.DET_SCORE_THRESH
    
    # Filter by detection threshold
    filtered_faces = []
    for f in faces:
        score = float(getattr(f, "det_score", 1.0))
        if score >= thresh:
            filtered_faces.append(f)
    
    return filtered_faces


def detect_faces(img_np_bgr: np.ndarray) -> List[Dict]:
    """
    Run detector -> return list of dicts:
      { "bbox": [x1,y1,x2,y2], "landmarks": [[x,y],...5], "confidence": float }
    """
    faces = detect_faces_raw(img_np_bgr)
    out: List[Dict] = []
    for f in faces:
        # f.bbox: ndarray [x1,y1,x2,y2], f.kps: (5,2), f.det_score
        score = float(getattr(f, "det_score", 1.0))
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