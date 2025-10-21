from __future__ import annotations
import threading
import numpy as np
import cv2

from insightface.app import FaceAnalysis

from config.settings import settings

_embed_lock = threading.Lock()
_embed_model = None  # ArcFace-like model

def load_model():
    """
    Load the ArcFace embedding model once. We'll use the buffalo_l app which includes the recognition model.
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    with _embed_lock:
        if _embed_model is not None:
            return _embed_model

        # Use buffalo_l app which includes the recognition model
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0)  # 0 for GPU if available; falls back gracefully
        _embed_model = app
        return _embed_model

def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < eps:
        return vec
    return vec / n

def embed(aligned_bgr_112: "np.ndarray") -> "np.ndarray":
    """
    aligned_bgr_112: ndarray HxWxC, 112x112x3, BGR aligned crop (from detector.align_and_crop).
    Returns: np.ndarray shape (512,), dtype float32
    """
    app = load_model()

    if aligned_bgr_112 is None or aligned_bgr_112.ndim != 3 or aligned_bgr_112.shape[:2] != (settings.IMAGE_SIZE, settings.IMAGE_SIZE):
        raise ValueError(f"Expected aligned {settings.IMAGE_SIZE}x{settings.IMAGE_SIZE} BGR image, got {None if aligned_bgr_112 is None else aligned_bgr_112.shape}")

    # InsightFace expects BGR aligned crop (uint8)
    if aligned_bgr_112.dtype != np.uint8:
        img = np.clip(aligned_bgr_112, 0, 255).astype(np.uint8)
    else:
        img = aligned_bgr_112

    # Use the buffalo_l app to get embeddings
    faces = app.get(img)
    if not faces:
        raise ValueError("No faces found in aligned crop")
    
    # Get the first face's embedding
    face = faces[0]
    feat = face.embedding  # This should be the 512-dim embedding
    feat = feat.astype(np.float32, copy=False)
    feat = l2_normalize(feat).astype(np.float32, copy=False)
    
    if feat.ndim == 2 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.shape[0] != 512:
        raise ValueError(f"Expected 512-dim embedding, got {feat.shape}")
    return feat