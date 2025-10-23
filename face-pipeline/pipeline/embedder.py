from __future__ import annotations
import numpy as np

from config.settings import settings
from pipeline.models import load_insightface_app

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
    app = load_insightface_app()

    if aligned_bgr_112 is None or aligned_bgr_112.ndim != 3 or aligned_bgr_112.shape[:2] != (settings.IMAGE_SIZE, settings.IMAGE_SIZE):
        raise ValueError(f"Expected aligned {settings.IMAGE_SIZE}x{settings.IMAGE_SIZE} BGR image, got {None if aligned_bgr_112 is None else aligned_bgr_112.shape}")

    # InsightFace expects BGR aligned crop (uint8)
    if aligned_bgr_112.dtype != np.uint8:
        img = np.clip(aligned_bgr_112, 0, 255).astype(np.uint8)
    else:
        img = aligned_bgr_112

    # Use the recognition model directly instead of re-running detection
    # This avoids the "No faces found in aligned crop" error
    rec_model = app.models['recognition']
    
    # Preprocess the image for the recognition model
    # The recognition model expects normalized input
    img_float = img.astype(np.float32)
    img_float = (img_float - 127.5) / 127.5  # Normalize to [-1, 1]
    img_float = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
    img_float = np.expand_dims(img_float, axis=0)  # Add batch dimension
    
    # Use the recognition model's forward method directly
    feat = rec_model.forward(img_float)
    feat = feat.astype(np.float32, copy=False)
    feat = l2_normalize(feat).astype(np.float32, copy=False)
    
    if feat.ndim == 2 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.shape[0] != 512:
        raise ValueError(f"Expected 512-dim embedding, got {feat.shape}")
    return feat