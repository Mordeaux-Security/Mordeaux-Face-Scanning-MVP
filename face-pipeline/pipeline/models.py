"""Shared model loading for face detection and recognition."""
from __future__ import annotations
import threading
from typing import Tuple

from insightface.app import FaceAnalysis

from config.settings import settings

_model_lock = threading.Lock()
_shared_app: FaceAnalysis | None = None


def load_insightface_app() -> FaceAnalysis:
    """
    Load buffalo_l model once, shared by detector and embedder.
    
    This consolidates model loading to avoid duplicate instances and reduce memory usage.
    The model is loaded with thread-safe singleton pattern.
    """
    global _shared_app
    if _shared_app is not None:
        return _shared_app
    
    with _model_lock:
        if _shared_app is not None:
            return _shared_app
        
        # Parse providers from settings
        providers = [p.strip() for p in settings.ONNX_PROVIDERS_CSV.split(",") if p.strip()]
        
        # Parse detection size from settings
        try:
            w, h = settings.DET_SIZE.split(",")
            det_size = (int(w), int(h))
        except Exception:
            det_size = (640, 640)
        
        # Create unified InsightFace app
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        _shared_app = app
        return _shared_app

