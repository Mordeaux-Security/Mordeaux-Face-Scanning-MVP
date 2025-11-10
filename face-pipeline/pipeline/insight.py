import os
import threading
from insightface.app import FaceAnalysis

_lock = threading.Lock()
_app = None

def get_app() -> FaceAnalysis:
    """
    Singleton InsightFace FaceAnalysis pack.
    Uses 'buffalo_l' (detector + recognition) by default.
    """
    global _app
    if _app is not None:
        return _app
    with _lock:
        if _app is not None:
            return _app
        pack_name = os.getenv("INSIGHTFACE_PACK", "buffalo_l")
        det_size = tuple(map(int, os.getenv("DETECT_SIZE", "640,640").split(",")))
        # Let InsightFace choose providers; if you need CPU only:
        providers = ["CPUExecutionProvider"]
        _app = FaceAnalysis(name=pack_name, providers=providers)
        _app.prepare(ctx_id=0, det_size=det_size)  # ctx_id has no effect with CPU provider
        return _app

