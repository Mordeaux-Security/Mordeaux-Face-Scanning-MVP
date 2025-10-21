from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import cv2

from config.settings import settings

def laplacian_variance(img_bgr: "np.ndarray") -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def evaluate(img_bgr: "np.ndarray", min_size: int | None = None, min_blur_var: float | None = None) -> Dict:
    if min_size is None:
        min_size = max(80, settings.IMAGE_SIZE)  # ensure crop is ≥112 anyway
    if min_blur_var is None:
        min_blur_var = 120.0

    h, w = img_bgr.shape[:2]
    blur = laplacian_variance(img_bgr)
    ok_size = (h >= min_size) and (w >= min_size)
    ok_blur = blur >= min_blur_var

    passed = bool(ok_size and ok_blur)
    reason = "ok" if passed else f"fail:size={ok_size},blur={ok_blur}({blur:.1f})"

    return {
        "pass": passed,
        "reason": reason,
        "blur": float(blur),
        "size": (int(w), int(h)),
    }