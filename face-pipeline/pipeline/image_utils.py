import base64
import re
import numpy as np
import cv2

_DATAURL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", re.IGNORECASE)

def decode_image_b64(data: str):
    """
    Accepts either a data URL (data:image/jpeg;base64,...) or a raw base64 string.
    Returns BGR np.ndarray (OpenCV) or None if decode fails.
    """
    try:
        m = _DATAURL_RE.match(data.strip())
        if m:
            b64 = m.group("b64")
        else:
            b64 = data.strip()
        buf = base64.b64decode(b64, validate=False)
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None
