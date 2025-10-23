import logging
import time
from contextlib import contextmanager
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
from PIL import Image
import imagehash
from io import BytesIO

"""
Pipeline Utility Functions

TODO: Implement common utility functions used across pipeline
TODO: Add image preprocessing helpers
TODO: Add format conversion utilities
TODO: Add validation helpers
TODO: Add timing/profiling decorators
"""

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Core Utility Functions
# ============================================================================

def l2_normalize(vec: "np.ndarray") -> "np.ndarray":
    """
    L2-normalize a vector to unit length.

    Converts a vector to unit length by dividing by its L2 norm.
    Essential for embedding normalization before vector search.

    Args:
        vec: Input vector (any length)

    Returns:
        L2-normalized vector with unit length (norm = 1.0)

    Example:
        >>> vec = np.array([3.0, 4.0])
        >>> normalized = l2_normalize(vec)
        >>> np.linalg.norm(normalized)  # Should be 1.0
    """
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def compute_phash(pil_img: Image.Image) -> str:
    # 64-bit pHash -> 16 hex chars
    return imagehash.phash(pil_img, hash_size=8).__str__()


def hamming_distance_hex(a: str, b: str) -> int:
    # length-safe: pad shorter
    a = a.strip().lower()
    b = b.strip().lower()
    maxlen = max(len(a), len(b))
    a = a.zfill(maxlen); b = b.zfill(maxlen)
    x = int(a, 16) ^ int(b, 16)
    # count bits
    return x.bit_count()


def phash_prefix(hex_str: str, bits: int = 16) -> str:
    # first 16 bits = first 4 hex chars
    n_hex = bits // 4
    return hex_str[:n_hex]


def bytes_to_numpy(image_bytes: bytes) -> "np.ndarray":
    """
    Convert image bytes to numpy array.

    TODO: Implement conversion using PIL or cv2
    TODO: Handle different image formats
    TODO: Add error handling
    """
    pass


def numpy_to_bytes(
    image: "np.ndarray", format: str = "JPEG", quality: int = 95
) -> bytes:
    """
    Convert numpy array to image bytes.

    TODO: Implement conversion
    TODO: Support different formats (JPEG, PNG, WebP)
    """
    pass


def resize_image(
    image: "np.ndarray", max_size: Tuple[int, int]
) -> "np.ndarray":
    """
    Resize image maintaining aspect ratio.

    TODO: Implement smart resizing
    TODO: Support different interpolation methods
    """
    pass


def crop_with_margin(
    image: "np.ndarray",
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2
) -> "np.ndarray":
    """
    Crop region from image with margin.

    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        margin: Margin as fraction of bbox size

    TODO: Implement cropping with boundary checks
    """
    pass


def compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """
    Compute Intersection over Union for two bounding boxes.

    TODO: Implement IoU calculation
    """
    pass


def validate_image_bytes(image_bytes: bytes) -> bool:
    """
    Validate that bytes represent a valid image.

    TODO: Check file signature/magic bytes
    TODO: Try to decode image
    TODO: Check minimum size requirements
    """
    pass


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate unique identifier for faces/images.

    TODO: Implement UUID or timestamp-based ID generation
    """
    pass


# ============================================================================
# Timing & Observability
# ============================================================================

@contextmanager
def timer(section: str):
    """
    Context manager for timing code sections with logging.

    Measures elapsed time in milliseconds and logs the duration.
    Useful for profiling pipeline steps and identifying bottlenecks.

    Args:
        section: Name of the section being timed (for logging)

    Yields:
        None

    Example:
        >>> with timer("face_detection"):
        ...     faces = detect_faces(image)
        # Logs: "face_detection completed in 45.23ms"

    TODO - DEV2 Enhancement:
    - Add support for structured logging (JSON format)
    - Add metric export to Prometheus/StatsD
    - Add configurable log levels (debug for dev, info for prod)
    - Add optional return value for elapsed time
    - Add exception handling to log failures

    Usage in pipeline:
        with timer("download_image"):
            image_bytes = storage.get_bytes(bucket, key)

        with timer("detect_faces"):
            faces = detector.detect_faces(image_np)
    """
    start_time = time.perf_counter()

    # TODO: Add structured logging support
    # TODO: Add metric export (Prometheus, StatsD)
    # TODO: Make logging configurable (debug/info levels)

    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"⏱️  {section} completed in {elapsed_ms:.2f}ms")

        # TODO: Export to metrics backend
        # TODO: Track in application metrics (avg, p50, p95, p99)


class Timer:
    """
    Context manager for timing operations (legacy class-based approach).

    Note: Prefer using the `timer()` function above for simpler usage.
    This class is kept for backward compatibility.
    """

    def __enter__(self):
        # TODO: Record start time
        pass

    def __exit__(self, *args):
        # TODO: Calculate elapsed time
        # TODO: Log timing information
        pass


def profile_async(func):
    """
    Decorator to profile async functions.

    TODO: Implement async profiling decorator
    TODO: Log execution time and memory usage
    """
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
