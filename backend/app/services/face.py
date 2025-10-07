import io
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import imagehash
from insightface.app import FaceAnalysis
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List, Dict
import logging
import gc
import psutil

logger = logging.getLogger(__name__)

_face_app = None
_thread_pool = None

def _get_thread_pool() -> ThreadPoolExecutor:
    """Get thread pool for CPU-intensive operations."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face_processing")
    return _thread_pool

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
    """Read image with memory optimization."""
    try:
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # fallback via PIL if needed
            pil = Image.open(io.BytesIO(b)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            # Clean up PIL image immediately
            del pil
        return img
    except Exception as e:
        logger.warning(f"Failed to read image: {str(e)}")
        return None

def enhance_image_for_face_detection(image_bytes: bytes) -> Tuple[bytes, float]:
    """
    Enhance image quality for better face detection, especially for low-resolution images.
    
    Args:
        image_bytes: Original image data
        
    Returns:
        Tuple of (enhanced_image_bytes, scale_factor)
        scale_factor is 1.0 if no enhancement was applied
    """
    try:
        # Load image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = pil_image.size
        
        # Determine if this is a low-resolution image that needs enhancement
        is_low_res = width < 500 or height < 400
        
        if is_low_res:
            logger.info(f"Enhancing low-resolution image ({width}x{height}) for better face detection")
            
            # Apply enhancement techniques
            # 1. Upscale using LANCZOS resampling (better than default)
            scale_factor = max(2.0, 800 / min(width, height))  # Scale to at least 800px on smallest side
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Use LANCZOS for better quality upscaling - use PIL version compatible constant
            try:
                # Try new PIL constant first (PIL 9.0+)
                enhanced = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError:
                # Fall back to old PIL constant (PIL < 9.0)
                enhanced = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 2. Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)  # 20% more contrast
            
            # 3. Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)  # 10% more sharpness
            
            # 4. Apply slight unsharp mask for better detail
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            # Convert back to bytes
            output = io.BytesIO()
            enhanced.save(output, format='JPEG', quality=95, optimize=True)
            return output.getvalue(), scale_factor
        else:
            # For high-resolution images, just ensure good quality
            output = io.BytesIO()
            pil_image.save(output, format='JPEG', quality=95, optimize=True)
            return output.getvalue(), 1.0
            
    except Exception as e:
        logger.warning(f"Failed to enhance image, using original: {str(e)}")
        return image_bytes, 1.0

def _check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:  # If memory usage is above 85%
            logger.warning(f"High memory usage detected: {memory_percent}%")
            gc.collect()  # Force garbage collection
            return False
        return True
    except Exception:
        return True  # If we can't check memory, assume it's OK

def detect_and_embed(image_bytes: bytes, enhancement_scale: float = 1.0, min_size: int = 0) -> List[Dict]:
    """
    Detect faces using multiple scales for better small face detection.
    
    Args:
        image_bytes: Image data (already enhanced)
        enhancement_scale: Scale factor applied during image enhancement (1.0 if no enhancement)
        min_size: Minimum face size in pixels (in original image coordinates)
        
    Returns:
        List of detected faces with metadata (coordinates in original image space)
    """
    # Check memory usage before processing
    _check_memory_usage()
    
    app = _load_app()
    img = _read_image(image_bytes)
    
    if img is None:
        logger.warning("Failed to read image for face detection")
        return []
    
    height, width = img.shape[:2]
    
    # Calculate original image dimensions
    original_height = int(height / enhancement_scale)
    original_width = int(width / enhancement_scale)
    
    all_faces = []
    
    # Try detection at multiple scales
    scales = [1.0, 2.0]  # Original and 2x
    
    for scale in scales:
        try:
            if scale != 1.0:
                # Scale image for detection
                new_width = int(width * scale)
                new_height = int(height * scale)
                scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                scaled_img = img
            
            # Detect faces at this scale
            faces = app.get(scaled_img)
            
            for face in faces:
                if not hasattr(face, "embedding") or face.embedding is None:
                    continue
                
                # Convert bounding box back to original image coordinates
                # First convert from detection scale back to enhanced image scale
                x1, y1, x2, y2 = face.bbox
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                # Then convert from enhanced image scale back to original image coordinates
                x1, y1, x2, y2 = x1/enhancement_scale, y1/enhancement_scale, x2/enhancement_scale, y2/enhancement_scale
                
                # Clamp coordinates to original image boundaries
                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))
                x2 = max(0, min(x2, original_width))
                y2 = max(0, min(y2, original_height))
                
                # Calculate face size in original image coordinates
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Skip very small faces (in original image coordinates)
                if face_width < min_size or face_height < min_size:
                    continue
                
                # Store face data
                face_data = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "embedding": face.embedding.astype(np.float32).tolist(),
                    "det_score": float(getattr(face, "det_score", 0.0)),
                    "scale": scale,
                    "enhancement_scale": enhancement_scale,
                    "face_size": (face_width, face_height)
                }
                
                
                # Check for duplicates (same face detected at multiple scales)
                is_duplicate = False
                for existing_face in all_faces:
                    existing_bbox = existing_face["bbox"]
                    # Calculate overlap
                    overlap_x = max(0, min(x2, existing_bbox[2]) - max(x1, existing_bbox[0]))
                    overlap_y = max(0, min(y2, existing_bbox[3]) - max(y1, existing_bbox[1]))
                    overlap_area = overlap_x * overlap_y
                    
                    # Calculate areas
                    face_area = face_width * face_height
                    existing_area = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
                    
                    # If overlap is more than 50% of either face, consider it a duplicate
                    if overlap_area > 0.5 * min(face_area, existing_area):
                        # Keep the detection with higher confidence
                        if face_data["det_score"] > existing_face["det_score"]:
                            all_faces.remove(existing_face)
                            break
                        else:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    all_faces.append(face_data)
                    
        except Exception as e:
            logger.warning(f"Failed face detection at scale {scale}: {str(e)}")
            continue
    
    # Sort by detection confidence
    all_faces.sort(key=lambda x: x["det_score"], reverse=True)
    
    # Clean up memory
    del img
    gc.collect()
    
    logger.info(f"Multi-scale detection found {len(all_faces)} faces (scales: {scales}, enhancement_scale: {enhancement_scale})")
    return all_faces

def compute_phash(b: bytes) -> str:
    """Compute perceptual hash for image content."""
    pil = Image.open(io.BytesIO(b)).convert("RGB")
    return str(imagehash.phash(pil))

def compute_tolerant_phash(b: bytes) -> tuple:
    """
    Compute multiple perceptual hashes for robust duplicate detection.
    
    Returns:
        Tuple of (phash, dhash, whash, ahash) - different hash types for tolerance
    """
    try:
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        
        # Compute multiple hash types for better duplicate detection
        phash = str(imagehash.phash(pil))           # Perceptual hash (most robust)
        dhash = str(imagehash.dhash(pil))           # Difference hash (good for minor changes)
        whash = str(imagehash.whash(pil))           # Wavelet hash (good for compression artifacts)
        ahash = str(imagehash.average_hash(pil))    # Average hash (simple, fast)
        
        return phash, dhash, whash, ahash
    except Exception as e:
        logger.warning(f"Failed to compute tolerant phash: {str(e)}")
        return "", "", "", ""

def compute_phash_similarity(hash1: tuple, hash2: tuple, threshold: int = 5) -> bool:
    """
    Check if two tolerant hash tuples are similar enough to be considered duplicates.
    
    Args:
        hash1: First hash tuple (phash, dhash, whash, ahash)
        hash2: Second hash tuple (phash, dhash, whash, ahash)
        threshold: Maximum Hamming distance for similarity (default: 5)
        
    Returns:
        True if any hash type is similar enough
    """
    if not hash1 or not hash2 or len(hash1) != 4 or len(hash2) != 4:
        return False
    
    try:
        # Check similarity for each hash type
        hash_types = [hash1, hash2]
        
        for i in range(4):  # Check all 4 hash types
            if not hash_types[0][i] or not hash_types[1][i]:
                continue
                
            # Convert to imagehash objects for comparison
            hash_obj1 = imagehash.hex_to_hash(hash_types[0][i])
            hash_obj2 = imagehash.hex_to_hash(hash_types[1][i])
            
            # Calculate Hamming distance
            distance = hash_obj1 - hash_obj2
            
            if distance <= threshold:
                logger.debug(f"Hash similarity found: type {i}, distance {distance}")
                return True
                
        return False
        
    except Exception as e:
        logger.warning(f"Failed to compare hashes: {str(e)}")
        return False


async def detect_and_embed_async(content: bytes):
    """Async wrapper for face detection and embedding."""
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()
    
    try:
        # Run CPU-intensive face detection in thread pool
        result = await loop.run_in_executor(thread_pool, detect_and_embed, content)
        return result
    except Exception as e:
        logger.error(f"Error in async face detection: {e}")
        raise

async def compute_phash_async(content: bytes):
    """Async wrapper for perceptual hash computation."""
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()
    
    try:
        result = await loop.run_in_executor(thread_pool, compute_phash, content)
        return result
    except Exception as e:
        logger.error(f"Error in async phash computation: {e}")
        raise

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
    
    # Extract bounding box coordinates and ensure they're valid
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are in correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    # Ensure we have a valid bounding box
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
        return image_bytes  # Return original image if invalid bbox
    
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
    
    # Final validation
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        logger.warning(f"Invalid crop coordinates: ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})")
        return image_bytes  # Return original image if invalid crop
    
    # Crop the image
    cropped_image = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Convert back to bytes
    output = io.BytesIO()
    cropped_image.save(output, format='JPEG', quality=95)
    return output.getvalue()


def crop_face_and_create_thumbnail(image_bytes: bytes, face_data: dict, margin: float = 0.2) -> bytes:
    """
    Crop face region from image and create a thumbnail.
    
    Args:
        image_bytes: Original image data
        face_data: Face data dictionary containing bbox
        margin: Margin around face as fraction of face size (default: 0.2 = 20%)
        
    Returns:
        Thumbnail image as bytes
    """
    # Extract bounding box from face data
    bbox = face_data.get("bbox", [])
    if not bbox or len(bbox) != 4:
        logger.warning("Invalid face data: missing or invalid bbox")
        return create_thumbnail(image_bytes)
    
    # Crop the face
    cropped_face = crop_face_from_image(image_bytes, bbox, margin)
    
    # Create thumbnail from cropped face
    return create_thumbnail(cropped_face)

def create_thumbnail(image_bytes: bytes, size: tuple = (150, 150)) -> bytes:
    """
    Create a thumbnail from image bytes.
    
    Args:
        image_bytes: Original image data
        size: Thumbnail size (width, height) - default (150, 150)
        
    Returns:
        Thumbnail image as bytes
    """
    import io
    
    # Load image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Create thumbnail - use PIL version compatible constant
    try:
        # Try new PIL constant first (PIL 9.0+)
        pil_image.thumbnail(size, Image.Resampling.LANCZOS)
    except AttributeError:
        # Fall back to old PIL constant (PIL < 9.0)
        pil_image.thumbnail(size, Image.LANCZOS)
    
    # Convert back to bytes
    output = io.BytesIO()
    pil_image.save(output, format='JPEG', quality=95)
    return output.getvalue()

def get_face_service():
    # simple accessor; in real app you might have a class
    class _FaceSvc:
        detect_and_embed = staticmethod(detect_and_embed)
        detect_and_embed_async = staticmethod(detect_and_embed_async)
        compute_phash = staticmethod(compute_phash)
        compute_phash_async = staticmethod(compute_phash_async)
        crop_face_from_image = staticmethod(crop_face_from_image)
        enhance_image_for_face_detection = staticmethod(enhance_image_for_face_detection)
        crop_face_and_create_thumbnail = staticmethod(crop_face_and_create_thumbnail)
        create_thumbnail = staticmethod(create_thumbnail)
    return _FaceSvc()
