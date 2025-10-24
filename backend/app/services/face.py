import io
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
import imagehash
from insightface.app import FaceAnalysis
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List, Dict
import logging
import gc
import psutil
import threading
import atexit

# Import GPU manager and settings
from .gpu_manager import get_gpu_manager
from ..core.settings import get_settings

# Image safety configuration - set once on import
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

_face_app = None
_thread_pool = None
_model_lock = threading.Lock()  # Thread synchronization for model loading
_early_exit_flag = False  # Tracks whether last detection used early exit

# Face similarity cache for deduplication
_face_similarity_cache = {}  # face_id -> face_metadata
_face_embeddings_cache = []  # List of normalized embeddings
_face_ids_cache = []  # Corresponding face IDs
_similarity_lock = threading.Lock()  # Thread synchronization for similarity cache
_similarity_threshold = 0.7  # Cosine similarity threshold for considering faces similar

def _get_thread_pool() -> ThreadPoolExecutor:
    """Get thread pool for CPU-intensive operations."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face_processing")
    return _thread_pool

def _load_app() -> FaceAnalysis:
    """
    Load face analysis model with thread-safe singleton pattern.
    Prevents race condition where multiple threads load the model simultaneously.
    Now supports GPU acceleration based on configuration.
    """
    global _face_app
    with _model_lock:
        if _face_app is None:
            logger.info("Loading face analysis model (first time)")
            
            # Get GPU configuration
            gpu_manager = get_gpu_manager()
            settings = get_settings()
            
            # Determine if GPU should be used
            use_gpu = gpu_manager.is_operation_gpu_enabled('face_detection')
            ctx_id = gpu_manager.get_gpu_context_id()
            
            home = os.path.expanduser("~/.insightface")
            os.makedirs(home, exist_ok=True)
            app = FaceAnalysis(name=settings.face_model_name, root=home)
            
            # Prepare with GPU context if available
            det_size = settings.face_detection_size
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            
            _face_app = app
            
            if use_gpu:
                device = gpu_manager.get_preferred_device()
                if device:
                    logger.info(f"Face analysis model loaded with GPU acceleration: {device}")
                else:
                    logger.warning("GPU requested but no device available, using CPU")
            else:
                logger.info("Face analysis model loaded with CPU")
        else:
            logger.debug("Using existing face analysis model")
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
    
    This function combines the advanced image enhancement from basic_crawler1.1
    with robust error handling for production use. Now supports GPU acceleration.
    
    Args:
        image_bytes: Original image data
        
    Returns:
        Tuple of (enhanced_image_bytes, scale_factor)
        scale_factor is 1.0 if no enhancement was applied
    """
    gpu_manager = get_gpu_manager()
    
    # Check if GPU enhancement is enabled
    if gpu_manager.is_operation_gpu_enabled('image_enhancement'):
        try:
            return _enhance_image_gpu(image_bytes)
        except Exception as e:
            logger.warning(f"GPU image enhancement failed, falling back to CPU: {e}")
    
    # CPU enhancement (original implementation)
    try:
        t_start = time.perf_counter()
        # Load image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = pil_image.size
        
        # Determine if this is a low-resolution image; avoid heavy upscaling here.
        is_low_res = width < 500 or height < 400
        
        if is_low_res:
            logger.info(f"Enhancing low-resolution image ({width}x{height}) for better face detection (no resize)")
            
            # Apply light enhancement only (no resizing). Multi-scale detection handles scaling.
            enhanced = pil_image
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.15)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            output = io.BytesIO()
            enhanced.save(output, format='JPEG', quality=95, optimize=True)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(f"Image enhancement completed in {elapsed_ms:.1f} ms (scale=1.0, no resize)")
            return output.getvalue(), 1.0
        else:
            # For high-resolution images, just ensure good quality
            output = io.BytesIO()
            pil_image.save(output, format='JPEG', quality=95, optimize=True)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(f"Image enhancement (hi-res passthrough) completed in {elapsed_ms:.1f} ms (scale=1.0)")
            return output.getvalue(), 1.0
            
    except Exception as e:
        logger.warning(f"Failed to enhance image, using original: {str(e)}")
        return image_bytes, 1.0


def _enhance_image_gpu(image_bytes: bytes) -> Tuple[bytes, float]:
    """
    GPU-accelerated image enhancement.
    
    Args:
        image_bytes: Original image data
        
    Returns:
        Tuple of (enhanced_image_bytes, scale_factor)
    """
    try:
        import torch
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import adjust_contrast, adjust_sharpness
        
        t_start = time.perf_counter()
        
        # Load image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = pil_image.size
        
        # Determine if this is a low-resolution image
        is_low_res = width < 500 or height < 400
        
        if is_low_res:
            logger.info(f"GPU enhancing low-resolution image ({width}x{height}) for better face detection")
            
            # Convert to tensor and move to GPU
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            # Get GPU device
            gpu_manager = get_gpu_manager()
            device = gpu_manager.get_preferred_device()
            
            if device and device.backend.value != 'cpu':
                if device.backend == GPUBackend.CUDA:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.ROCM:
                    torch_device = torch.device('cuda')  # ROCm uses CUDA API
                elif device.backend == GPUBackend.MPS:
                    torch_device = torch.device('mps')
                else:
                    torch_device = torch.device('cpu')
            else:
                torch_device = torch.device('cpu')
            
            # Convert to tensor
            tensor = transform(pil_image).unsqueeze(0).to(torch_device)
            
            # Apply GPU-accelerated enhancements
            # Contrast enhancement
            enhanced_tensor = adjust_contrast(tensor, 1.15)
            
            # Sharpness enhancement
            enhanced_tensor = adjust_sharpness(enhanced_tensor, 1.1)
            
            # Convert back to PIL
            enhanced_tensor = enhanced_tensor.squeeze(0).cpu()
            enhanced_pil = transforms.ToPILImage()(enhanced_tensor)
            
            # Apply unsharp mask (CPU operation as it's not easily GPU-accelerated)
            enhanced_pil = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            output = io.BytesIO()
            enhanced_pil.save(output, format='JPEG', quality=95, optimize=True)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(f"GPU image enhancement completed in {elapsed_ms:.1f} ms (scale=1.0)")
            return output.getvalue(), 1.0
        else:
            # For high-resolution images, just ensure good quality
            output = io.BytesIO()
            pil_image.save(output, format='JPEG', quality=95, optimize=True)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(f"GPU image enhancement (hi-res passthrough) completed in {elapsed_ms:.1f} ms (scale=1.0)")
            return output.getvalue(), 1.0
            
    except ImportError:
        logger.warning("PyTorch not available for GPU image enhancement, falling back to CPU")
        raise
    except Exception as e:
        logger.warning(f"GPU image enhancement failed: {e}")
        raise

def _check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    try:
        memory_percent = psutil.virtual_memory().percent
        memory_gb = psutil.virtual_memory().used / (1024**3)
        if memory_percent > 85:  # If memory usage is above 85%
            logger.warning(f"High memory usage detected: {memory_percent}% ({memory_gb:.1f}GB used)")
            gc.collect()  # Force garbage collection
            return False
        else:
            logger.debug(f"Memory usage: {memory_percent}% ({memory_gb:.1f}GB used)")
        return True
    except Exception:
        return True  # If we can't check memory, assume it's OK

def detect_and_embed(image_bytes: bytes, enhancement_scale: float = 1.0, min_size: int = 0) -> List[Dict]:
    """
    Detect faces using single-pass upscaled detection for optimal performance.
    
    This function uses tiered upscaling based on image dimensions to achieve
    better face detection while maintaining performance.
    
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
    t_total_start = time.perf_counter()
    img = _read_image(image_bytes)
    
    if img is None:
        logger.warning("Failed to read image for face detection")
        return []
    
    height, width = img.shape[:2]
    
    # Calculate original image dimensions
    original_height = int(height / enhancement_scale)
    original_width = int(width / enhancement_scale)
    
    # Determine upscale factor based on image dimensions
    if width <= 150 or height <= 150:
        upscale_factor = 4.0
    elif width <= 250 or height <= 250:
        upscale_factor = 3.0
    elif width <= 600 or height <= 600:
        upscale_factor = 2.0
    else:
        upscale_factor = 1.0  # No upscaling for large images
    
    # Upscale image once for optimal face detection
    t_upscale_start = time.perf_counter()
    if upscale_factor != 1.0:
        new_width = int(width * upscale_factor)
        new_height = int(height * upscale_factor)
        upscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        t_upscale_ms = (time.perf_counter() - t_upscale_start) * 1000.0
        logger.info(f"Upscaling {width}x{height} by {upscale_factor}x completed in {t_upscale_ms:.1f} ms")
    else:
        upscaled_img = img
        t_upscale_ms = (time.perf_counter() - t_upscale_start) * 1000.0
        logger.info(f"No upscaling needed for {width}x{height} image (upscale_factor=1.0) in {t_upscale_ms:.1f} ms")
    
    # Apply enhancement once to upscaled image
    t_enhance_start = time.perf_counter()
    try:
        success, encoded = cv2.imencode('.jpg', upscaled_img)
        if success:
            enhanced_bytes, _ = enhance_image_for_face_detection(encoded.tobytes())
            arr = np.frombuffer(enhanced_bytes, dtype=np.uint8)
            enhanced_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if enhanced_img is not None:
                upscaled_img = enhanced_img
    except Exception as _:
        # If enhancement fails, proceed with the upscaled image
        pass
    
    t_enhance_ms = (time.perf_counter() - t_enhance_start) * 1000.0
    logger.info(f"Enhancement of upscaled image completed in {t_enhance_ms:.1f} ms")
    
    # Detect faces once on upscaled+enhanced image
    t_detect_start = time.perf_counter()
    faces = app.get(upscaled_img)
    t_detect_ms = (time.perf_counter() - t_detect_start) * 1000.0
    logger.info(f"Single-pass face detection completed in {t_detect_ms:.1f} ms (found {len(faces) if faces is not None else 0} raw detections)")
    
    all_faces = []
    
    # Process detected faces
    for face in faces:
        if not hasattr(face, "embedding") or face.embedding is None:
            continue
        
        # Convert bounding box back to original image coordinates
        # First convert from upscaled scale back to enhanced image scale
        x1, y1, x2, y2 = face.bbox
        if upscale_factor != 1.0:
            x1, y1, x2, y2 = x1/upscale_factor, y1/upscale_factor, x2/upscale_factor, y2/upscale_factor
        
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
            "scale": upscale_factor,
            "enhancement_scale": enhancement_scale,
            "face_size": (face_width, face_height)
        }
        
        # Since we're using single-pass detection, no duplicate checking needed
        all_faces.append(face_data)
    
    # Log total processing time
    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
    logger.info(f"Single-pass face detection completed in {t_total_ms:.1f} ms total (found {len(all_faces)} faces)")
    
    return all_faces

def consume_early_exit_flag() -> bool:
    """Return and reset the early-exit flag set during the last detection call."""
    global _early_exit_flag
    value = _early_exit_flag
    _early_exit_flag = False
    return value

def _normalize_embedding(embedding: List[float]) -> np.ndarray:
    """
    Normalize face embedding to unit vector for cosine similarity.
    
    Args:
        embedding: Raw face embedding from InsightFace
        
    Returns:
        Normalized embedding as numpy array
    """
    embedding_array = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding_array)
    if norm == 0:
        raise ValueError("Zero-norm embedding detected")
    return embedding_array / norm

def _cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two normalized embeddings.
    
    Args:
        embedding1: First normalized embedding
        embedding2: Second normalized embedding
        
    Returns:
        Cosine similarity score (0.0-1.0, higher = more similar)
    """
    return float(np.dot(embedding1, embedding2))

def _find_most_similar_face(embedding: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Find the most similar face in the cache.
    
    Args:
        embedding: Normalized face embedding to compare
        
    Returns:
        Tuple of (face_id, similarity_score) or (None, 0.0) if no similar face found
    """
    global _face_embeddings_cache, _face_ids_cache, _similarity_threshold
    
    if not _face_embeddings_cache:
        return None, 0.0
    
    # Calculate similarities with all cached faces
    similarities = np.dot(_face_embeddings_cache, embedding)
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]
    
    if max_similarity >= _similarity_threshold:
        face_id = _face_ids_cache[max_similarity_idx]
        return face_id, float(max_similarity)
    
    return None, 0.0

def _generate_face_id(face_metadata: Dict) -> str:
    """
    Generate a unique ID for a face based on its metadata.
    
    Args:
        face_metadata: Face metadata including bbox, det_score, etc.
        
    Returns:
        Unique face ID string
    """
    # Use a combination of bbox coordinates and detection score for ID
    bbox = face_metadata.get('bbox', [0, 0, 0, 0])
    det_score = face_metadata.get('det_score', 0.0)
    
    # Create a hash-based ID from key face characteristics
    import hashlib
    face_data = f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},{det_score:.3f}"
    return hashlib.md5(face_data.encode()).hexdigest()[:16]

def check_face_similarity(face_embedding: List[float], face_metadata: Dict) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a face is similar to any previously seen faces.
    
    Args:
        face_embedding: Raw face embedding from InsightFace
        face_metadata: Additional metadata about the face (bbox, det_score, etc.)
        
    Returns:
        Tuple of (is_similar, similarity_score, similar_face_id)
    """
    global _face_similarity_cache, _face_embeddings_cache, _face_ids_cache, _similarity_lock
    
    try:
        # Normalize the embedding
        normalized_embedding = _normalize_embedding(face_embedding)
        
        # Check for similar faces in cache
        with _similarity_lock:
            similar_face_id, similarity_score = _find_most_similar_face(normalized_embedding)
        
        if similar_face_id:
            # Found a similar face
            logger.debug(f"Found similar face: {similar_face_id} (similarity: {similarity_score:.3f})")
            return True, similarity_score, similar_face_id
        else:
            # No similar face found, add to cache
            face_id = _generate_face_id(face_metadata)
            
            with _similarity_lock:
                _face_similarity_cache[face_id] = face_metadata.copy()
                _face_embeddings_cache.append(normalized_embedding)
                _face_ids_cache.append(face_id)
            
            logger.debug(f"New unique face added to similarity cache: {face_id}")
            return False, 0.0, None
            
    except Exception as e:
        logger.error(f"Error checking face similarity: {e}")
        return False, 0.0, None

def get_face_similarity_stats() -> Dict:
    """
    Get statistics about the face similarity cache.
    
    Returns:
        Dictionary with cache statistics
    """
    global _face_similarity_cache, _similarity_threshold
    
    with _similarity_lock:
        return {
            'total_faces': len(_face_similarity_cache),
            'similarity_threshold': _similarity_threshold,
            'cache_size_mb': len(str(_face_similarity_cache)) / (1024 * 1024)
        }

def clear_face_similarity_cache():
    """Clear the face similarity cache."""
    global _face_similarity_cache, _face_embeddings_cache, _face_ids_cache
    
    with _similarity_lock:
        _face_similarity_cache.clear()
        _face_embeddings_cache.clear()
        _face_ids_cache.clear()
    logger.info("Face similarity cache cleared")

def set_similarity_threshold(threshold: float):
    """Set the similarity threshold for face comparison."""
    global _similarity_threshold
    _similarity_threshold = max(0.0, min(1.0, threshold))
    logger.info(f"Face similarity threshold set to {_similarity_threshold}")

def compute_phash(b: bytes) -> str:
    """Compute perceptual hash for image content."""
    pil = Image.open(io.BytesIO(b)).convert("RGB")
    return str(imagehash.phash(pil))




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


def detect_and_embed_batch_gpu(images: List[bytes], batch_size: int = 32) -> List[List[Dict]]:
    """
    GPU-accelerated batch face detection and embedding.
    
    Args:
        images: List of image bytes
        batch_size: Number of images to process in each batch
        
    Returns:
        List of face detection results for each image
    """
    gpu_manager = get_gpu_manager()
    
    if not gpu_manager.is_operation_gpu_enabled('face_detection'):
        # Fallback to CPU batch processing
        logger.info("GPU not enabled for face detection, using CPU batch processing")
        return [detect_and_embed(img) for img in images]
    
    try:
        app = _load_app()
        results = []
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = []
            
            # Convert batch to numpy arrays
            batch_arrays = []
            for img_bytes in batch_images:
                img_array = _read_image(img_bytes)
                if img_array is not None:
                    batch_arrays.append(img_array)
                else:
                    batch_arrays.append(None)
            
            # Process batch on GPU
            for img_array in batch_arrays:
                if img_array is not None:
                    try:
                        faces = app.get(img_array)
                        face_data = []
                        
                        for face in faces:
                            if hasattr(face, "embedding") and face.embedding is not None:
                                face_info = {
                                    "bbox": face.bbox.tolist(),
                                    "embedding": face.embedding.astype(np.float32).tolist(),
                                    "det_score": float(getattr(face, "det_score", 0.0)),
                                    "scale": 1.0,
                                    "enhancement_scale": 1.0,
                                    "face_size": (face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
                                }
                                face_data.append(face_info)
                        
                        batch_results.append(face_data)
                    except Exception as e:
                        logger.warning(f"Error processing image in GPU batch: {e}")
                        batch_results.append([])
                else:
                    batch_results.append([])
            
            results.extend(batch_results)
            
            # GPU memory cleanup
            if hasattr(app, 'clear_memory'):
                app.clear_memory()
        
        logger.info(f"GPU batch processing completed: {len(images)} images, {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in GPU batch face detection: {e}")
        # Fallback to CPU processing
        logger.info("Falling back to CPU batch processing")
        return [detect_and_embed(img) for img in images]


async def detect_and_embed_batch_async(images: List[bytes], batch_size: int = 32) -> List[List[Dict]]:
    """
    Async wrapper for GPU batch face detection.
    
    Args:
        images: List of image bytes
        batch_size: Number of images to process in each batch
        
    Returns:
        List of face detection results for each image
    """
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()
    
    try:
        result = await loop.run_in_executor(
            thread_pool, detect_and_embed_batch_gpu, images, batch_size
        )
        return result
    except Exception as e:
        logger.error(f"Error in async batch face detection: {e}")
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
    """Get face service instance with all available methods."""
    class _FaceSvc:
        detect_and_embed = staticmethod(detect_and_embed)
        detect_and_embed_async = staticmethod(detect_and_embed_async)
        detect_and_embed_batch_gpu = staticmethod(detect_and_embed_batch_gpu)
        detect_and_embed_batch_async = staticmethod(detect_and_embed_batch_async)
        compute_phash = staticmethod(compute_phash)
        compute_phash_async = staticmethod(compute_phash_async)
        crop_face_from_image = staticmethod(crop_face_from_image)
        enhance_image_for_face_detection = staticmethod(enhance_image_for_face_detection)
        crop_face_and_create_thumbnail = staticmethod(crop_face_and_create_thumbnail)
        create_thumbnail = staticmethod(create_thumbnail)
        consume_early_exit_flag = staticmethod(consume_early_exit_flag)
    return _FaceSvc()


def close_face_service():
    """
    Clean shutdown of face service resources.
    
    This function:
    1. Shuts down thread pools with wait=True
    2. Clears model references to free memory
    3. Resets global variables to None
    4. Forces garbage collection
    """
    global _face_app, _thread_pool, _model_lock, _early_exit_flag
    
    logger.info("Closing face service resources...")
    
    try:
        # Shutdown thread pool if it exists
        if _thread_pool is not None:
            logger.info("Shutting down face service thread pool...")
            _thread_pool.shutdown(wait=True)
            logger.info("Face service thread pool shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down face service thread pool: {e}")
    
    try:
        # Clear model reference
        if _face_app is not None:
            logger.info("Clearing face analysis model reference...")
            # InsightFace models don't have explicit cleanup, but we can clear the reference
            del _face_app
            logger.info("Face analysis model reference cleared")
    except Exception as e:
        logger.warning(f"Error clearing face analysis model: {e}")
    
    try:
        # Reset global variables
        _face_app = None
        _thread_pool = None
        _early_exit_flag = False
        logger.info("Face service global variables reset")
    except Exception as e:
        logger.warning(f"Error resetting face service globals: {e}")
    
    try:
        # Force garbage collection to free memory
        gc.collect()
        logger.info("Face service cleanup complete - garbage collection triggered")
    except Exception as e:
        logger.warning(f"Error during face service garbage collection: {e}")


# Register cleanup function with atexit
atexit.register(close_face_service)
