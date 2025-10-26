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

# Import GPU client for Windows worker integration
from .gpu_client import get_gpu_client, close_gpu_client
from ..core.settings import get_settings

# Image safety configuration - set once on import
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

_face_app = None
_thread_pool = None
_model_lock = threading.Lock()  # Thread synchronization for model loading
_early_exit_flag = False  # Tracks whether last detection used early exit


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
    """
    global _face_app
    with _model_lock:
        if _face_app is None:
            logger.info("Loading face analysis model (first time)")
            home = os.path.expanduser("~/.insightface")
            os.makedirs(home, exist_ok=True)
            app = FaceAnalysis(name="buffalo_l", root=home)
            # CPU default (onnxruntime)
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _face_app = app
            logger.info("Face analysis model loaded successfully")
        else:
            logger.debug("Using existing face analysis model")
        return _face_app

async def _try_gpu_worker_batch(
    image_bytes_list: List[bytes],
    min_face_quality: float = 0.5,
    require_face: bool = True,
    crop_faces: bool = True,
    face_margin: float = 0.2
) -> Optional[List[List[Dict]]]:
    """
    Try to process images using GPU worker.

    Returns:
        List of face detection results for each image, or None if GPU worker unavailable
    """
    try:
        settings = get_settings()
        if not settings.gpu_worker_enabled:
            logger.info("[CPU-FALLBACK] GPU worker disabled in settings")
            return None

        logger.info(f"[GPU-WORKER-ATTEMPT] Processing {len(image_bytes_list)} images with GPU worker")
        start_time = time.time()
        
        gpu_client = await get_gpu_client()
        results = await gpu_client.detect_faces_batch_async(
            image_bytes_list, min_face_quality, require_face, crop_faces, face_margin
        )

        processing_time = time.time() - start_time
        
        # Convert GPU client results to our format
        converted_results = []
        total_faces = 0
        for image_results in results:
            face_detections = []
            for detection in image_results:
                face_detection = {
                    'bbox': detection.bbox,
                    'landmarks': detection.landmarks,
                    'embedding': detection.embedding,
                    'quality': detection.quality,
                    'age': detection.age,
                    'gender': detection.gender
                }
                face_detections.append(face_detection)
                total_faces += 1
            converted_results.append(face_detections)

        logger.info(f"[GPU-WORKER-SUCCESS] Processed {len(image_bytes_list)} images in {processing_time:.3f}s, found {total_faces} faces")
        return converted_results

    except Exception as e:
        logger.warning(f"[GPU-WORKER-FAIL] GPU worker processing failed: {e}")
        return None

def _try_gpu_worker_sync(
    image_bytes_list: List[bytes],
    min_face_quality: float = 0.5,
    require_face: bool = True,
    crop_faces: bool = True,
    face_margin: float = 0.2
) -> Optional[List[List[Dict]]]:
    """
    Synchronous version of GPU worker processing to avoid event loop conflicts.
    
    Returns:
        List of face detection results for each image, or None if GPU worker unavailable
    """
    try:
        settings = get_settings()
        if not settings.gpu_worker_enabled:
            logger.info("[CPU-FALLBACK] GPU worker disabled in settings")
            return None

        logger.info(f"[GPU-WORKER-ATTEMPT] Processing {len(image_bytes_list)} images with GPU worker (sync)")
        start_time = time.time()
        
        # Use synchronous HTTP client for GPU worker communication
        import httpx
        
        # Encode images for GPU worker
        encoded_images = []
        for i, image_bytes in enumerate(image_bytes_list):
            import base64
            encoded_data = base64.b64encode(image_bytes).decode('utf-8')
            encoded_images.append({
                "data": encoded_data,
                "image_id": f"sync_{i}"
            })
        
        # Prepare request payload
        payload = {
            "images": encoded_images,
            "min_face_quality": min_face_quality,
            "require_face": require_face,
            "crop_faces": crop_faces,
            "face_margin": face_margin
        }
        
        # Make synchronous HTTP request to GPU worker
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.gpu_worker_url}/detect_faces_batch",
                json=payload
            )
            
            if response.status_code == 200:
                gpu_response = response.json()
                processing_time = time.time() - start_time
                
                # Extract results from GPU worker response
                gpu_results = gpu_response.get('results', [])
                
                # Convert GPU worker results to our format
                converted_results = []
                total_faces = 0
                for image_results in gpu_results:
                    face_detections = []
                    for detection in image_results:
                        face_detection = {
                            'bbox': detection['bbox'],
                            'landmarks': detection['landmarks'],
                            'embedding': detection['embedding'],
                            'quality': detection['quality'],
                            'age': detection.get('age'),
                            'gender': detection.get('gender')
                        }
                        face_detections.append(face_detection)
                        total_faces += 1
                    converted_results.append(face_detections)

                logger.info(f"[GPU-WORKER-SUCCESS] Processed {len(image_bytes_list)} images in {processing_time:.3f}s, found {total_faces} faces")
                return converted_results
            else:
                logger.warning(f"[GPU-WORKER-FAIL] GPU worker returned status {response.status_code}")
                return None

    except Exception as e:
        logger.warning(f"[GPU-WORKER-FAIL] GPU worker processing failed: {e}")
        return None

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
    with robust error handling for production use.
    
    Args:
        image_bytes: Original image data
        
    Returns:
        Tuple of (enhanced_image_bytes, scale_factor)
        scale_factor is 1.0 if no enhancement was applied
    """
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

def detect_and_embed_batch_optimized(image_bytes_list: List[bytes], 
                                   min_face_quality: float = 0.5,
                                   require_face: bool = True,
                                   crop_faces: bool = True,
                                   face_margin: float = 0.2) -> List[List[Dict]]:
    """
    Optimized batch face detection that sends multiple images to GPU worker at once.
    
    Args:
        image_bytes_list: List of image bytes
        min_face_quality: Minimum face quality threshold
        require_face: Whether to require at least one face
        crop_faces: Whether to crop face regions
        face_margin: Margin around face as fraction of face size
        
    Returns:
        List of face detection results for each image
    """
    if not image_bytes_list:
        return []
    
    # Try GPU worker batch processing
    try:
        settings = get_settings()
        if settings.gpu_worker_enabled:
            logger.info(f"[GPU-WORKER-BATCH] Processing {len(image_bytes_list)} images with GPU worker")
            
            # Use synchronous GPU worker call for batch processing
            results = _try_gpu_worker_sync(
                image_bytes_list,
                min_face_quality=min_face_quality,
                require_face=require_face,
                crop_faces=crop_faces,
                face_margin=face_margin
            )
            
            if results is not None:
                logger.info(f"[GPU-WORKER-BATCH-SUCCESS] Processed {len(image_bytes_list)} images via GPU worker")
                return results
    except Exception as e:
        logger.warning(f"[GPU-WORKER-BATCH-FAIL] GPU worker batch processing failed: {e}")
    
    # Fallback to individual processing
    logger.info(f"[CPU-FALLBACK-BATCH] Processing {len(image_bytes_list)} images with CPU fallback")
    results = []
    for image_bytes in image_bytes_list:
        try:
            faces = detect_and_embed(image_bytes)
            results.append(faces)
        except Exception as e:
            logger.error(f"Error processing image in batch: {e}")
            results.append([])
    
    return results


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
        
    Note: This function performs CPU-only face detection. For GPU acceleration,
    use detect_and_embed_batch_optimized() with multiple images.
    """
    logger.debug("[CPU-FALLBACK] Processing with CPU")
    
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


def detect_faces_batch(
    image_bytes_list: List[bytes], 
    min_face_quality: float = 0.5,
    require_face: bool = True,
    crop_faces: bool = True,
    face_margin: float = 0.2
) -> List[List[Dict]]:
    """
    Detect faces in a batch of images with GPU acceleration fallback.
    
    Args:
        image_bytes_list: List of image bytes
        min_face_quality: Minimum face quality threshold
        require_face: Whether to require at least one face
        crop_faces: Whether to crop face regions
        face_margin: Margin around face as fraction of face size
        
    Returns:
        List of face detection results for each image
    """
    if not image_bytes_list:
        return []
    
    logger.info(f"[FACE-DETECTION] Processing {len(image_bytes_list)} images")
    
    # Try GPU worker first if enabled
    try:
        settings = get_settings()
        if settings.gpu_worker_enabled:
            logger.info(f"[FACE-DETECTION] GPU worker enabled, attempting GPU processing")
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(_try_gpu_worker_batch(
                            image_bytes_list, min_face_quality, require_face, crop_faces, face_margin
                        ))
                    )
                    results = future.result(timeout=30)  # 30 second timeout
                    if results is not None:
                        images_with_faces = sum(1 for faces in results if len(faces) > 0)
                        logger.info(f"[FACE-DETECTION] GPU worker returned results: {images_with_faces}/{len(results)} images with faces")
                        return results
            except RuntimeError:
                # No event loop running, check if we have an event loop set
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        # Loop is closed, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    results = loop.run_until_complete(
                        _try_gpu_worker_batch(
                            image_bytes_list, min_face_quality, require_face, crop_faces, face_margin
                        )
                    )
                    if results is not None:
                        images_with_faces = sum(1 for faces in results if len(faces) > 0)
                        logger.info(f"[FACE-DETECTION] GPU worker returned results: {images_with_faces}/{len(results)} images with faces")
                        return results
                except Exception as e:
                    logger.warning(f"Event loop error: {e}")
                    # Fall through to CPU processing
        else:
            logger.warning(f"[FACE-DETECTION] GPU worker disabled, using CPU fallback")
    except Exception as e:
        logger.warning(f"[FACE-DETECTION] GPU worker attempt failed: {e}")
    
    # Fallback to CPU processing
    logger.info(f"[FACE-DETECTION] Using CPU fallback for {len(image_bytes_list)} images")
    
    # Load model
    app = _load_app()
    logger.info(f"[CPU-FALLBACK] Face detection model loaded")
    
    # Process images in batches to manage memory
    batch_size = 8  # Smaller batches for CPU processing
    results = []
    
    for i in range(0, len(image_bytes_list), batch_size):
        batch = image_bytes_list[i:i + batch_size]
        batch_results = []
        
        logger.info(f"[CPU-FALLBACK] Processing batch {i//batch_size + 1} ({len(batch)} images)")
        
        for j, image_bytes in enumerate(batch):
            try:
                # Read image
                image = _read_image(image_bytes)
                if image is None:
                    logger.warning(f"[CPU-FALLBACK] Failed to read image {i+j}")
                    batch_results.append([])
                    continue
                
                logger.debug(f"[CPU-FALLBACK] Image {i+j} shape: {image.shape}")
                
                # Detect faces
                faces = app.get(image)
                logger.debug(f"[CPU-FALLBACK] Image {i+j} detected {len(faces)} faces")
                
                # Convert to our format
                face_detections = []
                for face in faces:
                    if face.det_score >= min_face_quality:
                        face_detection = {
                            'bbox': face.bbox.tolist(),
                            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else [],
                            'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None,
                            'quality': float(face.det_score),
                            'age': int(face.age) if hasattr(face, 'age') else None,
                            'gender': str(face.gender) if hasattr(face, 'gender') else None
                        }
                        face_detections.append(face_detection)
                
                batch_results.append(face_detections)
                
            except Exception as e:
                logger.error(f"[CPU-FALLBACK] Error detecting faces in image {i+j}: {e}", exc_info=True)
                batch_results.append([])
        
        results.extend(batch_results)
        logger.info(f"[CPU-FALLBACK] Batch {i//batch_size + 1} complete: {sum(len(r) for r in batch_results)} faces found")
    
    logger.info(f"[CPU-FALLBACK] Total faces found: {sum(len(r) for r in results)}")
    return results

async def detect_faces_batch_async(
    image_bytes_list: List[bytes], 
    min_face_quality: float = 0.5,
    require_face: bool = True,
    crop_faces: bool = True,
    face_margin: float = 0.2
) -> List[List[Dict]]:
    """
    Async version of detect_faces_batch with GPU worker support.
    
    Args:
        image_bytes_list: List of image bytes
        min_face_quality: Minimum face quality threshold
        require_face: Whether to require at least one face
        crop_faces: Whether to crop face regions
        face_margin: Margin around face as fraction of face size
        
    Returns:
        List of face detection results for each image
    """
    if not image_bytes_list:
        return []
    
    # Try GPU worker first if enabled
    try:
        settings = get_settings()
        if settings.gpu_worker_enabled:
            results = await _try_gpu_worker_batch(
                image_bytes_list, min_face_quality, require_face, crop_faces, face_margin
            )
            if results is not None:
                return results
    except Exception as e:
        logger.warning(f"GPU worker attempt failed: {e}")
    
    # Fallback to CPU processing in thread pool
    logger.info(f"Processing {len(image_bytes_list)} images with CPU fallback (async)")
    
    loop = asyncio.get_event_loop()
    thread_pool = _get_thread_pool()
    
    # Run CPU processing in thread pool
    results = await loop.run_in_executor(
        thread_pool,
        detect_faces_batch,
        image_bytes_list,
        min_face_quality,
        require_face,
        crop_faces,
        face_margin
    )
    
    return results

# Register cleanup function with atexit
atexit.register(close_face_service)
