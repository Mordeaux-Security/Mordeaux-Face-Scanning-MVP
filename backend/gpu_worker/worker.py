"""
Windows GPU Worker Service

FastAPI service that runs natively on Windows with DirectML support for GPU-accelerated
face detection and embedding operations. Provides REST API endpoints for batch processing.
"""

import asyncio
import base64
import io
import logging
import os
import sys
import threading
import time
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model management
_face_app = None
_model_lock = threading.Lock()
_initialization_lock = threading.Lock()
_is_initialized = False

# Thread pool for CPU operations
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gpu_worker")

# Resource management (optional, won't block startup if it fails)
try:
    import psutil
    PSUTIL_AVAILABLE = True
    
    _last_inference_time = time.time()
    _total_inferences = 0
    _memory_cleanup_threshold = 100
    _max_memory_mb = 2048
    _warmup_interval = 300
    _cleanup_task = None
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource management features disabled")
    
    # Set defaults for when psutil is not available
    _last_inference_time = time.time()
    _total_inferences = 0
    _memory_cleanup_threshold = 100
    _max_memory_mb = 2048
    _warmup_interval = 300
    _cleanup_task = None

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    directml_available: bool
    model_loaded: bool
    uptime_seconds: float

class ImageData(BaseModel):
    data: str = Field(..., description="Base64-encoded image data")
    image_id: Optional[str] = Field(None, description="Optional image identifier")

class BatchRequest(BaseModel):
    images: List[ImageData] = Field(..., description="List of images to process")
    min_face_quality: float = Field(default=0.5, description="Minimum face quality threshold")
    require_face: bool = Field(default=True, description="Whether to require at least one face")
    crop_faces: bool = Field(default=True, description="Whether to crop face regions")
    face_margin: float = Field(default=0.2, description="Margin around face as fraction of face size")

class FaceDetection(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    landmarks: List[List[float]] = Field(..., description="Facial landmarks")
    embedding: Optional[List[float]] = Field(None, description="Face embedding vector")
    quality: float = Field(..., description="Detection quality score")
    age: Optional[int] = Field(None, description="Estimated age")
    gender: Optional[str] = Field(None, description="Estimated gender")

class BatchResponse(BaseModel):
    results: List[List[FaceDetection]] = Field(..., description="Face detection results for each image")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    gpu_used: bool = Field(..., description="Whether GPU was used for processing")

class EnhanceImageRequest(BaseModel):
    image_data: str = Field(..., description="Base64-encoded image data")
    scale_factor: float = Field(default=2.0, description="Upscaling factor")

class EnhanceImageResponse(BaseModel):
    enhanced_data: str = Field(..., description="Base64-encoded enhanced image")
    original_size: Tuple[int, int] = Field(..., description="Original image dimensions")
    enhanced_size: Tuple[int, int] = Field(..., description="Enhanced image dimensions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

# FastAPI app
app = FastAPI(
    title="GPU Worker Service",
    description="Windows GPU worker for face detection and embedding",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time for uptime calculation
_startup_time = time.time()

def _check_directml_availability() -> bool:
    """Check if DirectML execution provider is available."""
    try:
        available_providers = ort.get_available_providers()
        return 'DmlExecutionProvider' in available_providers
    except Exception as e:
        logger.error(f"Error checking DirectML availability: {e}")
        return False

def _check_actual_gpu_usage() -> bool:
    """Check if GPU is actually being used by examining model providers."""
    try:
        if _face_app is None:
            return False
        
        # Check if any model is using DirectML
        if hasattr(_face_app, 'models'):
            for model_name, model in _face_app.models.items():
                if hasattr(model, 'session'):
                    providers = model.session.get_providers()
                    # If the first provider is DirectML, GPU is being used
                    if providers and providers[0] == 'DmlExecutionProvider':
                        return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking actual GPU usage: {e}")
        return False

def _patch_insightface_for_directml():
    """Patch InsightFace to use DirectML execution provider."""
    try:
        from insightface.model_zoo import model_zoo
        
        # Get available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        # Prefer DirectML, fallback to CPU
        if 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            logger.info("Using DirectML execution provider")
        else:
            providers = ['CPUExecutionProvider']
            logger.warning("DirectML not available, using CPU")
        
        # Monkey-patch model loading
        original_get_model = model_zoo.get_model
        
        def patched_get_model(name, **kwargs):
            model = original_get_model(name, **kwargs)
            
            # Recreate ONNX session with preferred providers
            if hasattr(model, 'session') and model.session is not None:
                try:
                    model_path = getattr(model, 'model_file', None)
                    if model_path:
                        # Filter to available providers
                        filtered_providers = [p for p in providers if p in available_providers]
                        
                        if not filtered_providers:
                            logger.warning("No preferred providers available, using CPU")
                            filtered_providers = ['CPUExecutionProvider']
                        
                        logger.info(f"Creating ONNX session with providers: {filtered_providers}")
                        
                        # Create session options
                        sess_options = ort.SessionOptions()
                        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        
                        # Recreate session
                        model.session = ort.InferenceSession(
                            model_path,
                            sess_options=sess_options,
                            providers=filtered_providers
                        )
                        
                        # Log actual providers used
                        actual_providers = model.session.get_providers()
                        logger.info(f"ONNX session created with providers: {actual_providers}")
                        
                        if actual_providers[0] != 'CPUExecutionProvider':
                            logger.info(f"[GPU] GPU acceleration enabled: {actual_providers[0]}")
                        else:
                            logger.warning("[WARNING] Using CPU execution")
                            
                except Exception as e:
                    logger.error(f"Failed to patch ONNX session: {e}")
                    logger.warning("Continuing with default execution")
            
            return model
        
        # Apply the patch
        model_zoo.get_model = patched_get_model
        logger.info("InsightFace patched for DirectML execution")
        
    except Exception as e:
        logger.error(f"Failed to patch InsightFace: {e}")

def _cleanup_gpu_resources():
    """Force cleanup of GPU resources to prevent memory leaks."""
    if not PSUTIL_AVAILABLE:
        logger.debug("psutil not available, skipping cleanup")
        return None
    
    import gc
    
    try:
        logger.info("Performing GPU resource cleanup...")
        gc.collect()
        
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after cleanup: {memory_mb:.1f}MB")
        return memory_mb
        
    except Exception as e:
        logger.error(f"Error during GPU cleanup: {e}", exc_info=True)
        return None

def _check_memory_pressure() -> bool:
    """Check if memory pressure requires cleanup."""
    if not PSUTIL_AVAILABLE:
        return False
    
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > _max_memory_mb:
            logger.warning(f"Memory pressure detected: {memory_mb:.1f}MB > {_max_memory_mb}MB")
            return True
        
        return False
    except Exception as e:
        logger.debug(f"Could not check memory pressure: {e}")
        return False

def _load_face_model() -> FaceAnalysis:
    """Load InsightFace model with DirectML support."""
    global _face_app
    
    with _model_lock:
        if _face_app is None:
            logger.info("Loading InsightFace model...")
            
            # Patch InsightFace for DirectML
            _patch_insightface_for_directml()
            
            # Create model directory
            home = os.path.expanduser("~/.insightface")
            os.makedirs(home, exist_ok=True)
            
            # Load model
            app = FaceAnalysis(name='buffalo_l', root=home)
            app.prepare(ctx_id=0)  # Let InsightFace use native image dimensions
            
            _face_app = app
            logger.info("InsightFace model loaded successfully")
            
            # Log execution providers for each model
            if hasattr(_face_app, 'models'):
                for model_name, model in _face_app.models.items():
                    if hasattr(model, 'session'):
                        providers = model.session.get_providers()
                        logger.info(f"Model '{model_name}' using providers: {providers}")
    
    return _face_app

def _warmup_inference():
    """Perform dummy inference to keep GPU resources active."""
    global _last_inference_time
    
    try:
        logger.debug("Performing warmup inference to keep GPU active...")
        
        # Create a small dummy image (100x100 RGB)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_image[:] = (128, 128, 128)  # Gray image
        
        # Run inference
        face_app = _load_face_model()
        _ = face_app.get(dummy_image)
        
        _last_inference_time = time.time()
        logger.debug("Warmup inference completed successfully")
        
    except Exception as e:
        logger.warning(f"Warmup inference failed: {e}")

async def _background_maintenance():
    """Background task for GPU maintenance and warmup."""
    global _last_inference_time, _total_inferences
    
    if not PSUTIL_AVAILABLE:
        logger.info("psutil not available, skipping background maintenance")
        return
    
    logger.info("Starting background maintenance task...")
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = time.time()
            time_since_inference = current_time - _last_inference_time
            
            # Warmup if idle for too long
            if time_since_inference > _warmup_interval:
                logger.info(f"GPU idle for {time_since_inference:.1f}s, performing warmup...")
                _warmup_inference()
            
            # Periodic cleanup based on inference count
            if _total_inferences > 0 and _total_inferences % _memory_cleanup_threshold == 0:
                logger.info(f"Reached {_total_inferences} inferences, performing cleanup...")
                _cleanup_gpu_resources()
            
            # Force cleanup on memory pressure
            if _check_memory_pressure():
                logger.warning("Memory pressure detected, forcing cleanup...")
                _cleanup_gpu_resources()
                
        except asyncio.CancelledError:
            logger.info("Background maintenance task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in background maintenance: {e}", exc_info=True)
            await asyncio.sleep(60)

def _decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array with better error handling."""
    try:
        # Validate base64 string
        if not image_data or len(image_data) < 10:
            raise ValueError("Invalid base64 data: too short")
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Validate decoded bytes
        if not image_bytes or len(image_bytes) < 10:
            raise ValueError("Invalid image data: decoded bytes too small")
        
        # Check for common image headers
        if not (image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                image_bytes.startswith(b'\x89PNG') or       # PNG
                image_bytes.startswith(b'GIF8') or          # GIF
                image_bytes.startswith(b'BM')):            # BMP
            raise ValueError("Invalid image data: unrecognized format")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("OpenCV failed to decode image - possibly corrupted data")
        
        logger.debug(f"Successfully decoded image: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        logger.error(f"Base64 data length: {len(image_data) if image_data else 0}")
        logger.error(f"Base64 data preview: {image_data[:50] if image_data else 'None'}...")
        raise ValueError(f"Failed to decode image: {e}")

def _encode_image(image: np.ndarray) -> str:
    """Encode numpy array image to base64."""
    try:
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        
        # Convert to base64
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image: {e}")

def _detect_faces_batch(images: List[np.ndarray], min_quality: float = 0.5) -> List[List[FaceDetection]]:
    """Detect faces in a batch of images."""
    global _last_inference_time, _total_inferences
    
    logger.info(f"Processing batch of {len(images)} images with GPU")
    
    try:
        # Check memory before processing (optional)
        if PSUTIL_AVAILABLE and _check_memory_pressure():
            logger.warning("Memory pressure before batch, performing cleanup...")
            _cleanup_gpu_resources()
        
        face_app = _load_face_model()
        results = []
        
        for i, image in enumerate(images):
            try:
                logger.debug(f"Processing image {i+1}/{len(images)}")
                
                # Detect faces
                faces = face_app.get(image)
                
                # Convert to our format
                face_detections = []
                for face in faces:
                    if face.det_score >= min_quality:
                        # Convert numpy types to Python types for Pydantic validation
                        gender_value = getattr(face, 'gender', None)
                        if gender_value is not None:
                            if isinstance(gender_value, np.integer):
                                gender_value = str(int(gender_value))
                            else:
                                gender_value = str(gender_value)
                        
                        age_value = getattr(face, 'age', None)
                        if age_value is not None and isinstance(age_value, np.integer):
                            age_value = int(age_value)
                        
                        face_detection = FaceDetection(
                            bbox=face.bbox.tolist(),
                            landmarks=face.kps.tolist() if hasattr(face, 'kps') else [],
                            embedding=face.embedding.tolist() if hasattr(face, 'embedding') else None,
                            quality=float(face.det_score),
                            age=age_value,
                            gender=gender_value
                        )
                        face_detections.append(face_detection)
                
                results.append(face_detections)
                logger.debug(f"Image {i+1}: Found {len(face_detections)} faces")
                
            except Exception as e:
                logger.error(f"Error detecting faces in image {i+1}: {type(e).__name__}: {e}", exc_info=True)
                results.append([])
        
        # Update inference tracking (optional)
        if PSUTIL_AVAILABLE:
            _last_inference_time = time.time()
            _total_inferences += len(images)
            logger.info(f"Successfully processed batch: {len(results)} results (total inferences: {_total_inferences})")
            
            # Cleanup after large batches
            if len(images) >= 32:
                logger.debug("Large batch completed, performing cleanup...")
                _cleanup_gpu_resources()
        else:
            logger.info(f"Successfully processed batch: {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.error(f"GPU processing error: {type(e).__name__}: {e}", exc_info=True)
        raise

def _enhance_image(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    """Enhance image using upscaling."""
    try:
        # Convert to PIL for better upscaling
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Calculate new size
        width, height = pil_image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Upscale using LANCZOS resampling
        enhanced = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert back to OpenCV format
        enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        return enhanced_cv
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image

@app.on_event("startup")
async def startup_event():
    """Initialize the GPU worker on startup."""
    global _is_initialized, _cleanup_task
    
    with _initialization_lock:
        if not _is_initialized:
            logger.info("=== GPU Worker Service Starting ===")
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Working directory: {os.getcwd()}")
            
            try:
                # Check DirectML availability
                directml_available = _check_directml_availability()
                logger.info(f"DirectML available: {directml_available}")
                
                # Load face model
                logger.info("Loading face detection model...")
                _load_face_model()
                logger.info("Face detection model loaded successfully")
                
                # Perform initial warmup (optional)
                if PSUTIL_AVAILABLE:
                    try:
                        logger.info("Performing initial GPU warmup...")
                        _warmup_inference()
                        logger.info("GPU warmup completed")
                    except Exception as e:
                        logger.warning(f"GPU warmup failed (non-critical): {e}")
                
                # Start background maintenance task (optional)
                if PSUTIL_AVAILABLE:
                    try:
                        logger.info("Starting background maintenance task...")
                        _cleanup_task = asyncio.create_task(_background_maintenance())
                        logger.info("Background maintenance task started")
                    except Exception as e:
                        logger.warning(f"Background maintenance task failed (non-critical): {e}")
                
                _is_initialized = True
                logger.info("=== GPU Worker Service Started Successfully ===")
                
            except Exception as e:
                logger.error(f"CRITICAL: Failed to initialize GPU worker: {e}", exc_info=True)
                raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _cleanup_task, _face_app
    
    logger.info("Shutting down GPU Worker Service...")
    
    # Cancel background task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    
    # Cleanup GPU resources
    try:
        _cleanup_gpu_resources()
        
        # Clear model reference
        _face_app = None
        
        logger.info("GPU Worker Service shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with resource metrics."""
    directml_available = _check_directml_availability()
    gpu_actually_used = _check_actual_gpu_usage()
    model_loaded = _face_app is not None
    uptime = time.time() - _startup_time
    
    # Add resource metrics (optional)
    if PSUTIL_AVAILABLE:
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            logger.debug(f"Health check: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%, Inferences={_total_inferences}")
        except Exception as e:
            logger.debug(f"Could not get resource metrics: {e}")
    
    return HealthResponse(
        status="healthy" if _is_initialized else "unhealthy",
        gpu_available=directml_available,
        directml_available=directml_available,
        model_loaded=model_loaded,
        uptime_seconds=uptime
    )

@app.post("/cleanup")
async def manual_cleanup():
    """Manually trigger GPU resource cleanup."""
    try:
        memory_mb = _cleanup_gpu_resources()
        return {
            "status": "success",
            "message": "GPU resources cleaned up",
            "memory_mb": memory_mb,
            "total_inferences": _total_inferences
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

@app.post("/detect_faces_batch", response_model=BatchResponse)
async def detect_faces_batch(request: BatchRequest):
    """Detect faces in a batch of images."""
    start_time = time.time()
    
    try:
        # Decode images
        images = []
        for img_data in request.images:
            try:
                image = _decode_image(img_data.data)
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to decode image {img_data.image_id}: {e}")
                images.append(None)
        
        # Filter out None images
        valid_images = [img for img in images if img is not None]
        
        if not valid_images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Detect faces in batch
        results = await asyncio.get_event_loop().run_in_executor(
            _thread_pool,
            _detect_faces_batch,
            valid_images,
            request.min_face_quality
        )
        
        # Pad results for None images
        full_results = []
        result_idx = 0
        for img in images:
            if img is not None:
                full_results.append(results[result_idx])
                result_idx += 1
            else:
                full_results.append([])
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchResponse(
            results=full_results,
            processing_time_ms=processing_time,
            gpu_used=_check_actual_gpu_usage()
        )
        
    except Exception as e:
        logger.error(f"Error in detect_faces_batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhance_image", response_model=EnhanceImageResponse)
async def enhance_image(request: EnhanceImageRequest):
    """Enhance a single image using upscaling."""
    start_time = time.time()
    
    try:
        # Decode image
        image = _decode_image(request.image_data)
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Enhance image
        enhanced = await asyncio.get_event_loop().run_in_executor(
            _thread_pool,
            _enhance_image,
            image,
            request.scale_factor
        )
        
        # Encode enhanced image
        enhanced_data = _encode_image(enhanced)
        enhanced_size = (enhanced.shape[1], enhanced.shape[0])
        
        processing_time = (time.time() - start_time) * 1000
        
        return EnhanceImageResponse(
            enhanced_data=enhanced_data,
            original_size=original_size,
            enhanced_size=enhanced_size,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in enhance_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu_info")
async def get_gpu_info():
    """Get detailed GPU utilization information."""
    directml_available = _check_directml_availability()
    gpu_actually_used = _check_actual_gpu_usage()
    
    model_info = {}
    if _face_app and hasattr(_face_app, 'models'):
        for model_name, model in _face_app.models.items():
            if hasattr(model, 'session'):
                providers = model.session.get_providers()
                model_info[model_name] = {
                    'providers': providers,
                    'using_gpu': providers[0] == 'DmlExecutionProvider' if providers else False
                }
    
    return {
        'directml_available': directml_available,
        'gpu_actually_used': gpu_actually_used,
        'model_info': model_info,
        'timestamp': time.time()
    }

@app.post("/detect_and_embed_batch", response_model=BatchResponse)
async def detect_and_embed_batch(request: BatchRequest):
    """Detect faces and compute embeddings in a batch."""
    # For now, this is the same as detect_faces_batch
    # The embedding is already included in the face detection results
    return await detect_faces_batch(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
