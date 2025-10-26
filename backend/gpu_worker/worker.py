"""
Windows GPU Worker Service

Robust FastAPI service for GPU-accelerated face detection using DirectML.
Features process management, health monitoring, and graceful degradation.
"""

import asyncio
import base64
import io
import logging
import os
import threading
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import signal
import sys
import atexit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_worker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model management
_face_app = None
_model_lock = threading.Lock()
_initialization_lock = threading.Lock()
_is_initialized = False
_shutdown_event = threading.Event()

# Request queue for batch processing
_request_queue = Queue(maxsize=1000)
_processing_thread = None
_worker_id = str(uuid.uuid4())[:8]

# Thread pool for CPU operations
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gpu_worker")

# Dynamic batch size management
_current_batch_size = 128  # Start aggressive for better GPU utilization
_max_batch_size = 1024     # Allow large batches (computers like powers of 2)
_min_batch_size = 4       # Minimum batch size

# Metrics tracking
_metrics = {
    'requests_processed': 0,
    'requests_failed': 0,
    'total_processing_time': 0.0,
    'gpu_utilization': 0.0,
    'start_time': time.time(),
    'last_health_check': time.time(),
    'throughput_rps': 0.0,
    'avg_latency_ms': 0.0,
    'recent_requests': [],
    'recent_latencies': []
}

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    directml_available: bool
    model_loaded: bool
    uptime_seconds: float
    worker_id: str
    queue_size: int
    metrics: Dict[str, Any]

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
    worker_id: str = Field(..., description="Worker instance ID")

class BatchConfigRequest(BaseModel):
    batch_size: int = Field(..., description="New batch size", ge=1, le=128)

class BatchConfigResponse(BaseModel):
    current_batch_size: int = Field(..., description="Current batch size")
    max_batch_size: int = Field(..., description="Maximum allowed batch size")
    min_batch_size: int = Field(..., description="Minimum allowed batch size")
    success: bool = Field(..., description="Whether the update was successful")
    message: str = Field(..., description="Status message")

# FastAPI app
app = FastAPI(
    title="GPU Worker Service",
    description="Windows GPU worker for face detection and embedding with DirectML",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _check_directml_availability() -> bool:
    """Check if DirectML execution provider is available."""
    try:
        available_providers = ort.get_available_providers()
        directml_available = 'DmlExecutionProvider' in available_providers
        logger.info(f"Available ONNX providers: {available_providers}")
        logger.info(f"DirectML available: {directml_available}")
        return directml_available
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
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            _face_app = app
            logger.info("InsightFace model loaded successfully")
            
            # Log execution providers for each model
            if hasattr(_face_app, 'models'):
                for model_name, model in _face_app.models.items():
                    if hasattr(model, 'session'):
                        providers = model.session.get_providers()
                        logger.info(f"Model '{model_name}' using providers: {providers}")
    
    return _face_app

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

def _detect_faces_batch(images: List[np.ndarray], min_quality: float = 0.5) -> List[List[FaceDetection]]:
    """Detect faces in a batch of images using dynamic batch sizing."""
    face_app = _load_face_model()
    results = []
    
    # Process images in chunks based on current batch size
    batch_size = _current_batch_size
    total_images = len(images)
    
    logger.debug(f"Processing {total_images} images in batches of {batch_size}")
    
    for i in range(0, total_images, batch_size):
        chunk = images[i:i + batch_size]
        chunk_results = []
        
        for image in chunk:
            try:
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
                
                chunk_results.append(face_detections)
                
            except Exception as e:
                logger.error(f"Error detecting faces in image: {e}")
                chunk_results.append([])
        
        # Add chunk results to overall results
        results.extend(chunk_results)
    
    return results

def _process_request_queue():
    """Process requests from the queue in a separate thread."""
    global _metrics
    
    while not _shutdown_event.is_set():
        try:
            # Get request from queue with timeout
            request_data = _request_queue.get(timeout=1.0)
            
            start_time = time.time()
            
            try:
                # Decode images
                images = []
                for img_data in request_data['images']:
                    try:
                        image = _decode_image(img_data['data'])
                        images.append(image)
                    except Exception as e:
                        logger.error(f"Failed to decode image {img_data.get('image_id', 'unknown')}: {e}")
                        images.append(None)
                
                # Filter out None images
                valid_images = [img for img in images if img is not None]
                
                if not valid_images:
                    # Return empty results for all images
                    results = [[] for _ in images]
                else:
                    # Detect faces in batch
                    face_results = _detect_faces_batch(
                        valid_images, 
                        request_data.get('min_face_quality', 0.5)
                    )
                    
                    # Pad results for None images
                    results = []
                    result_idx = 0
                    for img in images:
                        if img is not None:
                            results.append(face_results[result_idx])
                            result_idx += 1
                        else:
                            results.append([])
                
                # Update metrics
                processing_time = time.time() - start_time
                _metrics['requests_processed'] += 1
                _metrics['total_processing_time'] += processing_time
                
                # Track recent requests and latencies for throughput calculation
                current_time = time.time()
                _metrics['recent_requests'].append(current_time)
                _metrics['recent_latencies'].append({
                    'timestamp': current_time,
                    'latency': processing_time
                })
                
                # Clean up old metrics (keep last 100 entries)
                _metrics['recent_requests'] = [req_time for req_time in _metrics['recent_requests'] if current_time - req_time < 300.0]  # Last 5 minutes
                _metrics['recent_latencies'] = [lat for lat in _metrics['recent_latencies'] if current_time - lat['timestamp'] < 300.0]  # Last 5 minutes
                
                # Put result back in request
                request_data['result'] = results
                request_data['processing_time'] = processing_time
                request_data['gpu_used'] = _check_actual_gpu_usage()
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                _metrics['requests_failed'] += 1
                request_data['result'] = None
                request_data['error'] = str(e)
            
            # Mark task as done
            _request_queue.task_done()
            
        except Empty:
            # Timeout - continue loop
            continue
        except Exception as e:
            logger.error(f"Error in request processing loop: {e}")
            time.sleep(1)

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_event.set()
    
    # Wait for processing thread to finish
    if _processing_thread and _processing_thread.is_alive():
        _processing_thread.join(timeout=5)
    
    # Close thread pool
    _thread_pool.shutdown(wait=True)
    
    logger.info("GPU Worker shutdown complete")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

@app.on_event("startup")
async def startup_event():
    """Initialize the GPU worker on startup."""
    global _is_initialized, _processing_thread
    
    with _initialization_lock:
        if not _is_initialized:
            logger.info(f"=== GPU Worker Service Starting (ID: {_worker_id}) ===")
            
            # Check DirectML availability
            directml_available = _check_directml_availability()
            logger.info(f"[GPU-DETECTED] DirectML available: {directml_available}")
            
            # Load face model
            try:
                logger.info("[GPU-DETECTED] Loading face analysis model...")
                _load_face_model()
                
                # Verify GPU is actually being used
                gpu_actually_used = _check_actual_gpu_usage()
                if gpu_actually_used:
                    logger.info("[GPU-ACTIVE] Models are using DirectML/GPU acceleration")
                else:
                    logger.warning("[GPU-WARNING] Models are using CPU execution (GPU not active)")
                
                _is_initialized = True
                logger.info("[GPU-DETECTED] GPU Worker Service started successfully")
                
                # Start processing thread
                _processing_thread = threading.Thread(
                    target=_process_request_queue,
                    name="request_processor",
                    daemon=True
                )
                _processing_thread.start()
                logger.info("Request processing thread started")
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU worker: {e}")
                raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    directml_available = _check_directml_availability()
    gpu_actually_used = _check_actual_gpu_usage()
    model_loaded = _face_app is not None
    uptime = time.time() - _metrics['start_time']
    
    return HealthResponse(
        status="healthy" if _is_initialized else "unhealthy",
        gpu_available=directml_available,
        directml_available=directml_available,
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        worker_id=_worker_id,
        queue_size=_request_queue.qsize(),
        metrics=_metrics.copy()
    )

@app.post("/detect_faces_batch", response_model=BatchResponse)
async def detect_faces_batch(request: BatchRequest):
    """Detect faces in a batch of images."""
    start_time = time.time()
    
    try:
        # Prepare request data
        request_data = {
            'images': [{'data': img.data, 'image_id': img.image_id} for img in request.images],
            'min_face_quality': request.min_face_quality,
            'require_face': request.require_face,
            'crop_faces': request.crop_faces,
            'face_margin': request.face_margin
        }
        
        # Add to processing queue
        _request_queue.put(request_data)
        
        # Wait for processing to complete
        timeout = 30.0  # 30 second timeout
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            if 'result' in request_data:
                break
            await asyncio.sleep(0.1)
        
        if 'result' not in request_data:
            raise HTTPException(status_code=504, detail="Request processing timeout")
        
        if 'error' in request_data:
            raise HTTPException(status_code=500, detail=request_data['error'])
        
        results = request_data['result']
        processing_time = request_data.get('processing_time', time.time() - start_time)
        gpu_used = request_data.get('gpu_used', False)
        
        return BatchResponse(
            results=results,
            processing_time_ms=processing_time * 1000,
            gpu_used=gpu_used,
            worker_id=_worker_id
        )
        
    except Exception as e:
        logger.error(f"Error in detect_faces_batch: {e}")
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
    
    # Calculate recent throughput and latency
    current_time = time.time()
    recent_requests = [req_time for req_time in _metrics['recent_requests'] if current_time - req_time < 60.0]  # Last 60 seconds
    recent_latencies = [lat for lat in _metrics['recent_latencies'] if current_time - lat['timestamp'] < 60.0]  # Last 60 seconds
    
    throughput_rps = len(recent_requests) / 60.0 if recent_requests else 0.0
    avg_latency_ms = sum(lat['latency'] for lat in recent_latencies) / len(recent_latencies) * 1000 if recent_latencies else 0.0
    
    # Get system memory usage
    system_memory_percent = psutil.virtual_memory().percent
    
    return {
        'directml_available': directml_available,
        'gpu_actually_used': gpu_actually_used,
        'model_info': model_info,
        'worker_id': _worker_id,
        'queue_size': _request_queue.qsize(),
        'current_batch_size': _current_batch_size,
        'max_batch_size': _max_batch_size,
        'min_batch_size': _min_batch_size,
        'throughput_rps': throughput_rps,
        'avg_latency_ms': avg_latency_ms,
        'system_memory_percent': system_memory_percent,
        'metrics': _metrics.copy(),
        'timestamp': current_time
    }

@app.get("/batch_config", response_model=BatchConfigResponse)
async def get_batch_config():
    """Get current batch size configuration."""
    return BatchConfigResponse(
        current_batch_size=_current_batch_size,
        max_batch_size=_max_batch_size,
        min_batch_size=_min_batch_size,
        success=True,
        message="Batch configuration retrieved successfully"
    )

@app.post("/batch_config", response_model=BatchConfigResponse)
async def set_batch_config(request: BatchConfigRequest):
    """Update batch size configuration."""
    global _current_batch_size
    
    new_batch_size = request.batch_size
    
    # Validate batch size
    if new_batch_size < _min_batch_size or new_batch_size > _max_batch_size:
        return BatchConfigResponse(
            current_batch_size=_current_batch_size,
            max_batch_size=_max_batch_size,
            min_batch_size=_min_batch_size,
            success=False,
            message=f"Batch size must be between {_min_batch_size} and {_max_batch_size}"
        )
    
    # Update batch size
    old_batch_size = _current_batch_size
    _current_batch_size = new_batch_size
    
    logger.info(f"Batch size updated: {old_batch_size} -> {new_batch_size}")
    
    return BatchConfigResponse(
        current_batch_size=_current_batch_size,
        max_batch_size=_max_batch_size,
        min_batch_size=_min_batch_size,
        success=True,
        message=f"Batch size updated from {old_batch_size} to {new_batch_size}"
    )

@app.post("/detect_and_embed_batch", response_model=BatchResponse)
async def detect_and_embed_batch(request: BatchRequest):
    """Detect faces and compute embeddings in a batch."""
    # For now, this is the same as detect_faces_batch
    # The embedding is already included in the face detection results
    return await detect_faces_batch(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)