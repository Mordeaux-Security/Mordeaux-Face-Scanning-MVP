"""
Windows GPU Worker Service

FastAPI service that runs natively on Windows with DirectML support for GPU-accelerated
face detection and embedding operations. Provides REST API endpoints for batch processing.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# Batched detector support
_batched_detector = None
_batched_detector_enabled = os.getenv('BATCHED_DETECTOR_ENABLED', 'false').lower() == 'true'
_detect_target_batch = int(os.getenv('DETECT_TARGET_BATCH', '16'))
_detect_min_launch_ms = int(os.getenv('DETECT_MIN_LAUNCH_MS', '180'))
_original_target_batch = _detect_target_batch
_original_min_launch_ms = _detect_min_launch_ms

# Auto-backoff state
_backoff_active = False
_backoff_until = 0.0
_backoff_duration = 30.0  # seconds

# Metrics tracking for batched detector
_batch_metrics = {
    'total_batches': 0,
    'total_images': 0,
    'batch_sizes': [],
    'latencies': [],
    'det_forward_times': [],
    'last_launch_time': 0.0,
    'interlaunch_times': [],
    'inflight_batches': 0,
    'max_inflight': 0
}
_metrics_lock = threading.Lock()
_last_summary_time = time.time()

# Thread pool for CPU operations
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gpu_worker")

# Startup time for uptime calculation
_startup_time = time.time()

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    directml_available: bool
    model_loaded: bool
    uptime_seconds: float

class FaceDetection(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    landmarks: List[List[float]] = Field(..., description="Facial landmarks")
    embedding: Optional[List[float]] = Field(None, description="Face embedding vector")
    quality: float = Field(..., description="Detection quality score")
    age: Optional[int] = Field(None, description="Estimated age")
    gender: Optional[str] = Field(None, description="Estimated gender")

class BatchResponsePhash(BaseModel):
    results: Dict[str, List[FaceDetection]] = Field(..., description="Face detection results keyed by phash")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    gpu_used: bool = Field(..., description="Whether GPU was used for processing")
    worker_id: str = Field(default="gpu-worker-1", description="Worker identifier")

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
                        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
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
            app.prepare(ctx_id=0)
            
            _face_app = app
            logger.info("InsightFace model loaded successfully")
            
            # Log execution providers for each model
            if hasattr(_face_app, 'models'):
                for model_name, model in _face_app.models.items():
                    if hasattr(model, 'session'):
                        providers = model.session.get_providers()
                        logger.info(f"Model '{model_name}' using providers: {providers}")
            
            # Load batched detector if enabled
            if _batched_detector_enabled:
                _load_batched_detector()
    
    return _face_app

def _activate_backoff():
    """Activate auto-backoff: reduce batch size and increase launch gap."""
    global _backoff_active, _backoff_until, _detect_target_batch, _detect_min_launch_ms
    global _original_target_batch, _original_min_launch_ms
    
    if _backoff_active:
        return  # Already in backoff
    
    _backoff_active = True
    _backoff_until = time.time() + _backoff_duration
    
    # Reduce batch size by 20-25%
    reduction_factor = 0.75  # 25% reduction
    _detect_target_batch = max(4, int(_original_target_batch * reduction_factor))
    
    # Increase launch gap by +50ms
    _detect_min_launch_ms = _original_min_launch_ms + 50
    
    logger.warning(f"[AUTO-BACKOFF] Activated: target_batch={_detect_target_batch} (was {_original_target_batch}), "
                   f"min_launch_ms={_detect_min_launch_ms} (was {_original_min_launch_ms}), "
                   f"duration={_backoff_duration}s")
    
    # Schedule restoration using threading (works in sync context)
    def restore_thread():
        global _detect_target_batch, _detect_min_launch_ms, _backoff_active, _backoff_until
        
        time.sleep(_backoff_duration)
        if _backoff_active and time.time() >= _backoff_until:
            _detect_target_batch = _original_target_batch
            _detect_min_launch_ms = _original_min_launch_ms
            _backoff_active = False
            
            logger.info(f"[AUTO-BACKOFF] Restored: target_batch={_detect_target_batch}, "
                       f"min_launch_ms={_detect_min_launch_ms}")
    
    threading.Thread(target=restore_thread, daemon=True).start()

def _load_batched_detector():
    """Load batched detector if enabled."""
    global _batched_detector, _face_app
    
    try:
        from .detectors.scrfd_onnx import SCRFDOnnx
        
        if _face_app is None:
            logger.warning("Face app not loaded, cannot initialize batched detector")
            return
        
        # Get execution provider from face app's detection model
        detector_model = _face_app.models.get('detection')
        provider = "DmlExecutionProvider"
        if detector_model and hasattr(detector_model, 'session'):
            providers = detector_model.session.get_providers()
            if providers:
                provider = providers[0]
        
        # Detection parameters tuned for accuracy:
        # - Input size increased to 1024 for better small face detection
        # - Score threshold increased to 0.65 to reduce false positives
        input_size = int(os.getenv('DETECT_INPUT_SIZE', '1024'))
        score_thr = float(os.getenv('DETECT_SCORE_THR', '0.65'))
        nms_iou = float(os.getenv('DETECT_NMS_IOU', '0.4'))
        
        _batched_detector = SCRFDOnnx(
            face_app=_face_app,
            provider=provider,
            device_id=0,
            input_size=input_size,
            score_thr=score_thr,
            nms_iou=nms_iou
        )
        
        logger.info(f"Batched detector loaded successfully (provider={provider}, input_size={input_size})")
    except Exception as e:
        logger.error(f"Failed to load batched detector: {e}", exc_info=True)
        _batched_detector = None

def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode binary image bytes to numpy array."""
    try:
        if not image_bytes or len(image_bytes) < 10:
            raise ValueError("Invalid image data: too small")
        
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
            raise ValueError("OpenCV failed to decode image")
        
        return image
    except Exception as e:
        logger.error(f"Failed to decode binary image: {e}")
        raise ValueError(f"Failed to decode image: {e}")

def _log_metrics_summary():
    """Log periodic metrics summary."""
    global _batch_metrics
    
    with _metrics_lock:
        if _batch_metrics['total_batches'] == 0:
            return
        
        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100.0)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        avg_batch = sum(_batch_metrics['batch_sizes']) / len(_batch_metrics['batch_sizes']) if _batch_metrics['batch_sizes'] else 0.0
        
        p50_latency = percentile(_batch_metrics['latencies'], 50)
        p95_latency = percentile(_batch_metrics['latencies'], 95)
        
        p50_det_forward = percentile(_batch_metrics['det_forward_times'], 50)
        p95_det_forward = percentile(_batch_metrics['det_forward_times'], 95)
        
        p50_interlaunch = percentile(_batch_metrics['interlaunch_times'], 50) if _batch_metrics['interlaunch_times'] else 0.0
        p95_interlaunch = percentile(_batch_metrics['interlaunch_times'], 95) if _batch_metrics['interlaunch_times'] else 0.0
        
        # Calculate idle time (interlaunch - latency, clipped >= 0)
        idle_times = []
        for i in range(min(len(_batch_metrics['interlaunch_times']), len(_batch_metrics['latencies']))):
            idle = max(0.0, _batch_metrics['interlaunch_times'][i] - _batch_metrics['latencies'][i])
            idle_times.append(idle)
        p50_idle = percentile(idle_times, 50) if idle_times else 0.0
        p95_idle = percentile(idle_times, 95) if idle_times else 0.0
        
        # Calculate average GPU percentage (from detector times - approximate since we don't track GPU separately)
        avg_detector_time = sum(_batch_metrics['det_forward_times']) / len(_batch_metrics['det_forward_times']) if _batch_metrics['det_forward_times'] else 0
        avg_total_time = sum(_batch_metrics['latencies']) / len(_batch_metrics['latencies']) if _batch_metrics['latencies'] else 0
        est_gpu_pct = (avg_detector_time / avg_total_time * 100) if avg_total_time > 0 else 0
        
        logger.info(f"[BATCHED-DET-SUMMARY] batches={_batch_metrics['total_batches']}, "
                   f"images={_batch_metrics['total_images']}, "
                   f"avg_batch={avg_batch:.1f}, "
                   f"p50_latency={p50_latency:.1f}ms, p95_latency={p95_latency:.1f}ms, "
                   f"p50_det_forward={p50_det_forward:.1f}ms, p95_det_forward={p95_det_forward:.1f}ms, "
                   f"est_gpu_pct={est_gpu_pct:.1f}%, "
                   f"p50_interlaunch={p50_interlaunch:.1f}ms, p95_interlaunch={p95_interlaunch:.1f}ms, "
                   f"p50_idle={p50_idle:.1f}ms, p95_idle={p95_idle:.1f}ms, "
                   f"inflight_max={_batch_metrics['max_inflight']}")
        
        # Reset metrics for next window (keep totals)
        _batch_metrics['batch_sizes'] = []
        _batch_metrics['latencies'] = []
        _batch_metrics['det_forward_times'] = []
        _batch_metrics['interlaunch_times'] = []
        _batch_metrics['max_inflight'] = 0

def _detect_faces_batch(images: List[np.ndarray], min_quality: float) -> List[List[FaceDetection]]:
    """Detect faces in a batch of images."""
    global _batched_detector, _batched_detector_enabled, _batch_metrics, _metrics_lock, _detect_min_launch_ms, _last_summary_time
    
    if min_quality is None:
        raise ValueError("min_quality must be provided")
    
    batch_start_time = time.perf_counter()
    batch_size = len(images)
    
    logger.info(f"Processing batch of {batch_size} images with GPU")
    
    # Try batched detector if enabled and batch size > 1
    if _batched_detector_enabled and _batched_detector is not None and batch_size > 1:
        try:
            # Calculate interlaunch time
            current_time = time.perf_counter()
            interlaunch_ms = 0.0
            with _metrics_lock:
                if _batch_metrics['last_launch_time'] > 0:
                    interlaunch_ms = (current_time - _batch_metrics['last_launch_time']) * 1000
                _batch_metrics['last_launch_time'] = current_time
                _batch_metrics['inflight_batches'] += 1
                _batch_metrics['max_inflight'] = max(_batch_metrics['max_inflight'], 
                                                      _batch_metrics['inflight_batches'])
                if interlaunch_ms > 0:
                    _batch_metrics['interlaunch_times'].append(interlaunch_ms)
                    if interlaunch_ms < _detect_min_launch_ms:
                        logger.warning(f"[BATCHED-DET] Pace violation: interlaunch_ms={interlaunch_ms:.1f}ms < MIN={_detect_min_launch_ms}ms")
            
            # Log launch
            logger.info(f"[BATCHED-DET-LAUNCH] stage=det, batch_id={int(current_time*1000)}, "
                       f"size={batch_size}, inflight={_batch_metrics['inflight_batches']}, "
                       f"interlaunch_ms={interlaunch_ms:.1f}")
            
            # Detailed timing breakdown
            # The detector's process_batch logs internal timings, but we track overall here
            t_detector_start = time.perf_counter()
            detection_results = _batched_detector.process_batch(images)
            t_detector_end = time.perf_counter()
            t_detector_total_ms = (t_detector_end - t_detector_start) * 1000
            
            # Convert DetectionResult to FaceDetection format (post-processing)
            results = []
            t_post_start = time.perf_counter()
            for i, det_result in enumerate(detection_results):
                face_detections = []
                for j in range(len(det_result.boxes)):
                    bbox = det_result.boxes[j]
                    score = det_result.scores[j]
                    kps = det_result.kps[j] if det_result.kps is not None else None
                    
                    if score >= min_quality:
                        face_detection = FaceDetection(
                            bbox=bbox.tolist(),
                            landmarks=kps.tolist() if kps is not None else [],
                            embedding=None,
                            quality=float(score),
                            age=None,
                            gender=None
                        )
                        face_detections.append(face_detection)
                
                results.append(face_detections)
            
            t_post_ms = (time.perf_counter() - t_post_start) * 1000
            t_total_ms = (time.perf_counter() - batch_start_time) * 1000
            
            # Calculate overhead (metrics tracking, lock contention, etc.)
            t_overhead_ms = t_total_ms - t_detector_total_ms - t_post_ms
            
            # Log detailed timing breakdown with GPU vs CPU breakdown
            pct_detector = (t_detector_total_ms / t_total_ms * 100) if t_total_ms > 0 else 0
            pct_post = (t_post_ms / t_total_ms * 100) if t_total_ms > 0 else 0
            pct_overhead = (t_overhead_ms / t_total_ms * 100) if t_total_ms > 0 else 0
            
            logger.info(f"[BATCHED-DET-TIMING] batch_id={int(current_time*1000)}, size={batch_size}, "
                       f"total={t_total_ms:.2f}ms (100%) | "
                       f"detector={t_detector_total_ms:.2f}ms ({pct_detector:.1f}%) | "
                       f"format_convert={t_post_ms:.2f}ms ({pct_post:.1f}%) | "
                       f"overhead={t_overhead_ms:.2f}ms ({pct_overhead:.1f}%) | "
                       f"per_image={t_total_ms/batch_size:.2f}ms")
            
            # Detector logs [SCRFD-TIMING] with detailed GPU vs CPU breakdown
            
            # Update metrics
            with _metrics_lock:
                _batch_metrics['total_batches'] += 1
                _batch_metrics['total_images'] += batch_size
                _batch_metrics['batch_sizes'].append(batch_size)
                _batch_metrics['latencies'].append(t_total_ms)
                _batch_metrics['det_forward_times'].append(t_detector_total_ms)
                _batch_metrics['inflight_batches'] -= 1
            
            # Log completion
            logger.info(f"[BATCHED-DET-COMPLETE] batch_id={int(current_time*1000)}, "
                       f"size={batch_size}, total_ms={t_total_ms:.1f}, "
                       f"detector_ms={t_detector_total_ms:.1f}, "
                       f"format_convert_ms={t_post_ms:.1f}, "
                       f"inflight={_batch_metrics['inflight_batches']}, "
                       f"faces={sum(len(r) for r in results)}")
            
            # Periodic summary (every 10s)
            if time.time() - _last_summary_time >= 10.0:
                _log_metrics_summary()
                _last_summary_time = time.time()
            
            return results
            
        except Exception as e:
            error_msg = str(e).lower()
            is_dml_error = 'dml' in error_msg or 'directml' in error_msg or 'driver' in error_msg or 'gpu' in error_msg
            
            if is_dml_error:
                logger.error(f"[BATCHED-DET] DML/driver error detected: {e}")
                _activate_backoff()
            
            logger.warning(f"Batched detector failed, falling back to single-image: {e}", exc_info=True)
            # Update metrics on failure
            with _metrics_lock:
                if _batch_metrics['inflight_batches'] > 0:
                    _batch_metrics['inflight_batches'] -= 1
            # Fall through to single-image processing
    
    # Fallback: single-image processing
    logger.info(f"[SINGLE-IMAGE-FALLBACK] Processing {batch_size} images one-by-one")
    face_app = _load_face_model()
    results = []
    
    t_total_start = time.perf_counter()
    t_infer_times = []
    t_format_times = []
    
    for i, image in enumerate(images):
        try:
            # Detect faces (includes GPU inference if available)
            t_detect_start = time.perf_counter()
            faces = face_app.get(image)
            t_detect_ms = (time.perf_counter() - t_detect_start) * 1000
            t_infer_times.append(t_detect_ms)
            
            # Convert to our format
            t_format_start = time.perf_counter()
            face_detections = []
            for face in faces:
                if face.det_score >= min_quality:
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
            
            t_format_ms = (time.perf_counter() - t_format_start) * 1000
            t_format_times.append(t_format_ms)
            
            results.append(face_detections)
            
        except Exception as e:
            logger.error(f"Error detecting faces in image {i+1}: {type(e).__name__}: {e}", exc_info=True)
            results.append([])
    
    t_total_ms = (time.perf_counter() - t_total_start) * 1000
    avg_infer_ms = sum(t_infer_times) / len(t_infer_times) if t_infer_times else 0
    avg_format_ms = sum(t_format_times) / len(t_format_times) if t_format_times else 0
    t_overhead_ms = t_total_ms - sum(t_infer_times) - sum(t_format_times)
    
    pct_infer = (sum(t_infer_times) / t_total_ms * 100) if t_total_ms > 0 else 0
    pct_format = (sum(t_format_times) / t_total_ms * 100) if t_total_ms > 0 else 0
    pct_overhead = (t_overhead_ms / t_total_ms * 100) if t_total_ms > 0 else 0
    
    logger.info(f"[SINGLE-IMAGE-TIMING] batch_size={batch_size}, "
               f"total={t_total_ms:.2f}ms (100%) | "
               f"detection={sum(t_infer_times):.2f}ms ({pct_infer:.1f}%, avg={avg_infer_ms:.2f}ms/img) | "
               f"format_convert={sum(t_format_times):.2f}ms ({pct_format:.1f}%, avg={avg_format_ms:.2f}ms/img) | "
               f"overhead={t_overhead_ms:.2f}ms ({pct_overhead:.1f}%) | "
               f"per_image={t_total_ms/batch_size:.2f}ms")
    
    return results

@app.on_event("startup")
async def startup_event():
    """Initialize the GPU worker on startup."""
    global _is_initialized
    
    with _initialization_lock:
        if not _is_initialized:
            logger.info("=== GPU Worker Service Starting ===")
            
            try:
                # Check DirectML availability
                directml_available = _check_directml_availability()
                logger.info(f"DirectML available: {directml_available}")
                
                # Load face model
                logger.info("Loading face detection model...")
                _load_face_model()
                logger.info("Face detection model loaded successfully")
                
                _is_initialized = True
                logger.info("=== GPU Worker Service Started Successfully ===")
                
            except Exception as e:
                logger.error(f"CRITICAL: Failed to initialize GPU worker: {e}", exc_info=True)
                raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down GPU Worker Service...")
    logger.info("GPU Worker Service shutdown complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    directml_available = _check_directml_availability()
    gpu_actually_used = _check_actual_gpu_usage()
    model_loaded = _face_app is not None
    uptime = time.time() - _startup_time
    
    return HealthResponse(
        status="healthy" if _is_initialized else "unhealthy",
        gpu_available=directml_available,
        directml_available=directml_available,
        model_loaded=model_loaded,
        uptime_seconds=uptime
    )

@app.post("/detect_faces_batch_multipart", response_model=BatchResponsePhash)
async def detect_faces_batch_multipart(
    images: List[UploadFile] = File(...),
    image_hashes: str = Form(...),
    min_face_quality: float = Form(...),
    require_face: bool = Form(False),
    crop_faces: bool = Form(True),
    face_margin: float = Form(0.2)
):
    """
    Detect faces in a batch of images using multipart/form-data.
    
    Accepts binary image files directly (no base64 encoding).
    Results are keyed by phash for reliable linkage.
    """
    start_time = time.time()
    batch_id = f"api_{int(start_time * 1000)}"
    
    try:
        # Log request start
        logger.info(f"[GPU-WORKER-API] REQUEST-START: batch_id={batch_id}, image_count={len(images)}, timestamp={start_time:.3f}")
        # Parse image_hashes JSON
        try:
            hash_mapping = json.loads(image_hashes)
            index_to_phash = {item["index"]: item["phash"] for item in hash_mapping}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse image_hashes JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image_hashes format: {e}")
        
        if len(images) != len(hash_mapping):
            logger.warning(f"Image count mismatch: {len(images)} files vs {len(hash_mapping)} hash entries")
            raise HTTPException(status_code=400, detail=f"Image count ({len(images)}) doesn't match hash count ({len(hash_mapping)})")
        
        # Decode images from binary
        decoded_images = []
        phash_list = []
        failed_indices = []
        
        for idx, upload_file in enumerate(images):
            try:
                image_bytes = await upload_file.read()
                
                if not image_bytes:
                    logger.warning(f"Empty image at index {idx}")
                    failed_indices.append(idx)
                    continue
                
                # Decode binary to numpy array
                image = await asyncio.get_event_loop().run_in_executor(
                    _thread_pool,
                    _decode_image_bytes,
                    image_bytes
                )
                
                decoded_images.append(image)
                phash = index_to_phash.get(idx)
                if phash:
                    phash_list.append(phash)
                else:
                    logger.warning(f"No phash mapping for index {idx}, using placeholder")
                    phash_list.append(f"unknown_{idx}")
                    
            except Exception as e:
                logger.error(f"Failed to decode image at index {idx}: {e}")
                failed_indices.append(idx)
                continue
        
        if not decoded_images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        if failed_indices:
            logger.warning(f"Failed to decode {len(failed_indices)} images: {failed_indices}")
        
        # Detect faces in batch
        results = await asyncio.get_event_loop().run_in_executor(
            _thread_pool,
            _detect_faces_batch,
            decoded_images,
            min_face_quality
        )
        
        # Build results dictionary keyed by phash
        results_dict = {}
        for idx, (phash, face_detections) in enumerate(zip(phash_list, results)):
            results_dict[phash] = face_detections
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate additional metrics
        total_faces = sum(len(f) for f in results_dict.values())
        avg_faces_per_image = total_faces / len(decoded_images) if decoded_images else 0
        gpu_used = _check_actual_gpu_usage()
        
        logger.info(f"[MULTIPART-BATCH] processed={len(decoded_images)} images, "
                   f"results={len(results_dict)}, faces={total_faces} "
                   f"(avg={avg_faces_per_image:.2f}/img), "
                   f"time={processing_time:.1f}ms ({processing_time/len(decoded_images):.2f}ms/img), "
                   f"gpu_used={gpu_used}")
        
        # Log request complete
        logger.info(f"[GPU-WORKER-API] REQUEST-COMPLETE: batch_id={batch_id}, images={len(decoded_images)}, "
                   f"faces={total_faces}, duration_ms={processing_time:.1f}, status=200, gpu_used={gpu_used}")
        
        return BatchResponsePhash(
            results=results_dict,
            processing_time_ms=processing_time,
            gpu_used=gpu_used
        )
        
    except HTTPException as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[GPU-WORKER-API] REQUEST-ERROR: batch_id={batch_id}, error={str(e)}, duration_ms={duration_ms:.1f}")
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[GPU-WORKER-API] REQUEST-ERROR: batch_id={batch_id}, error={str(e)}, duration_ms={duration_ms:.1f}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
