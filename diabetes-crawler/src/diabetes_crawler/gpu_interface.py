"""
GPU Interface for New Crawler System

Provides GPU worker client with CPU fallback for face detection and embedding.
Handles consistent data structures and proper batching to avoid single-image requests.
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import time
from typing import List, Optional, Dict, Any, Tuple
import traceback

import cv2
import httpx
import numpy as np
from PIL import Image

from .config import get_config
from .data_structures import ImageTask, FaceDetection, BatchRequest, BatchResponse
from .gpu_worker_logger import GPUWorkerLogger

logger = logging.getLogger(__name__)

# Singleton pattern per process
_gpu_interface_instance = None


class GPUInterface:
    """GPU worker interface with CPU fallback."""
    
    def __init__(self):
        self.config = get_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._client_event_loop_id: Optional[int] = None  # Track which event loop the client is bound to
        self._client_lock: Optional[asyncio.Lock] = None  # Lock for client creation/recreation (lazy init)
        self.gpu_logger = GPUWorkerLogger(0)  # Use 0 as default worker ID
        
        # Circuit breaker
        self._is_available = False
        self._failure_count = 0
        self._last_health_check = 0
        self._circuit_open_until = 0
        self._last_error_type: Optional[str] = None  # Track last error type for circuit breaker logging
        
        # Metrics tracking
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        
        # CPU fallback model cache (lazy-loaded and reused)
        self._cpu_app = None
        self._cpu_app_lock = None  # Created lazily in the correct event loop
    
    def _get_client_lock(self) -> asyncio.Lock:
        """Get or create the client lock (lazy initialization in correct event loop)."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        return self._client_lock
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client, recreating if bound to different event loop.
        
        This handles the case where a process is forked and inherits a client bound
        to the parent's event loop. When the child process creates a new event loop,
        we detect the mismatch and recreate the client.
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            logger.error(f"[GPU-INTERFACE] [TRACE] Creating httpx.AsyncClient - NO event loop running! This will fail!")
            raise RuntimeError("Cannot create httpx.AsyncClient: no event loop running")
        
        # Use lock to prevent race conditions during client recreation
        lock = self._get_client_lock()
        async with lock:
            # Check if we need to recreate the client (different event loop or doesn't exist)
            if self._client is None or self._client_event_loop_id != current_loop_id:
                # Close existing client if it exists but is bound to a different loop
                if self._client is not None:
                    logger.warning(f"[GPU-INTERFACE] [TRACE] Existing client bound to different event loop (old={self._client_event_loop_id}, new={current_loop_id}), recreating")
                    try:
                        await self._client.aclose()
                    except Exception as e:
                        logger.warning(f"[GPU-INTERFACE] [TRACE] Error closing old client: {e}")
                    self._client = None
                    self._client_event_loop_id = None
                    self._client_lock = None  # Reset lock so it's recreated in the correct event loop
                
                logger.info(f"[GPU-INTERFACE] [TRACE] Creating httpx.AsyncClient - event loop exists: id={current_loop_id}")
                
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self.config.gpu_worker_timeout,
                    write=15.0,
                    pool=10.0
                )
                
                limits = httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=10,
                    keepalive_expiry=120.0
                )
                
                logger.info(f"[GPU-INTERFACE] [TRACE] Instantiating httpx.AsyncClient with base_url={self.config.gpu_worker_host_resolved}")
                try:
                    self._client = httpx.AsyncClient(
                        base_url=self.config.gpu_worker_host_resolved,
                        timeout=timeout,
                        limits=limits,
                        http2=True,
                        headers={
                            "Connection": "keep-alive",
                            "Keep-Alive": "timeout=120, max=1000",
                            "User-Agent": "New-Crawler-GPU-Client/1.0"
                        }
                    )
                    self._client_event_loop_id = current_loop_id  # Track the event loop ID
                    logger.info(f"[GPU-INTERFACE] [TRACE] httpx.AsyncClient created successfully, bound to event loop: id={current_loop_id}")
                except Exception as e:
                    logger.error(f"[GPU-INTERFACE] [TRACE] Failed to create httpx.AsyncClient: {e}", exc_info=True)
                    raise
            else:
                logger.debug(f"[GPU-INTERFACE] [TRACE] Returning existing httpx.AsyncClient (bound to event loop: id={current_loop_id})")
        
        return self._client
    
    async def _check_health(self) -> bool:
        """Check if GPU worker is available (optimized with circuit breaker)."""
        current_time = time.time()
        
        # Skip if checked recently (10s cache)
        if current_time - self._last_health_check < 10.0:
            return self._is_available
        
        # Circuit breaker check (should be checked in process_batch, but double-check here)
        if current_time < self._circuit_open_until:
            return False
        
        try:
            client = await self._get_client()
            # Use shorter timeout for health check (5s is reasonable)
            response = await client.get("/health", timeout=5.0)
            
            if response.status_code == 200:
                health_data = response.json()
                self._is_available = health_data.get("status") == "healthy"
                was_open = self._circuit_open_until > current_time
                self._failure_count = 0
                # Reset circuit breaker on success
                self._circuit_open_until = 0
                self.gpu_logger.log_health_check(True)
                # Log if circuit breaker was just closed
                if was_open:
                    logger.info(f"[GPU-SCHEDULER] Re-enabling GPU interface after cooldown")
            else:
                self._is_available = False
                self.gpu_logger.log_health_check(False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self._is_available = False
            error_type = self._classify_error_type(e)
            self._last_error_type = error_type
            self._failure_count += 1
            self.gpu_logger.log_health_check(False, str(e))
            
            # Open circuit breaker after 3 failures (60s cooldown)
            if self._failure_count >= 3:
                cooldown_seconds = 60.0
                self._circuit_open_until = current_time + cooldown_seconds
                self.gpu_logger.log_circuit_breaker_open(f"{self._failure_count} failures")
                logger.warning(f"[GPU-SCHEDULER] Disabling GPU interface after error type={error_type}, cooldown={cooldown_seconds}s")
        
        self._last_health_check = current_time
        return self._is_available
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 with validation."""
        try:
            # Validate image bytes first
            if not image_bytes or len(image_bytes) < 10:
                raise ValueError("Invalid image data: too small")
            
            # Check for common image headers
            if not (image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                    image_bytes.startswith(b'\x89PNG') or       # PNG
                    image_bytes.startswith(b'GIF8') or          # GIF
                    image_bytes.startswith(b'BM')):            # BMP
                raise ValueError("Invalid image data: unrecognized format")
            
            encoded = base64.b64encode(image_bytes).decode('utf-8')
            logger.debug(f"Encoded image: {len(image_bytes)} bytes -> {len(encoded)} chars")
            return encoded
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            logger.error(f"Image bytes length: {len(image_bytes) if image_bytes else 0}")
            raise
    
    def _decode_face_detection(self, detection_data: Dict[str, Any]) -> FaceDetection:
        """Decode face detection from API response."""
        return FaceDetection(
            bbox=detection_data["bbox"],
            landmarks=detection_data.get("landmarks", []),
            embedding=detection_data.get("embedding"),
            quality=detection_data["quality"],
            age=detection_data.get("age"),
            gender=detection_data.get("gender")
        )
    
    async def _gpu_worker_request(self, image_tasks: List[ImageTask]) -> Optional[Dict[str, List[FaceDetection]]]:
        """Make request to GPU worker using multipart/form-data (no base64 encoding)."""
        try:
            try:
                loop = asyncio.get_running_loop()
                loop_id = id(loop)
                logger.info(f"[GPU-INTERFACE] [TRACE] _gpu_worker_request() called - event loop: id={loop_id}")
            except RuntimeError:
                logger.error(f"[GPU-INTERFACE] [TRACE] _gpu_worker_request() called - NO event loop!")
                raise RuntimeError("No event loop running in _gpu_worker_request")
            # Load images from temp paths
            valid_tasks = []
            for task in image_tasks:
                try:
                    with open(task.temp_path, 'rb') as f:
                        image_bytes = f.read()
                    if image_bytes:
                        valid_tasks.append((task, image_bytes))
                except Exception as e:
                    logger.error(f"Failed to load image from {task.temp_path}: {e}")
                    continue
            
            if not valid_tasks:
                logger.warning("No valid images in batch")
                return None
            
            # Prepare multipart form data
            # Build image_hashes JSON: [{"phash": "abc123", "index": 0}, ...]
            image_hashes = []
            files_data = []
            
            for idx, (task, image_bytes) in enumerate(valid_tasks):
                # Add phash mapping
                image_hashes.append({
                    "phash": task.phash,
                    "index": idx
                })
                # Add file data (binary, no encoding)
                files_data.append(
                    ("images", (f"{task.phash}.jpg", image_bytes, "image/jpeg"))
                )
            
            # Prepare form data
            form_data = {
                "image_hashes": json.dumps(image_hashes),
                "min_face_quality": str(self.config.min_face_quality),
                "require_face": "false",
                "crop_faces": "true",
                "face_margin": str(self.config.face_margin)
            }
            
            # Make multipart request
            logger.info(f"[GPU-INTERFACE] [TRACE] About to call _get_client() to get httpx client")
            client = await self._get_client()
            logger.info(f"[GPU-INTERFACE] [TRACE] Got httpx client, about to make POST request")
            start_time = time.time()
            
            self.gpu_logger.log_request_start(len(valid_tasks), f"{self.config.gpu_worker_host_resolved}/detect_faces_batch_multipart")
            # No encoding needed for multipart (saves CPU cycles)
            
            response = await client.post(
                f"{self.config.gpu_worker_host_resolved}/detect_faces_batch_multipart",
                files=files_data,
                data=form_data
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result_data = response.json()
                self.gpu_logger.log_result_decoding_start(len(str(result_data)))
                
                # Results are already keyed by phash: {"results": {"phash1": [...faces...], ...}}
                results_dict = result_data.get("results", {})
                
                logger.info(f"[GPU-INTERFACE] [TRACE] GPU worker returned results for {len(results_dict)} images")
                
                # Log raw response structure for debugging
                total_raw_faces = sum(len(face_list) for face_list in results_dict.values())
                logger.info(f"[GPU-INTERFACE] [TRACE] Raw GPU response: {total_raw_faces} total faces across {len(results_dict)} images")
                
                # Convert to our format (results_dict already has FaceDetection objects)
                # But we need to ensure they're properly converted
                converted_results = {}
                total_faces_before_pose = 0
                for phash, face_list in results_dict.items():
                    face_detections = []
                    for detection_data in face_list:
                        # Detection data may be dict or already FaceDetection
                        if isinstance(detection_data, dict):
                            face_detection = self._decode_face_detection(detection_data)
                            # Log quality score for debugging
                            quality = detection_data.get("quality", detection_data.get("det_score", "unknown"))
                            logger.debug(f"[GPU-INTERFACE] [TRACE] Decoded face for {phash[:8]}...: quality={quality}")
                        else:
                            # Already FaceDetection object from GPU worker
                            face_detection = detection_data
                            quality = getattr(face_detection, "quality", "unknown")
                            logger.debug(f"[GPU-INTERFACE] [TRACE] Face object for {phash[:8]}...: quality={quality}")
                        face_detections.append(face_detection)
                        total_faces_before_pose += 1
                    converted_results[phash] = face_detections
                    if face_list:
                        logger.info(f"[GPU-INTERFACE] [TRACE] Image {phash[:8]}... has {len(face_list)} faces before pose filtering")

                logger.info(f"[GPU-INTERFACE] [TRACE] Total faces before pose filtering: {total_faces_before_pose}")
                converted_results, pose_stats = self._filter_faces_by_pose(converted_results)
                self._log_pose_filter_stats(pose_stats, source="gpu")
                
                total_faces_after_pose = sum(len(faces) for faces in converted_results.values())
                logger.info(f"[GPU-INTERFACE] [TRACE] Total faces after pose filtering: {total_faces_after_pose} (rejected: {pose_stats.get('rejected', 0)})")
                
                gpu_used = result_data.get('gpu_used', False)
                worker_id = result_data.get('worker_id', 'unknown')
                
                # Count total faces found
                total_faces = sum(len(faces) for faces in converted_results.values())
                
                processed_count = len(converted_results)
                
                logger.info(f"✓ GPU worker processed {processed_count}/{len(image_tasks)} images "
                           f"via multipart (no encoding) in {processing_time:.1f}ms "
                           f"(GPU: {gpu_used}, worker: {worker_id})")
                
                # Use GPU worker logger
                self.gpu_logger.log_result_decoding_complete(total_faces)
                self.gpu_logger.log_request_complete(processed_count, total_faces, processing_time, gpu_used)
                
                # Update metrics
                self._request_count += 1
                self._success_count += 1
                self._total_latency += processing_time
                
                return converted_results
            else:
                logger.warning(f"GPU worker returned status {response.status_code}")
                self.gpu_logger.log_http_error(response.status_code, "Bad response")
                self._request_count += 1
                self._failure_count += 1
                return None
                
        except Exception as e:
            try:
                loop = asyncio.get_running_loop()
                loop_id = id(loop)
                logger.error(f"[GPU-INTERFACE] [TRACE] Exception in _gpu_worker_request - event loop: id={loop_id}")
            except RuntimeError:
                logger.error(f"[GPU-INTERFACE] [TRACE] Exception in _gpu_worker_request - NO event loop!")
            
            logger.error(f"[GPU-INTERFACE] [TRACE] Exception details: {type(e).__name__}: {e}")
            logger.error(f"[GPU-INTERFACE] [TRACE] Full traceback:\n{traceback.format_exc()}")
            logger.warning(f"GPU worker request failed: {e}")
            self.gpu_logger.log_request_failed(str(e))
            self._request_count += 1
            self._failure_count += 1
            return None
    
    async def _get_cpu_app(self):
        """Get or load CPU face detection model (thread-safe, cached)."""
        # Create lock lazily in the correct event loop context
        if self._cpu_app_lock is None:
            self._cpu_app_lock = asyncio.Lock()
        
        if self._cpu_app is None:
            async with self._cpu_app_lock:
                # Double-check after acquiring lock
                if self._cpu_app is None:
                    from insightface.app import FaceAnalysis
                    home = os.path.expanduser("~/.insightface")
                    os.makedirs(home, exist_ok=True)
                    logger.debug("CPU fallback: Loading face detection model (first time)")
                    app = FaceAnalysis(name="buffalo_l", root=home)
                    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU: ctx_id=-1
                    self._cpu_app = app
                    logger.debug("CPU fallback: Face detection model loaded successfully")
        return self._cpu_app
    
    async def _preload_images(self, image_tasks: List[ImageTask]) -> List[Tuple[ImageTask, bytes]]:
        """Preload all images into memory with parallel async I/O and validation.
        
        Returns list of (task, image_bytes) tuples for valid images only.
        Filters out invalid images early to avoid wasted processing.
        """
        async def load_single_image(task: ImageTask) -> Optional[Tuple[ImageTask, bytes]]:
            """Load and validate a single image."""
            try:
                if not task.phash or not task.temp_path:
                    return None
                
                # Read file asynchronously
                loop = asyncio.get_event_loop()
                def read_file():
                    if not os.path.exists(task.temp_path):
                        return None
                    with open(task.temp_path, 'rb') as f:
                        return f.read()
                image_bytes = await loop.run_in_executor(None, read_file)
                
                if not image_bytes or len(image_bytes) < 10:
                    return None
                
                # Quick format validation (same as _validate_image_task)
                if not (image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                        image_bytes.startswith(b'\x89PNG') or       # PNG
                        image_bytes.startswith(b'GIF8') or          # GIF
                        image_bytes.startswith(b'BM')):            # BMP
                    return None
                
                return (task, image_bytes)
            except Exception as e:
                logger.debug(f"CPU fallback: Failed to preload {task.phash[:8] if task.phash else 'unknown'}...: {e}")
                return None
        
        # Load all images in parallel (no semaphore needed for I/O)
        results = await asyncio.gather(*[load_single_image(task) for task in image_tasks], return_exceptions=True)
        
        # Filter out None and exceptions
        valid_images = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if result is not None:
                valid_images.append(result)
        
        return valid_images
    
    def _process_single_image_cpu(self, task: ImageTask, app) -> Tuple[str, List[FaceDetection], int, int]:
        """Process a single image using CPU fallback (sync function for thread execution).
        
        Returns: (phash, face_detections, faces_found, faces_skipped)
        """
        try:
            # Validate phash exists
            if not task.phash:
                logger.error(f"CPU fallback: Image task missing phash, skipping")
                return (None, [], 0, 0)
            
            # Load image from temp path
            if not os.path.exists(task.temp_path):
                logger.warning(f"CPU fallback: Image file not found for {task.phash[:8]}...: {task.temp_path}")
                return (task.phash, [], 0, 0)
            
            with open(task.temp_path, 'rb') as f:
                image_bytes = f.read()
            
            if not image_bytes or len(image_bytes) < 10:
                logger.warning(f"CPU fallback: Invalid image data for {task.phash[:8]}... (size: {len(image_bytes) if image_bytes else 0})")
                return (task.phash, [], 0, 0)
            
            # Read image using OpenCV (same as GPU worker preprocessing)
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if img is None:
                # Fallback to PIL if OpenCV fails
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                logger.warning(f"CPU fallback: Failed to decode image for {task.phash[:8]}...")
                return (task.phash, [], 0, 0)
            
            # Detect faces using InsightFace
            faces = app.get(img)
            
            # Convert to FaceDetection objects with quality filtering (same as GPU)
            face_detections = []
            faces_found = 0
            faces_skipped = 0
            
            for face in faces:
                # Check if face has required attributes
                if not hasattr(face, "embedding") or face.embedding is None:
                    faces_skipped += 1
                    continue
                
                # Get detection score (quality) - same threshold as GPU
                det_score = float(getattr(face, "det_score", 0.0))
                
                # Apply quality threshold (EXACT same as GPU: min_face_quality)
                if det_score < self.config.min_face_quality:
                    logger.debug(f"CPU fallback: Face quality {det_score:.3f} below threshold {self.config.min_face_quality} for {task.phash[:8]}...")
                    faces_skipped += 1
                    continue
                
                # Get bounding box (convert to list format matching GPU)
                bbox = face.bbox.tolist() if hasattr(face.bbox, 'tolist') else list(face.bbox)
                
                # Get landmarks (keypoints) - same format as GPU
                landmarks = []
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.tolist() if hasattr(face.kps, 'tolist') else list(face.kps)
                
                # Get embedding (same format as GPU)
                embedding = face.embedding.tolist() if hasattr(face.embedding, 'tolist') else face.embedding
                
                # Get age and gender if available (same as GPU)
                age = None
                if hasattr(face, 'age') and face.age is not None:
                    try:
                        age = int(face.age)
                    except (ValueError, TypeError):
                        pass
                
                gender = None
                if hasattr(face, 'gender') and face.gender is not None:
                    try:
                        gender = str(face.gender)
                    except (ValueError, TypeError):
                        pass
                
                # Create FaceDetection object (EXACT same structure as GPU)
                face_detection = FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    embedding=embedding,
                    quality=det_score,  # Same quality field as GPU
                    age=age,
                    gender=gender
                )
                face_detections.append(face_detection)
                faces_found += 1
            
            return (task.phash, face_detections, faces_found, faces_skipped)
            
        except Exception as e:
            logger.error(f"CPU fallback: Error processing image {task.phash[:8] if task.phash else 'unknown'}...: {e}", exc_info=True)
            return (task.phash if task.phash else None, [], 0, 0)
    
    def _process_single_image_cpu_preloaded(self, task: ImageTask, image_bytes: bytes, app) -> Tuple[str, List[FaceDetection], int, int]:
        """Process a preloaded image using CPU fallback (no disk I/O).
        
        Returns: (phash, face_detections, faces_found, faces_skipped)
        """
        try:
            if not task.phash:
                return (None, [], 0, 0)
            
            # Decode image directly from bytes (no file I/O)
            # Try OpenCV first (faster for common formats)
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if img is None:
                # Fallback to PIL (more reliable for edge cases)
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.debug(f"CPU fallback: Failed to decode image {task.phash[:8]}...: {e}")
                    return (task.phash, [], 0, 0)
            
            if img is None:
                return (task.phash, [], 0, 0)
            
            # Detect faces using InsightFace
            faces = app.get(img)
            
            # Convert to FaceDetection objects with quality filtering
            face_detections = []
            faces_found = 0
            faces_skipped = 0
            
            for face in faces:
                if not hasattr(face, "embedding") or face.embedding is None:
                    logger.debug(f"CPU fallback: Face missing embedding for {task.phash[:8]}...")
                    faces_skipped += 1
                    continue
                
                det_score = float(getattr(face, "det_score", 0.0))
                if det_score < self.config.min_face_quality:
                    logger.debug(f"CPU fallback: Face quality {det_score:.3f} < {self.config.min_face_quality} for {task.phash[:8]}...")
                    faces_skipped += 1
                    continue
                
                logger.debug(f"CPU fallback: Face passed quality check: score={det_score:.3f} >= {self.config.min_face_quality} for {task.phash[:8]}...")
                
                # Efficient conversion (avoid redundant checks)
                bbox = face.bbox.tolist() if hasattr(face.bbox, 'tolist') else list(face.bbox)
                landmarks = face.kps.tolist() if (hasattr(face, 'kps') and face.kps is not None and hasattr(face.kps, 'tolist')) else (list(face.kps) if (hasattr(face, 'kps') and face.kps is not None) else [])
                embedding = face.embedding.tolist() if hasattr(face.embedding, 'tolist') else face.embedding
                
                # Extract age/gender with minimal overhead
                age = None
                if hasattr(face, 'age') and face.age is not None:
                    try:
                        age = int(face.age)
                    except (ValueError, TypeError):
                        pass
                
                gender = None
                if hasattr(face, 'gender') and face.gender is not None:
                    try:
                        gender = str(face.gender)
                    except (ValueError, TypeError):
                        pass
                
                face_detection = FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    embedding=embedding,
                    quality=det_score,
                    age=age,
                    gender=gender
                )
                face_detections.append(face_detection)
                faces_found += 1
            
            return (task.phash, face_detections, faces_found, faces_skipped)
            
        except Exception as e:
            logger.error(f"CPU fallback: Error processing image {task.phash[:8] if task.phash else 'unknown'}...: {e}", exc_info=True)
            return (task.phash if task.phash else None, [], 0, 0)
    
    async def _cpu_fallback(self, image_tasks: List[ImageTask], batch_id: Optional[str] = None) -> Dict[str, List[FaceDetection]]:
        """CPU fallback with preloading and optimized processing pipeline.
        
        Pipeline stages:
        1. Preload all images into memory (parallel async I/O)
        2. Process images in parallel (CPU-bound, limited by cores)
        3. Collect and format results
        
        Returns dict keyed by phash: {phash: [FaceDetection, ...], ...}
        """
        if not image_tasks:
            return {}
        
        import os as os_module
        cpu_count = os_module.cpu_count() or 4
        max_workers = min(len(image_tasks), cpu_count)
        
        logger.debug(f"CPU fallback: Processing {len(image_tasks)} images (max {max_workers} workers)")
        
        try:
            # Stage 1: Preload all images into memory (parallel async I/O)
            preload_start = time.time()
            valid_images = await self._preload_images(image_tasks)
            preload_time = (time.time() - preload_start) * 1000
            
            if not valid_images:
                logger.warning(f"CPU fallback: No valid images after preloading")
                return {task.phash: [] for task in image_tasks if task.phash}
            
            logger.debug(f"CPU fallback: Preloaded {len(valid_images)}/{len(image_tasks)} images in {preload_time:.1f}ms")
            
            # Stage 2: Load model (cached for subsequent calls)
            app = await self._get_cpu_app()
            
            # Stage 3: Process images in parallel (CPU-bound, limited by cores)
            process_start = time.time()
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_with_semaphore(task_and_bytes):
                task, image_bytes = task_and_bytes
                async with semaphore:
                    return await asyncio.to_thread(
                        self._process_single_image_cpu_preloaded, 
                        task, 
                        image_bytes, 
                        app
                    )
            
            # Process all valid images in parallel
            results = await asyncio.gather(
                *[process_with_semaphore(img_data) for img_data in valid_images], 
                return_exceptions=True
            )
            
            processing_time = (time.time() - process_start) * 1000
            
            # Stage 4: Collect results efficiently
            converted_results = {}
            total_faces = 0
            processed_count = 0
            failed_count = 0
            skipped_count = 0
            
            # Initialize with all tasks (even invalid ones get empty results)
            for task in image_tasks:
                if task.phash:
                    converted_results[task.phash] = []
            
            # Process results in single pass
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.debug(f"CPU fallback: Exception processing image: {result}")
                    failed_count += 1
                    continue
                
                if result is None or result[0] is None:
                    failed_count += 1
                    continue
                
                phash, face_detections, faces_found, faces_skipped = result
                if phash:
                    converted_results[phash] = face_detections
                    total_faces += faces_found
                    skipped_count += faces_skipped
                    processed_count += 1
            
            total_time = (time.time() - preload_start) * 1000
            
            logger.info(f"✓ CPU fallback: Processed {processed_count}/{len(image_tasks)} images "
                       f"({max_workers} workers) in {total_time:.1f}ms "
                       f"(preload: {preload_time:.1f}ms, process: {processing_time:.1f}ms, "
                       f"{total_faces} faces, {failed_count} failed, {skipped_count} skipped)")
            
            logger.info(f"[GPU-INTERFACE] [TRACE] CPU fallback: {total_faces} faces before pose filtering, "
                       f"quality_threshold={self.config.min_face_quality}, "
                       f"yaw_limit={self.config.max_face_yaw_deg}°, pitch_limit={self.config.max_face_pitch_deg}°")
            
            filtered_results, pose_stats = self._filter_faces_by_pose(converted_results)
            self._log_pose_filter_stats(pose_stats, source="cpu-fallback")
            
            total_faces_after_pose = sum(len(faces) for faces in filtered_results.values())
            logger.info(f"[GPU-INTERFACE] [TRACE] CPU fallback: {total_faces_after_pose} faces after pose filtering "
                       f"(rejected: {pose_stats.get('rejected', 0)})")
            logger.info(f"[GPU-INTERFACE] [TRACE] CPU fallback completed: returning {len(filtered_results)} image results")
            
            # Log FALLBACK-DONE with batch_id if provided
            if batch_id:
                self.gpu_logger.log_fallback_done(batch_id, len(image_tasks), total_faces_after_pose, total_time)
            else:
                # Fallback to old method if no batch_id
                self.gpu_logger.log_fallback_complete(len(image_tasks), total_faces_after_pose, total_time)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return {task.phash: [] for task in image_tasks if task.phash}

    def _filter_faces_by_pose(self, results: Dict[str, List[FaceDetection]]) -> Tuple[Dict[str, List[FaceDetection]], Dict[str, int]]:
        """Filter face detections using lightweight pose heuristics."""
        if not results:
            logger.debug(f"[GPU-INTERFACE] [TRACE] _filter_faces_by_pose: No results to filter")
            return results, {"total": 0, "rejected": 0}
        
        filtered: Dict[str, List[FaceDetection]] = {}
        total = 0
        rejected = 0
        
        for phash, faces in results.items():
            kept: List[FaceDetection] = []
            for face in faces:
                total += 1
                quality = getattr(face, "quality", "unknown")
                yaw, pitch = self._extract_pose_angles(face)
                logger.debug(f"[GPU-INTERFACE] [TRACE] Filtering face for {phash[:8]}...: quality={quality}, yaw={yaw}, pitch={pitch}")
                
                if self._pose_within_limits(face):
                    kept.append(face)
                    logger.debug(f"[GPU-INTERFACE] [TRACE] Face kept for {phash[:8]}...")
                else:
                    rejected += 1
                    logger.debug(f"[GPU-INTERFACE] [TRACE] Face rejected for {phash[:8]}... (yaw={yaw}, pitch={pitch})")
            filtered[phash] = kept
        
        logger.info(f"[GPU-INTERFACE] [TRACE] Pose filter: {total} total, {rejected} rejected, {total - rejected} kept")
        return filtered, {"total": total, "rejected": rejected}

    def _log_pose_filter_stats(self, stats: Dict[str, int], source: str) -> None:
        """Log pose rejection stats for observability."""
        rejected = stats.get("rejected", 0)
        total = stats.get("total", 0)
        if total > 0:
            logger.info(
                f"Pose filter: {total} total faces, {rejected} rejected, {total - rejected} kept "
                f"(source={source}, yaw_limit={self.config.max_face_yaw_deg}°, "
                f"pitch_limit={self.config.max_face_pitch_deg}°)"
            )
        elif rejected > 0:
            logger.warning(
                f"Pose filter rejected {rejected} faces but total was 0 (inconsistent stats)"
            )

    def _pose_within_limits(self, face: FaceDetection) -> bool:
        """Return True when pose is within configured limits or unavailable."""
        yaw, pitch = self._extract_pose_angles(face)
        
        # If pose angles are unavailable (None), allow the face (don't reject due to missing data)
        if yaw is None and pitch is None:
            return True  # No pose data available, allow face
        
        if yaw is not None and abs(yaw) > self.config.max_face_yaw_deg:
            logger.debug(f"Pose filter: Rejecting face - yaw={yaw:.1f}° > {self.config.max_face_yaw_deg}°")
            return False
        if pitch is not None and abs(pitch) > self.config.max_face_pitch_deg:
            logger.debug(f"Pose filter: Rejecting face - pitch={pitch:.1f}° > {self.config.max_face_pitch_deg}°")
            return False
        return True

    def _extract_pose_angles(self, face: FaceDetection) -> Tuple[Optional[float], Optional[float]]:
        """Get pose angles directly or estimate from landmarks."""
        yaw = getattr(face, "yaw", None)
        pitch = getattr(face, "pitch", None)
        
        if yaw is not None or pitch is not None:
            try:
                yaw_val = float(yaw) if yaw is not None else None
                pitch_val = float(pitch) if pitch is not None else None
                return yaw_val, pitch_val
            except (TypeError, ValueError):
                pass
        
        return self._estimate_pose_from_landmarks(face.landmarks)

    def _estimate_pose_from_landmarks(self, landmarks: List[List[float]]) -> Tuple[Optional[float], Optional[float]]:
        """Estimate yaw/pitch using five-point landmarks."""
        if not landmarks or len(landmarks) < 5:
            return None, None
        
        try:
            left_eye, right_eye, nose, mouth_left, mouth_right = landmarks[:5]
        except (ValueError, TypeError):
            return None, None
        
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye
        nose_x, nose_y = nose
        mouth_left_y = mouth_left[1]
        mouth_right_y = mouth_right[1]
        
        eye_dx = right_eye_x - left_eye_x
        eye_mid_x = (left_eye_x + right_eye_x) / 2.0
        eye_mid_y = (left_eye_y + right_eye_y) / 2.0
        
        yaw = None
        if abs(eye_dx) > 1e-5:
            yaw = math.degrees(math.atan((nose_x - eye_mid_x) / eye_dx))
        
        mouth_mid_y = (mouth_left_y + mouth_right_y) / 2.0
        vertical_span = mouth_mid_y - eye_mid_y
        pitch = None
        if abs(vertical_span) > 1e-5:
            reference_y = (eye_mid_y + mouth_mid_y) / 2.0
            pitch = math.degrees(math.atan((nose_y - reference_y) / vertical_span))
        
        return yaw, pitch
    
    def _classify_error_type(self, error: Exception) -> str:
        """Classify error type for logging purposes."""
        error_str = str(error)
        error_type = type(error).__name__
        
        if "bound to a different event loop" in error_str or "event loop" in error_str.lower():
            return "LoopMismatch"
        elif isinstance(error, (ConnectionError, OSError)) or "connection" in error_str.lower():
            return "ConnectionError"
        elif isinstance(error, TimeoutError) or "timeout" in error_str.lower():
            return "Timeout"
        elif hasattr(httpx, 'HTTPStatusError') and isinstance(error, httpx.HTTPStatusError):
            return "HTTPError"
        elif "http" in error_str.lower() or "status" in error_str.lower() or error_type == "HTTPStatusError":
            return "HTTPError"
        elif "processing" in error_str.lower() or "detection" in error_str.lower():
            return "ProcessingError"
        else:
            return "Unknown"
    
    async def process_batch(self, image_tasks: List[ImageTask], batch_id: Optional[str] = None) -> Optional[Dict[str, List[FaceDetection]]]:
        """Process a batch of images with GPU worker and CPU fallback.
        
        Args:
            image_tasks: List of image tasks to process
            batch_id: Optional batch ID for logging correlation
        
        Returns dict keyed by phash: {phash: [FaceDetection, ...], ...}
        """
        batch_start_time = time.time()
        
        # Generate batch_id if not provided
        if batch_id is None:
            batch_id = f"batch_{int(time.time()*1000)}"
        
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            logger.info(f"[GPU-INTERFACE] [TRACE] process_batch() called - event loop: id={loop_id}, images={len(image_tasks)}, batch_id={batch_id}")
        except RuntimeError:
            logger.error(f"[GPU-INTERFACE] [TRACE] process_batch() called - NO event loop!")
        
        if not image_tasks:
            return {}
        
        logger.debug(f"Processing batch of {len(image_tasks)} images")
        self.gpu_logger.log_batch_start(batch_id, len(image_tasks))
        
        # Check if GPU worker is enabled
        if not self.config.gpu_worker_enabled:
            logger.info("GPU worker disabled, using CPU fallback")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            result = await self._cpu_fallback(image_tasks, batch_id)
            duration_ms = (time.time() - batch_start_time) * 1000
            total_faces = sum(len(faces) for faces in result.values()) if result else 0
            self.gpu_logger.log_fallback_done(batch_id, len(image_tasks), total_faces, duration_ms)
            return result
        
        # Check circuit breaker FIRST (avoid unnecessary health check)
        current_time = time.time()
        if current_time < self._circuit_open_until:
            logger.debug(f"Circuit breaker open, using CPU fallback (opens until {self._circuit_open_until - current_time:.1f}s)")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            result = await self._cpu_fallback(image_tasks, batch_id)
            duration_ms = (time.time() - batch_start_time) * 1000
            total_faces = sum(len(faces) for faces in result.values()) if result else 0
            self.gpu_logger.log_fallback_done(batch_id, len(image_tasks), total_faces, duration_ms)
            return result
        
        # Check GPU worker health (only if circuit breaker is closed)
        if not await self._check_health():
            logger.warning("GPU worker not available, using CPU fallback")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            result = await self._cpu_fallback(image_tasks, batch_id)
            duration_ms = (time.time() - batch_start_time) * 1000
            total_faces = sum(len(faces) for faces in result.values()) if result else 0
            self.gpu_logger.log_fallback_done(batch_id, len(image_tasks), total_faces, duration_ms)
            return result
        
        # Try GPU worker first (httpx client timeout handles hung requests)
        try:
            # Let httpx client timeout handle it (configured to gpu_worker_timeout, typically 60s)
            # This allows legitimate batch processing (11-30s) to complete while still
            # timing out on truly hung requests
            results = await self._gpu_worker_request(image_tasks)
            
            if results is not None:
                duration_ms = (time.time() - batch_start_time) * 1000
                total_faces = sum(len(faces) for faces in results.values()) if results else 0
                self.gpu_logger.log_batch_result_ok(batch_id, len(image_tasks), total_faces, duration_ms)
                return results
        except Exception as e:
            error_type = self._classify_error_type(e)
            logger.warning(f"GPU worker request failed: {e}")
            self.gpu_logger.log_request_failed(str(e))
            duration_ms = (time.time() - batch_start_time) * 1000
            self.gpu_logger.log_batch_result_error(batch_id, error_type, "CPU", duration_ms)
        
        # Fallback to CPU
        logger.info("Falling back to CPU processing")
        self.gpu_logger.log_fallback_start(len(image_tasks))
        result = await self._cpu_fallback(image_tasks, batch_id)
        duration_ms = (time.time() - batch_start_time) * 1000
        total_faces = sum(len(faces) for faces in result.values()) if result else 0
        self.gpu_logger.log_fallback_done(batch_id, len(image_tasks), total_faces, duration_ms)
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        success_rate = (self._success_count / self._request_count * 100) if self._request_count > 0 else 0
        avg_latency = (self._total_latency / self._success_count) if self._success_count > 0 else 0
        
        return {
            'request_count': self._request_count,
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'gpu_worker_available': self._is_available,
            'gpu_worker_enabled': self.config.gpu_worker_enabled
        }
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._client_event_loop_id = None
            self._client_event_loop_id = None




def get_gpu_interface() -> GPUInterface:
    """Get singleton GPU interface instance."""
    global _gpu_interface_instance
    if _gpu_interface_instance is None:
        _gpu_interface_instance = GPUInterface()
    return _gpu_interface_instance


async def close_gpu_interface():
    """Close singleton GPU interface."""
    global _gpu_interface_instance
    if _gpu_interface_instance:
        await _gpu_interface_instance.close()
        _gpu_interface_instance = None
