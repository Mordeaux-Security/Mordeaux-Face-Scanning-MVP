"""
GPU Interface for New Crawler System

Provides GPU worker client with CPU fallback for face detection and embedding.
Handles consistent data structures and proper batching to avoid single-image requests.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import List, Optional, Dict, Any, Tuple
import httpx
import numpy as np
import cv2
from PIL import Image
import io

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
        self.gpu_logger = GPUWorkerLogger(0)  # Use 0 as default worker ID
        
        # Circuit breaker
        self._is_available = False
        self._failure_count = 0
        self._last_health_check = 0
        self._circuit_open_until = 0
        
        # Metrics tracking
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        
        # CPU fallback model cache (lazy-loaded and reused)
        self._cpu_app = None
        self._cpu_app_lock = asyncio.Lock()
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
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
                self._failure_count = 0
                # Reset circuit breaker on success
                self._circuit_open_until = 0
                self.gpu_logger.log_health_check(True)
            else:
                self._is_available = False
                self.gpu_logger.log_health_check(False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self._is_available = False
            self._failure_count += 1
            self.gpu_logger.log_health_check(False, str(e))
            
            # Open circuit breaker after 3 failures (60s cooldown)
            if self._failure_count >= 3:
                self._circuit_open_until = current_time + 60.0
                self.gpu_logger.log_circuit_breaker_open(f"{self._failure_count} failures")
        
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
            client = await self._get_client()
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
                
                # Convert to our format (results_dict already has FaceDetection objects)
                # But we need to ensure they're properly converted
                converted_results = {}
                for phash, face_list in results_dict.items():
                    face_detections = []
                    for detection_data in face_list:
                        # Detection data may be dict or already FaceDetection
                        if isinstance(detection_data, dict):
                            face_detection = self._decode_face_detection(detection_data)
                        else:
                            # Already FaceDetection object from GPU worker
                            face_detection = detection_data
                        face_detections.append(face_detection)
                    converted_results[phash] = face_detections
                
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
            logger.warning(f"GPU worker request failed: {e}")
            self.gpu_logger.log_request_failed(str(e))
            self._request_count += 1
            self._failure_count += 1
            return None
    
    async def _get_cpu_app(self):
        """Get or load CPU face detection model (thread-safe, cached)."""
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
                    faces_skipped += 1
                    continue
                
                det_score = float(getattr(face, "det_score", 0.0))
                if det_score < self.config.min_face_quality:
                    faces_skipped += 1
                    continue
                
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
    
    async def _cpu_fallback(self, image_tasks: List[ImageTask]) -> Dict[str, List[FaceDetection]]:
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
            
            # Log completion
            self.gpu_logger.log_fallback_complete(len(image_tasks), total_faces, total_time)
            
            logger.info(f"✓ CPU fallback: Processed {processed_count}/{len(image_tasks)} images "
                       f"({max_workers} workers) in {total_time:.1f}ms "
                       f"(preload: {preload_time:.1f}ms, process: {processing_time:.1f}ms, "
                       f"{total_faces} faces, {failed_count} failed, {skipped_count} skipped)")
            
            return converted_results
            
        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return {task.phash: [] for task in image_tasks if task.phash}
    
    async def process_batch(self, image_tasks: List[ImageTask]) -> Optional[Dict[str, List[FaceDetection]]]:
        """Process a batch of images with GPU worker and CPU fallback.
        
        Returns dict keyed by phash: {phash: [FaceDetection, ...], ...}
        """
        if not image_tasks:
            return {}
        
        logger.debug(f"Processing batch of {len(image_tasks)} images")
        self.gpu_logger.log_batch_start(f"batch_{int(time.time())}", len(image_tasks))
        
        # Check if GPU worker is enabled
        if not self.config.gpu_worker_enabled:
            logger.info("GPU worker disabled, using CPU fallback")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            return await self._cpu_fallback(image_tasks)
        
        # Check circuit breaker FIRST (avoid unnecessary health check)
        current_time = time.time()
        if current_time < self._circuit_open_until:
            logger.debug(f"Circuit breaker open, using CPU fallback (opens until {self._circuit_open_until - current_time:.1f}s)")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            return await self._cpu_fallback(image_tasks)
        
        # Check GPU worker health (only if circuit breaker is closed)
        if not await self._check_health():
            logger.warning("GPU worker not available, using CPU fallback")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            return await self._cpu_fallback(image_tasks)
        
        # Try GPU worker first (with shorter timeout for fast failure)
        try:
            # Use shorter timeout for initial attempt to fail fast
            fast_timeout = min(10.0, self.config.gpu_worker_timeout)  # Max 10s for initial attempt
            
            # Create a timeout wrapper
            results = await asyncio.wait_for(
                self._gpu_worker_request(image_tasks),
                timeout=fast_timeout
            )
            
            if results is not None:
                return results
        except asyncio.TimeoutError:
            logger.warning(f"GPU worker request timed out after {fast_timeout}s, falling back to CPU")
            self.gpu_logger.log_request_failed("timeout")
        except Exception as e:
            logger.warning(f"GPU worker request failed: {e}")
            self.gpu_logger.log_request_failed(str(e))
        
        # Fallback to CPU
        logger.info("Falling back to CPU processing")
        self.gpu_logger.log_fallback_start(len(image_tasks))
        return await self._cpu_fallback(image_tasks)
    
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
