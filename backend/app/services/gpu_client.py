"""
GPU Client Service

Robust HTTP client for communicating with the Windows GPU worker service.
Features smart circuit breaker, connection pooling, adaptive retry logic,
and comprehensive monitoring for reliable GPU acceleration.
"""

import asyncio
import base64
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import statistics

import httpx
import numpy as np
from ..core.settings import get_settings

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

class CircuitBreaker:
    """Smart circuit breaker with adaptive thresholds."""
    
    def __init__(self, 
                 failure_threshold: int = 10, 
                 recovery_timeout: float = 30.0, 
                 success_threshold: int = 5,
                 min_requests: int = 20):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.min_requests = min_requests
        
        self.failure_count = 0
        self.success_count = 0
        self.total_requests = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        
        # Adaptive thresholds based on success rate
        self._base_failure_threshold = failure_threshold
        self._base_recovery_timeout = recovery_timeout
        
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit state."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker: Moving to HALF_OPEN state")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record a successful request."""
        self.total_requests += 1
        self.success_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker: Moving to CLOSED state (service recovered)")
        elif self.state == CircuitState.CLOSED:
            # Gradually reduce failure count on success
            self.failure_count = max(0, self.failure_count - 1)
            
            # Adapt thresholds based on success rate
            self._adapt_thresholds()
    
    def record_failure(self):
        """Record a failed request."""
        self.total_requests += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.total_requests >= self.min_requests and self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: Moving to OPEN state (failure count: {self.failure_count})")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker: Moving back to OPEN state (service still failing)")
    
    def _adapt_thresholds(self):
        """Adapt circuit breaker thresholds based on success rate."""
        if self.total_requests < self.min_requests:
            return
            
        success_rate = self.success_count / self.total_requests
        
        # If success rate is high, be more lenient
        if success_rate > 0.95:
            self.failure_threshold = min(self._base_failure_threshold * 2, 50)
            self.recovery_timeout = max(self._base_recovery_timeout * 0.5, 10.0)
        elif success_rate > 0.8:
            self.failure_threshold = int(self._base_failure_threshold * 1.5)
            self.recovery_timeout = self._base_recovery_timeout
        else:
            self.failure_threshold = self._base_failure_threshold
            self.recovery_timeout = self._base_recovery_timeout

@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: List[float]
    landmarks: List[List[float]]
    embedding: Optional[List[float]]
    quality: float
    age: Optional[int] = None
    gender: Optional[str] = None

class GPUClient:
    """
    Robust HTTP client for GPU worker service with advanced error handling.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None
        self._is_available = False
        self._last_health_check = 0
        self._health_check_interval = 10.0  # Check health every 10 seconds
        
        # Smart circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=15,     # Higher threshold for stability
            recovery_timeout=20.0,   # Faster recovery
            success_threshold=3,     # Fewer successes needed to close
            min_requests=10          # Minimum requests before opening circuit
        )
        
        # Metrics tracking
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        self._latency_history = []
        self._last_metrics_log = 0
        self._current_batch_size = 0  # Track current batch size for metrics
        
        # Connection management
        self._connection_lock = asyncio.Lock()
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with optimized connection pooling."""
        if self._client is None:
            async with self._connection_lock:
                if self._client is None:
                    # Optimized timeout configuration
                    timeout = httpx.Timeout(
                        connect=10.0,  # Longer connection timeout
                        read=60.0,     # Allow time for GPU processing
                        write=15.0,     # Reasonable write timeout
                        pool=10.0      # Pool timeout
                    )
                    
                    # Connection limits optimized for GPU worker
                    limits = httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=20,
                        keepalive_expiry=120.0
                    )
                    
                    self._client = httpx.AsyncClient(
                        base_url=self.settings.gpu_worker_url,
                        timeout=timeout,
                        limits=limits,
                        http2=True,  # Enable HTTP/2 for better multiplexing
                        headers={
                            "Connection": "keep-alive",
                            "Keep-Alive": "timeout=120, max=1000",
                            "User-Agent": "Mordeaux-GPU-Client/2.0"
                        }
                    )
                    
                    logger.info("GPU client initialized with optimized connection pooling")
        
        return self._client
    
    async def _check_health(self) -> bool:
        """Check if GPU worker is available with caching."""
        current_time = time.time()
        
        # Skip if checked recently
        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_available
        
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            
            if response.status_code == 200:
                health_data = response.json()
                self._is_available = health_data.get("status") == "healthy"
                
                # Log worker info
                worker_id = health_data.get("worker_id", "unknown")
                queue_size = health_data.get("queue_size", 0)
                uptime = health_data.get("uptime_seconds", 0)
                
                logger.info(f"GPU worker health check: status={health_data.get('status')}, "
                           f"worker_id={worker_id}, queue_size={queue_size}, uptime={uptime:.1f}s")
                
                # Check if GPU is actually being used
                gpu_available = health_data.get("gpu_available", False)
                if not gpu_available:
                    logger.warning("GPU worker reports GPU not available")
                
            else:
                self._is_available = False
                logger.warning(f"GPU worker health check failed: {response.status_code}")
                
        except Exception as e:
            self._is_available = False
            logger.warning(f"GPU worker health check error: {e}")
            
            # Log specific error types for better debugging
            if "Connection refused" in str(e):
                logger.error(f"GPU worker not reachable at {self.settings.gpu_worker_url}. Is the Windows GPU worker running?")
            elif "timeout" in str(e).lower():
                logger.error(f"GPU worker timeout at {self.settings.gpu_worker_url}. Check network connectivity.")
            else:
                logger.error(f"GPU worker connection failed: {e}")
        
        self._last_health_check = current_time
        return self._is_available
    
    async def _make_request_with_retry(self, method: str, endpoint: str, **kwargs) -> Optional[httpx.Response]:
        """Make HTTP request with smart retry logic and circuit breaker."""
        if not self.settings.gpu_worker_enabled:
            logger.info("[GPU-WORKER-SKIP] GPU worker disabled in settings")
            return None
        
        # Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            logger.warning(f"[GPU-WORKER-SKIP] Circuit breaker is OPEN (state: {self._circuit_breaker.state.value})")
            return None
        
        # Check health first
        if not await self._check_health():
            logger.warning("[GPU-WORKER-SKIP] GPU worker health check failed")
            return None
        
        client = await self._get_client()
        start_time = time.time()
        
        # Log request details
        payload_size = 0
        if 'json' in kwargs:
            import json
            payload_size = len(json.dumps(kwargs['json']))
        
        logger.info(f"[GPU-WORKER-ATTEMPT] {method} {endpoint} (payload: {payload_size} bytes)")
        
        # Adaptive retry logic based on circuit state
        max_retries = self.settings.gpu_worker_max_retries
        if self._circuit_breaker.state == CircuitState.HALF_OPEN:
            max_retries = 1  # Only one retry in half-open state
        
        for attempt in range(max_retries + 1):
            try:
                response = await client.request(method, endpoint, **kwargs)
                
                if response.status_code == 200:
                    # Record success metrics
                    latency = time.time() - start_time
                    self._record_success(latency)
                    logger.info(f"[GPU-WORKER-SUCCESS] Request completed in {latency:.3f}s (response: {len(response.content)} bytes)")
                    return response
                elif response.status_code >= 500:
                    # Server error, retry with exponential backoff
                    if attempt < max_retries:
                        delay = self._calculate_backoff_delay(attempt)
                        logger.warning(f"[GPU-WORKER-RETRY] Server error {response.status_code}, retrying in {delay:.2f}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Client error (4xx), don't retry
                    logger.error(f"[GPU-WORKER-FAIL] Client error: {response.status_code}")
                    return None
                    
            except httpx.TimeoutException:
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"[GPU-WORKER-RETRY] Timeout, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
            except httpx.ConnectError as e:
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"[GPU-WORKER-RETRY] Connection error: {e}, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
            except Exception as e:
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"[GPU-WORKER-RETRY] Request error: {e}, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
        
        logger.error("[GPU-WORKER-FAIL] Request failed after all retries")
        self._record_failure()
        return None
    
    def _record_success(self, latency: float):
        """Record a successful request."""
        self._request_count += 1
        self._success_count += 1
        self._total_latency += latency
        self._latency_history.append(latency)
        
        # Keep only last 100 latency measurements
        if len(self._latency_history) > 100:
            self._latency_history = self._latency_history[-100:]
        
        self._circuit_breaker.record_success()
        self._log_metrics()
    
    def _record_failure(self):
        """Record a failed request."""
        self._request_count += 1
        self._failure_count += 1
        self._circuit_breaker.record_failure()
        self._log_metrics()
    
    
    def _log_metrics(self):
        """Log connection health metrics."""
        current_time = time.time()
        if current_time - self._last_metrics_log >= 30.0:
            if self._request_count > 0:
                success_rate = (self._success_count / self._request_count) * 100
                avg_latency = self._total_latency / self._success_count if self._success_count > 0 else 0
                
                # Calculate latency percentiles
                if self._latency_history:
                    p50 = statistics.median(self._latency_history)
                    p95 = statistics.quantiles(self._latency_history, n=20)[18] if len(self._latency_history) >= 20 else p50
                else:
                    p50 = p95 = 0
                
                throughput = self._success_count / 30.0  # requests per second over last 30s
                
                logger.info(f"GPU worker metrics: {self._success_count}/{self._request_count} requests successful "
                           f"({success_rate:.1f}%), avg latency: {avg_latency:.2f}s, p50: {p50:.2f}s, p95: {p95:.2f}s, "
                           f"throughput: {throughput:.1f} req/s, batch_size: {self._current_batch_size}, "
                           f"circuit state: {self._circuit_breaker.state.value}")
            self._last_metrics_log = current_time
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        import random
        
        # Base delays: 0.5s, 1s, 2s, 4s, 8s
        base_delay = 0.5 * (2 ** attempt)
        
        # Add jitter to prevent thundering herd (±25% random variation)
        jitter = random.uniform(-0.25, 0.25) * base_delay
        
        return max(0.1, base_delay + jitter)
    
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
            landmarks=detection_data["landmarks"],
            embedding=detection_data.get("embedding"),
            quality=detection_data["quality"],
            age=detection_data.get("age"),
            gender=detection_data.get("gender")
        )
    
    async def detect_faces_batch_async(
        self, 
        image_bytes_list: List[bytes], 
        min_face_quality: float = 0.5,
        require_face: bool = True,
        crop_faces: bool = True,
        face_margin: float = 0.2
    ) -> List[List[FaceDetection]]:
        """
        Detect faces in a batch of images using GPU worker.
        
        Args:
            image_bytes_list: List of image bytes
            min_face_quality: Minimum face quality threshold
            require_face: Whether to require at least one face
            crop_faces: Whether to crop face regions
            face_margin: Margin around face as fraction of face size
            
        Returns:
            List of face detection results for each image
        """
        if not self.settings.gpu_worker_enabled:
            logger.debug("GPU worker disabled, returning empty results")
            return [[] for _ in image_bytes_list]
        
        logger.info(f"Attempting GPU worker processing for {len(image_bytes_list)} images "
                   f"at {self.settings.gpu_worker_url}")
        
        # Update batch size for metrics
        self._current_batch_size = len(image_bytes_list)
        
        # Prepare request data
        images_data = []
        for i, image_bytes in enumerate(image_bytes_list):
            images_data.append({
                "data": self._encode_image(image_bytes),
                "image_id": f"image_{i}"
            })
        
        request_data = {
            "images": images_data,
            "min_face_quality": min_face_quality,
            "require_face": require_face,
            "crop_faces": crop_faces,
            "face_margin": face_margin
        }
        
        # Make request
        response = await self._make_request_with_retry(
            "POST", 
            "/detect_faces_batch", 
            json=request_data
        )
        
        if response is None:
            logger.warning("GPU worker request failed, falling back to CPU processing")
            return [[] for _ in image_bytes_list]
        
        try:
            result_data = response.json()
            results = []
            
            for image_results in result_data["results"]:
                face_detections = []
                for detection_data in image_results:
                    face_detection = self._decode_face_detection(detection_data)
                    face_detections.append(face_detection)
                results.append(face_detections)
            
            # Pad results for remaining images if batch was truncated
            while len(results) < len(image_bytes_list):
                results.append([])
            
            processing_time = result_data.get('processing_time_ms', 0)
            gpu_used = result_data.get('gpu_used', False)
            worker_id = result_data.get('worker_id', 'unknown')
            
            logger.info(f"✓ GPU worker successfully processed {len(image_bytes_list)} images in "
                       f"{processing_time:.1f}ms (GPU: {gpu_used}, worker: {worker_id})")
            return results
            
        except Exception as e:
            logger.error(f"Error parsing GPU worker response: {e}")
            return [[] for _ in image_bytes_list]
    
    async def detect_and_embed_batch_async(
        self, 
        image_bytes_list: List[bytes], 
        min_face_quality: float = 0.5,
        require_face: bool = True,
        crop_faces: bool = True,
        face_margin: float = 0.2
    ) -> List[List[FaceDetection]]:
        """
        Detect faces and compute embeddings in a batch of images.
        This is the same as detect_faces_batch_async since embeddings are included.
        """
        return await self.detect_faces_batch_async(
            image_bytes_list, min_face_quality, require_face, crop_faces, face_margin
        )
    
    async def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU worker information and metrics."""
        try:
            client = await self._get_client()
            response = await client.get("/gpu_info", timeout=10.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get GPU info: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return None
    
    async def set_batch_size(self, batch_size: int) -> bool:
        """Set GPU worker batch size."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/batch_config",
                json={"batch_size": batch_size},
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                logger.warning(f"Failed to set batch size: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting batch size: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

# Global GPU client instance
_gpu_client: Optional[GPUClient] = None
_client_lock = asyncio.Lock()

async def get_gpu_client() -> GPUClient:
    """Get or create the global GPU client instance."""
    global _gpu_client
    
    async with _client_lock:
        if _gpu_client is None:
            _gpu_client = GPUClient()
        return _gpu_client

async def close_gpu_client():
    """Close the global GPU client."""
    global _gpu_client
    
    async with _client_lock:
        if _gpu_client:
            await _gpu_client.close()
            _gpu_client = None