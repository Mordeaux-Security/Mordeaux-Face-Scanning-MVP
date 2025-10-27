"""
GPU Interface for New Crawler System

Provides GPU worker client with CPU fallback for face detection and embedding.
Handles consistent data structures and proper batching to avoid single-image requests.
"""

import asyncio
import base64
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
import httpx
import numpy as np
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
        """Check if GPU worker is available."""
        current_time = time.time()
        
        # Skip if checked recently
        if current_time - self._last_health_check < 10.0:
            return self._is_available
        
        # Check circuit breaker
        if current_time < self._circuit_open_until:
            return False
        
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            
            if response.status_code == 200:
                health_data = response.json()
                self._is_available = health_data.get("status") == "healthy"
                self._failure_count = 0
                self.gpu_logger.log_health_check(True)
            else:
                self._is_available = False
                self.gpu_logger.log_health_check(False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self._is_available = False
            self._failure_count += 1
            self.gpu_logger.log_health_check(False, str(e))
            
            # Open circuit breaker after 3 failures
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
    
    async def _gpu_worker_request(self, image_tasks: List[ImageTask]) -> Optional[List[List[FaceDetection]]]:
        """Make request to GPU worker."""
        try:
            # Load images from temp paths
            image_bytes_list = []
            for task in image_tasks:
                try:
                    with open(task.temp_path, 'rb') as f:
                        image_bytes = f.read()
                    image_bytes_list.append(image_bytes)
                except Exception as e:
                    logger.error(f"Failed to load image from {task.temp_path}: {e}")
                    image_bytes_list.append(b'')  # Empty bytes for failed loads
            
            # Filter out empty images
            valid_images = [(i, img_bytes) for i, img_bytes in enumerate(image_bytes_list) if img_bytes]
            if not valid_images:
                logger.warning("No valid images in batch")
                return None
            
            # Prepare request data with consistent structure
            images_data = []
            for i, (original_idx, image_bytes) in enumerate(valid_images):
                images_data.append({
                    "data": self._encode_image(image_bytes),
                    "image_id": f"img_{original_idx}"
                })
            
            request_data = {
                "images": images_data,
                "min_face_quality": self.config.min_face_quality,
                "require_face": False,
                "crop_faces": True,
                "face_margin": self.config.face_margin
            }
            
            # Make request
            client = await self._get_client()
            start_time = time.time()
            
            self.gpu_logger.log_request_start(len(valid_images), f"{self.config.gpu_worker_host_resolved}/detect_faces_batch")
            self.gpu_logger.log_batch_encoding_start(len(valid_images))
            
            response = await client.post(
                f"{self.config.gpu_worker_host_resolved}/detect_faces_batch",
                json=request_data
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result_data = response.json()
                self.gpu_logger.log_result_decoding_start(len(str(result_data)))
                results = []
                
                # Convert GPU worker results to our format
                for image_results in result_data["results"]:
                    face_detections = []
                    for detection_data in image_results:
                        face_detection = self._decode_face_detection(detection_data)
                        face_detections.append(face_detection)
                    results.append(face_detections)
                
                # Pad results for failed image loads
                full_results = []
                valid_idx = 0
                for i, img_bytes in enumerate(image_bytes_list):
                    if img_bytes:
                        full_results.append(results[valid_idx])
                        valid_idx += 1
                    else:
                        full_results.append([])
                
                gpu_used = result_data.get('gpu_used', False)
                worker_id = result_data.get('worker_id', 'unknown')
                
                # Count total faces found
                total_faces = sum(len(faces) for faces in results)
                
                logger.info(f"✓ GPU worker processed {len(image_tasks)} images in "
                           f"{processing_time:.1f}ms (GPU: {gpu_used}, worker: {worker_id})")
                
                # Use GPU worker logger
                self.gpu_logger.log_result_decoding_complete(total_faces)
                self.gpu_logger.log_request_complete(len(image_tasks), total_faces, processing_time, gpu_used)
                
                # Update metrics
                self._request_count += 1
                self._success_count += 1
                self._total_latency += processing_time
                
                return full_results
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
    
    async def _cpu_fallback(self, image_tasks: List[ImageTask]) -> List[List[FaceDetection]]:
        """CPU fallback for face detection."""
        logger.info(f"Using CPU fallback for {len(image_tasks)} images")
        
        try:
            # Import CPU face detection (reuse existing face.py)
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            
            from app.services.face import detect_faces_batch
            
            # Load images from temp paths
            image_bytes_list = []
            for task in image_tasks:
                try:
                    with open(task.temp_path, 'rb') as f:
                        image_bytes = f.read()
                    image_bytes_list.append(image_bytes)
                except Exception as e:
                    logger.error(f"Failed to load image from {task.temp_path}: {e}")
                    image_bytes_list.append(b'')
            
            # Filter out empty images
            valid_images = [img_bytes for img_bytes in image_bytes_list if img_bytes]
            if not valid_images:
                logger.warning("No valid images for CPU fallback")
                return [[] for _ in image_tasks]
            
            # Run CPU face detection
            start_time = time.time()
            results = detect_faces_batch(
                valid_images,
                min_face_quality=self.config.min_face_quality,
                require_face=False,
                crop_faces=True,
                face_margin=self.config.face_margin
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Count total faces found
            total_faces = sum(len(faces) for faces in results)
            
            # Log completion
            self.gpu_logger.log_fallback_complete(len(image_tasks), total_faces, processing_time)
            
            # Convert to our format
            converted_results = []
            valid_idx = 0
            for img_bytes in image_bytes_list:
                if img_bytes:
                    face_detections = []
                    for detection in results[valid_idx]:
                        face_detection = FaceDetection(
                            bbox=detection['bbox'],
                            landmarks=detection.get('landmarks', []),
                            embedding=detection.get('embedding'),
                            quality=detection.get('quality', 0.0),
                            age=detection.get('age'),
                            gender=detection.get('gender')
                        )
                        face_detections.append(face_detection)
                    converted_results.append(face_detections)
                    valid_idx += 1
                else:
                    converted_results.append([])
            
            logger.info(f"✓ CPU fallback processed {len(image_tasks)} images in {processing_time:.1f}ms")
            return converted_results
            
        except Exception as e:
            logger.error(f"CPU fallback failed: {e}")
            return [[] for _ in image_tasks]
    
    async def process_batch(self, image_tasks: List[ImageTask]) -> List[List[FaceDetection]]:
        """Process a batch of images with GPU worker and CPU fallback."""
        if not image_tasks:
            return []
        
        logger.info(f"Processing batch of {len(image_tasks)} images")
        self.gpu_logger.log_batch_start(f"batch_{int(time.time())}", len(image_tasks))
        
        # Check if GPU worker is enabled and available
        if not self.config.gpu_worker_enabled:
            logger.info("GPU worker disabled, using CPU fallback")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            return await self._cpu_fallback(image_tasks)
        
        # Check GPU worker health
        if not await self._check_health():
            logger.warning("GPU worker not available, using CPU fallback")
            self.gpu_logger.log_fallback_start(len(image_tasks))
            return await self._cpu_fallback(image_tasks)
        
        # Try GPU worker first
        try:
            results = await self._gpu_worker_request(image_tasks)
            if results is not None:
                return results
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
