"""
GPU Worker Logger for New Crawler System

Dedicated logger for GPU operations with consistent formatting and comprehensive tracking.
"""

import logging
from typing import Optional


class GPUWorkerLogger:
    """Dedicated logger for GPU worker operations."""
    
    def __init__(self, worker_id: int):
        self.logger = logging.getLogger(f"gpu_worker_{worker_id}")
        self.worker_id = worker_id
    
    def log_batch_start(self, batch_id: str, image_count: int):
        """Log batch processing start."""
        self.logger.info(f"[GPU-WORKER-{self.worker_id}] BATCH-START: {batch_id}, images={image_count}")
    
    def log_health_check(self, available: bool, error: Optional[str] = None):
        """Log health check results."""
        if available:
            self.logger.info(f"[GPU-WORKER-{self.worker_id}] HEALTH-CHECK: OK")
        else:
            self.logger.warning(f"[GPU-WORKER-{self.worker_id}] HEALTH-CHECK: FAILED - {error}")
    
    def log_request_start(self, image_count: int, url: str):
        """Log request start."""
        self.logger.debug(f"[GPU-WORKER-{self.worker_id}] REQUEST-START: {image_count} images to {url}")
    
    def log_request_complete(self, image_count: int, faces_found: int, time_ms: float, gpu_used: bool):
        """Log request completion."""
        mode = "GPU" if gpu_used else "CPU"
        self.logger.info(f"[GPU-WORKER-{self.worker_id}] REQUEST-COMPLETE: {image_count} images, {faces_found} faces, {time_ms:.1f}ms, mode={mode}")
    
    def log_request_failed(self, error: str):
        """Log request failure."""
        self.logger.error(f"[GPU-WORKER-{self.worker_id}] REQUEST-FAILED: {error}")
    
    def log_fallback_start(self, image_count: int):
        """Log CPU fallback start."""
        self.logger.warning(f"[GPU-WORKER-{self.worker_id}] FALLBACK-START: CPU processing {image_count} images")
    
    def log_fallback_complete(self, image_count: int, faces_found: int, time_ms: float):
        """Log CPU fallback completion."""
        self.logger.info(f"[GPU-WORKER-{self.worker_id}] FALLBACK-COMPLETE: {image_count} images, {faces_found} faces, {time_ms:.1f}ms")
    
    def log_batch_encoding_start(self, image_count: int):
        """Log batch encoding start."""
        self.logger.debug(f"[GPU-WORKER-{self.worker_id}] BATCH-ENCODING-START: encoding {image_count} images")
    
    def log_batch_encoding_complete(self, image_count: int, encoded_size: int):
        """Log batch encoding completion."""
        self.logger.debug(f"[GPU-WORKER-{self.worker_id}] BATCH-ENCODING-COMPLETE: {image_count} images, {encoded_size} bytes")
    
    def log_result_decoding_start(self, response_size: int):
        """Log result decoding start."""
        self.logger.debug(f"[GPU-WORKER-{self.worker_id}] RESULT-DECODING-START: {response_size} bytes")
    
    def log_result_decoding_complete(self, faces_found: int):
        """Log result decoding completion."""
        self.logger.debug(f"[GPU-WORKER-{self.worker_id}] RESULT-DECODING-COMPLETE: {faces_found} faces")
    
    def log_circuit_breaker_open(self, reason: str):
        """Log circuit breaker opening."""
        self.logger.warning(f"[GPU-WORKER-{self.worker_id}] CIRCUIT-BREAKER-OPEN: {reason}")
    
    def log_circuit_breaker_close(self):
        """Log circuit breaker closing."""
        self.logger.info(f"[GPU-WORKER-{self.worker_id}] CIRCUIT-BREAKER-CLOSE: attempting GPU again")
    
    def log_retry_attempt(self, attempt: int, max_retries: int, error: str):
        """Log retry attempt."""
        self.logger.warning(f"[GPU-WORKER-{self.worker_id}] RETRY-ATTEMPT: {attempt}/{max_retries}, error: {error}")
    
    def log_timeout(self, timeout_seconds: float):
        """Log timeout occurrence."""
        self.logger.warning(f"[GPU-WORKER-{self.worker_id}] TIMEOUT: {timeout_seconds}s exceeded")
    
    def log_connection_error(self, error: str):
        """Log connection error."""
        self.logger.error(f"[GPU-WORKER-{self.worker_id}] CONNECTION-ERROR: {error}")
    
    def log_http_error(self, status_code: int, error: str):
        """Log HTTP error."""
        self.logger.error(f"[GPU-WORKER-{self.worker_id}] HTTP-ERROR: {status_code}, {error}")
    
    def log_processing_error(self, error: str):
        """Log processing error."""
        self.logger.error(f"[GPU-WORKER-{self.worker_id}] PROCESSING-ERROR: {error}")
    
    def log_metrics(self, metrics: dict):
        """Log performance metrics."""
        self.logger.info(f"[GPU-WORKER-{self.worker_id}] METRICS: {metrics}")




