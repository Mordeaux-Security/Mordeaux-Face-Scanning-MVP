"""
Batch Queue Manager for GPU Processing

Manages a buffered queue that accumulates images and sends them to the GPU worker
in batches for maximum throughput and GPU utilization.
"""

import asyncio
import logging
import threading
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
import uuid

from .gpu_client import get_gpu_client
from ..core.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class QueuedImage:
    """Represents an image queued for batch processing."""
    image_bytes: bytes
    image_id: str
    image_info: Any  # Original ImageInfo object
    timestamp: float = 0.0


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    batch_id: str
    results: List[List[Dict]]  # Face detection results for each image
    processing_time: float
    success: bool
    error: Optional[str] = None


class BatchQueueManager:
    """
    Manages a buffered queue that accumulates images and sends them to GPU worker in batches.
    
    Features:
    - Configurable batch size and queue depth
    - Automatic batch flushing on timeout
    - Bounded queue to prevent memory exhaustion
    - Background worker thread for processing
    - Thread-safe result queue for async result retrieval
    """
    
    def __init__(self, 
                 batch_size: int = 64,
                 max_queue_depth: int = 5,
                 flush_timeout: float = 0.5,
                 enabled: bool = True):
        """
        Initialize the batch queue manager.
        
        Args:
            batch_size: Number of images per batch
            max_queue_depth: Maximum number of batches in queue
            flush_timeout: Timeout in seconds before flushing partial batch
            enabled: Whether batch queue is enabled
        """
        self.batch_size = batch_size
        self.max_queue_depth = max_queue_depth
        self.flush_timeout = flush_timeout
        self.enabled = enabled
        
        # Current batch being built
        self._current_batch: List[QueuedImage] = []
        self._batch_lock = threading.Lock()
        
        # Queue of completed batches waiting for GPU processing
        self._batch_queue: Queue = Queue(maxsize=max_queue_depth)
        
        # Result queue for async result retrieval (thread-safe)
        self._result_queue: Queue = Queue()
        self._pending_image_ids: set = set()
        self._results_lock = threading.Lock()
        
        # Background worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._worker_running = False
        
        # Statistics
        self._stats = {
            'images_queued': 0,
            'batches_processed': 0,
            'total_processing_time': 0.0,
            'queue_full_count': 0,
            'flush_timeout_count': 0
        }
        
        logger.info(f"BatchQueueManager initialized: batch_size={batch_size}, "
                   f"max_queue_depth={max_queue_depth}, flush_timeout={flush_timeout}s")
    
    def start(self) -> None:
        """Start the background worker thread."""
        if not self.enabled:
            logger.info("Batch queue disabled, not starting worker thread")
            return
            
        if self._worker_running:
            logger.warning("Batch queue worker already running")
            return
        
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="batch-queue-worker",
            daemon=True
        )
        self._worker_thread.start()
        self._worker_running = True
        logger.info("Batch queue worker started")
    
    def stop(self) -> None:
        """Stop the background worker thread and flush any remaining batches."""
        if not self._worker_running:
            return
        
        logger.info("Stopping batch queue worker...")
        self._shutdown_event.set()
        
        # Flush any remaining images in current batch
        with self._batch_lock:
            if self._current_batch:
                logger.info(f"Flushing {len(self._current_batch)} remaining images")
                self._flush_current_batch()
        
        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        self._worker_running = False
        logger.info("Batch queue worker stopped")
    
    def add_image(self, 
                  image_bytes: bytes, 
                  image_info: Any) -> str:
        """
        Add an image to the batch queue.
        
        Args:
            image_bytes: Image data
            image_info: Original ImageInfo object
            
        Returns:
            Image ID for tracking
        """
        if not self.enabled:
            # Fallback to immediate processing
            logger.debug("Batch queue disabled, processing image immediately")
            image_id = str(uuid.uuid4())
            return image_id
        
        image_id = str(uuid.uuid4())
        queued_image = QueuedImage(
            image_bytes=image_bytes,
            image_id=image_id,
            image_info=image_info,
            timestamp=time.time()
        )
        
        with self._batch_lock:
            self._current_batch.append(queued_image)
            self._stats['images_queued'] += 1
            
            # Track pending image IDs
            with self._results_lock:
                self._pending_image_ids.add(image_id)
            
            # Check if batch is full
            if len(self._current_batch) >= self.batch_size:
                logger.debug(f"Batch full ({len(self._current_batch)} images), flushing")
                self._flush_current_batch()
        
        return image_id
    
    async def get_result_async(self, image_id: str, timeout: float = 30.0) -> List[Dict]:
        """
        Get the result for an image ID by polling the result queue asynchronously.
        
        Args:
            image_id: The image ID to get results for
            timeout: Maximum time to wait for results
            
        Returns:
            List of face detection results
            
        Raises:
            TimeoutError: If result is not available within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to get result from queue
                result_id, result_data = self._result_queue.get_nowait()
                
                if result_id == image_id:
                    logger.debug(f"Got result for image {image_id}")
                    return result_data
                else:
                    # Put back if not our result
                    self._result_queue.put((result_id, result_data))
                    
            except Empty:
                pass  # No results available yet
            
            # Small delay before retry
            await asyncio.sleep(0.01)
        
        # Timeout reached
        raise TimeoutError(f"Timeout waiting for result {image_id}")
    
    def _flush_current_batch(self) -> None:
        """Flush the current batch to the processing queue."""
        if not self._current_batch:
            return
        
        # Check if queue is full
        if self._batch_queue.full():
            logger.warning("Batch queue full, dropping oldest batch")
            try:
                self._batch_queue.get_nowait()  # Remove oldest batch
                self._stats['queue_full_count'] += 1
            except Empty:
                pass
        
        # Create batch for processing
        batch_id = str(uuid.uuid4())
        batch_images = self._current_batch.copy()
        self._current_batch.clear()
        
        # Add to processing queue
        self._batch_queue.put({
            'batch_id': batch_id,
            'images': batch_images,
            'timestamp': time.time()
        })
        
        logger.debug(f"Flushed batch {batch_id} with {len(batch_images)} images")
    
    def _worker_loop(self) -> None:
        """Background worker loop that processes batches."""
        logger.info("Batch queue worker loop started")
        
        while not self._shutdown_event.is_set():
            try:
                # Check for timeout flush
                with self._batch_lock:
                    if self._current_batch:
                        oldest_image = min(self._current_batch, key=lambda x: x.timestamp)
                        if time.time() - oldest_image.timestamp > self.flush_timeout:
                            logger.debug(f"Flush timeout reached, flushing {len(self._current_batch)} images")
                            self._flush_current_batch()
                            self._stats['flush_timeout_count'] += 1
                
                # Adaptive batch sizing - adjust based on timeout flush ratio
                if self._stats['batches_processed'] > 10:  # After warmup
                    timeout_ratio = self._stats['flush_timeout_count'] / self._stats['batches_processed']
                    if timeout_ratio < 0.2 and self.batch_size < 128:  # Mostly full batches
                        self.batch_size = min(128, self.batch_size + 16)
                        logger.info(f"Increasing batch size to {self.batch_size} (low timeout ratio: {timeout_ratio:.2f})")
                    elif timeout_ratio > 0.7 and self.batch_size > 32:  # Mostly timeout flushes
                        self.batch_size = max(32, self.batch_size - 16)
                        logger.info(f"Decreasing batch size to {self.batch_size} (high timeout ratio: {timeout_ratio:.2f})")
                
                # Process next batch
                try:
                    batch_data = self._batch_queue.get(timeout=1.0)
                    self._process_batch(batch_data)
                    self._batch_queue.task_done()
                except Empty:
                    # No batches to process, continue loop
                    continue
                    
            except Exception as e:
                logger.error(f"Error in batch queue worker loop: {e}")
                time.sleep(1.0)
        
        logger.info("Batch queue worker loop stopped")
    
    def _process_batch(self, batch_data: Dict) -> None:
        """Process a batch of images through the GPU worker."""
        batch_id = batch_data['batch_id']
        images = batch_data['images']
        
        logger.info(f"Processing batch {batch_id} with {len(images)} images")
        start_time = time.time()
        
        try:
            # Extract image bytes for GPU worker
            image_bytes_list = [img.image_bytes for img in images]
            
            # Get GPU client and process batch
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                gpu_client = loop.run_until_complete(get_gpu_client())
                results = loop.run_until_complete(
                    gpu_client.detect_faces_batch_async(
                        image_bytes_list,
                        min_face_quality=0.5,
                        require_face=False,
                        crop_faces=False,
                        face_margin=0.2
                    )
                )
                
                processing_time = time.time() - start_time
                
                # Put results in result queue for async retrieval
                for image, result in zip(images, results):
                    self._result_queue.put((image.image_id, result))
                    
                    # Remove from pending set
                    with self._results_lock:
                        self._pending_image_ids.discard(image.image_id)
                
                self._stats['batches_processed'] += 1
                self._stats['total_processing_time'] += processing_time
                
                logger.info(f"Batch {batch_id} processed successfully in {processing_time:.3f}s")
                
            finally:
                loop.close()
                
        except Exception as e:
            import traceback
            logger.error(f"Error processing batch {batch_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Put empty results for all images in the batch
            for image in images:
                self._result_queue.put((image.image_id, []))
                
                # Remove from pending set
                with self._results_lock:
                    self._pending_image_ids.discard(image.image_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._batch_lock:
            current_batch_size = len(self._current_batch)
        
        queue_size = self._batch_queue.qsize()
        result_queue_size = self._result_queue.qsize()
        
        with self._results_lock:
            pending_count = len(self._pending_image_ids)
        
        # Calculate timeout flush ratio
        timeout_flush_ratio = 0.0
        if self._stats['batches_processed'] > 0:
            timeout_flush_ratio = self._stats['flush_timeout_count'] / self._stats['batches_processed']
        
        return {
            **self._stats,
            'current_batch_size': current_batch_size,
            'queue_size': queue_size,
            'result_queue_size': result_queue_size,
            'pending_image_count': pending_count,
            'worker_running': self._worker_running,
            'enabled': self.enabled,
            'current_batch_size_config': self.batch_size,
            'timeout_flush_ratio': timeout_flush_ratio
        }


# Global batch queue manager instance
_batch_queue_manager: Optional[BatchQueueManager] = None


def get_batch_queue_manager() -> BatchQueueManager:
    """Get the global batch queue manager instance."""
    global _batch_queue_manager
    
    if _batch_queue_manager is None:
        settings = get_settings()
        _batch_queue_manager = BatchQueueManager(
            batch_size=getattr(settings, 'batch_queue_size', 20),
            max_queue_depth=getattr(settings, 'batch_queue_max_depth', 5),
            flush_timeout=getattr(settings, 'batch_queue_flush_timeout', 0.5),
            enabled=getattr(settings, 'batch_queue_enabled', True)
        )
    
    return _batch_queue_manager


def start_batch_queue() -> None:
    """Start the global batch queue manager."""
    manager = get_batch_queue_manager()
    manager.start()


def stop_batch_queue() -> None:
    """Stop the global batch queue manager."""
    global _batch_queue_manager
    
    if _batch_queue_manager:
        _batch_queue_manager.stop()
        _batch_queue_manager = None
