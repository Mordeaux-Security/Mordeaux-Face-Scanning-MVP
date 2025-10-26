"""
Batch Queue Manager

Manages batching of images for GPU processing. Uses a shared multiprocessing queue
to collect images from worker processes and send them to the GPU worker in batches.
"""

import asyncio
import logging
import threading
import time
from typing import Optional, List, Dict, Any, Tuple
from queue import Queue
from dataclasses import dataclass

# Support both threading and multiprocessing
try:
    from multiprocessing import Queue as MPQueue, Lock as MPLock
    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False
    MPQueue = None
    MPLock = None

logger = logging.getLogger(__name__)


@dataclass
class QueuedImage:
    """Data class for queued image."""
    image_bytes: bytes
    image_info: Dict[str, Any]
    timestamp: float


class BatchQueueManager:
    """
    Manages batching of images for GPU processing.
    
    Collects images from multiple workers and batches them for efficient GPU processing.
    Supports both threading and multiprocessing modes.
    """
    
    def __init__(
        self,
        batch_size: int = 64,
        max_queue_depth: int = 10,
        flush_timeout: float = 2.0,
        enabled: bool = True,
        shared_queue: Optional[Any] = None,
        max_images_per_site: Optional[int] = None
    ):
        """
        Initialize the batch queue manager.
        
        Args:
            batch_size: Target batch size for GPU processing
            max_queue_depth: Maximum number of batches to queue
            flush_timeout: Time in seconds before flushing incomplete batches
            enabled: Whether batching is enabled
            shared_queue: Shared multiprocessing queue (if in multiprocessing mode)
            max_images_per_site: Maximum images to process per site (None = unlimited)
        """
        self.batch_size = batch_size
        self.max_queue_depth = max_queue_depth
        self.flush_timeout = flush_timeout
        self.enabled = enabled
        self.max_images_per_site = max_images_per_site
        self._is_multiprocess = shared_queue is not None
        
        # Current batch being built
        self._current_batch: List[QueuedImage] = []
        
        # Use multiprocessing lock if using shared queue, else threading lock
        if self._is_multiprocess and HAS_MULTIPROCESSING and MPLock:
            self._batch_lock = MPLock()
        else:
            self._batch_lock = threading.Lock()
        
        # Queue of completed batches waiting for GPU processing
        if shared_queue:
            self._batch_queue = shared_queue  # Use shared queue if provided
        else:
            self._batch_queue: Queue = Queue(maxsize=max_queue_depth)
        
        # Batch queue worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Stats
        self._stats = {
            'batches_sent': 0,
            'images_processed': 0,
            'timeout_flushes': 0
        }
        
        # Site thumbnail counters (for limiting)
        self._site_thumbnail_counts = {}
    
    def start(self):
        """Start the batch queue worker thread."""
        if self._worker_thread is not None:
            logger.warning("Batch queue worker already running")
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Batch queue manager started")
    
    def stop(self):
        """Stop the batch queue worker thread."""
        if self._worker_thread is None:
            return
        
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)
        self._worker_thread = None
        logger.info("Batch queue manager stopped")
    
    def _batch_worker(self):
        """Worker thread that processes batches from the queue."""
        logger.info("Batch queue worker started")
        
        while not self._stop_event.is_set():
            try:
                # Try to get a batch from queue (with timeout)
                try:
                    batch = self._batch_queue.get(timeout=1.0)
                except:
                    continue
                
                # Process batch (send to GPU worker)
                self._process_batch(batch)
                self._batch_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in batch worker: {e}", exc_info=True)
        
        logger.info("Batch queue worker stopped")
    
    def _get_site_thumbnail_count(self, site: str) -> int:
        """Get the current thumbnail count for a site."""
        return self._site_thumbnail_counts.get(site, 0)
    
    def _increment_site_thumbnail_count(self, site: str, count: int = 1):
        """Increment the thumbnail count for a site."""
        self._site_thumbnail_counts[site] = self._site_thumbnail_counts.get(site, 0) + count
    
    def _process_batch(self, batch: List[QueuedImage]):
        """
        Process a batch of images by sending to GPU worker.
        
        Args:
            batch: List of QueuedImage objects
        """
        try:
            logger.info(f"Processing batch of {len(batch)} images via GPU worker")
            
            # Extract image bytes from queued images
            image_bytes_list = [qi.image_bytes for qi in batch]
            
            # Call GPU worker synchronously
            from .face import _try_gpu_worker_sync
            results = _try_gpu_worker_sync(
                image_bytes_list,
                min_face_quality=0.5,
                require_face=False,
                crop_faces=True,
                face_margin=0.2
            )
            
            if results:
                # Store results back with image info
                for i, (result, queued_img) in enumerate(zip(results, batch)):
                    queued_img.image_info['faces'] = result
                logger.info(f"GPU worker processed batch successfully: {len(results)} results")
                
                # Increment thumbnail counts for each site in this batch
                site_counts = {}
                for queued_img in batch:
                    site = queued_img.image_info.get('site', 'unknown')
                    site_counts[site] = site_counts.get(site, 0) + 1
                
                for site, count in site_counts.items():
                    self._increment_site_thumbnail_count(site, count)
                    logger.info(f"Site {site} thumbnail count: {self._get_site_thumbnail_count(site)}")
            else:
                logger.warning("GPU worker returned no results for batch")
            
            # Update stats
            with self._batch_lock:
                self._stats['batches_sent'] += 1
                self._stats['images_processed'] += len(batch)
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
    
    def add_image(self, image_bytes: bytes, image_info: Dict[str, Any]) -> bool:
        """
        Add an image to the batch queue.
        
        Args:
            image_bytes: Raw image bytes
            image_info: Image metadata
            
        Returns:
            True if image was added, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Check thumbnail limit per site if specified
            if self.max_images_per_site:
                site = image_info.get('site', 'unknown')
                current_thumbnails = self._get_site_thumbnail_count(site)
                
                if current_thumbnails >= self.max_images_per_site:
                    logger.info(f"Site {site} already has {current_thumbnails} thumbnails (limit: {self.max_images_per_site}), rejecting image")
                    return False
            
            queued_image = QueuedImage(
                image_bytes=image_bytes,
                image_info=image_info,
                timestamp=time.time()
            )
            
            with self._batch_lock:
                self._current_batch.append(queued_image)
                
                # Check if batch is full
                if len(self._current_batch) >= self.batch_size:
                    # Flush batch
                    self._flush_batch()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding image to batch: {e}")
            return False
    
    def _flush_batch(self):
        """Flush the current batch to the batch queue."""
        if not self._current_batch:
            return
        
        try:
            # Copy current batch
            batch = self._current_batch.copy()
            self._current_batch.clear()
            
            # Add to batch queue
            self._batch_queue.put(batch, block=False)
            
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        with self._batch_lock:
            return self._stats.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

