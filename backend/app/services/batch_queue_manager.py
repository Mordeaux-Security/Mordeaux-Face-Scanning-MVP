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
        site_results_dict: Optional[Any] = None
    ):
        """
        Initialize the batch queue manager.
        
        Args:
            batch_size: Target batch size for GPU processing
            max_queue_depth: Maximum number of batches to queue
            flush_timeout: Time in seconds before flushing incomplete batches
            enabled: Whether batching is enabled
            shared_queue: Shared multiprocessing queue (if in multiprocessing mode)
            site_results_dict: Shared dictionary for updating site statistics
        """
        self.batch_size = batch_size
        self.max_queue_depth = max_queue_depth
        self.flush_timeout = flush_timeout
        self.enabled = enabled
        self._is_multiprocess = shared_queue is not None
        self._site_results_dict = site_results_dict
        
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
                logger.info(f"[DATAFLOW] BatchQueueManager: QueueSize={self._batch_queue.qsize()}, Processing batch of {len(batch)}")
                self._process_batch(batch)
                logger.info(f"[DATAFLOW] BatchQueueManager: Batch processed, QueueSize={self._batch_queue.qsize()}")
                self._batch_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in batch worker: {e}", exc_info=True)
        
        logger.info("Batch queue worker stopped")
    
    
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
                # Process results and save to MinIO
                raw_images_saved = 0
                thumbnails_saved = 0
                
                for i, (result, queued_img) in enumerate(zip(results, batch)):
                    queued_img.image_info['faces'] = result
                    
                    # Save raw image and face thumbnails to MinIO with proper bucket separation
                    try:
                        from .storage import save_raw_and_thumb_content_addressed, save_image, _minio, _boto3_s3, get_settings
                        from PIL import Image
                        import io
                        import asyncio
                        
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            # Process each image and its detected faces
                            if result and len(result) > 0:
                                # Load the image once
                                image = Image.open(io.BytesIO(queued_img.image_bytes))
                                
                                # Crop and save each face as a separate thumbnail
                                for face_idx, face in enumerate(result):
                                    try:
                                        # Extract bounding box
                                        bbox = face.get('bbox', [])
                                        if len(bbox) >= 4:
                                            x1, y1, x2, y2 = bbox[:4]
                                            
                                            # Convert to integers and ensure valid bounds
                                            x1, y1 = max(0, int(x1)), max(0, int(y1))
                                            x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
                                            
                                            if x2 > x1 and y2 > y1:
                                                # Crop face
                                                face_crop = image.crop((x1, y1, x2, y2))
                                                
                                                # Resize to thumbnail size
                                                face_crop.thumbnail((256, 256), Image.Resampling.LANCZOS)
                                                
                                                # Convert to bytes
                                                face_bytes = io.BytesIO()
                                                face_crop.save(face_bytes, format='JPEG', quality=85)
                                                face_bytes_data = face_bytes.getvalue()
                                                
                                                # Save raw image and thumbnail with matching hash names
                                                # This saves:
                                                # - Raw: s3_bucket_raw/default/<hash>.jpg
                                                # - Thumb: s3_bucket_thumbs/default/<hash>_thumb.jpg
                                                raw_key, raw_url, thumb_key, thumb_url, metadata = loop.run_until_complete(
                                                    asyncio.get_event_loop().run_in_executor(
                                                        None,
                                                        save_raw_and_thumb_content_addressed,
                                                        queued_img.image_bytes,
                                                        face_bytes_data,
                                                        'default',  # tenant_id
                                                        queued_img.image_info.get('url'),  # source_url
                                                        None  # video_url
                                                    )
                                                )
                                                
                                                raw_images_saved += 1
                                                thumbnails_saved += 1
                                                
                                                # Also save JSON metadata sidecar using save_image()
                                                # This creates the meta.json file next to the image
                                                site = queued_img.image_info.get('site', 'unknown')
                                                settings = get_settings()
                                                client = _minio() if settings.using_minio else _boto3_s3()
                                                
                                                loop.run_until_complete(
                                                    save_image(
                                                        image_bytes=queued_img.image_bytes,
                                                        mime="image/jpeg",
                                                        filename=raw_key.split('/')[-1],  # Extract filename from key
                                                        bucket=settings.s3_bucket_raw,
                                                        client=client,
                                                        site=site,
                                                        page_url=queued_img.image_info.get('url'),
                                                        source_image_url=queued_img.image_info.get('url')
                                                    )
                                                )
                                                
                                                logger.info(f"Saved raw image to {raw_key} and thumbnail to {thumb_key}")
                                                
                                    except Exception as face_error:
                                        logger.error(f"Error cropping face {face_idx}: {face_error}")
                            else:
                                # No faces detected, just save the raw image
                                # Use save_image() to get proper metadata sidecars
                                site = queued_img.image_info.get('site', 'unknown')
                                settings = get_settings()
                                client = _minio() if settings.using_minio else _boto3_s3()
                                
                                result_dict = loop.run_until_complete(
                                    save_image(
                                        image_bytes=queued_img.image_bytes,
                                        mime="image/jpeg",
                                        filename=f"image_{i}.jpg",
                                        bucket=settings.s3_bucket_raw,
                                        client=client,
                                        site=site,
                                        page_url=queued_img.image_info.get('url'),
                                        source_image_url=queued_img.image_info.get('url')
                                    )
                                )
                                raw_images_saved += 1
                                logger.info(f"Saved raw image (no faces) to {result_dict['image_key']}")
                                        
                        finally:
                            loop.close()
                            
                    except Exception as e:
                        logger.error(f"Error saving image to MinIO: {e}")
                
                logger.info(f"GPU worker processed batch successfully: {len(results)} results, {raw_images_saved} raw images saved, {thumbnails_saved} thumbnails saved")
                
                # Update stats with MinIO saves
                with self._batch_lock:
                    self._stats['raw_images_saved'] = self._stats.get('raw_images_saved', 0) + raw_images_saved
                    self._stats['thumbnails_saved'] = self._stats.get('thumbnails_saved', 0) + thumbnails_saved
                
                # Update site results if available
                if self._site_results_dict:
                    # Track saves per site
                    site_saves = {}
                    for queued_img in batch:
                        site = queued_img.image_info.get('site', 'unknown')
                        if site not in site_saves:
                            site_saves[site] = {'raw': 0, 'thumbs': 0}
                        # Each image in the batch contributes to raw saves
                        site_saves[site]['raw'] += 1
                        # Thumbnails depend on face detection results
                        faces = queued_img.image_info.get('faces', [])
                        if faces:
                            site_saves[site]['thumbs'] += len(faces)
                    
                    # Update site results
                    for site, saves in site_saves.items():
                        if site in self._site_results_dict:
                            stats = self._site_results_dict[site]
                            stats['raw_images_saved'] += saves['raw']
                            stats['thumbnails_saved'] += saves['thumbs']
                            self._site_results_dict[site] = stats
                    
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

