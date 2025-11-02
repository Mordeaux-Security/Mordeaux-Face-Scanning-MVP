"""
GPU Processor Worker for New Crawler System

Processes image batches with GPU worker and CPU fallback.
Handles face detection, embedding, and saves results to storage.
"""

import asyncio
import io
import logging
import multiprocessing
import os
import signal
import tempfile
import time
import shutil
from datetime import datetime
from typing import List, Optional, Dict

from .config import get_config
from .redis_manager import get_redis_manager
from .cache_manager import get_cache_manager
from .gpu_interface import get_gpu_interface
from .gpu_scheduler import GPUScheduler
from .data_structures import BatchRequest, FaceResult, FaceDetection, TaskStatus, ImageTask, StorageTask
from .timing_logger import get_timing_logger

logger = logging.getLogger(__name__)


class GPUProcessorWorker:
    """GPU processor worker for face detection and storage."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.config = get_config()
        self.redis = get_redis_manager()
        self.cache = get_cache_manager()
        self.gpu_interface = get_gpu_interface()
        self.timing_logger = get_timing_logger()
        
        # Worker state
        self.running = False
        self.processed_batches = 0
        self.processed_images = 0
        self.faces_detected = 0
        self.raw_images_saved = 0
        self.thumbnails_saved = 0
        self.cached_results = 0
        
        
        # GPU scheduler for centralized batching
        self.scheduler = GPUScheduler(
            redis_mgr=self.redis,
            deserializer=self.redis.deserialize_image_task,
            inbox_key=getattr(self.config, 'gpu_inbox_key', 'gpu:inbox'),
            target_batch=int(getattr(self.config, 'gpu_target_batch', 32)),
            max_wait_ms=int(getattr(self.config, 'gpu_max_wait_ms', 12)),
            min_launch_ms=int(getattr(self.config, 'gpu_min_launch_ms', 200)),
            config=self.config,
        )
        
        # No per-site tracking needed - limits handled by extractor
    
    async def process_batch(self, batch_request: BatchRequest) -> int:
        """Process a batch of image tasks."""
        batch_start_time = time.time()
        try:
            logger.info(f"[GPU Processor {self.worker_id}] Processing batch: {batch_request.batch_id}, "
                       f"{len(batch_request.image_tasks)} images")
            
            # Log GPU batch start
            self.timing_logger.log_gpu_batch_start(batch_request.batch_id, len(batch_request.image_tasks))
            await self.redis.set_gpu_processing_async(batch_request.batch_id)
            
            # CRITICAL FIX: Copy temp files to GPU processor ownership
            # This prevents race condition where extractor deletes temp files before GPU can access them
            gpu_temp_dir = tempfile.mkdtemp(prefix=f"gpu_{self.worker_id}_")
            logger.debug(f"[GPU Processor {self.worker_id}] Created temp dir: {gpu_temp_dir}")
            
            # Track GPU temp paths separately (don't mutate image_task)
            gpu_temp_paths = {}  # Map: original_path -> gpu_path
            
            # Copy all temp files to GPU's ownership
            for image_task in batch_request.image_tasks:
                original_path = image_task.temp_path
                gpu_path = os.path.join(gpu_temp_dir, os.path.basename(original_path))
                
                try:
                    if os.path.exists(original_path):
                        shutil.copy2(original_path, gpu_path)
                        gpu_temp_paths[original_path] = gpu_path
                        logger.debug(f"[GPU Processor {self.worker_id}] Copied temp file: {original_path} -> {gpu_path}")
                    else:
                        logger.warning(f"[GPU Processor {self.worker_id}] Temp file missing: {original_path}")
                except Exception as e:
                    logger.error(f"[GPU Processor {self.worker_id}] Failed to copy temp file {original_path}: {e}")
            
            # Pre-validate images and fill batch to target size
            target_batch_size = int(getattr(self.config, 'gpu_target_batch', 512))
            valid_tasks = []
            invalid_tasks = []
            
            # Validate initial batch
            for image_task in batch_request.image_tasks:
                if self._validate_image_task(image_task):
                    valid_tasks.append(image_task)
                else:
                    invalid_tasks.append(image_task)
            
            # If we're short of target, pull more images from scheduler until we have enough valid ones
            # Pull from scheduler staging first, then from queue if needed
            while len(valid_tasks) < target_batch_size:
                # Try to get from scheduler staging first
                additional_task = None
                if self.scheduler._staging:
                    # Pop from staging
                    additional_task = self.scheduler._staging.pop(0)
                else:
                    # Pull directly from queue if staging is empty
                    inbox_key = getattr(self.config, 'gpu_inbox_key', 'gpu:inbox')
                    raw = await asyncio.to_thread(
                        self.redis.blpop_many, inbox_key, max_n=1, timeout=0.1
                    )
                    if raw:
                        try:
                            additional_task = self.redis.deserialize_image_task(raw[0])
                        except Exception as e:
                            logger.debug(f"[GPU Processor {self.worker_id}] Failed to deserialize task: {e}")
                            continue
                
                if not additional_task:
                    # No more images available, break
                    break
                
                # Copy temp file for additional image
                original_path = additional_task.temp_path
                gpu_path = os.path.join(gpu_temp_dir, os.path.basename(original_path))
                try:
                    if os.path.exists(original_path):
                        shutil.copy2(original_path, gpu_path)
                        gpu_temp_paths[original_path] = gpu_path
                    else:
                        # Temp file missing, skip
                        invalid_tasks.append(additional_task)
                        continue
                except Exception as e:
                    logger.error(f"[GPU Processor {self.worker_id}] Failed to copy temp file {original_path}: {e}")
                    invalid_tasks.append(additional_task)
                    continue
                
                if self._validate_image_task(additional_task):
                    valid_tasks.append(additional_task)
                else:
                    invalid_tasks.append(additional_task)
            
            # Log validation results
            if invalid_tasks:
                logger.info(f"[GPU Processor {self.worker_id}] Validated batch: {len(valid_tasks)} valid, {len(invalid_tasks)} invalid (skipped)")
            
            # Use only valid tasks for processing
            batch_request.image_tasks = valid_tasks[:target_batch_size]  # Cap at target
            
            # Temporarily replace temp paths for GPU processing
            original_temp_paths = {}  # Map: task_index -> original_path
            for i, image_task in enumerate(batch_request.image_tasks):
                original_temp_paths[i] = image_task.temp_path
                if image_task.temp_path in gpu_temp_paths:
                    image_task.temp_path = gpu_temp_paths[image_task.temp_path]
            
            start_time = time.time()
            compute_type = "GPU"  # Track compute type
            
            # Process batch with GPU interface (now guaranteed to have target valid images)
            recognition_start = time.time()
            self.timing_logger.log_gpu_recognition_start(batch_request.batch_id)
            face_results = await self.gpu_interface.process_batch(batch_request.image_tasks)
            recognition_duration = (time.time() - recognition_start) * 1000
            face_count = sum(len(faces) for faces in face_results) if face_results else 0
            self.timing_logger.log_gpu_recognition_end(batch_request.batch_id, recognition_duration, face_count)
            
            # Log GPU results summary for debugging
            if face_results:
                images_with_faces = sum(1 for faces in face_results if faces)
                logger.info(f"[GPU Processor {self.worker_id}] GPU batch {batch_request.batch_id} results: "
                           f"{len(face_results)} images, {images_with_faces} with faces, "
                           f"{face_count} total faces detected in {recognition_duration:.1f}ms")
            else:
                logger.warning(f"[GPU Processor {self.worker_id}] GPU batch {batch_request.batch_id} returned None")
            
            # Check if GPU fallback occurred
            if not face_results or all(not faces for faces in face_results):
                compute_type = "CPU (fallback)"
            
            # CRITICAL: Restore original temp paths so storage can access extractor files
            for i, image_task in enumerate(batch_request.image_tasks):
                if i in original_temp_paths:
                    image_task.temp_path = original_temp_paths[i]
                    logger.debug(f"[GPU Processor {self.worker_id}] Restored original temp path: {image_task.temp_path}")
            
            # Only drop batch if GPU completely failed (returned None)
            # Empty face lists per image are valid (images with no faces detected)
            if face_results is None:
                logger.warning(f"[GPU Processor {self.worker_id}] GPU processing failed for batch: {batch_request.batch_id}")
                return 0
            
            # Ensure face_results matches input batch size
            if len(face_results) != len(batch_request.image_tasks):
                logger.warning(f"[GPU Processor {self.worker_id}] Result count mismatch: {len(face_results)} != {len(batch_request.image_tasks)} for batch {batch_request.batch_id}")
                # Continue anyway, process what we got (truncate or pad as needed)
                min_len = min(len(face_results), len(batch_request.image_tasks))
                face_results = face_results[:min_len]
                batch_request.image_tasks = batch_request.image_tasks[:min_len]
            
            # Process each image result: crop faces and queue storage tasks
            # This decouples GPU processing from storage I/O
            storage_tasks_created = 0
            storage_tasks_pushed = 0
            total_faces_in_batch = 0
            cached_in_batch = 0
            
            for i, (image_task, face_detections) in enumerate(zip(batch_request.image_tasks, face_results)):
                # Track faces detected
                total_faces_in_batch += len(face_detections)
                
                # Prepare storage task (crops faces, creates task)
                storage_task = await self._prepare_storage_task(image_task, face_detections, start_time)
                
                if storage_task:
                    storage_tasks_created += 1
                    # Push to storage queue (non-blocking)
                    pushed = await self.redis.push_storage_task_async(storage_task)
                    if pushed:
                        storage_tasks_pushed += 1
                        logger.debug(f"[GPU Processor {self.worker_id}] Pushed storage task: {image_task.phash[:8]}..., faces={len(face_detections)}")
                    else:
                        logger.warning(f"[GPU Processor {self.worker_id}] Failed to push storage task: {image_task.phash[:8]}...")
                else:
                    # Storage task was None (likely cached)
                    cached_in_batch += 1
            
            # Count based on GPU results - all images that were processed by GPU
            # This allows batch to return immediately while storage runs in background
            # face_results has one entry per input image (even if empty list for no faces)
            processed_count = len(face_results)
            
            # Log batch processing summary
            logger.info(f"[GPU Processor {self.worker_id}] Batch {batch_request.batch_id} summary: "
                       f"processed={processed_count}, cached={cached_in_batch}, "
                       f"storage_tasks={storage_tasks_created} (pushed={storage_tasks_pushed}), "
                       f"faces_detected={total_faces_in_batch}")
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"[GPU Processor {self.worker_id}] Batch completed: {processed_count}/{len(batch_request.image_tasks)} "
                       f"images processed in {processing_time:.1f}ms using {compute_type}")
            
            # Log GPU batch end
            batch_duration = (time.time() - batch_start_time) * 1000
            self.timing_logger.log_gpu_batch_end(batch_request.batch_id, batch_duration, processed_count)
            await self.redis.clear_gpu_processing_async()
            
            return processed_count
            
        except Exception as e:
            logger.error(f"[GPU Processor {self.worker_id}] Error processing batch {batch_request.batch_id}: {e}")
            await self.redis.clear_gpu_processing_async()
            return 0
        finally:
            # Cleanup GPU temp directory after batch processing
            # (Original extractor temp files still exist for storage access)
            try:
                if 'gpu_temp_dir' in locals() and os.path.exists(gpu_temp_dir):
                    shutil.rmtree(gpu_temp_dir, ignore_errors=True)
                    logger.debug(f"[GPU Processor {self.worker_id}] Cleaned up GPU temp dir: {gpu_temp_dir}")
            except Exception as e:
                logger.error(f"[GPU Processor {self.worker_id}] Error cleaning up GPU temp dir: {e}")

    def _validate_image_task(self, image_task: ImageTask) -> bool:
        """Validate that an image task can be encoded successfully."""
        try:
            # Check if temp file exists
            if not os.path.exists(image_task.temp_path):
                return False
            
            # Try to load and validate image bytes
            with open(image_task.temp_path, 'rb') as f:
                image_bytes = f.read()
            
            if not image_bytes or len(image_bytes) < 10:
                return False
            
            # Check for common image headers (same validation as _encode_image)
            if not (image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                    image_bytes.startswith(b'\x89PNG') or       # PNG
                    image_bytes.startswith(b'GIF8') or          # GIF
                    image_bytes.startswith(b'BM')):            # BMP
                return False
            
            return True
        except Exception as e:
            logger.debug(f"[GPU Processor {self.worker_id}] Image validation failed for {image_task.phash[:8]}...: {e}")
            return False
    
    def _crop_face_from_image(self, image_path: str, face_detection: FaceDetection, 
                               margin: float = None) -> Optional[bytes]:
        """Crop face from image using PIL (compute operation - moved from storage_manager)."""
        try:
            from PIL import Image
            
            if margin is None:
                margin = self.config.face_margin
            
            # Check file exists
            if not os.path.exists(image_path):
                logger.error(f"[GPU Processor {self.worker_id}] File not found for cropping: {image_path}")
                return None
            
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Extract bounding box coordinates
                bbox = face_detection.bbox
                x1, y1, x2, y2 = bbox
                
                # Check if coordinates are already normalized (0-1 range)
                if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                    # Coordinates are in pixels, normalize them
                    x1 = x1 / width
                    y1 = y1 / height
                    x2 = x2 / width
                    y2 = y2 / height
                
                # Convert to pixel coordinates
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                
                # Add margin
                face_width = x2 - x1
                face_height = y2 - y1
                margin_x = int(face_width * margin)
                margin_y = int(face_height * margin)
                
                # Expand bounding box with margin
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(width, x2 + margin_x)
                y2 = min(height, y2 + margin_y)
                
                # Ensure valid bounds
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"[GPU Processor {self.worker_id}] Invalid face bounds: ({x1}, {y1}, {x2}, {y2})")
                    return None
                
                if x2 - x1 < self.config.min_face_size or y2 - y1 < self.config.min_face_size:
                    logger.debug(f"[GPU Processor {self.worker_id}] Skipping tiny face: ({x2 - x1}x{y2 - y1}) < {self.config.min_face_size}")
                    return None

                # Crop the face
                face_crop = img.crop((x1, y1, x2, y2))
                
                # Convert to bytes
                output = io.BytesIO()
                face_crop.save(output, format='JPEG', quality=85)
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"[GPU Processor {self.worker_id}] Failed to crop face from {image_path}: {e}", exc_info=True)
            return None
    
    async def _crop_faces(self, image_task: ImageTask, 
                         face_detections: List[FaceDetection]) -> List[bytes]:
        """Crop all faces from an image (async wrapper for PIL operations)."""
        if not face_detections:
            return []
        
        # Crop all faces concurrently
        crop_tasks = []
        for face_detection in face_detections:
            task = asyncio.to_thread(
                self._crop_face_from_image, 
                image_task.temp_path, 
                face_detection
            )
            crop_tasks.append(task)
        
        results = await asyncio.gather(*crop_tasks, return_exceptions=True)
        
        # Filter out None and exceptions
        face_crops = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[GPU Processor {self.worker_id}] Failed to crop face {i}: {result}")
            elif result is not None:
                face_crops.append(result)
        
        return face_crops
    
    async def _prepare_storage_task(self, image_task: ImageTask, 
                                   face_detections: List[FaceDetection],
                                   batch_start_time: float) -> Optional[StorageTask]:
        """Prepare storage task by cropping faces and creating StorageTask."""
        try:
            site_id = image_task.candidate.site_id
            
            # Check cache
            if self.cache.is_image_cached(image_task.phash):
                logger.debug(f"[GPU Processor {self.worker_id}] Result already cached: {image_task.phash[:8]}..., faces={len(face_detections)}")
                self.cached_results += 1
                
                # Still create a result for statistics tracking
                face_result = FaceResult(
                    image_task=image_task,
                    faces=face_detections,
                    processing_time_ms=(time.time() - batch_start_time) * 1000,
                    gpu_used=False,
                    saved_to_raw=False,
                    saved_to_thumbs=False,
                    skip_reason="cached"
                )
                await self.redis.push_face_result_async(face_result)
                
                # FIX: Update statistics for cached images too (include face counts)
                stats_update = {
                    'faces_detected': len(face_detections),  # Count faces even if cached
                    'images_cached': 1
                }
                await self.redis.update_site_stats_async(site_id, stats_update)
                logger.debug(f"[GPU Processor {self.worker_id}] Updated stats for cached image: {len(face_detections)} faces, site={site_id}")
                
                return None
            
            # Create face result
            face_result = FaceResult(
                image_task=image_task,
                faces=face_detections,
                processing_time_ms=(time.time() - batch_start_time) * 1000,
                gpu_used=True
            )
            
            # Crop faces using PIL (compute operation happens here)
            face_crops = await self._crop_faces(image_task, face_detections)
            
            # Log cropping results for debugging
            if not face_detections:
                logger.debug(f"[GPU Processor {self.worker_id}] Image {image_task.phash[:8]}... has 0 faces, will save raw image only")
            elif face_detections and not face_crops:
                logger.warning(f"[GPU Processor {self.worker_id}] All {len(face_detections)} face crops failed for {image_task.phash[:8]}... (faces detected but crops failed)")
            elif len(face_crops) < len(face_detections):
                logger.warning(f"[GPU Processor {self.worker_id}] Only {len(face_crops)}/{len(face_detections)} crops succeeded for {image_task.phash[:8]}...")
            else:
                logger.debug(f"[GPU Processor {self.worker_id}] Image {image_task.phash[:8]}... has {len(face_detections)} faces, {len(face_crops)} crops successful")
            
            # Create storage task with pre-cropped faces
            storage_task = StorageTask(
                image_task=image_task,
                face_result=face_result,
                face_crops=face_crops,
                batch_start_time=batch_start_time
            )
            
            return storage_task
            
        except Exception as e:
            logger.error(f"[GPU Processor {self.worker_id}] Error preparing storage task for {image_task.phash[:8]}...: {e}", exc_info=True)
            return None
    
    async def run(self):
        """Main worker loop with GPU scheduler."""
        logger.info(f"[GPU Processor {self.worker_id}] Starting GPU processor worker with scheduler")
        self.running = True
        
        # Diagnostic tracking
        if self.config.nc_diagnostic_logging:
            self._last_batch_completion_time = None
            self._batch_gaps = []
            self._staging_waits = []
            self._queue_depths = []
        
        while self.running:
            try:
                # Keep staging warm by feeding items from Redis
                added = self.scheduler.feed()
                
                # Diagnostic: Log feed() results
                if self.config.nc_diagnostic_logging:
                    queue_depth = await self.redis.get_queue_length_by_key_async('gpu:inbox')
                    if added > 0:
                        logger.debug(f"[GPU-PROC-DIAG-{self.worker_id}] feed() added={added}, "
                                   f"staging={len(self.scheduler._staging)}, queue_depth={queue_depth}")
                
                # Build a batch if it's time
                batch_tasks = self.scheduler.next_batch()
                if batch_tasks:
                    batch_id = f"{int(time.time()*1000)}-{len(batch_tasks)}"
                    batch_start_time = time.time()
                    
                    # Calculate time since last batch completion
                    gap_seconds = 0.0
                    if self.config.nc_diagnostic_logging:
                        if self._last_batch_completion_time is not None:
                            gap_seconds = batch_start_time - self._last_batch_completion_time
                            self._batch_gaps.append(gap_seconds)
                        
                        queue_depth = await self.redis.get_queue_length_by_key_async('gpu:inbox')
                        logger.info(f"[GPU-PROC-DIAG-{self.worker_id}] Starting batch {batch_id}: "
                               f"size={len(batch_tasks)}, staging={len(self.scheduler._staging)}, "
                               f"queue_depth={queue_depth}, time_since_last_batch={gap_seconds:.2f}s")
                    
                    self.scheduler.mark_launched(batch_id)
                    
                    # Create BatchRequest from image tasks
                    batch_request = BatchRequest(
                        image_tasks=batch_tasks,
                        batch_id=batch_id,
                        min_face_quality=self.config.min_face_quality,
                        require_face=False,
                        crop_faces=True,
                        face_margin=self.config.face_margin
                    )
                    
                    # Process synchronously (scheduler enforces max 2 inflight)
                    processed_count = await self.process_batch(batch_request)
                    
                    # Update statistics
                    self.processed_batches += 1
                    self.processed_images += processed_count
                    
                    # Mark batch as completed
                    self.scheduler.mark_completed(batch_id)
                    
                    # Diagnostic: Log batch completion
                    batch_duration = time.time() - batch_start_time
                    if self.config.nc_diagnostic_logging:
                        queue_depth = await self.redis.get_queue_length_by_key_async('gpu:inbox')
                        self._last_batch_completion_time = time.time()
                        logger.info(f"[GPU-PROC-DIAG-{self.worker_id}] Batch {batch_id} completed: "
                                  f"processed={processed_count}, time_since_start={batch_duration:.1f}s, "
                                  f"staging={len(self.scheduler._staging)}, queue_depth={queue_depth}")
                        self._queue_depths.append(queue_depth)
                    
                    # Log progress periodically
                    if self.processed_batches % 10 == 0:
                        logger.info(f"[GPU Processor {self.worker_id}] Processed {self.processed_batches} batches, "
                                   f"{self.processed_images} images, {self.faces_detected} faces, "
                                   f"{self.raw_images_saved} raw images, {self.thumbnails_saved} thumbnails")
                    
                    # Diagnostic: Periodic summary
                    if self.config.nc_diagnostic_logging and self.processed_batches % 5 == 0 and self.processed_batches > 0:
                        if self._batch_gaps:
                            avg_gap = sum(self._batch_gaps) / len(self._batch_gaps)
                            max_gap = max(self._batch_gaps)
                            min_gap = min(self._batch_gaps)
                        else:
                            avg_gap = max_gap = min_gap = 0.0
                        
                        avg_queue_depth = sum(self._queue_depths) / len(self._queue_depths) if self._queue_depths else 0.0
                        
                        logger.info(f"[GPU-PROC-DIAG-{self.worker_id}] Summary: batches={self.processed_batches}, "
                                  f"avg_batch_gap={avg_gap:.2f}s (min={min_gap:.2f}s, max={max_gap:.2f}s), "
                                  f"avg_queue_depth={avg_queue_depth:.1f}")
                        
                        # Reset tracking for next window
                        self._batch_gaps = []
                        self._queue_depths = []
                
                
                # Avoid busy-spin (2ms = 500 iterations/sec max)
                await asyncio.sleep(0.002)
                
            except KeyboardInterrupt:
                logger.info(f"[GPU Processor {self.worker_id}] Interrupted, shutting down")
                break
            except Exception as e:
                logger.error(f"[GPU Processor {self.worker_id}] Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
        
        logger.info(f"[GPU Processor {self.worker_id}] GPU processor worker stopped")
    
    def stop(self):
        """Stop the worker."""
        self.running = False
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'processed_batches': self.processed_batches,
            'processed_images': self.processed_images,
            'faces_detected': self.faces_detected,
            'raw_images_saved': self.raw_images_saved,
            'thumbnails_saved': self.thumbnails_saved,
            'cached_results': self.cached_results,
            'running': self.running
        }


def gpu_processor_worker_process(worker_id: int):
    """GPU processor worker process entry point."""
    # Reset signal handlers to default (let parent handle)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    
    # Configure logging for multiprocessing
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - GPU-Processor-{worker_id} - %(levelname)s - %(message)s',
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"[GPU Processor {worker_id}] Starting GPU processor worker process")
    
    try:
        # Create worker
        worker = GPUProcessorWorker(worker_id)
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run worker
            loop.run_until_complete(worker.run())
        finally:
            # Close loop
            loop.close()
            
    except Exception as e:
        logger.error(f"[GPU Processor {worker_id}] Fatal error: {e}", exc_info=True)
    finally:
        logger.info(f"[GPU Processor {worker_id}] GPU processor worker process ended")


async def _process_batch_cpu_fallback(image_tasks: List[ImageTask]) -> List[List[FaceDetection]]:
    """Process batch using CPU fallback without attempting GPU connection."""
    try:
        start_time = time.time()
        # Use the GPU interface's CPU fallback method
        gpu_interface = get_gpu_interface()
        result = await gpu_interface._cpu_fallback(image_tasks)
        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"✓ CPU fallback processed {len(image_tasks)} images in {elapsed_time:.1f}ms")
        return result
    except Exception as e:
        logger.error(f"CPU fallback processing failed: {e}")
        # Return empty results for all images
        return [[] for _ in image_tasks]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    
    args = parser.parse_args()
    gpu_processor_worker_process(args.worker_id)
