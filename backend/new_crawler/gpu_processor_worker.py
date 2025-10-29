"""
GPU Processor Worker for New Crawler System

Processes image batches with GPU worker and CPU fallback.
Handles face detection, embedding, and saves results to storage.
"""

import asyncio
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
from .storage_manager import get_storage_manager
from .data_structures import BatchRequest, FaceResult, FaceDetection, TaskStatus, ImageTask
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
        self.storage = get_storage_manager()
        self.timing_logger = get_timing_logger()
        
        # Worker state
        self.running = False
        self.processed_batches = 0
        self.processed_images = 0
        self.faces_detected = 0
        self.raw_images_saved = 0
        self.thumbnails_saved = 0
        self.cached_results = 0
        
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
            
            # Temporarily replace temp paths for GPU processing
            original_temp_paths = {}  # Map: task_index -> original_path
            for i, image_task in enumerate(batch_request.image_tasks):
                original_temp_paths[i] = image_task.temp_path
                if image_task.temp_path in gpu_temp_paths:
                    image_task.temp_path = gpu_temp_paths[image_task.temp_path]
            
            start_time = time.time()
            compute_type = "GPU"  # Track compute type
            
            # Check if batch is large enough for GPU processing
            if len(batch_request.image_tasks) < self.config.gpu_min_batch_size:
                logger.info(f"[GPU-WORKER-{self.worker_id}] Batch size ({len(batch_request.image_tasks)}) below minimum threshold ({self.config.gpu_min_batch_size}), using CPU for efficiency")
                compute_type = "CPU"
                # Skip directly to CPU fallback processing
                face_results = await _process_batch_cpu_fallback(batch_request.image_tasks)
            else:
                # Process batch with GPU interface
                recognition_start = time.time()
                self.timing_logger.log_gpu_recognition_start(batch_request.batch_id)
                face_results = await self.gpu_interface.process_batch(batch_request.image_tasks)
                recognition_duration = (time.time() - recognition_start) * 1000
                face_count = sum(len(faces) for faces in face_results) if face_results else 0
                self.timing_logger.log_gpu_recognition_end(batch_request.batch_id, recognition_duration, face_count)
                # Check if GPU fallback occurred
                if not face_results or all(not faces for faces in face_results):
                    compute_type = "CPU (fallback)"
            
            # CRITICAL: Restore original temp paths so storage can access extractor files
            for i, image_task in enumerate(batch_request.image_tasks):
                if i in original_temp_paths:
                    image_task.temp_path = original_temp_paths[i]
                    logger.debug(f"[GPU Processor {self.worker_id}] Restored original temp path: {image_task.temp_path}")
            
            if not face_results:
                logger.warning(f"[GPU Processor {self.worker_id}] No face results from batch: {batch_request.batch_id}")
                return 0
            
            # Process each image result concurrently
            processed_count = 0
            
            # Create tasks for concurrent image processing
            save_tasks = []
            for i, (image_task, face_detections) in enumerate(zip(batch_request.image_tasks, face_results)):
                task = self._process_and_save_single_image(
                    image_task, face_detections, start_time
                )
                save_tasks.append(task)
            
            # Execute all saves concurrently
            save_results = await asyncio.gather(*save_tasks, return_exceptions=True)
            
            # Process results and count successes
            for result in save_results:
                if isinstance(result, Exception):
                    logger.error(f"[GPU Processor {self.worker_id}] Error in batch processing: {result}")
                elif result:
                    processed_count += 1
            
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

    async def _process_and_save_single_image(self, image_task: ImageTask, 
                                            face_detections: List[FaceDetection],
                                            batch_start_time: float) -> bool:
        """Process and save a single image (helper for concurrent execution)."""
        try:
            site_id = image_task.candidate.site_id
            
            # Check cache
            if self.cache.is_image_cached(image_task.phash):
                logger.debug(f"[GPU Processor {self.worker_id}] Result already cached: {image_task.phash[:8]}...")
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
                return False
            
            # Create face result
            face_result = FaceResult(
                image_task=image_task,
                faces=face_detections,
                processing_time_ms=(time.time() - batch_start_time) * 1000,
                gpu_used=True
            )
            
            # TODO: ARCHITECTURE ISSUE - Storage Responsibility Mixing
            # Currently storage_manager does image cropping (compute-heavy PIL operations).
            # This should be refactored so:
            #   1. GPU processor (this file) handles ALL compute: ML inference + image manipulation
            #   2. Storage manager handles ONLY I/O: save/load from MinIO
            # 
            # Proposed flow:
            #   - After GPU returns bboxes, crop faces HERE using PIL
            #   - Pass pre-cropped bytes to storage
            #   - Storage just saves files, no image processing
            #
            # See: docs/architecture/clean-storage-separation.md (future)
            
            # Save to storage (async with concurrent thumbnails)
            storage_start = time.time()
            self.timing_logger.log_gpu_storage_start(image_task.phash[:8], "raw")
            face_result, save_counts = await self.storage.save_face_result_async(
                image_task, face_result
            )
            storage_duration = (time.time() - storage_start) * 1000
            self.timing_logger.log_gpu_storage_end(image_task.phash[:8], "raw", storage_duration)
            
            # No local tracking needed - Redis stats are source of truth
            
            # Update face result fields
            face_result.saved_to_raw = save_counts.get('saved_raw', 0) > 0
            face_result.saved_to_thumbs = save_counts.get('saved_thumbs', 0) > 0
            
            # Cache result
            await asyncio.to_thread(
                self.cache.store_processing_result, image_task, face_result
            )
            
            # Update statistics
            stats_update = {}
            if face_detections:
                stats_update['faces_detected'] = len(face_detections)
            if face_result.saved_to_raw:
                stats_update['images_saved_raw'] = 1
            if face_result.saved_to_thumbs:
                stats_update['images_saved_thumbs'] = 1
            
            await self.redis.update_site_stats_async(site_id, stats_update)
            
            # Push result
            await self.redis.push_face_result_async(face_result)
            
            return True
            
        except Exception as e:
            logger.error(f"[GPU Processor {self.worker_id}] Error processing image: {e}")
            return False
    
    async def _process_batch_task(self, batch_request: BatchRequest):
        """Process a single batch task with statistics tracking."""
        try:
            # Process batch
            processed_count = await self.process_batch(batch_request)
            
            # Update statistics
            self.processed_batches += 1
            self.processed_images += processed_count
            
            # Log progress periodically
            if self.processed_batches % 10 == 0:
                logger.info(f"[GPU Processor {self.worker_id}] Processed {self.processed_batches} batches, "
                           f"{self.processed_images} images, {self.faces_detected} faces, "
                           f"{self.raw_images_saved} raw images, {self.thumbnails_saved} thumbnails")
        
        except Exception as e:
            logger.error(f"[GPU Processor {self.worker_id}] Error processing batch {batch_request.batch_id}: {e}")
    
    async def run(self):
        """Main worker loop with concurrent batch processing."""
        logger.info(f"[GPU Processor {self.worker_id}] Starting GPU processor worker with concurrent batch processing")
        self.running = True
        
        # Track active batch processing tasks
        active_batch_tasks = set()
        max_concurrent_batches = self.config.nc_max_concurrent_batches_per_worker
        
        while self.running:
            try:
                # Clean up completed tasks
                active_batch_tasks = {t for t in active_batch_tasks if not t.done()}
                
                # If we have capacity, try to pop a new batch
                if len(active_batch_tasks) < max_concurrent_batches:
                    batch_request = self.redis.pop_image_batch(timeout=2.0)  # Reduced from 5.0
                    
                    if batch_request:
                        # Start processing batch in background
                        task = asyncio.create_task(self._process_batch_task(batch_request))
                        active_batch_tasks.add(task)
                        logger.info(f"[GPU Processor {self.worker_id}] Started batch {batch_request.batch_id} "
                                   f"({len(active_batch_tasks)}/{max_concurrent_batches} active)")
                
                # Brief pause before checking for more batches
                await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info(f"[GPU Processor {self.worker_id}] Interrupted, shutting down")
                break
            except Exception as e:
                logger.error(f"[GPU Processor {self.worker_id}] Error in main loop: {e}")
                await asyncio.sleep(1)
        
        # Wait for active tasks to complete
        if active_batch_tasks:
            logger.info(f"[GPU Processor {self.worker_id}] Waiting for {len(active_batch_tasks)} active batches to complete")
            await asyncio.gather(*active_batch_tasks, return_exceptions=True)
        
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
