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
import time
from typing import List, Optional, Dict

from .config import get_config
from .redis_manager import get_redis_manager
from .cache_manager import get_cache_manager
from .gpu_interface import get_gpu_interface
from .storage_manager import get_storage_manager
from .data_structures import BatchRequest, FaceResult, FaceDetection, TaskStatus, ImageTask

logger = logging.getLogger(__name__)


class GPUProcessorWorker:
    """GPU processor worker for face detection and storage."""
    
    # Minimum batch size to justify GPU overhead
    MIN_GPU_BATCH_SIZE = 8
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.config = get_config()
        self.redis = get_redis_manager()
        self.cache = get_cache_manager()
        self.gpu_interface = get_gpu_interface()
        self.storage = get_storage_manager()
        
        # Worker state
        self.running = False
        self.processed_batches = 0
        self.processed_images = 0
        self.faces_detected = 0
        self.raw_images_saved = 0
        self.thumbnails_saved = 0
        self.cached_results = 0
        
        # Per-site tracking
        self.site_image_counts: Dict[str, Dict[str, int]] = {}
        self._counts_lock = asyncio.Lock()
    
    async def process_batch(self, batch_request: BatchRequest) -> int:
        """Process a batch of image tasks."""
        try:
            logger.info(f"[GPU Processor {self.worker_id}] Processing batch: {batch_request.batch_id}, "
                       f"{len(batch_request.image_tasks)} images")
            
            start_time = time.time()
            
            # Check if batch is large enough for GPU processing
            if len(batch_request.image_tasks) < self.MIN_GPU_BATCH_SIZE:
                logger.info(f"[GPU-WORKER-{self.worker_id}] Batch size ({len(batch_request.image_tasks)}) below minimum threshold ({self.MIN_GPU_BATCH_SIZE}), using CPU for efficiency")
                # Skip directly to CPU fallback processing
                face_results = await _process_batch_cpu_fallback(batch_request.image_tasks)
            else:
                # Process batch with GPU interface
                face_results = await self.gpu_interface.process_batch(batch_request.image_tasks)
            
            if not face_results:
                logger.warning(f"[GPU Processor {self.worker_id}] No face results from batch: {batch_request.batch_id}")
                return 0
            
            # Process each image result
            processed_count = 0
            for i, (image_task, face_detections) in enumerate(zip(batch_request.image_tasks, face_results)):
                try:
                    site_id = image_task.candidate.site_id
                    
                    # Initialize site tracking if needed
                    async with self._counts_lock:
                        if site_id not in self.site_image_counts:
                            self.site_image_counts[site_id] = {
                                'processed': 0,
                                'saved_raw': 0,
                                'saved_thumbs': 0,
                                'skipped': 0
                            }
                        self.site_image_counts[site_id]['processed'] += 1
                    
                    # Check if result is already cached
                    if self.cache.is_image_cached(image_task.phash):
                        logger.debug(f"[GPU Processor {self.worker_id}] Result already cached: {image_task.phash[:8]}...")
                        self.cached_results += 1
                        continue
                    
                    # Check per-site image limit
                    max_images_per_site = self.config.nc_max_images_per_site
                    async with self._counts_lock:
                        if self.site_image_counts[site_id]['saved_thumbs'] >= max_images_per_site:
                            logger.info(f"[GPU-PROC-{self.worker_id}] Site {site_id} reached image limit ({max_images_per_site}), skipping save")
                            self.site_image_counts[site_id]['skipped'] += 1
                            
                            # Create face result without saving
                            face_result = FaceResult(
                                image_task=image_task,
                                faces=face_detections,
                                processing_time_ms=(time.time() - start_time) * 1000,
                                gpu_used=True,
                                saved_to_raw=False,
                                saved_to_thumbs=False,
                                skip_reason="site_limit_reached"
                            )
                            
                            # Update statistics in Redis
                            stats_update = {}
                            if face_detections:
                                stats_update['faces_detected'] = len(face_detections)
                            if face_result.saved_to_raw:
                                stats_update['images_saved_raw'] = 1
                            if face_result.saved_to_thumbs:
                                stats_update['images_saved_thumbs'] = 1
                            if face_result.skip_reason == "site_limit_reached":
                                stats_update['images_skipped_limit'] = 1
                            
                            await self.redis.update_site_stats_async(site_id, stats_update)
                            
                            # Push result to results queue
                            await self.redis.push_face_result_async(face_result)
                            processed_count += 1
                            continue
                    
                    # Create face result
                    face_result = FaceResult(
                        image_task=image_task,
                        faces=face_detections,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        gpu_used=True  # Will be updated by GPU interface
                    )
                    
                    # Save to storage (run in thread pool - blocking I/O)
                    face_result, save_counts = await asyncio.to_thread(
                        self.storage.save_face_result, image_task, face_result
                    )
                    
                    # Update site counters
                    async with self._counts_lock:
                        self.site_image_counts[site_id]['saved_raw'] += save_counts.get('saved_raw', 0)
                        self.site_image_counts[site_id]['saved_thumbs'] += save_counts.get('saved_thumbs', 0)
                    
                    # Update face result fields
                    face_result.saved_to_raw = save_counts.get('saved_raw', 0) > 0
                    face_result.saved_to_thumbs = save_counts.get('saved_thumbs', 0) > 0
                    
                    # Cache the result (run in thread pool - blocking I/O)
                    await asyncio.to_thread(
                        self.cache.store_processing_result, image_task, face_result
                    )
                    
                    # Update statistics
                    self.faces_detected += len(face_detections)
                    if face_result.raw_image_key:
                        self.raw_images_saved += 1
                    self.thumbnails_saved += len(face_result.thumbnail_keys)
                    
                    # Log save details
                    logger.debug(f"[GPU-PROC-{self.worker_id}] Image saved: raw={face_result.raw_image_key}, thumbs={len(face_result.thumbnail_keys)}")
                    
                    # Update statistics in Redis
                    stats_update = {}
                    if face_detections:
                        stats_update['faces_detected'] = len(face_detections)
                    if face_result.saved_to_raw:
                        stats_update['images_saved_raw'] = 1
                    if face_result.saved_to_thumbs:
                        stats_update['images_saved_thumbs'] = 1
                    
                    await self.redis.update_site_stats_async(site_id, stats_update)
                    
                    # Log site progress
                    counts = self.site_image_counts[site_id]
                    logger.debug(f"[GPU-PROC-{self.worker_id}] Site {site_id}: processed={counts['processed']}, raw={counts['saved_raw']}, thumbs={counts['saved_thumbs']}, skipped={counts['skipped']}")
                    
                    # Push result to results queue
                    await self.redis.push_face_result_async(face_result)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"[GPU Processor {self.worker_id}] Error processing image {i} in batch: {e}")
                    continue
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"[GPU Processor {self.worker_id}] Batch completed: {processed_count}/{len(batch_request.image_tasks)} "
                       f"images processed in {processing_time:.1f}ms")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"[GPU Processor {self.worker_id}] Error processing batch {batch_request.batch_id}: {e}")
            return 0
    
    async def run(self):
        """Main worker loop."""
        logger.info(f"[GPU Processor {self.worker_id}] Starting GPU processor worker")
        self.running = True
        
        while self.running:
            try:
                # Get batch from queue
                batch_request = self.redis.pop_image_batch(timeout=5)
                
                if batch_request:
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
                
            except KeyboardInterrupt:
                logger.info(f"[GPU Processor {self.worker_id}] Interrupted, shutting down")
                break
            except Exception as e:
                logger.error(f"[GPU Processor {self.worker_id}] Error in main loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
        
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
        # Use the GPU interface's CPU fallback method
        gpu_interface = get_gpu_interface()
        return await gpu_interface._cpu_fallback(image_tasks)
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
