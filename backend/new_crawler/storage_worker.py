"""
Storage Worker for New Crawler System

Consumes storage tasks from Redis queue and saves them to MinIO.
Handles I/O operations only - all compute (cropping) happens in GPU processor.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import Optional

from .config import get_config
from .redis_manager import get_redis_manager
from .storage_manager import get_storage_manager
from .cache_manager import get_cache_manager
from .timing_logger import get_timing_logger
from .data_structures import StorageTask, FaceResult

logger = logging.getLogger(__name__)


class StorageWorker:
    """Storage worker that consumes storage queue and saves to MinIO."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.config = get_config()
        self.redis = get_redis_manager()
        self.storage = get_storage_manager()
        self.cache = get_cache_manager()
        self.timing_logger = get_timing_logger()
        
        # Worker state
        self.running = False
        self.processed_tasks = 0
        self.failed_tasks = 0
        
    async def run(self):
        """Main worker loop - consumes storage queue."""
        logger.info(f"[STORAGE-{self.worker_id}] Starting storage worker")
        self.running = True
        
        # Throughput tracking
        start_time = time.time()
        last_log_time = start_time
        
        while self.running:
            try:
                # Pop storage task from queue
                storage_task = await self.redis.pop_storage_task_async(timeout=2.0)
                
                if storage_task:
                    # Log queue depth for monitoring
                    queue_depth = await self.redis.get_queue_length_by_key_async(self.config.get_queue_name('storage'))
                    image_task = storage_task.image_task
                    faces_count = len(storage_task.face_result.faces) if storage_task.face_result else 0
                    crops_count = len(storage_task.face_crops) if storage_task.face_crops else 0
                    temp_path_exists = image_task.temp_path and os.path.exists(image_task.temp_path) if image_task.temp_path else False
                    logger.info(f"[STORAGE-{self.worker_id}] DIAG: Popped task from queue: "
                               f"image={image_task.phash[:8] if image_task.phash else 'NO_PHASH'}..., "
                               f"faces={faces_count}, crops={crops_count}, "
                               f"queue_depth={queue_depth}, "
                               f"temp_path={'EXISTS' if temp_path_exists else 'MISSING'}")
                    
                    await self._process_storage_task(storage_task)
                    self.processed_tasks += 1
                    
                    # Periodic throughput logging
                    now = time.time()
                    if now - last_log_time >= 10.0:  # Log every 10 seconds
                        elapsed = now - start_time
                        tasks_per_sec = self.processed_tasks / elapsed if elapsed > 0 else 0
                        logger.info(f"[STORAGE-{self.worker_id}] Throughput: {tasks_per_sec:.2f} tasks/sec, "
                                   f"processed={self.processed_tasks}, failed={self.failed_tasks}")
                        last_log_time = now
                else:
                    # No tasks available, brief sleep to avoid tight loop
                    queue_depth = await self.redis.get_queue_length_by_key_async(self.config.get_queue_name('storage'))
                    # Log every 10 seconds to avoid spam
                    current_time = int(time.time())
                    if not hasattr(self, '_last_no_task_log') or current_time - self._last_no_task_log >= 10:
                        logger.info(f"[STORAGE-{self.worker_id}] DIAG: No tasks in queue, depth={queue_depth}, "
                                   f"processed={self.processed_tasks}, waiting...")
                        self._last_no_task_log = current_time
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"[STORAGE-{self.worker_id}] Error in storage worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        
        logger.info(f"[STORAGE-{self.worker_id}] Storage worker stopped. Processed: {self.processed_tasks}, Failed: {self.failed_tasks}")
    
    async def _process_storage_task(self, storage_task: StorageTask):
        """Process a single storage task."""
        storage_start_time = time.time()
        image_task = storage_task.image_task
        site_id = image_task.candidate.site_id
        image_id = image_task.phash[:8] if image_task.phash else 'NO_PHASH'
        
        try:
            # Validate storage task structure
            faces_count = len(storage_task.face_result.faces) if storage_task.face_result else 0
            crops_count = len(storage_task.face_crops) if storage_task.face_crops else 0
            if faces_count != crops_count:
                logger.warning(f"[STORAGE-{self.worker_id}] DIAG: Storage task mismatch: {faces_count} faces but {crops_count} crops "
                             f"for {image_id}...")
            
            # Log storage start
            self.timing_logger.log_storage_start(site_id, image_id)
            
            # Save to storage (pure I/O - pre-cropped faces already provided)
            face_result, save_counts = await self.storage.save_storage_task_async(storage_task)
            
            # Cache result
            await asyncio.to_thread(
                self.cache.store_processing_result, image_task, face_result
            )
            
            # Update statistics
            stats_update = {}
            # Always update faces_detected (even if 0, for accurate totals)
            stats_update['faces_detected'] = len(face_result.faces)
            if face_result.saved_to_raw:
                stats_update['images_saved_raw'] = 1
            saved_thumbs_count = save_counts.get('saved_thumbs', 0)
            if saved_thumbs_count > 0:
                stats_update['images_saved_thumbs'] = saved_thumbs_count
            
            await self.redis.update_site_stats_async(site_id, stats_update)
            
            # Push final result to results queue
            await self.redis.push_face_result_async(face_result)
            
            # Clean up temp file after successful save
            if image_task.temp_path:
                try:
                    if os.path.exists(image_task.temp_path):
                        await asyncio.to_thread(self.storage.cleanup_temp_file, image_task.temp_path)
                        logger.debug(f"[STORAGE-{self.worker_id}] Cleaned up temp file: {image_task.temp_path}")
                    else:
                        logger.info(f"[STORAGE-{self.worker_id}] [TEMP-FILE] Temp file already deleted before cleanup: "
                                  f"{image_task.temp_path}")
                except Exception as cleanup_err:
                    logger.warning(f"[STORAGE-{self.worker_id}] Failed to cleanup temp file {image_task.temp_path}: "
                                 f"{cleanup_err}")
            
            # Calculate duration
            storage_duration = (time.time() - storage_start_time) * 1000
            
            # Log storage end
            self.timing_logger.log_storage_end(
                site_id, 
                image_id, 
                storage_duration, 
                len(face_result.faces),
                face_result.saved_to_raw,
                saved_thumbs_count
            )
            
            logger.info(f"[STORAGE-{self.worker_id}] Saved: {image_id}..., "
                       f"raw={face_result.saved_to_raw}, thumbs={saved_thumbs_count}, "
                       f"faces={len(face_result.faces)}, duration={storage_duration:.1f}ms")
            
        except Exception as e:
            self.failed_tasks += 1
            storage_duration = (time.time() - storage_start_time) * 1000
            logger.error(f"[STORAGE-{self.worker_id}] Failed to process storage task {image_id}... "
                        f"(duration={storage_duration:.1f}ms): {e}", exc_info=True)
    
    def stop(self):
        """Stop the worker."""
        self.running = False


def storage_worker_process(worker_id: int):
    """Entry point for storage worker process."""
    import signal
    
    worker = None
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"[STORAGE-{worker_id}] Received signal {signum}, shutting down gracefully...")
        if worker:
            worker.stop()
        # Setting worker.running = False will cause the event loop to exit naturally
        # The while loop in worker.run() checks self.running
    
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - Storage-{worker_id} - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        
        # Create worker
        worker = StorageWorker(worker_id)
        
        # Register signal handlers AFTER creating worker but before running event loop
        # Note: Signal handlers work at the process level, so this should work
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Run async event loop
        asyncio.run(worker.run())
        
    except KeyboardInterrupt:
        logger.info(f"[STORAGE-{worker_id}] Received keyboard interrupt, shutting down...")
        if worker:
            worker.stop()
    except Exception as e:
        logger.error(f"[STORAGE-{worker_id}] Fatal error: {e}", exc_info=True)
        if worker:
            worker.stop()
        sys.exit(1)


if __name__ == '__main__':
    # For testing
    import sys
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    storage_worker_process(worker_id)

