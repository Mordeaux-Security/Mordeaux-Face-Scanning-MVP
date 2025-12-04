
"""
Storage Worker for New Crawler System

Consumes storage tasks from Redis queue and saves them to MinIO.
Handles I/O operations only - all compute (cropping) happens in GPU processor.
After saving to MinIO, upserts face embeddings to Qdrant vector database.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
import uuid
from typing import Optional, List, Dict, Any
import numpy as np

from .config import get_config
from .redis_manager import get_redis_manager
from .storage_manager import get_storage_manager
from .cache_manager import get_cache_manager
from .timing_logger import get_timing_logger
from .data_structures import StorageTask, FaceResult, ImageTask, FaceDetection

logger = logging.getLogger(__name__)

# Lazy-loaded vector client
_vector_client = None

def _get_vector_client():
    """Get or create Qdrant client for vector operations."""
    global _vector_client
    if _vector_client is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
            
            config = get_config()
            _vector_client = QdrantClient(url=config.qdrant_url)
            
            # Ensure collection exists
            try:
                _vector_client.create_collection(
                    collection_name=config.vector_index,
                    vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE),
                )
                logger.info(f"Created vector collection: {config.vector_index}")
            except Exception:
                # Collection already exists
                pass
                
            logger.info(f"Connected to Qdrant at {config.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    return _vector_client


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
        
        # Vectorization stats
        self.vectorized_faces = 0
        self.vectorization_errors = 0
        
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
                                   f"processed={self.processed_tasks}, failed={self.failed_tasks}, "
                                   f"vectorized={self.vectorized_faces}, vec_errors={self.vectorization_errors}")
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
        
        logger.info(f"[STORAGE-{self.worker_id}] Storage worker stopped. Processed: {self.processed_tasks}, "
                   f"Failed: {self.failed_tasks}, Vectorized: {self.vectorized_faces}, "
                   f"Vectorization Errors: {self.vectorization_errors}")
    
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
            
            # Vectorize embeddings to Qdrant (after successful storage)
            vectorized_count = 0
            if face_result.faces:
                vectorized_count = await self._upsert_embeddings_to_vector_db(face_result, image_task)
            
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
                       f"faces={len(face_result.faces)}, vectorized={vectorized_count}, "
                       f"duration={storage_duration:.1f}ms")
            
        except Exception as e:
            self.failed_tasks += 1
            storage_duration = (time.time() - storage_start_time) * 1000
            logger.error(f"[STORAGE-{self.worker_id}] Failed to process storage task {image_id}... "
                        f"(duration={storage_duration:.1f}ms): {e}", exc_info=True)
    
    def stop(self):
        """Stop the worker."""
        self.running = False
    
    async def _upsert_embeddings_to_vector_db(self, face_result: FaceResult, image_task: ImageTask) -> int:
        """
        Upsert face embeddings to Qdrant vector database.
        
        Returns:
            Number of faces successfully upserted
        """
        if not self.config.vectorization_enabled:
            logger.debug(f"[STORAGE-{self.worker_id}] Vectorization disabled, skipping")
            return 0
        
        if not face_result.faces:
            return 0
        
        # Filter faces that have embeddings
        faces_with_embeddings = [
            (i, face) for i, face in enumerate(face_result.faces) 
            if face.embedding is not None and len(face.embedding) > 0
        ]
        
        if not faces_with_embeddings:
            logger.debug(f"[STORAGE-{self.worker_id}] No faces with embeddings to vectorize")
            return 0
        
        try:
            from qdrant_client.http import models as qm
            
            client = _get_vector_client()
            tenant_id = self.config.default_tenant_id
            
            # Build points for Qdrant
            points = []
            for face_idx, face in faces_with_embeddings:
                # Generate unique face ID
                face_id = str(uuid.uuid4())
                
                # Get thumbnail key if available
                thumb_key = None
                if face_result.thumbnail_keys and face_idx < len(face_result.thumbnail_keys):
                    thumb_key = face_result.thumbnail_keys[face_idx]
                
                # Process embedding: ensure it's a numpy array, copy it, and normalize
                vec = face.embedding
                
                # Convert to numpy array and ensure it's float32
                vec = np.asarray(vec, dtype=np.float32).copy()
                
                # Normalize the embedding vector (L2 normalization for cosine similarity)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                else:
                    # Skip faces with near-zero embeddings (invalid embeddings)
                    logger.warning(f"[STORAGE-{self.worker_id}] Received near-zero embedding norm ({norm:.6f}) for face {face_id}, skipping")
                    continue
                
                # Build metadata payload
                payload = {
                    "tenant_id": tenant_id,
                    "raw_key": face_result.raw_image_key,
                    "thumb_key": thumb_key,
                    "source_url": image_task.candidate.img_url if image_task.candidate else None,
                    "page_url": image_task.candidate.page_url if image_task.candidate else None,
                    "site_id": image_task.candidate.site_id if image_task.candidate else None,
                    "phash": image_task.phash,
                    "bbox": face.bbox,
                    "quality": face.quality,
                    "age": face.age,
                    "gender": face.gender,
                    "indexed_at": time.time(),
                }
                
                # Convert normalized vector to list for Qdrant
                points.append(
                    qm.PointStruct(
                        id=face_id,
                        vector=vec.tolist(),
                        payload=payload
                    )
                )
            
            # Upsert to Qdrant
            if points:
                await asyncio.to_thread(
                    client.upsert,
                    collection_name=self.config.vector_index,
                    points=points
                )
                
                self.vectorized_faces += len(points)
                logger.info(f"[STORAGE-{self.worker_id}] Vectorized {len(points)} faces to Qdrant "
                           f"(total: {self.vectorized_faces})")
                return len(points)
            
            return 0
            
        except Exception as e:
            self.vectorization_errors += 1
            logger.error(f"[STORAGE-{self.worker_id}] Failed to upsert embeddings to Qdrant: {e}", 
                        exc_info=True)
            return 0


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

