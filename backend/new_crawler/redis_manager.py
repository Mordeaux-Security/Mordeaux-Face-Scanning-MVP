"""
Redis Manager for New Crawler System

Handles Redis queue operations, connection pooling, and back-pressure monitoring.
Provides clean interface for all queue operations with proper error handling.
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
import redis
import redis.asyncio as aioredis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from pydantic import BaseModel

from .config import get_config
from .data_structures import (
    SiteTask, CandidateImage, ImageTask, FaceResult, 
    BatchRequest, QueueMetrics, TaskStatus, StorageTask
)

logger = logging.getLogger(__name__)

# Singleton pattern per process
_redis_manager_instance = None


class RedisManager:
    """Redis manager for queue operations and connection pooling."""
    
    def __init__(self):
        self.config = get_config()
        self._pool: Optional[ConnectionPool] = None
        self._async_pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        # Active task counter key
        self._active_tasks_key = "nc:active_tasks"
        # Site limit flag prefix
        self._site_limit_flag_prefix = "nc:site:limit:"  # nc:site:limit:{site_id}
        # Track consecutive zero-depth reads for desync detection
        self._consecutive_zero_depth_count: int = 0
        self._last_non_zero_depth_time: float = time.time()  # Initialize to current time to avoid false positives
        
        # Log throttling for repetitive diagnostic logs (max every 500ms)
        self._last_diag_log_time: float = 0.0
        self._diag_log_interval: float = 0.5  # 500ms
        
    def _get_pool(self) -> ConnectionPool:
        """Get or create Redis connection pool."""
        if self._pool is None:
            self._pool = ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                decode_responses=False  # Handle bytes explicitly
            )
        return self._pool
    
    def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self._get_pool())
        return self._client
    
    # Consistent serialization
    def _serialize(self, obj: BaseModel) -> bytes:
        """Serialize Pydantic model to bytes."""
        return obj.model_dump_json().encode('utf-8')
    
    def _deserialize(self, data, model_class) -> BaseModel:
        """Deserialize bytes or string to Pydantic model."""
        # Handle both bytes and string (aioredis may return strings)
        if isinstance(data, bytes):
            return model_class.model_validate_json(data.decode('utf-8'))
        elif isinstance(data, str):
            return model_class.model_validate_json(data)
        else:
            # Try to convert to string
            return model_class.model_validate_json(str(data))
    
    async def _get_async_pool(self) -> aioredis.ConnectionPool:
        """Get or create async Redis connection pool."""
        if self._async_pool is None:
            self._async_pool = aioredis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                decode_responses=True
            )
        return self._async_pool
    
    async def _get_async_client(self) -> aioredis.Redis:
        """Get or create async Redis client."""
        if self._async_client is None:
            self._async_client = aioredis.Redis(connection_pool=await self._get_async_pool())
        return self._async_client
    
    def test_connection(self) -> bool:
        """Test Redis connection."""
        try:
            client = self._get_client()
            client.ping()
            logger.info("Redis connection successful")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
    
    async def test_async_connection(self) -> bool:
        """Test async Redis connection."""
        try:
            client = await self._get_async_client()
            await client.ping()
            logger.info("Async Redis connection successful")
            return True
        except Exception as e:
            logger.error(f"Async Redis connection failed: {e}")
            return False
    
    def reset_sync_connection_pool(self):
        """Reset sync connection pool to force fresh connections."""
        try:
            logger.warning("[REDIS] Resetting sync connection pool due to desync detection")
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
            if self._pool:
                try:
                    self._pool.disconnect()
                except Exception:
                    pass
                self._pool = None
            # Reset desync tracking
            self._consecutive_zero_depth_count = 0
            self._last_non_zero_depth_time = time.time()
            logger.info("[REDIS] Sync connection pool reset complete")
        except Exception as e:
            logger.error(f"[REDIS] Error resetting sync connection pool: {e}")
    
    async def reset_async_connection_pool(self):
        """Reset async connection pool to force fresh connections."""
        try:
            logger.warning("[REDIS] Resetting async connection pool due to desync detection")
            if self._async_client:
                try:
                    await self._async_client.close()
                except Exception:
                    pass
                self._async_client = None
            if self._async_pool:
                try:
                    await self._async_pool.disconnect()
                except Exception:
                    pass
                self._async_pool = None
            logger.info("[REDIS] Async connection pool reset complete")
        except Exception as e:
            logger.error(f"[REDIS] Error resetting async connection pool: {e}")
    
    # Queue Operations
    
    def push_site(self, site_task: SiteTask, timeout: float = 5.0) -> bool:
        """Push site task to sites queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('sites')
            data = self._serialize(site_task)
            result = client.rpush(queue_name, data)
            logger.debug(f"Pushed site task to queue: {site_task.site_id}")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push error: {e}")
            return False
    
    def pop_site(self, timeout: float = 5.0) -> Optional[SiteTask]:
        """Pop site task from sites queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('sites')
            result = client.brpop(queue_name, timeout=timeout)
            
            if result:
                _, data = result
                return self._deserialize(data, SiteTask)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis pop error: {e}")
            return None
    
    def push_candidate(self, candidate: CandidateImage) -> bool:
        """Push candidate image to candidates queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('candidates')
            data = self._serialize(candidate)
            result = client.rpush(queue_name, data)
            logger.debug(f"Pushed candidate to queue: {candidate.img_url}")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push error: {e}")
            return False
    
    async def push_candidate_async(self, candidate: CandidateImage) -> bool:
        """Push candidate image to candidates queue (async version)."""
        try:
            client = await self._get_async_client()
            queue_name = self.config.get_queue_name('candidates')
            data = self._serialize(candidate)
            result = await client.rpush(queue_name, data)
            logger.debug(f"Pushed candidate to queue (async): {candidate.img_url}")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push error (async): {e}")
            return False
    
    def pop_candidate(self, timeout: float = 1.0) -> Optional[CandidateImage]:
        """Pop candidate image from candidates queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('candidates')
            result = client.brpop(queue_name, timeout=timeout)
            
            if result:
                _, data = result
                return self._deserialize(data, CandidateImage)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis pop error: {e}")
            return None
    
    def push_image_batch(self, image_tasks: List[ImageTask]) -> bool:
        """Push batch of image tasks to images queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('images')
            
            # Create batch request
            batch_request = BatchRequest(
                image_tasks=image_tasks,
                batch_id=f"batch_{int(time.time() * 1000)}"
            )
            
            data = self._serialize(batch_request)
            result = client.rpush(queue_name, data)
            logger.info(f"Pushed image batch to queue: {len(image_tasks)} images")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push error: {e}")
            return False
    
    async def push_image_batch_async(self, image_tasks: List[ImageTask]) -> bool:
        """Push batch of image tasks to images queue (async version)."""
        try:
            client = await self._get_async_client()
            queue_name = self.config.get_queue_name('images')
            
            # Create batch request
            batch_request = BatchRequest(
                image_tasks=image_tasks,
                batch_id=f"batch_{int(time.time() * 1000)}"
            )
            
            data = self._serialize(batch_request)
            result = await client.rpush(queue_name, data)
            logger.info(f"Pushed image batch to queue (async): {len(image_tasks)} images")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push error (async): {e}")
            return False
    
    def pop_image_batch(self, timeout: float = 5.0) -> Optional[BatchRequest]:
        """Pop image batch from images queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('images')
            result = client.brpop(queue_name, timeout=timeout)
            
            if result:
                _, data = result
                return self._deserialize(data, BatchRequest)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis pop error: {e}")
            return None
    
    def push_face_result(self, face_result: FaceResult) -> bool:
        """Push face result to results queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('results')
            data = self._serialize(face_result)
            result = client.rpush(queue_name, data)
            logger.debug(f"Pushed face result to queue: {face_result.image_task.candidate.img_url}")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push error: {e}")
            return False
    
    async def push_face_result_async(self, face_result: FaceResult) -> bool:
        """Push face result to results queue (async version)."""
        try:
            client = await self._get_async_client()
            queue_name = self.config.get_queue_name('results')
            data = self._serialize(face_result)
            result = await client.rpush(queue_name, data)
            pushed = bool(result)
            queue_depth = await self.get_queue_length_by_key_async(queue_name)
            image_phash = face_result.image_task.phash[:8] if (face_result.image_task and face_result.image_task.phash) else 'NO_PHASH'
            faces_count = len(face_result.faces) if face_result.faces else 0
            logger.debug(f"[REDIS] DIAG: Pushed face result: {image_phash}..., "
                       f"success={pushed}, faces={faces_count}, "
                       f"queue_depth={queue_depth}")
            logger.debug(f"Pushed face result to queue (async): {face_result.image_task.candidate.img_url}")
            return pushed
        except redis.RedisError as e:
            logger.error(f"Redis push error (async): {e}")
            return False
    
    def pop_face_result(self, timeout: float = 5.0) -> Optional[FaceResult]:
        """Pop face result from results queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('results')
            result = client.brpop(queue_name, timeout=timeout)
            
            if result:
                _, data = result
                return self._deserialize(data, FaceResult)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis pop error: {e}")
            return None
    
    # Storage Queue Operations
    
    def push_storage_task(self, storage_task: StorageTask) -> bool:
        """Push storage task to storage queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('storage')
            data = self._serialize(storage_task)
            result = client.rpush(queue_name, data)
            logger.debug(f"Pushed storage task to queue: {storage_task.image_task.phash[:8]}...")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis push storage task error: {e}")
            return False
    
    async def push_storage_task_async(self, storage_task: StorageTask) -> bool:
        """Push storage task to storage queue (async version)."""
        try:
            client = await self._get_async_client()
            queue_name = self.config.get_queue_name('storage')
            data = self._serialize(storage_task)
            result = await client.rpush(queue_name, data)
            pushed = bool(result)
            queue_depth = await self.get_queue_length_by_key_async(queue_name)
            image_phash = storage_task.image_task.phash[:8] if storage_task.image_task.phash else 'NO_PHASH'
            logger.debug(f"[REDIS] DIAG: Pushed storage task: {image_phash}..., "
                       f"success={pushed}, queue_depth={queue_depth}")
            logger.debug(f"Pushed storage task to queue (async): {storage_task.image_task.phash[:8]}...")
            return pushed
        except redis.RedisError as e:
            logger.error(f"Redis push storage task error (async): {e}")
            return False
    
    def pop_storage_task(self, timeout: float = 5.0) -> Optional[StorageTask]:
        """Pop storage task from storage queue."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name('storage')
            result = client.brpop(queue_name, timeout=timeout)
            
            if result:
                _, data = result
                return self._deserialize(data, StorageTask)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis pop storage task error: {e}")
            return None
    
    async def pop_storage_task_async(self, timeout: float = 5.0) -> Optional[StorageTask]:
        """Pop storage task from storage queue (async version)."""
        try:
            client = await self._get_async_client()
            queue_name = self.config.get_queue_name('storage')
            result = await client.brpop(queue_name, timeout=timeout)
            
            if result:
                _, data = result
                return self._deserialize(data, StorageTask)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis pop storage task error (async): {e}")
            return None
    
    # GPU Inbox Operations (new centralized batching)
    
    def push_many(self, key: str, payloads: list[bytes]) -> int:
        """
        Push multiple payloads to a Redis queue atomically.
        
        Args:
            key: Redis queue key
            payloads: List of bytes payloads to push
            
        Returns:
            Number of items successfully pushed
        """
        if not payloads:
            return 0
        try:
            # RPUSH = FIFO with BLPOP on consumer
            client = self._get_client()
            with client.pipeline() as p:
                p.rpush(key, *payloads)
                p.execute()
            return len(payloads)
        except redis.RedisError as e:
            logger.error(f"Redis push_many error: {e}")
            return 0
    
    def blpop_many(self, key: str, max_n: int, timeout: float = 0.5) -> list[bytes]:
        """
        Block for first item, then drain up to max_n-1 without blocking.
        
        Args:
            key: Redis queue key
            max_n: Maximum number of items to retrieve
            timeout: Blocking timeout for first item (seconds)
            
        Returns:
            List of bytes payloads (up to max_n items)
        """
        items: list[bytes] = []
        try:
            client = self._get_client()
            
            # DIAGNOSTIC: Test connection and get connection info
            try:
                ping_result = client.ping()
                connection_info = {
                    'host': client.connection_pool.connection_kwargs.get('host', 'unknown'),
                    'port': client.connection_pool.connection_kwargs.get('port', 'unknown'),
                    'db': client.connection_pool.connection_kwargs.get('db', 'unknown'),
                    'decode_responses': client.connection_pool.connection_kwargs.get('decode_responses', 'unknown'),
                }
            except Exception as conn_e:
                ping_result = False
                connection_info = {'error': str(conn_e)}
            
            # Check actual queue depth before blpop for diagnostics
            actual_depth = client.llen(key)
            
            # Track desync: if we see zero depth multiple times, it might indicate a stale connection
            current_time = time.time()
            if actual_depth == 0:
                self._consecutive_zero_depth_count += 1
            else:
                self._consecutive_zero_depth_count = 0
                self._last_non_zero_depth_time = current_time
            
            # Detect desync: if we've seen zero depth 10+ times in a row and it's been >5 seconds since last non-zero,
            # or if we've seen zero for >30 seconds continuously, reset the connection pool
            should_reset = False
            if key == 'gpu:inbox' and actual_depth == 0:
                if (self._consecutive_zero_depth_count >= 10 and 
                    (current_time - self._last_non_zero_depth_time) > 5.0):
                    should_reset = True
                    logger.warning(f"[REDIS] DESYNC DETECTED: {self._consecutive_zero_depth_count} consecutive zero-depth reads, "
                                 f"last non-zero was {current_time - self._last_non_zero_depth_time:.1f}s ago")
                elif self._consecutive_zero_depth_count > 0 and (current_time - self._last_non_zero_depth_time) > 30.0:
                    should_reset = True
                    logger.warning(f"[REDIS] DESYNC DETECTED: Zero depth for {current_time - self._last_non_zero_depth_time:.1f}s")
            
            if should_reset:
                self.reset_sync_connection_pool()
                # Get fresh client and retry depth check
                client = self._get_client()
                actual_depth = client.llen(key)
                logger.info(f"[REDIS] After pool reset: sync_depth={actual_depth}")
            
            # Log comprehensive diagnostics with throttling (max every 500ms)
            current_time = time.time()
            should_log_diag = (current_time - self._last_diag_log_time) >= self._diag_log_interval
            if should_log_diag:
                redis_url_redacted = self.config.redis_url.replace('://', '://***@') if '@' in self.config.redis_url else self.config.redis_url
                logger.debug(f"[REDIS] DIAG: blpop_many(key='{key}', max_n={max_n}, timeout={timeout}, "
                           f"sync_depth={actual_depth}, ping={ping_result}, conn={connection_info}, "
                           f"redis_url={redis_url_redacted})")
                self._last_diag_log_time = current_time
            
            # 1) Block for first item
            res = client.blpop(key, timeout=timeout)
            if not res:
                # Log when blpop times out - include actual queue depth to diagnose (throttled)
                if should_log_diag:
                    logger.debug(f"[REDIS] DIAG: blpop_many(key='{key}') TIMEOUT after {timeout}s, "
                              f"sync_depth={actual_depth}, ping={ping_result}, conn={connection_info}, returning 0 items")
                return items
            _, first = res
            items.append(first)
            
            # 2) Drain tail quickly (non-blocking)
            drain = max_n - 1
            if drain > 0:
                with client.pipeline() as p:
                    for _ in range(drain):
                        p.lpop(key)  # Continue from the head (RPUSH/BLPOP/LPOP = FIFO)
                    drained = p.execute()
                items.extend([x for x in drained if x is not None])
            
            # Log when items are retrieved (count of items)
            logger.debug(f"[REDIS] DIAG: blpop_many({key}) retrieved {len(items)} items")
            return items
        except redis.RedisError as e:
            logger.error(f"Redis blpop_many error: {e}", exc_info=True)
            return items
    
    def serialize_image_task(self, task: ImageTask) -> bytes:
        """
        Serialize ImageTask to bytes.
        
        Args:
            task: ImageTask to serialize
            
        Returns:
            Serialized bytes
        """
        return self._serialize(task)
    
    def deserialize_image_task(self, data: bytes) -> ImageTask:
        """
        Deserialize bytes to ImageTask.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized ImageTask
        """
        return self._deserialize(data, ImageTask)
    
    # Queue Monitoring
    
    def get_queue_depth(self, queue_type: str) -> int:
        """Get current queue depth."""
        try:
            client = self._get_client()
            queue_name = self.config.get_queue_name(queue_type)
            return client.llen(queue_name)
        except Exception as e:
            logger.error(f"Failed to get queue depth for {queue_type}: {e}")
            return 0
    
    async def get_queue_depth_async(self, queue_type: str) -> int:
        """Get current queue depth (async version)."""
        try:
            client = await self._get_async_client()
            queue_name = self.config.get_queue_name(queue_type)
            return await client.llen(queue_name)
        except Exception as e:
            logger.error(f"Failed to get queue depth for {queue_type} (async): {e}")
            return 0
    
    def get_queue_length_by_key(self, key: str) -> int:
        """Get queue length directly by Redis key name (not queue type)."""
        try:
            client = self._get_client()
            return client.llen(key)
        except Exception as e:
            logger.debug(f"Failed to get queue length for key {key}: {e}")
            return 0
    
    async def get_queue_length_by_key_async(self, key: str) -> int:
        """Get queue length directly by Redis key name (async version)."""
        try:
            client = await self._get_async_client()
            
            # DIAGNOSTIC: Test connection and get connection info
            try:
                ping_result = await client.ping()
                connection_info = {
                    'host': client.connection_pool.connection_kwargs.get('host', 'unknown'),
                    'port': client.connection_pool.connection_kwargs.get('port', 'unknown'),
                    'db': client.connection_pool.connection_kwargs.get('db', 'unknown'),
                    'decode_responses': client.connection_pool.connection_kwargs.get('decode_responses', 'unknown'),
                }
            except Exception as conn_e:
                ping_result = False
                connection_info = {'error': str(conn_e)}
            
            async_depth = await client.llen(key)
            
            # DIAGNOSTIC: Also check with sync client for comparison
            sync_depth = None
            try:
                sync_client = self._get_client()
                sync_depth = sync_client.llen(key)
            except Exception as sync_e:
                sync_depth = f"Error: {sync_e}"
            
            # DESYNC DETECTION: If async and sync see significantly different depths, reset sync pool
            if key == 'gpu:inbox' and isinstance(sync_depth, int):
                depth_diff = abs(async_depth - sync_depth)
                # If async sees items but sync sees zero (or vice versa with large difference), it's a desync
                if (async_depth > 10 and sync_depth == 0) or (sync_depth > 10 and async_depth == 0):
                    logger.warning(f"[REDIS] DESYNC DETECTED: async_depth={async_depth}, sync_depth={sync_depth}, "
                                 f"difference={depth_diff}. Resetting sync connection pool.")
                    self.reset_sync_connection_pool()
                    # Retry sync depth check after reset
                    try:
                        sync_client = self._get_client()
                        sync_depth = sync_client.llen(key)
                        logger.info(f"[REDIS] After pool reset: sync_depth={sync_depth}, async_depth={async_depth}")
                    except Exception as retry_e:
                        sync_depth = f"Error after reset: {retry_e}"
                elif depth_diff > 50:  # Large difference even if both non-zero
                    logger.warning(f"[REDIS] DESYNC DETECTED: Large depth difference (async={async_depth}, sync={sync_depth}, "
                                 f"diff={depth_diff}). Resetting sync connection pool.")
                    self.reset_sync_connection_pool()
                    try:
                        sync_client = self._get_client()
                        sync_depth = sync_client.llen(key)
                        logger.info(f"[REDIS] After pool reset: sync_depth={sync_depth}, async_depth={async_depth}")
                    except Exception as retry_e:
                        sync_depth = f"Error after reset: {retry_e}"
            
            # Log comprehensive diagnostics with throttling (only for gpu:inbox to avoid spam)
            if key == 'gpu:inbox':
                current_time = time.time()
                should_log_diag = (current_time - self._last_diag_log_time) >= self._diag_log_interval
                if should_log_diag:
                    redis_url_redacted = self.config.redis_url.replace('://', '://***@') if '@' in self.config.redis_url else self.config.redis_url
                    logger.debug(f"[REDIS] DIAG: get_queue_length_by_key_async(key='{key}', "
                               f"async_depth={async_depth}, sync_depth={sync_depth}, "
                               f"ping={ping_result}, conn={connection_info}, redis_url={redis_url_redacted})")
                    self._last_diag_log_time = current_time
            
            return async_depth
        except Exception as e:
            logger.debug(f"Failed to get queue length for key {key} (async): {e}")
            return 0
    
    def get_queue_metrics(self, queue_type: str) -> QueueMetrics:
        """Get queue metrics."""
        try:
            depth = self.get_queue_depth(queue_type)
            utilization = (depth / self.config.nc_max_queue_depth) * 100
            
            return QueueMetrics(
                queue_name=queue_type,
                depth=depth,
                max_depth=self.config.nc_max_queue_depth,
                utilization_percent=utilization
            )
        except Exception as e:
            logger.error(f"Failed to get queue metrics for {queue_type}: {e}")
            return QueueMetrics(
                queue_name=queue_type,
                depth=0,
                max_depth=self.config.nc_max_queue_depth,
                utilization_percent=0.0
            )
    
    async def get_queue_metrics_async(self, queue_type: str) -> QueueMetrics:
        """Get queue metrics (async version)."""
        try:
            depth = await self.get_queue_depth_async(queue_type)
            utilization = (depth / self.config.nc_max_queue_depth) * 100
            
            return QueueMetrics(
                queue_name=queue_type,
                depth=depth,
                max_depth=self.config.nc_max_queue_depth,
                utilization_percent=utilization
            )
        except Exception as e:
            logger.error(f"Failed to get queue metrics for {queue_type} (async): {e}")
            return QueueMetrics(
                queue_name=queue_type,
                depth=0,
                max_depth=self.config.nc_max_queue_depth,
                utilization_percent=0.0
            )
    
    def get_all_queue_metrics(self) -> List[QueueMetrics]:
        """Get metrics for all queues."""
        queue_types = ['sites', 'candidates', 'images', 'results']
        return [self.get_queue_metrics(qt) for qt in queue_types]
    
    async def get_all_queue_metrics_async(self) -> List[QueueMetrics]:
        """Get metrics for all queues (async version)."""
        queue_types = ['sites', 'candidates', 'images', 'results']
        metrics = []
        for qt in queue_types:
            metrics.append(await self.get_queue_metrics_async(qt))
        return metrics
    
    def is_queue_full(self, queue_type: str, threshold: float = None) -> bool:
        """Check if queue is full based on threshold."""
        if threshold is None:
            threshold = self.config.backpressure_threshold
        
        metrics = self.get_queue_metrics(queue_type)
        return metrics.utilization_percent > (threshold * 100)
    
    async def is_queue_full_async(self, queue_type: str, threshold: float = None) -> bool:
        """Check if queue is full based on threshold (async version)."""
        if threshold is None:
            threshold = self.config.backpressure_threshold
        
        metrics = await self.get_queue_metrics_async(queue_type)
        return metrics.utilization_percent > (threshold * 100)
    
    def should_apply_backpressure(self) -> bool:
        """Check if back-pressure should be applied."""
        # Apply back-pressure if any critical queue is too full
        critical_queues = ['images', 'results']
        for queue_type in critical_queues:
            if self.is_queue_full(queue_type):
                logger.warning(f"Back-pressure triggered by {queue_type} queue")
                return True
        return False
    
    async def should_apply_backpressure_async(self) -> bool:
        """Check if back-pressure should be applied (async version)."""
        # Apply back-pressure if any critical queue is too full
        critical_queues = ['images', 'results']
        for queue_type in critical_queues:
            if await self.is_queue_full_async(queue_type):
                logger.warning(f"Back-pressure triggered by {queue_type} queue (async)")
                return True
        return False
    
    # Cache Operations
    
    def set_cache(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        """Set cache value with TTL."""
        try:
            client = self._get_client()
            if ttl_seconds is None:
                ttl_seconds = self.config.nc_cache_ttl_days * 24 * 3600
            
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = client.setex(key, ttl_seconds, value)
            logger.debug(f"Set cache key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value."""
        try:
            client = self._get_client()
            value = client.get(key)
            
            if value is None:
                return None
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def delete_cache(self, key: str) -> bool:
        """Delete cache key."""
        try:
            client = self._get_client()
            result = client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists_cache(self, key: str) -> bool:
        """Check if cache key exists."""
        try:
            client = self._get_client()
            return bool(client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def url_seen(self, url: str) -> bool:
        """Check if URL has been seen (for deduplication)."""
        try:
            client = self._get_client()
            key = "candidates:urls_seen"
            normalized_url = self._normalize_url(url)
            return bool(client.sismember(key, normalized_url))
        except redis.RedisError as e:
            logger.error(f"Redis url_seen error: {e}")
            return False
    
    async def url_seen_async(self, url: str) -> bool:
        """Check if URL has been seen (async version)."""
        try:
            client = await self._get_async_client()
            key = "candidates:urls_seen"
            normalized_url = self._normalize_url(url)
            return bool(await client.sismember(key, normalized_url))
        except redis.RedisError as e:
            logger.error(f"Redis url_seen_async error: {e}")
            return False
    
    def mark_url_seen(self, url: str, ttl_seconds: int = None) -> bool:
        """Mark URL as seen (for deduplication)."""
        try:
            client = self._get_client()
            key = "candidates:urls_seen"
            normalized_url = self._normalize_url(url)
            client.sadd(key, normalized_url)
            if ttl_seconds:
                client.expire(key, ttl_seconds)
            return True
        except redis.RedisError as e:
            logger.error(f"Redis mark_url_seen error: {e}")
            return False
    
    async def mark_url_seen_async(self, url: str, ttl_seconds: int = None) -> bool:
        """Mark URL as seen (async version)."""
        try:
            client = await self._get_async_client()
            key = "candidates:urls_seen"
            normalized_url = self._normalize_url(url)
            await client.sadd(key, normalized_url)
            if ttl_seconds:
                await client.expire(key, ttl_seconds)
            return True
        except redis.RedisError as e:
            logger.error(f"Redis mark_url_seen_async error: {e}")
            return False
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication (strip query params/fragments that don't affect image)."""
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        # Keep scheme, netloc, path - strip params, query, fragment
        # This handles URLs like "image.jpg?v=123&size=large#fragment" -> "image.jpg"
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        return normalized
    
    # Queue Management
    
    def clear_queues(self) -> bool:
        """Clear all crawler queues."""
        try:
            client = self._get_client()
            queue_names = list(self.config.queue_names.values())
            result = client.delete(*queue_names)
            logger.info(f"Cleared {result} crawler queues")
            # Reset active tasks counter
            client.set(self._active_tasks_key, 0)
            return True
        except Exception as e:
            logger.error(f"Failed to clear queues: {e}")
            return False
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """Get lengths of all queues."""
        try:
            client = self._get_client()
            queue_names = self.config.queue_names
            lengths = {}
            
            for queue_type, queue_name in queue_names.items():
                lengths[queue_type] = client.llen(queue_name)
            
            return lengths
        except Exception as e:
            logger.error(f"Failed to get queue lengths: {e}")
            return {}

    # Active Task Counters (for orchestrator to wait on JS tasks)
    def incr_active_tasks(self, amount: int = 1) -> int:
        try:
            client = self._get_client()
            return client.incrby(self._active_tasks_key, amount)
        except Exception as e:
            logger.error(f"Failed to increment active tasks: {e}")
            return 0
    
    def decr_active_tasks(self, amount: int = 1) -> int:
        try:
            client = self._get_client()
            new_val = client.decrby(self._active_tasks_key, amount)
            if new_val < 0:
                client.set(self._active_tasks_key, 0)
                return 0
            return new_val
        except Exception as e:
            logger.error(f"Failed to decrement active tasks: {e}")
            return 0

    def get_active_task_count(self) -> int:
        try:
            client = self._get_client()
            val = client.get(self._active_tasks_key)
            return int(val or 0)
        except Exception:
            return 0
    
    async def get_queue_lengths_async(self) -> Dict[str, int]:
        """Get lengths of all queues (async version)."""
        try:
            client = await self._get_async_client()
            queue_names = self.config.queue_names
            lengths = {}
            
            for queue_type, queue_name in queue_names.items():
                lengths[queue_type] = await client.llen(queue_name)
            
            return lengths
        except Exception as e:
            logger.error(f"Failed to get queue lengths (async): {e}")
            return {}
    
    # Health Check
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            client = self._get_client()
            
            # Test basic operations
            client.ping()
            
            # Get queue metrics
            queue_metrics = self.get_all_queue_metrics()
            queue_lengths = self.get_queue_lengths()
            
            return {
                'status': 'healthy',
                'connection': True,
                'queue_metrics': [qm.dict() for qm in queue_metrics],
                'queue_lengths': queue_lengths,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                'status': 'unhealthy',
                'connection': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    # Statistics Operations
    
    def update_site_stats(self, site_id: str, stats_update: Dict[str, int]) -> bool:
        """Update site statistics in Redis."""
        try:
            client = self._get_client()
            key = f"stats:{site_id}"
            for field, value in stats_update.items():
                client.hincrby(key, field, value)
            logger.debug(f"Updated stats for site {site_id}: {stats_update}")
            return True
        except Exception as e:
            logger.error(f"Failed to update stats for site {site_id}: {e}")
            return False
    
    async def update_site_stats_async(self, site_id: str, stats_update: Dict[str, Any]) -> bool:
        """Update site statistics in Redis (async)."""
        try:
            client = await self._get_async_client()
            key = f"stats:{site_id}"
            
            # Handle datetime fields by converting to ISO format
            processed_stats = {}
            for field, value in stats_update.items():
                if isinstance(value, datetime):
                    processed_stats[field] = value.isoformat()
                else:
                    processed_stats[field] = value
            
            # Use HINCRBY for integers and HINCRBYFLOAT for floats (both atomic)
            # For non-numeric values, use HSET
            async with client.pipeline() as pipeline:
                for field, value in processed_stats.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, int):
                            await pipeline.hincrby(key, field, value)
                        else:
                            # Use HINCRBYFLOAT for atomic float increment
                            # This is atomic, unlike read-modify-write
                            await pipeline.hincrbyfloat(key, field, value)
                    else:
                        await pipeline.hset(key, field, str(value))
                await pipeline.execute()
            
            logger.debug(f"Updated stats for site {site_id} (async): {processed_stats}")
            return True
        except Exception as e:
            logger.error(f"Failed to update stats for site {site_id} (async): {e}")
            return False
    
    def get_site_stats(self, site_id: str) -> Dict[str, Any]:
        """Get site statistics from Redis."""
        try:
            client = self._get_client()
            key = f"stats:{site_id}"
            raw_stats = client.hgetall(key)
            # Convert bytes keys/values to strings/appropriate types
            stats = {}
            for key_bytes, value_bytes in raw_stats.items():
                key_str = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                value_str = value_bytes.decode() if isinstance(value_bytes, bytes) else value_bytes
                
                # Try to convert to int, float, or keep as string
                try:
                    if '.' in value_str:
                        stats[key_str] = float(value_str)
                    else:
                        stats[key_str] = int(value_str)
                except ValueError:
                    # Keep as string (for datetime fields)
                    stats[key_str] = value_str
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for site {site_id}: {e}")
            return {}
    
    def get_all_site_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all site statistics."""
        try:
            client = self._get_client()
            stats = {}
            keys = client.keys("stats:*")
            for key_bytes in keys:
                key_str = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                site_id = key_str.split(":", 1)[1]
                stats[site_id] = self.get_site_stats(site_id)
            return stats
        except Exception as e:
            logger.error(f"Failed to get all site stats: {e}")
            return {}
    
    def clear_site_stats(self) -> bool:
        """Clear all site statistics."""
        try:
            client = self._get_client()
            keys = client.keys("stats:*")
            if keys:
                client.delete(*keys)
                logger.info(f"Cleared {len(keys)} site stats")
            return True
        except Exception as e:
            logger.error(f"Failed to clear site stats: {e}")
            return False
    
    # Shared Batch Management
    # DEPRECATED: These methods are no longer used with GPU scheduler migration.
    # Extractors now push directly to gpu:inbox queue, and scheduler handles batching.
    # Kept for backward compatibility during migration period.
    
    def push_to_shared_batch(self, image_task: ImageTask) -> int:
        """DEPRECATED: Push image task to shared batch accumulator and return new size.
        
        Replaced by: push_many() to gpu:inbox queue.
        """
        try:
            client = self._get_client()
            batch_key = "batch:accumulator"
            data = self._serialize(image_task)
            # Push and get new length atomically
            pipeline = client.pipeline()
            pipeline.rpush(batch_key, data)
            pipeline.llen(batch_key)
            results = pipeline.execute()
            new_size = results[1]  # Length after push
            return new_size
        except Exception as e:
            logger.error(f"Failed to push to shared batch: {e}")
            return 0

    async def push_to_shared_batch_async(self, image_task: ImageTask) -> int:
        """DEPRECATED: Push image task to shared batch accumulator (async).
        
        Replaced by: push_many() to gpu:inbox queue.
        """
        try:
            client = await self._get_async_client()
            batch_key = "batch:accumulator"
            ts_key = "batch:accumulator:last_ts"
            data = self._serialize(image_task)
            # Push and get new length atomically
            async with client.pipeline() as pipeline:
                await pipeline.rpush(batch_key, data)
                await pipeline.llen(batch_key)
                # Only set timestamp if batch was empty (first push after flush)
                await pipeline.setnx(ts_key, int(time.time() * 1000))
                results = await pipeline.execute()
            new_size = results[1]
            return new_size
        except Exception as e:
            logger.error(f"Failed to push to shared batch (async): {e}")
            return 0

    async def flush_shared_batch_if_ready_async(self, batch_size_threshold: int) -> bool:
        """DEPRECATED: Atomically flush shared batch if it reaches threshold. Returns True if flushed.
        
        Replaced by: GPU scheduler handles batching automatically.
        """
        try:
            client = await self._get_async_client()
            batch_key = "batch:accumulator"
            
            # Use Lua script for atomic check-and-flush
            lua_script = """
            local batch_key = KEYS[1]
            local threshold = tonumber(ARGV[1])
            local current_size = redis.call('LLEN', batch_key)
            
            if current_size >= threshold then
                -- Pop exactly 'threshold' items
                local items = redis.call('LRANGE', batch_key, 0, threshold - 1)
                redis.call('LTRIM', batch_key, threshold, -1)
                return {1, items}  -- Return success=1 and items
            else
                return {0, {}}  -- Return success=0
            end
            """
            
            result = await client.eval(lua_script, 1, batch_key, batch_size_threshold)
            
            if result[0] == 1:
                # Successfully flushed - deserialize and push to images queue
                items = result[1]
                image_tasks = []
                for item in items:
                    # Handle both bytes and string returns from Lua
                    if isinstance(item, str):
                        item = item.encode('utf-8')
                    image_tasks.append(self._deserialize(item, ImageTask))
                
                # Push batch to images queue
                await self.push_image_batch_async(image_tasks)
                logger.info(f"Flushed shared batch: {len(image_tasks)} images")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to flush shared batch: {e}")
            return False

    async def flush_shared_batch_if_stale_async(self, stale_ms: int, min_items: int, max_batch_size: int) -> bool:
        """DEPRECATED: Flush shared batch if stale time exceeded and at least min_items present. Returns True if flushed.
        
        Replaced by: GPU scheduler handles timing automatically.
        """
        try:
            client = await self._get_async_client()
            batch_key = "batch:accumulator"
            ts_key = "batch:accumulator:last_ts"
            now_ms = int(time.time() * 1000)
            lua_script = """
            local batch_key = KEYS[1]
            local ts_key = KEYS[2]
            local stale_ms = tonumber(ARGV[1])
            local min_items = tonumber(ARGV[2])
            local max_batch = tonumber(ARGV[3])
            local now_ms = tonumber(ARGV[4])
            local current_size = redis.call('LLEN', batch_key)
            if current_size == 0 then
                return {0, {}}
            end
            local last_ts = tonumber(redis.call('GET', ts_key) or '0')
            -- Flush if ANY condition is true:
            -- 1. Batch full (size-based immediate flush)
            -- 2. Batch has min items AND is stale (time-based flush)
            if (current_size >= max_batch) or (current_size >= min_items and (now_ms - last_ts) >= stale_ms) then
                local n = max_batch
                if current_size < n then n = current_size end
                local items = redis.call('LRANGE', batch_key, 0, n - 1)
                redis.call('LTRIM', batch_key, n, -1)
                -- CRITICAL: Delete timestamp so next push creates new one
                redis.call('DEL', ts_key)
                return {1, items}
            else
                return {0, {}}
            end
            """
            result = await client.eval(lua_script, 2, batch_key, ts_key, stale_ms, min_items, max_batch_size, now_ms)
            if result[0] == 1:
                items = result[1]
                image_tasks = []
                for item in items:
                    if isinstance(item, str):
                        item = item.encode('utf-8')
                    image_tasks.append(self._deserialize(item, ImageTask))
                await self.push_image_batch_async(image_tasks)
                logger.info(f"Flushed stale shared batch: {len(image_tasks)} images")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to flush stale shared batch: {e}")
            return False

    # Site limit flags
    def set_site_limit_reached(self, site_id: str) -> bool:
        try:
            client = self._get_client()
            key = f"{self._site_limit_flag_prefix}{site_id}"
            client.set(key, 1)
            return True
        except Exception as e:
            logger.error(f"Failed to set site limit flag for {site_id}: {e}")
            return False

    async def set_site_limit_reached_async(self, site_id: str) -> bool:
        try:
            client = await self._get_async_client()
            key = f"{self._site_limit_flag_prefix}{site_id}"
            await client.set(key, 1)
            return True
        except Exception as e:
            logger.error(f"Failed to set site limit flag for {site_id} (async): {e}")
            return False

    def is_site_limit_reached(self, site_id: str) -> bool:
        try:
            client = self._get_client()
            key = f"{self._site_limit_flag_prefix}{site_id}"
            return bool(client.exists(key))
        except Exception:
            return False

    async def is_site_limit_reached_async(self, site_id: str) -> bool:
        try:
            client = await self._get_async_client()
            key = f"{self._site_limit_flag_prefix}{site_id}"
            return bool(await client.exists(key))
        except Exception:
            return False

    async def all_sites_limit_reached_async(self, site_ids: List[str]) -> bool:
        try:
            client = await self._get_async_client()
            keys = [f"{self._site_limit_flag_prefix}{sid}" for sid in site_ids]
            if not keys:
                return False
            exists = await client.exists(*keys)
            return int(exists) == len(keys)
        except Exception:
            return False

    # Queue clearing helpers
    def clear_queue(self, queue_type: str) -> bool:
        try:
            client = self._get_client()
            qn = self.config.get_queue_name(queue_type)
            client.delete(qn)
            return True
        except Exception as e:
            logger.error(f"Failed to clear queue {queue_type}: {e}")
            return False

    async def clear_queue_async(self, queue_type: str) -> bool:
        try:
            client = await self._get_async_client()
            qn = self.config.get_queue_name(queue_type)
            await client.delete(qn)
            return True
        except Exception as e:
            logger.error(f"Failed to clear queue {queue_type} (async): {e}")
            return False

    def get_shared_batch_size(self) -> int:
        """DEPRECATED: Get current size of shared batch accumulator.
        
        Replaced by: Direct LLEN check on gpu:inbox queue if needed.
        """
        try:
            client = self._get_client()
            return client.llen("batch:accumulator")
        except Exception as e:
            logger.error(f"Failed to get shared batch size: {e}")
            return 0

    async def get_shared_batch_size_async(self) -> int:
        """DEPRECATED: Get current size of shared batch accumulator (async).
        
        Replaced by: Direct LLEN check on gpu:inbox queue if needed.
        """
        try:
            client = await self._get_async_client()
            return await client.llen("batch:accumulator")
        except Exception as e:
            logger.error(f"Failed to get shared batch size (async): {e}")
            return 0

    async def set_gpu_processing_async(self, batch_id: str) -> bool:
        """Mark GPU as processing a batch."""
        try:
            client = await self._get_async_client()
            await client.setex("gpu:processing", 60, batch_id)
            return True
        except Exception as e:
            logger.error(f"Failed to set GPU processing: {e}")
            return False

    async def clear_gpu_processing_async(self) -> bool:
        """Mark GPU as idle."""
        try:
            client = await self._get_async_client()
            await client.delete("gpu:processing")
            return True
        except Exception as e:
            logger.error(f"Failed to clear GPU processing: {e}")
            return False

    async def is_gpu_idle_async(self) -> bool:
        """DEPRECATED: Check if GPU is idle (not processing a batch).
        
        No longer needed: GPU scheduler handles pacing automatically.
        """
        try:
            client = await self._get_async_client()
            return not await client.exists("gpu:processing")
        except Exception:
            return True

    # Cleanup
    
    def close(self):
        """Close Redis connections."""
        try:
            if self._client:
                self._client.close()
            if self._pool:
                self._pool.disconnect()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")
    
    async def close_async(self):
        """Close async Redis connections."""
        try:
            if self._async_client:
                await self._async_client.close()
            if self._async_pool:
                await self._async_pool.disconnect()
            logger.info("Async Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing async Redis connections: {e}")

    # Domain Rendering Strategy Management
    
    def set_domain_rendering_strategy(self, domain: str, use_js: bool, http_count: int, js_count: int) -> bool:
        """Store the winning rendering strategy for a domain."""
        try:
            client = self._get_client()
            key = f"domain:strategy:{domain}"
            data = {
                'use_js': use_js,
                'http_count': http_count,
                'js_count': js_count,
                'timestamp': time.time()
            }
            # Store with 7-day TTL (sites don't change rendering often)
            client.setex(key, 7 * 24 * 3600, json.dumps(data))
            logger.info(f"Stored rendering strategy for {domain}: use_js={use_js} (HTTP={http_count}, JS={js_count})")
            return True
        except Exception as e:
            logger.error(f"Failed to store domain strategy for {domain}: {e}")
            return False

    def get_domain_rendering_strategy(self, domain: str) -> Optional[bool]:
        """Get the stored rendering strategy for a domain. Returns None if not yet determined."""
        try:
            client = self._get_client()
            key = f"domain:strategy:{domain}"
            value = client.get(key)
            if value:
                data = json.loads(value)
                return data.get('use_js')
            return None
        except Exception as e:
            logger.debug(f"Failed to get domain strategy for {domain}: {e}")
            return None

    async def set_domain_rendering_strategy_async(self, domain: str, use_js: bool, http_count: int, js_count: int) -> bool:
        """Store the winning rendering strategy for a domain (async)."""
        try:
            client = await self._get_async_client()
            key = f"domain:strategy:{domain}"
            data = {
                'use_js': use_js,
                'http_count': http_count,
                'js_count': js_count,
                'timestamp': time.time()
            }
            await client.setex(key, 7 * 24 * 3600, json.dumps(data))
            logger.info(f"Stored rendering strategy for {domain}: use_js={use_js} (HTTP={http_count}, JS={js_count})")
            return True
        except Exception as e:
            logger.error(f"Failed to store domain strategy for {domain}: {e}")
            return False

    async def get_domain_rendering_strategy_async(self, domain: str) -> Optional[bool]:
        """Get the stored rendering strategy for a domain (async). Returns None if not yet determined."""
        try:
            client = await self._get_async_client()
            key = f"domain:strategy:{domain}"
            value = await client.get(key)
            if value:
                data = json.loads(value)
                return data.get('use_js')
            return None
        except Exception as e:
            logger.debug(f"Failed to get domain strategy for {domain}: {e}")
            return None




def get_redis_manager() -> RedisManager:
    """Get singleton Redis manager instance."""
    global _redis_manager_instance
    if _redis_manager_instance is None:
        _redis_manager_instance = RedisManager()
    return _redis_manager_instance


def close_redis_manager():
    """Close singleton Redis manager."""
    global _redis_manager_instance
    if _redis_manager_instance:
        _redis_manager_instance.close()
        _redis_manager_instance = None
