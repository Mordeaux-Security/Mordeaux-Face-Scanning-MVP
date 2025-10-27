"""
Redis Manager for New Crawler System

Handles Redis queue operations, connection pooling, and back-pressure monitoring.
Provides clean interface for all queue operations with proper error handling.
"""

import json
import logging
import time
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
    BatchRequest, QueueMetrics, TaskStatus
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
    
    def _deserialize(self, data: bytes, model_class) -> BaseModel:
        """Deserialize bytes to Pydantic model."""
        return model_class.model_validate_json(data.decode('utf-8'))
    
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
            logger.debug(f"Pushed face result to queue (async): {face_result.image_task.candidate.img_url}")
            return bool(result)
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
    
    # Queue Management
    
    def clear_queues(self) -> bool:
        """Clear all crawler queues."""
        try:
            client = self._get_client()
            queue_names = list(self.config.queue_names.values())
            result = client.delete(*queue_names)
            logger.info(f"Cleared {result} crawler queues")
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
    
    async def update_site_stats_async(self, site_id: str, stats_update: Dict[str, int]) -> bool:
        """Update site statistics in Redis (async)."""
        try:
            client = await self._get_async_client()
            key = f"stats:{site_id}"
            for field, value in stats_update.items():
                await client.hincrby(key, field, value)
            logger.debug(f"Updated stats for site {site_id} (async): {stats_update}")
            return True
        except Exception as e:
            logger.error(f"Failed to update stats for site {site_id} (async): {e}")
            return False
    
    def get_site_stats(self, site_id: str) -> Dict[str, int]:
        """Get site statistics from Redis."""
        try:
            client = self._get_client()
            key = f"stats:{site_id}"
            raw_stats = client.hgetall(key)
            # Convert bytes keys/values to strings/ints
            stats = {}
            for key_bytes, value_bytes in raw_stats.items():
                key_str = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                value_int = int(value_bytes.decode()) if isinstance(value_bytes, bytes) else int(value_bytes)
                stats[key_str] = value_int
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for site {site_id}: {e}")
            return {}
    
    def get_all_site_stats(self) -> Dict[str, Dict[str, int]]:
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
    
    def push_to_shared_batch(self, image_task: ImageTask) -> int:
        """Push image task to shared batch accumulator and return new size."""
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
        """Push image task to shared batch accumulator (async)."""
        try:
            client = await self._get_async_client()
            batch_key = "batch:accumulator"
            data = self._serialize(image_task)
            # Push and get new length atomically
            async with client.pipeline() as pipeline:
                await pipeline.rpush(batch_key, data)
                await pipeline.llen(batch_key)
                results = await pipeline.execute()
            new_size = results[1]
            return new_size
        except Exception as e:
            logger.error(f"Failed to push to shared batch (async): {e}")
            return 0

    async def flush_shared_batch_if_ready_async(self, batch_size_threshold: int) -> bool:
        """Atomically flush shared batch if it reaches threshold. Returns True if flushed."""
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

    def get_shared_batch_size(self) -> int:
        """Get current size of shared batch accumulator."""
        try:
            client = self._get_client()
            return client.llen("batch:accumulator")
        except Exception as e:
            logger.error(f"Failed to get shared batch size: {e}")
            return 0

    async def get_shared_batch_size_async(self) -> int:
        """Get current size of shared batch accumulator (async)."""
        try:
            client = await self._get_async_client()
            return await client.llen("batch:accumulator")
        except Exception as e:
            logger.error(f"Failed to get shared batch size (async): {e}")
            return 0

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
