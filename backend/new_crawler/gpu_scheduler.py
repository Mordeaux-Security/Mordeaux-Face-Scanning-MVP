"""
GPU Scheduler for New Crawler System

Centralized batching and pacing control for GPU processing.
Eliminates sawtooth utilization patterns by controlling batch timing and size.
"""

import time
import asyncio
import logging
from collections import deque
from typing import List, Optional

from .data_structures import ImageTask

logger = logging.getLogger(__name__)


class GPUScheduler:
    """
    Centralized GPU batch scheduler.
    
    Manages batching and pacing to prevent sawtooth GPU utilization.
    Controls batch size, launch timing, and maximum inflight batches.
    """
    
    def __init__(self, redis_mgr, deserializer, inbox_key: str,
                 target_batch: int = 32, max_wait_ms: int = 12, min_launch_ms: int = 200,
                 config=None):
        """
        Initialize GPU scheduler.
        
        Args:
            redis_mgr: RedisManager instance for queue operations
            deserializer: Function to deserialize bytes to ImageTask
            inbox_key: Redis queue key for GPU input
            target_batch: Target batch size (default 32)
            max_wait_ms: Max milliseconds to wait before launching early batch (default 12)
            min_launch_ms: Minimum milliseconds between batch launches (default 200)
            config: Optional config object for diagnostic logging
        """
        self.redis = redis_mgr
        self.deserialize = deserializer
        self.inbox_key = inbox_key
        self.config = config
        
        self.TARGET = int(target_batch)
        self.MAX_WAIT_MS = int(max_wait_ms)
        self.MIN_LAUNCH_MS = int(min_launch_ms)
        
        self._staging: List[ImageTask] = []
        self._last_launch_ms: float = 0.0
        self._inflight = deque(maxlen=2)  # Track up to 2 inflight batch IDs
        self._last_feed_time_ms: float = self._now_ms()  # Track when we last successfully fed items (init to now)
        
        # Queue depth caching for logging (reduces Redis calls)
        self._cached_queue_depth: Optional[int] = None
        self._cached_queue_depth_time: float = 0.0
        self._queue_depth_cache_ttl_ms: float = 75.0  # 75ms TTL - balance between freshness and Redis load
        
        # Log throttling for repetitive logs (max every 500ms)
        self._last_no_batch_log_time: float = 0.0
        self._no_batch_log_interval: float = 0.5  # 500ms
    
    @staticmethod
    def _now_ms() -> float:
        """Get current time in milliseconds using high-resolution timer."""
        return time.perf_counter() * 1000.0
    
    async def _get_queue_depth_cached(self, use_real_time: bool = False) -> int:
        """
        Get queue depth with caching for logging (non-critical).
        
        Args:
            use_real_time: If True, always fetch fresh value (for critical decisions)
        
        Returns:
            Queue depth
        """
        now_ms = self._now_ms()
        
        # Always use real-time for critical decisions
        if use_real_time:
            depth = await self.redis.get_queue_length_by_key_async(self.inbox_key)
            self._cached_queue_depth = depth
            self._cached_queue_depth_time = now_ms
            return depth
        
        # Use cached value if still valid
        if (self._cached_queue_depth is not None and 
            (now_ms - self._cached_queue_depth_time) < self._queue_depth_cache_ttl_ms):
            return self._cached_queue_depth
        
        # Cache expired, fetch new value
        depth = await self.redis.get_queue_length_by_key_async(self.inbox_key)
        self._cached_queue_depth = depth
        self._cached_queue_depth_time = now_ms
        return depth
    
    def _can_launch(self) -> bool:
        """
        Check if a new batch can be launched.
        
        Returns:
            True if launch is allowed (max 2 inflight, min spacing met)
        """
        if len(self._inflight) >= 2:
            return False
        return (self._now_ms() - self._last_launch_ms) >= self.MIN_LAUNCH_MS
    
    async def feed(self) -> int:
        """
        Bring in up to TARGET items to keep staging warm.
        
        Returns:
            Number of items added to staging
        """
        need = max(0, self.TARGET - len(self._staging))
        
        # Early exit: staging is full, skip all Redis calls
        if need == 0:
            return 0
        
        # Measure wait time and get queue depth
        feed_start = time.time()
        
        # Get REAL-TIME queue depth before blpop (not cached) to diagnose issues
        real_queue_depth = await self.redis.get_queue_length_by_key_async(self.inbox_key)
        queue_depth = await self._get_queue_depth_cached(use_real_time=False)  # For logging only
        
        # Log the key and real queue depth for debugging
        logger.debug(f"[GPU Scheduler] DIAG: feed() attempting to pull from key='{self.inbox_key}', "
                   f"real_queue_depth={real_queue_depth}, cached_queue_depth={queue_depth}, need={need}, staging={len(self._staging)}")
        
        # Use async wrapper for blpop_many (non-blocking)
        raw = await asyncio.to_thread(
            self.redis.blpop_many, 
            self.inbox_key, 
            max_n=need, 
            timeout=0.5
        )
        wait_time_ms = (time.time() - feed_start) * 1000.0
        
        if not raw:
            # Log when feed() times out - include real queue depth to diagnose
            logger.debug(f"[GPU Scheduler] DIAG: feed() TIMEOUT - added=0 items, need={need}, "
                           f"real_queue_depth={real_queue_depth}, cached_queue_depth={queue_depth}, "
                           f"staging={len(self._staging)}, wait_time_ms={wait_time_ms:.1f}, "
                           f"key='{self.inbox_key}'")
            return 0
        
        added = 0
        for r in raw:
            try:
                task = self.deserialize(r)
                self._staging.append(task)
                added += 1
            except Exception as e:
                # Malformed payload; skip and log at debug level
                logger.debug(f"[GPU Scheduler] Failed to deserialize payload: {e}")
                continue
        
        # Update last feed time if we added items
        if added > 0:
            self._last_feed_time_ms = self._now_ms()
        
        # Log when items are added (use cached value to reduce Redis calls)
        queue_depth_after = await self._get_queue_depth_cached(use_real_time=False)
        logger.debug(f"[GPU Scheduler] DIAG: feed() added={added} items, need={need}, "
                   f"queue_depth={queue_depth}, queue_depth_after={queue_depth_after}, "
                   f"staging={len(self._staging)}, wait_time_ms={wait_time_ms:.1f}")
        
        return added
    
    async def next_batch(self) -> Optional[List[ImageTask]]:
        """
        Return a list of ImageTasks to process now, or None.
        
        Returns:
            List of ImageTasks ready for processing, or None if not ready
        """
        staging_count = len(self._staging)
        
        # Log why batch isn't ready with throttling (max every 500ms)
        current_time = time.time()
        should_log = (current_time - self._last_no_batch_log_time) >= self._no_batch_log_interval
        
        if not self._staging:
            if should_log:
                logger.debug(f"[GPU-SCHEDULER] NO BATCH: staging is empty (0 items)")
                self._last_no_batch_log_time = current_time
            return None
        
        # Check launch conditions
        waited_ms = self._now_ms() - self._last_launch_ms
        can_launch = self._can_launch()
        inflight_count = len(self._inflight)
        time_since_last_feed_ms = self._now_ms() - self._last_feed_time_ms
        
        # Launch if we've reached target size and can launch
        if staging_count >= self.TARGET:
            if can_launch:
                batch = self._staging[:self.TARGET]
                self._staging = self._staging[self.TARGET:]
                logger.info(f"[GPU-SCHEDULER] BATCH READY: target reached ({staging_count} >= {self.TARGET}), "
                           f"can_launch=True, waited_ms={waited_ms:.1f}, inflight={inflight_count}")
                return batch
            else:
                # Can't launch - explain why
                if should_log:
                    if inflight_count >= 2:
                        logger.debug(f"[GPU-SCHEDULER] NO BATCH: target reached ({staging_count} >= {self.TARGET}) BUT "
                                   f"can_launch=False (inflight={inflight_count} >= 2), waited_ms={waited_ms:.1f}ms")
                    else:
                        min_launch_remaining = self.MIN_LAUNCH_MS - waited_ms
                        logger.debug(f"[GPU-SCHEDULER] NO BATCH: target reached ({staging_count} >= {self.TARGET}) BUT "
                                   f"can_launch=False (MIN_LAUNCH_MS not met: {waited_ms:.1f}ms < {self.MIN_LAUNCH_MS}ms, "
                                   f"need {min_launch_remaining:.1f}ms more), inflight={inflight_count}")
                    self._last_no_batch_log_time = current_time
        
        # Launch early if we've waited long enough
        floor = max(8, self.TARGET // 4)
        if can_launch and waited_ms >= self.MAX_WAIT_MS:
            if staging_count >= floor:
                n = min(staging_count, self.TARGET)
                batch = self._staging[:n]
                self._staging = self._staging[n:]
                logger.info(f"[GPU-SCHEDULER] BATCH READY: early launch ({n} items, floor={floor} met), "
                           f"waited_ms={waited_ms:.1f} >= {self.MAX_WAIT_MS}ms, staging={staging_count}")
                return batch
            else:
                if should_log:
                    logger.debug(f"[GPU-SCHEDULER] NO BATCH: early launch timeout met ({waited_ms:.1f}ms >= {self.MAX_WAIT_MS}ms) BUT "
                               f"staging={staging_count} < floor={floor} (need {floor - staging_count} more items)")
                    self._last_no_batch_log_time = current_time
        elif can_launch:
            if should_log:
                logger.debug(f"[GPU-SCHEDULER] NO BATCH: early launch timeout NOT met ({waited_ms:.1f}ms < {self.MAX_WAIT_MS}ms), "
                           f"staging={staging_count}, floor={floor}")
                self._last_no_batch_log_time = current_time
        else:
            # Can't launch - explain why
            if should_log:
                if inflight_count >= 2:
                    logger.debug(f"[GPU-SCHEDULER] NO BATCH: early launch blocked (can_launch=False, inflight={inflight_count} >= 2), "
                               f"waited_ms={waited_ms:.1f}ms, staging={staging_count}, floor={floor}")
                else:
                    min_launch_remaining = self.MIN_LAUNCH_MS - waited_ms
                    logger.debug(f"[GPU-SCHEDULER] NO BATCH: early launch blocked (can_launch=False, MIN_LAUNCH_MS not met: "
                               f"{waited_ms:.1f}ms < {self.MIN_LAUNCH_MS}ms, need {min_launch_remaining:.1f}ms more), "
                               f"staging={staging_count}, floor={floor}, inflight={inflight_count}")
                self._last_no_batch_log_time = current_time
        
        # Flush remaining items if queue is empty and we've waited long enough
        # This prevents the last batch from getting stuck in staging
        if can_launch and staging_count > 0:
            flush_timeout_ms = self.MAX_WAIT_MS * 3
            
            if time_since_last_feed_ms >= flush_timeout_ms:
                # Use real-time queue depth for critical flush decision
                queue_depth = await self._get_queue_depth_cached(use_real_time=True)
                if queue_depth == 0:
                    # Flush whatever is left in staging
                    batch = self._staging[:]
                    self._staging = []
                    logger.info(f"[GPU-SCHEDULER] BATCH FLUSHED: {len(batch)} items (queue empty, "
                               f"time_since_last_feed={time_since_last_feed_ms:.1f}ms >= {flush_timeout_ms}ms)")
                    return batch
                else:
                    if should_log:
                        logger.debug(f"[GPU-SCHEDULER] NO BATCH: flush timeout met ({time_since_last_feed_ms:.1f}ms >= {flush_timeout_ms}ms) BUT "
                                   f"queue NOT empty (queue_depth={queue_depth}), staging={staging_count}")
                        self._last_no_batch_log_time = current_time
            else:
                if should_log:
                    flush_remaining_ms = flush_timeout_ms - time_since_last_feed_ms
                    logger.debug(f"[GPU-SCHEDULER] NO BATCH: flush timeout NOT met (time_since_last_feed={time_since_last_feed_ms:.1f}ms < "
                               f"{flush_timeout_ms}ms, need {flush_remaining_ms:.1f}ms more), staging={staging_count}, "
                               f"can_launch={can_launch}")
                    self._last_no_batch_log_time = current_time
        elif staging_count > 0:
            # Can't launch - explain why
            if should_log:
                if inflight_count >= 2:
                    logger.debug(f"[GPU-SCHEDULER] NO BATCH: flush blocked (can_launch=False, inflight={inflight_count} >= 2), "
                               f"staging={staging_count}, time_since_last_feed={time_since_last_feed_ms:.1f}ms")
                else:
                    min_launch_remaining = self.MIN_LAUNCH_MS - waited_ms
                    logger.debug(f"[GPU-SCHEDULER] NO BATCH: flush blocked (can_launch=False, MIN_LAUNCH_MS not met: "
                             f"{waited_ms:.1f}ms < {self.MIN_LAUNCH_MS}ms, need {min_launch_remaining:.1f}ms more), "
                             f"staging={staging_count}, time_since_last_feed={time_since_last_feed_ms:.1f}ms, inflight={inflight_count}")
                self._last_no_batch_log_time = current_time
        
        return None
    
    def mark_launched(self, batch_id: str):
        """
        Mark a batch as launched.
        
        Args:
            batch_id: Unique batch identifier
        """
        self._last_launch_ms = self._now_ms()
        self._inflight.append(batch_id)
    
    def mark_completed(self, batch_id: str):
        """
        Mark a batch as completed.
        
        Args:
            batch_id: Unique batch identifier
        """
        try:
            self._inflight.remove(batch_id)
        except ValueError:
            # Batch ID not found in inflight (shouldn't happen, but handle gracefully)
            pass

