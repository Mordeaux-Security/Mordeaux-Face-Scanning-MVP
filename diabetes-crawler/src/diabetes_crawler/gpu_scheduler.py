"""
GPU Scheduler for New Crawler System

Centralized batching and pacing control for GPU processing.
Eliminates sawtooth utilization patterns by controlling batch timing and size.
"""

import time
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
    
    def __init__(self, redis_mgr, deserializer, inbox_key: str, metadata_deserializer=None,
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
        self.deserialize = metadata_deserializer or deserializer  # Use metadata deserializer if provided
        self.inbox_key = inbox_key
        self.config = config
        
        self.TARGET = int(target_batch)
        self.MAX_WAIT_MS = int(max_wait_ms)
        self.MIN_LAUNCH_MS = int(min_launch_ms)
        
        self._staging: List[ImageTask] = []
        self._last_launch_ms: float = 0.0
        self._inflight = deque(maxlen=2)  # Track up to 2 inflight batch IDs
    
    @staticmethod
    def _now_ms() -> float:
        """Get current time in milliseconds using high-resolution timer."""
        return time.perf_counter() * 1000.0
    
    def _can_launch(self) -> bool:
        """
        Check if a new batch can be launched.
        
        Returns:
            True if launch is allowed (max 2 inflight, min spacing met)
        """
        if len(self._inflight) >= 2:
            return False
        return (self._now_ms() - self._last_launch_ms) >= self.MIN_LAUNCH_MS
    
    def feed(self) -> int:
        """
        Bring in up to TARGET items to keep staging warm.
        
        Returns:
            Number of items added to staging
        """
        need = max(0, self.TARGET - len(self._staging))
        if need == 0:
            return 0
        
        # Measure wait time and get queue depth (always check, not just every 10th call)
        feed_start = time.time()
        queue_depth = self.redis.get_queue_length_by_key(self.inbox_key)
        
        raw = self.redis.blpop_many(self.inbox_key, max_n=need, timeout=0.5)
        wait_time_ms = (time.time() - feed_start) * 1000.0
        
        if not raw:
            # Log at INFO level when feed() returns 0 items (timeout case)
            logger.info(f"[GPU Scheduler] DIAG: feed() added=0 items, need={need}, "
                       f"queue_depth={queue_depth}, staging={len(self._staging)}, wait_time_ms={wait_time_ms:.1f}")
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
        
        # Log at INFO level when items are added (not just DEBUG)
        queue_depth_after = self.redis.get_queue_length_by_key(self.inbox_key)
        logger.info(f"[GPU Scheduler] DIAG: feed() added={added} items, need={need}, "
                   f"queue_depth={queue_depth}, queue_depth_after={queue_depth_after}, "
                   f"staging={len(self._staging)}, wait_time_ms={wait_time_ms:.1f}")
        
        return added
    
    def next_batch(self, force_flush: bool = False) -> Optional[List[ImageTask]]:
        """
        Return a list of ImageTasks to process now, or None.
        
        Args:
            force_flush: If True, flush all staging items regardless of size/thresholds
        
        Returns:
            List of ImageTasks ready for processing, or None if not ready
        """
        diagnostic_enabled = self.config and getattr(self.config, 'nc_diagnostic_logging', False)
        
        if not self._staging:
            return None
        
        # Force flush: if queues are empty, flush everything in staging
        if force_flush:
            batch = self._staging[:]
            self._staging = []
            if diagnostic_enabled:
                logger.debug(f"[GPU-SCHEDULER-DIAG] next_batch() FORCE FLUSH: returning {len(batch)} items, "
                           f"staging_remaining=0")
            logger.info(f"[GPU Scheduler] FORCE FLUSH: Flushed {len(batch)} items from staging (queues empty)")
            return batch
        
        # Launch if we've reached target size and can launch
        if len(self._staging) >= self.TARGET and self._can_launch():
            batch = self._staging[:self.TARGET]
            self._staging = self._staging[self.TARGET:]
            if diagnostic_enabled:
                logger.debug(f"[GPU-SCHEDULER-DIAG] next_batch() returning {len(batch)} items (target reached), "
                           f"staging_remaining={len(self._staging)}, waited_ms=0.0")
            return batch
        
        # Launch early if we've waited long enough
        waited = self._now_ms() - self._last_launch_ms
        can_launch = self._can_launch()
        if can_launch and waited >= self.MAX_WAIT_MS:
            # Avoid tiny batches: floor = max(8, TARGET//4) for stability
            floor = max(8, self.TARGET // 4)
            # Only launch if we have at least floor items (prevents tiny batches)
            if len(self._staging) >= floor:
                n = min(len(self._staging), self.TARGET)
                batch = self._staging[:n]
                self._staging = self._staging[n:]
                if diagnostic_enabled:
                    logger.debug(f"[GPU-SCHEDULER-DIAG] next_batch() returning {len(batch)} items (early launch), "
                               f"staging_remaining={len(self._staging)}, waited_ms={waited:.1f}")
                return batch
        
        # Diagnostic: Log when batch not ready
        if diagnostic_enabled:
            logger.debug(f"[GPU-SCHEDULER-DIAG] next_batch() not ready: staging={len(self._staging)}, "
                       f"can_launch={can_launch}, waited_ms={waited:.1f}, target={self.TARGET}, "
                       f"floor={max(8, self.TARGET // 4)}")
        
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

