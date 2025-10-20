
"""
Memory Management System

Handles memory monitoring, garbage collection coordination, and resource management
for the crawler system. Provides proactive memory management to prevent OOM crashes
and optimize performance.
"""

import gc
import psutil
import logging
import asyncio
import threading
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .config import CrawlerConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics snapshot."""
    system_memory_percent: float
    process_memory_mb: float
    available_memory_mb: float
    system_memory_total_mb: float
    system_memory_used_mb: float
    gc_count: int
    timestamp: float


class MemoryManager:
    """
    Manages memory usage, garbage collection, and resource monitoring.
    
    Provides proactive memory management to prevent out-of-memory crashes
    and optimize performance through intelligent garbage collection.
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.process = psutil.Process()
        self.gc_count = 0
        self.last_gc_time = 0.0
        self.memory_stats_history = []
        self.max_history_size = 100
        
        # Thread pool for non-blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory_mgmt")
        
        # Memory pressure callbacks
        self.pressure_callbacks: list[Callable] = []
        self.critical_callbacks: list[Callable] = []
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_stats = {
            'gc_forced_count': 0,
            'memory_pressure_events': 0,
            'critical_memory_events': 0,
            'total_gc_time': 0.0,
            'avg_gc_time': 0.0
        }
    
    def is_memory_pressured(self) -> bool:
        """
        Check if system is under memory pressure.
        
        Returns:
            True if memory usage exceeds pressure threshold
        """
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent > self.config.memory_pressure_threshold
        except Exception as e:
            logger.warning(f"Error checking memory pressure: {e}")
            return False
    
    def is_memory_critical(self) -> bool:
        """
        Check if system is under critical memory pressure.
        
        Returns:
            True if memory usage exceeds critical threshold
        """
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent > self.config.memory_critical_threshold
        except Exception as e:
            logger.warning(f"Error checking critical memory: {e}")
            return True  # Assume critical if we can't check
    
    def is_memory_low(self) -> bool:
        """
        Check if system has low memory usage.
        
        Returns:
            True if memory usage is below low threshold
        """
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent < self.config.memory_low_threshold
        except Exception as e:
            logger.warning(f"Error checking low memory: {e}")
            return False
    
    async def force_gc(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Force garbage collection with performance tracking.
        
        Args:
            reason: Reason for forcing GC (for logging)
            
        Returns:
            Dictionary with GC statistics
        """
        start_time = time.time()
        
        try:
            # Run GC in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            collected = await loop.run_in_executor(
                self.thread_pool,
                self._run_gc
            )
            
            gc_time = time.time() - start_time
            self.last_gc_time = time.time()
            
            # Update performance stats
            with self._lock:
                self.performance_stats['gc_forced_count'] += 1
                self.performance_stats['total_gc_time'] += gc_time
                self.performance_stats['avg_gc_time'] = (
                    self.performance_stats['total_gc_time'] / 
                    self.performance_stats['gc_forced_count']
                )
            
            logger.debug(f"Forced GC ({reason}): {collected} objects collected in {gc_time:.3f}s")
            
            return {
                'collected': collected,
                'gc_time': gc_time,
                'reason': reason,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error during forced GC: {e}")
            return {
                'collected': 0,
                'gc_time': time.time() - start_time,
                'reason': reason,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _run_gc(self) -> int:
        """Run garbage collection and return number of objects collected."""
        # Force collection of all generations
        collected = 0
        for generation in range(3):
            collected += gc.collect()
        return collected
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.
        
        Returns:
            MemoryStats object with current memory information
        """
        try:
            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            stats = MemoryStats(
                system_memory_percent=memory.percent,
                process_memory_mb=process_memory.rss / 1024 / 1024,
                available_memory_mb=memory.available / 1024 / 1024,
                system_memory_total_mb=memory.total / 1024 / 1024,
                system_memory_used_mb=memory.used / 1024 / 1024,
                gc_count=self.gc_count,
                timestamp=time.time()
            )
            
            # Store in history
            with self._lock:
                self.memory_stats_history.append(stats)
                if len(self.memory_stats_history) > self.max_history_size:
                    self.memory_stats_history.pop(0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return MemoryStats(
                system_memory_percent=0.0,
                process_memory_mb=0.0,
                available_memory_mb=0.0,
                system_memory_total_mb=0.0,
                system_memory_used_mb=0.0,
                gc_count=self.gc_count,
                timestamp=time.time()
            )
    
    def get_memory_trend(self, window_seconds: int = 60) -> Dict[str, float]:
        """
        Get memory usage trend over time window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with trend information
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self._lock:
            recent_stats = [
                stats for stats in self.memory_stats_history
                if stats.timestamp >= cutoff_time
            ]
        
        if len(recent_stats) < 2:
            return {
                'trend': 0.0,
                'volatility': 0.0,
                'peak_usage': 0.0,
                'avg_usage': 0.0
            }
        
        # Calculate trend (linear regression slope)
        timestamps = [stats.timestamp for stats in recent_stats]
        memory_percents = [stats.system_memory_percent for stats in recent_stats]
        
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(memory_percents)
        sum_xy = sum(x * y for x, y in zip(timestamps, memory_percents))
        sum_x2 = sum(x * x for x in timestamps)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            trend = 0.0
        else:
            trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate volatility (standard deviation)
        mean_usage = sum_y / n
        variance = sum((usage - mean_usage) ** 2 for usage in memory_percents) / n
        volatility = variance ** 0.5
        
        return {
            'trend': trend,
            'volatility': volatility,
            'peak_usage': max(memory_percents),
            'avg_usage': mean_usage
        }
    
    def add_pressure_callback(self, callback: Callable) -> None:
        """Add callback to be called when memory pressure is detected."""
        self.pressure_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable) -> None:
        """Add callback to be called when critical memory pressure is detected."""
        self.critical_callbacks.append(callback)
    
    async def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """
        Start continuous memory monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring:
            logger.warning("Memory monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval_seconds)
        )
        logger.info(f"Started memory monitoring (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped memory monitoring")
    
    async def _monitor_loop(self, interval_seconds: float) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Check memory pressure
                if self.is_memory_critical():
                    await self._handle_critical_memory()
                elif self.is_memory_pressured():
                    await self._handle_memory_pressure()
                
                # Periodic GC if needed
                if (time.time() - self.last_gc_time > 30.0 and 
                    self.is_memory_pressured()):
                    await self.force_gc("periodic")
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _handle_memory_pressure(self) -> None:
        """Handle memory pressure events."""
        with self._lock:
            self.performance_stats['memory_pressure_events'] += 1
        
        logger.warning("Memory pressure detected")
        
        # Force GC
        try:
            await self.force_gc("memory_pressure")
        except Exception as e:
            logger.debug(f"GC failed during memory pressure: {e}")
        
        # Call pressure callbacks
        for callback in self.pressure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in memory pressure callback: {e}")
    
    async def _handle_critical_memory(self) -> None:
        """Handle critical memory pressure events."""
        with self._lock:
            self.performance_stats['critical_memory_events'] += 1
        
        logger.error("Critical memory pressure detected")
        
        # Force aggressive GC
        try:
            await self.force_gc("critical_memory")
        except Exception as e:
            logger.debug(f"GC failed during critical memory: {e}")
        
        # Call critical callbacks
        for callback in self.critical_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in critical memory callback: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get memory management performance statistics."""
        with self._lock:
            return self.performance_stats.copy()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self.performance_stats = {
                'gc_forced_count': 0,
                'memory_pressure_events': 0,
                'critical_memory_events': 0,
                'total_gc_time': 0.0,
                'avg_gc_time': 0.0
            }
    
    async def cleanup(self) -> None:
        """Cleanup memory manager resources."""
        logger.info("Cleaning up memory manager")
        
        # Stop monitoring
        await self.stop_monitoring()
        
        # Final GC
        await self.force_gc("cleanup")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear callbacks
        self.pressure_callbacks.clear()
        self.critical_callbacks.clear()
        
        logger.info("Memory manager cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=False)


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(config: Optional[CrawlerConfig] = None) -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        if config is None:
            from .config import get_config
            config = get_config()
        _memory_manager = MemoryManager(config)
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset global memory manager instance."""
    global _memory_manager
    if _memory_manager is not None:
        # Note: We can't await here since this might be called from __del__
        # The cleanup will happen when the object is garbage collected
        _memory_manager = None
