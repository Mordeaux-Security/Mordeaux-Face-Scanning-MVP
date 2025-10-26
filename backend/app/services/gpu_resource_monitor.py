"""
GPU Resource Monitor Service

Dynamic resource management for GPU worker to maximize GPU utilization
while respecting memory constraints. Monitors GPU worker performance and
adaptively adjusts batch size to achieve >90% GPU utilization.
"""

import asyncio
import logging
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
import psutil

from ..core.settings import get_settings
from .gpu_client import get_gpu_client

logger = logging.getLogger(__name__)

@dataclass
class GPUResourceMetrics:
    """GPU resource metrics for monitoring."""
    current_batch_size: int
    queue_depth: int
    throughput_rps: float
    avg_latency_ms: float
    system_memory_percent: float
    gpu_utilization_proxy: float
    timestamp: float

class GPUResourceMonitor:
    """Dynamic GPU resource monitor for adaptive batch sizing."""
    
    def __init__(self):
        self.settings = get_settings()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Metrics tracking
        self._metrics_history: list[GPUResourceMetrics] = []
        self._max_history_size = 20  # Keep last 20 measurements
        
        # Batch size management
        self._current_batch_size = 128  # Start aggressive for better GPU utilization
        self._max_batch_size = 1024     # Allow large batches (computers like powers of 2)
        self._min_batch_size = 4
        
        # Adjustment parameters
        self._batch_increment = self.settings.gpu_batch_increment
        self._batch_decrement = self.settings.gpu_batch_decrement
        self._target_utilization = self.settings.gpu_target_utilization
        self._memory_threshold = self.settings.gpu_memory_threshold
        
        # Monitoring state
        self._last_adjustment_time = 0
        self._adjustment_cooldown = 5.0  # Minimum 5 seconds between adjustments
        
        logger.info(f"GPU Resource Monitor initialized - target utilization: {self._target_utilization:.1%}, "
                   f"memory threshold: {self._memory_threshold:.1%}")
    
    def start(self) -> None:
        """Start the GPU resource monitoring loop."""
        if self._monitoring:
            logger.warning("GPU Resource Monitor is already running")
            return
        
        if not self.settings.gpu_worker_enabled:
            logger.info("GPU worker disabled, skipping resource monitor")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="gpu-resource-monitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("GPU Resource Monitor started")
    
    def stop(self) -> None:
        """Stop the GPU resource monitoring loop."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("GPU Resource Monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        logger.info("GPU Resource Monitor loop started")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Run monitoring cycle
                    loop.run_until_complete(self._monitor_cycle())
                    
                    # Wait for next cycle
                    self._stop_event.wait(self.settings.gpu_resource_monitor_interval)
                    
                except Exception as e:
                    logger.error(f"Error in GPU Resource Monitor loop: {e}")
                    # Wait a bit before retrying
                    self._stop_event.wait(5.0)
        finally:
            loop.close()
        
        logger.info("GPU Resource Monitor loop stopped")
    
    async def _monitor_cycle(self) -> None:
        """Single monitoring cycle - collect metrics and adjust if needed."""
        try:
            # Get GPU worker metrics
            gpu_client = await get_gpu_client()
            gpu_info = await gpu_client.get_gpu_info()
            
            if not gpu_info:
                logger.warning("Failed to get GPU worker info")
                return
            
            # Get system memory usage
            system_memory_percent = psutil.virtual_memory().percent
            
            # Calculate GPU utilization proxy
            gpu_utilization_proxy = self._calculate_utilization_proxy(gpu_info)
            
            # Create metrics object
            metrics = GPUResourceMetrics(
                current_batch_size=gpu_info.get('current_batch_size', self._current_batch_size),
                queue_depth=gpu_info.get('queue_depth', 0),
                throughput_rps=gpu_info.get('throughput_rps', 0.0),
                avg_latency_ms=gpu_info.get('avg_latency_ms', 0.0),
                system_memory_percent=system_memory_percent,
                gpu_utilization_proxy=gpu_utilization_proxy,
                timestamp=time.time()
            )
            
            # Store metrics
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history.pop(0)
            
            # Check if we should adjust batch size
            if self._should_adjust_batch_size(metrics):
                await self._adjust_batch_size(metrics)
            
            # Log current status
            logger.debug(f"GPU Monitor - Batch: {metrics.current_batch_size}, "
                        f"Queue: {metrics.queue_depth}, Throughput: {metrics.throughput_rps:.1f} req/s, "
                        f"Latency: {metrics.avg_latency_ms:.1f}ms, Memory: {metrics.system_memory_percent:.1f}%, "
                        f"GPU Util: {metrics.gpu_utilization_proxy:.1%}")
            
        except Exception as e:
            logger.error(f"Error in GPU monitoring cycle: {e}")
    
    def _calculate_utilization_proxy(self, gpu_info: Dict[str, Any]) -> float:
        """Calculate GPU utilization proxy from available metrics."""
        # Use throughput and latency trends as proxy for GPU utilization
        throughput = gpu_info.get('throughput_rps', 0.0)
        latency = gpu_info.get('avg_latency_ms', 0.0)
        queue_depth = gpu_info.get('queue_depth', 0)
        
        # Simple heuristic: higher throughput + lower latency + some queue = higher utilization
        if throughput == 0:
            return 0.0
        
        # Normalize throughput (assume 10+ req/s is good)
        throughput_factor = min(throughput / 10.0, 1.0)
        
        # Normalize latency (assume <200ms is good)
        latency_factor = max(0.0, 1.0 - (latency / 200.0))
        
        # Queue depth indicates demand (some queue is good, too much is bad)
        queue_factor = min(queue_depth / 5.0, 1.0) if queue_depth > 0 else 0.5
        
        # Combine factors
        utilization = (throughput_factor * 0.5 + latency_factor * 0.3 + queue_factor * 0.2)
        return min(max(utilization, 0.0), 1.0)
    
    def _should_adjust_batch_size(self, metrics: GPUResourceMetrics) -> bool:
        """Determine if batch size should be adjusted."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_adjustment_time < self._adjustment_cooldown:
            return False
        
        # Check if we have enough history
        if len(self._metrics_history) < 3:
            return False
        
        # Get recent metrics for trend analysis
        recent_metrics = self._metrics_history[-3:]
        
        # Check for memory pressure (immediate decrease)
        if metrics.system_memory_percent > self._memory_threshold * 100:
            logger.warning(f"High memory usage detected: {metrics.system_memory_percent:.1f}% > {self._memory_threshold * 100:.1f}%")
            return True
        
        # Check for utilization trends
        avg_utilization = sum(m.gpu_utilization_proxy for m in recent_metrics) / len(recent_metrics)
        
        # Increase if utilization is low and stable
        if (avg_utilization < self._target_utilization and 
            metrics.current_batch_size < self._max_batch_size and
            metrics.system_memory_percent < self._memory_threshold * 100):
            return True
        
        # Decrease if utilization is very high (might be overloaded)
        if (avg_utilization > 0.95 and 
            metrics.current_batch_size > self._min_batch_size):
            return True
        
        return False
    
    async def _adjust_batch_size(self, metrics: GPUResourceMetrics) -> None:
        """Adjust GPU worker batch size based on current metrics."""
        try:
            current_batch = metrics.current_batch_size
            new_batch_size = current_batch
            
            # Memory pressure - decrease immediately
            if metrics.system_memory_percent > self._memory_threshold * 100:
                new_batch_size = max(current_batch - self._batch_decrement, self._min_batch_size)
                logger.warning(f"Memory pressure detected, decreasing batch size: {current_batch} -> {new_batch_size}")
            
            # Low utilization - increase
            elif metrics.gpu_utilization_proxy < self._target_utilization:
                new_batch_size = min(current_batch + self._batch_increment, self._max_batch_size)
                logger.info(f"Low GPU utilization, increasing batch size: {current_batch} -> {new_batch_size}")
            
            # Very high utilization - decrease
            elif metrics.gpu_utilization_proxy > 0.95:
                new_batch_size = max(current_batch - self._batch_decrement, self._min_batch_size)
                logger.info(f"High GPU utilization, decreasing batch size: {current_batch} -> {new_batch_size}")
            
            # Apply the change
            if new_batch_size != current_batch:
                gpu_client = await get_gpu_client()
                success = await gpu_client.set_batch_size(new_batch_size)
                
                if success:
                    self._current_batch_size = new_batch_size
                    self._last_adjustment_time = time.time()
                    logger.info(f"Successfully adjusted GPU worker batch size to {new_batch_size}")
                else:
                    logger.error(f"Failed to adjust GPU worker batch size to {new_batch_size}")
            
        except Exception as e:
            logger.error(f"Error adjusting batch size: {e}")
    
    def get_current_metrics(self) -> Optional[GPUResourceMetrics]:
        """Get the most recent metrics."""
        return self._metrics_history[-1] if self._metrics_history else None
    
    def get_metrics_history(self) -> list[GPUResourceMetrics]:
        """Get the metrics history."""
        return self._metrics_history.copy()

# Global monitor instance
_gpu_resource_monitor: Optional[GPUResourceMonitor] = None

def get_gpu_resource_monitor() -> GPUResourceMonitor:
    """Get the global GPU resource monitor instance."""
    global _gpu_resource_monitor
    if _gpu_resource_monitor is None:
        _gpu_resource_monitor = GPUResourceMonitor()
    return _gpu_resource_monitor

def start_gpu_resource_monitor() -> None:
    """Start the GPU resource monitor."""
    monitor = get_gpu_resource_monitor()
    monitor.start()

def stop_gpu_resource_monitor() -> None:
    """Stop the GPU resource monitor."""
    global _gpu_resource_monitor
    if _gpu_resource_monitor:
        _gpu_resource_monitor.stop()
        _gpu_resource_monitor = None
