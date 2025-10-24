"""
Comprehensive resource monitoring module for the crawler service.

This module contains the ResourceMonitor class and related resource management
functionality for adaptive resource management during crawling operations.
Expands the original MemoryMonitor with CPU, GPU, and I/O monitoring.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque
import psutil

# Import GPU manager for GPU monitoring
try:
    from ..gpu_manager import get_gpu_manager, GPUBackend
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False
    get_gpu_manager = None

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Comprehensive resource monitoring with adaptive thresholds for system resource management.
    
    Monitors CPU, GPU, memory, and I/O resources with trend analysis and bottleneck detection.
    Provides conservative adjustment recommendations to prevent oscillation.
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize resource monitor.
        
        Args:
            history_size: Number of samples to keep in history for trend analysis
        """
        self.history_size = history_size
        
        # Memory monitoring (existing functionality)
        self.initial_memory = psutil.virtual_memory().percent
        self.peak_memory = self.initial_memory
        self.memory_history = []
        self.gc_triggered = False
        
        # CPU monitoring
        self.cpu_history = deque(maxlen=history_size)
        self.cpu_io_wait_history = deque(maxlen=history_size)
        self.cpu_context_switches_history = deque(maxlen=history_size)
        
        # GPU monitoring
        self.gpu_manager = get_gpu_manager() if GPU_MANAGER_AVAILABLE else None
        self.gpu_utilization_history = deque(maxlen=history_size)
        self.gpu_memory_history = deque(maxlen=history_size)
        
        # I/O monitoring
        self.disk_io_history = deque(maxlen=history_size)
        self.network_io_history = deque(maxlen=history_size)
        
        # Pipeline monitoring
        self.queue_depths = {}
        self.processing_latency_history = deque(maxlen=history_size)
        self.throughput_history = deque(maxlen=history_size)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 0.1  # 100ms sampling
        self._last_sample_time = 0
        
        # Resource utilization scores (0-100)
        self._cpu_score = 0.0
        self._gpu_score = 0.0
        self._memory_score = 0.0
        self._io_score = 0.0
        self._overall_score = 0.0
        
        # Bottleneck detection
        self._bottleneck = "unknown"
        self._last_bottleneck_check = 0
        
    def start_monitoring(self, interval_ms: int = 100):
        """
        Start continuous resource monitoring.
        
        Args:
            interval_ms: Monitoring interval in milliseconds
        """
        if self._monitoring:
            return
            
        self._monitor_interval = interval_ms / 1000.0
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started resource monitoring with {interval_ms}ms interval")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._sample_resources()
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                time.sleep(self._monitor_interval)
    
    def _sample_resources(self):
        """Sample all resources and update history."""
        current_time = time.time()
        
        # Sample CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_times = psutil.cpu_times()
        self.cpu_history.append(cpu_percent)
        self.cpu_io_wait_history.append(cpu_times.iowait)
        self.cpu_context_switches_history.append(psutil.cpu_stats().ctx_switches)
        
        # Sample GPU
        if self.gpu_manager:
            try:
                gpu_memory_info = self.gpu_manager.get_memory_info()
                gpu_utilization = self._estimate_gpu_utilization()
                self.gpu_utilization_history.append(gpu_utilization)
                self.gpu_memory_history.append(gpu_memory_info.get('used', 0))
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
                self.gpu_utilization_history.append(0.0)
                self.gpu_memory_history.append(0)
        
        # Sample I/O
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        if disk_io:
            self.disk_io_history.append({
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_time': disk_io.read_time,
                'write_time': disk_io.write_time
            })
        if net_io:
            self.network_io_history.append({
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            })
        
        # Update utilization scores
        self._update_utilization_scores()
        
        # Check for bottlenecks (every 2 seconds)
        if current_time - self._last_bottleneck_check > 2.0:
            self._identify_bottleneck()
            self._last_bottleneck_check = current_time
        
        self._last_sample_time = current_time
    
    def _estimate_gpu_utilization(self) -> float:
        """Estimate GPU utilization (simplified approach)."""
        if not self.gpu_manager:
            return 0.0
        
        try:
            # This is a simplified estimation - in practice, you'd use
            # nvidia-smi, rocm-smi, or similar tools for accurate GPU utilization
            memory_info = self.gpu_manager.get_memory_info()
            memory_usage = memory_info.get('used', 0)
            memory_total = memory_info.get('total', 1)
            
            # Estimate utilization based on memory usage (rough approximation)
            memory_util = (memory_usage / memory_total) * 100 if memory_total > 0 else 0
            return min(memory_util * 1.2, 100.0)  # Assume compute is slightly higher than memory
        except Exception:
            return 0.0
    
    def _update_utilization_scores(self):
        """Update resource utilization scores."""
        # CPU score (0-100)
        if self.cpu_history:
            self._cpu_score = sum(self.cpu_history) / len(self.cpu_history)
        
        # GPU score (0-100)
        if self.gpu_utilization_history:
            self._gpu_score = sum(self.gpu_utilization_history) / len(self.gpu_utilization_history)
        
        # Memory score (0-100)
        memory_status = self.get_memory_status()
        self._memory_score = memory_status['percent']
        
        # I/O score (based on wait time and throughput)
        io_wait_avg = sum(self.cpu_io_wait_history) / len(self.cpu_io_wait_history) if self.cpu_io_wait_history else 0
        self._io_score = min(io_wait_avg * 10, 100)  # Convert to 0-100 scale
        
        # Overall score (weighted average)
        weights = {'cpu': 0.3, 'gpu': 0.2, 'memory': 0.3, 'io': 0.2}
        self._overall_score = (
            weights['cpu'] * self._cpu_score +
            weights['gpu'] * self._gpu_score +
            weights['memory'] * self._memory_score +
            weights['io'] * self._io_score
        )
    
    def _identify_bottleneck(self):
        """Identify the current system bottleneck."""
        scores = {
            'cpu': self._cpu_score,
            'gpu': self._gpu_score,
            'memory': self._memory_score,
            'io': self._io_score
        }
        
        # Find the resource with highest utilization
        max_resource = max(scores.items(), key=lambda x: x[1])
        
        # Determine bottleneck type
        if max_resource[1] > 85:
            self._bottleneck = max_resource[0]
        elif self._io_score > 20:  # High I/O wait
            self._bottleneck = 'io'
        elif self._memory_score > 80:
            self._bottleneck = 'memory'
        else:
            self._bottleneck = 'balanced'
    
    # ============================================================================
    # Memory Monitoring (Existing Functionality)
    # ============================================================================
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get comprehensive memory status."""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'pressure_level': self._calculate_pressure_level(memory.percent)
        }
    
    def _calculate_pressure_level(self, memory_percent: float) -> str:
        """Calculate memory pressure level."""
        if memory_percent < 60:
            return 'low'
        elif memory_percent < 75:
            return 'moderate'
        elif memory_percent < 85:
            return 'high'
        else:
            return 'critical'
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered."""
        status = self.get_memory_status()
        logger.debug(f"Memory status: {status}")
        
        # Always trigger GC if memory is critical
        if status['pressure_level'] == 'critical':
            return True
            
        # Trigger GC if memory is high and we haven't done it recently
        if status['pressure_level'] == 'high' and not self.gc_triggered:
            self.gc_triggered = True
            return True
            
        # Reset GC flag when memory is low
        if status['pressure_level'] == 'low':
            self.gc_triggered = False
            
        return False
    
    def get_safe_concurrency_limit(self, base_concurrency: int) -> int:
        """Calculate safe concurrency limit based on memory pressure."""
        status = self.get_memory_status()
        
        if status['pressure_level'] == 'critical':
            return max(1, base_concurrency // 4)
        elif status['pressure_level'] == 'high':
            return max(2, base_concurrency // 2)
        elif status['pressure_level'] == 'moderate':
            return int(base_concurrency * 0.75)
        else:  # low
            return min(base_concurrency * 2, 30)  # Cap at 30
    
    # ============================================================================
    # CPU Monitoring (New)
    # ============================================================================
    
    def get_cpu_status(self) -> Dict[str, float]:
        """Get comprehensive CPU status."""
        if not self.cpu_history:
            return {'utilization': 0.0, 'trend': 0.0, 'io_wait': 0.0}
        
        current_util = self.cpu_history[-1] if self.cpu_history else 0.0
        trend = self._calculate_trend(list(self.cpu_history))
        io_wait = self.cpu_io_wait_history[-1] if self.cpu_io_wait_history else 0.0
        
        return {
            'utilization': current_util,
            'trend': trend,
            'io_wait': io_wait,
            'context_switches': self.cpu_context_switches_history[-1] if self.cpu_context_switches_history else 0,
            'is_cpu_bound': self.is_cpu_bound()
        }
    
    def get_cpu_utilization_trend(self) -> List[float]:
        """Get CPU utilization trend over time."""
        return list(self.cpu_history)
    
    def is_cpu_bound(self) -> bool:
        """Determine if system is CPU-bound."""
        if not self.cpu_history:
            return False
        
        # CPU-bound if high utilization and low I/O wait
        avg_util = sum(self.cpu_history) / len(self.cpu_history)
        avg_io_wait = sum(self.cpu_io_wait_history) / len(self.cpu_io_wait_history) if self.cpu_io_wait_history else 0
        
        return avg_util > 80 and avg_io_wait < 5
    
    # ============================================================================
    # GPU Monitoring (New)
    # ============================================================================
    
    def get_gpu_status(self) -> Dict[str, float]:
        """Get comprehensive GPU status."""
        if not self.gpu_manager:
            return {'utilization': 0.0, 'memory_used': 0.0, 'memory_total': 0.0, 'available': False}
        
        try:
            memory_info = self.gpu_manager.get_memory_info()
            current_util = self.gpu_utilization_history[-1] if self.gpu_utilization_history else 0.0
            trend = self._calculate_trend(list(self.gpu_utilization_history))
            
            return {
                'utilization': current_util,
                'trend': trend,
                'memory_used': memory_info.get('used', 0),
                'memory_total': memory_info.get('total', 0),
                'memory_percent': (memory_info.get('used', 0) / memory_info.get('total', 1)) * 100 if memory_info.get('total', 0) > 0 else 0,
                'available': True,
                'is_gpu_bound': self.is_gpu_bound()
            }
        except Exception as e:
            logger.debug(f"GPU status error: {e}")
            return {'utilization': 0.0, 'memory_used': 0.0, 'memory_total': 0.0, 'available': False}
    
    def get_gpu_utilization_trend(self) -> List[float]:
        """Get GPU utilization trend over time."""
        return list(self.gpu_utilization_history)
    
    def is_gpu_bound(self) -> bool:
        """Determine if system is GPU-bound."""
        if not self.gpu_utilization_history:
            return False
        
        avg_util = sum(self.gpu_utilization_history) / len(self.gpu_utilization_history)
        return avg_util > 80
    
    # ============================================================================
    # I/O Monitoring (New)
    # ============================================================================
    
    def get_io_status(self) -> Dict[str, float]:
        """Get comprehensive I/O status."""
        if not self.disk_io_history or not self.network_io_history:
            return {'disk_utilization': 0.0, 'network_utilization': 0.0, 'io_wait': 0.0}
        
        # Calculate disk I/O utilization
        disk_util = self._calculate_disk_utilization()
        
        # Calculate network I/O utilization
        network_util = self._calculate_network_utilization()
        
        # I/O wait time
        io_wait = self.cpu_io_wait_history[-1] if self.cpu_io_wait_history else 0.0
        
        return {
            'disk_utilization': disk_util,
            'network_utilization': network_util,
            'io_wait': io_wait,
            'is_io_bound': self.is_io_bound()
        }
    
    def _calculate_disk_utilization(self) -> float:
        """Calculate disk utilization based on I/O history."""
        if len(self.disk_io_history) < 2:
            return 0.0
        
        # Calculate throughput over time
        recent = list(self.disk_io_history)[-5:]  # Last 5 samples
        if len(recent) < 2:
            return 0.0
        
        total_bytes = sum(io['read_bytes'] + io['write_bytes'] for io in recent)
        total_time = sum(io['read_time'] + io['write_time'] for io in recent)
        
        if total_time > 0:
            return min((total_bytes / total_time) / 1000000, 100.0)  # MB/s, capped at 100
        return 0.0
    
    def _calculate_network_utilization(self) -> float:
        """Calculate network utilization based on I/O history."""
        if len(self.network_io_history) < 2:
            return 0.0
        
        # Calculate throughput over time
        recent = list(self.network_io_history)[-5:]  # Last 5 samples
        if len(recent) < 2:
            return 0.0
        
        total_bytes = sum(io['bytes_sent'] + io['bytes_recv'] for io in recent)
        return min(total_bytes / 1000000, 100.0)  # MB, capped at 100
    
    def is_io_bound(self) -> bool:
        """Determine if system is I/O-bound."""
        if not self.cpu_io_wait_history:
            return False
        
        avg_io_wait = sum(self.cpu_io_wait_history) / len(self.cpu_io_wait_history)
        return avg_io_wait > 10  # High I/O wait indicates I/O bound
    
    # ============================================================================
    # Overall Resource Status (New)
    # ============================================================================
    
    def get_resource_summary(self) -> Dict[str, float]:
        """Get comprehensive resource summary."""
        return {
            'cpu_utilization': self._cpu_score,
            'gpu_utilization': self._gpu_score,
            'memory_utilization': self._memory_score,
            'io_utilization': self._io_score,
            'overall_utilization': self._overall_score,
            'bottleneck': self._bottleneck,
            'monitoring_active': self._monitoring,
            'samples_collected': len(self.cpu_history)
        }
    
    def identify_bottleneck(self) -> str:
        """Identify the current system bottleneck."""
        return self._bottleneck
    
    def get_utilization_score(self) -> float:
        """Get overall resource utilization score (0-100)."""
        return self._overall_score
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, negative = decreasing, positive = increasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return max(-1.0, min(1.0, slope / 10.0))  # Normalize to -1 to 1
    
    # ============================================================================
    # Pipeline Monitoring (New)
    # ============================================================================
    
    def update_queue_depth(self, queue_name: str, depth: int):
        """Update queue depth for pipeline monitoring."""
        self.queue_depths[queue_name] = depth
    
    def update_processing_latency(self, latency_ms: float):
        """Update processing latency."""
        self.processing_latency_history.append(latency_ms)
    
    def update_throughput(self, items_per_second: float):
        """Update throughput measurement."""
        self.throughput_history.append(items_per_second)
    
    def get_pipeline_status(self) -> Dict[str, float]:
        """Get pipeline performance status."""
        avg_latency = sum(self.processing_latency_history) / len(self.processing_latency_history) if self.processing_latency_history else 0.0
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0.0
        
        return {
            'queue_depths': dict(self.queue_depths),
            'avg_processing_latency_ms': avg_latency,
            'avg_throughput_items_per_sec': avg_throughput,
            'total_queued_items': sum(self.queue_depths.values())
        }


# Backward compatibility alias
MemoryMonitor = ResourceMonitor
