"""
Adaptive Resource Manager

Dynamically adjusts concurrency limits, batch sizes, and resource allocation
based on real-time resource monitoring. Uses conservative, small adjustments
to prevent oscillation and maintain stable resource utilization.
"""

import logging
import time
import threading
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

from .crawler_modules.resources import ResourceMonitor
from ..core.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for different operations."""
    min_concurrent_downloads: int = 5
    max_concurrent_downloads: int = 100
    min_batch_size: int = 1
    max_batch_size: int = 128
    min_concurrent_sites: int = 1
    max_concurrent_sites: int = 20
    min_gpu_batch_size: int = 8
    max_gpu_batch_size: int = 256
    min_storage_batch_size: int = 5
    max_storage_batch_size: int = 100


@dataclass
class AdjustmentConfig:
    """Configuration for resource adjustments."""
    smoothing_factor: float = 0.3
    adjustment_interval_s: float = 2.0
    step_size: int = 1
    step_percent: float = 5.0
    utilization_deadband: float = 5.0
    warmup_period_s: float = 30.0
    max_adjustments_per_minute: int = 10


class AdaptiveResourceManager:
    """
    Adaptive resource manager that dynamically adjusts system parameters
    based on real-time resource monitoring.
    
    Uses conservative, small adjustments to prevent oscillation and maintain
    stable resource utilization.
    """
    
    def __init__(self, resource_monitor: Optional[ResourceMonitor] = None):
        """
        Initialize adaptive resource manager.
        
        Args:
            resource_monitor: ResourceMonitor instance (creates new if None)
        """
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.settings = get_settings()
        
        # Load configuration
        self.limits = self._load_resource_limits()
        self.config = self._load_adjustment_config()
        
        # Current resource parameters
        self.current_params = {
            'concurrent_downloads': 10,
            'connection_pool_size': 20,
            'batch_size': 16,
            'concurrent_processing': 4,
            'gpu_batch_size': 32,
            'concurrent_sites': 3,
            'per_host_concurrency': 2,
            'storage_batch_size': 25,
            'concurrent_uploads': 5,
            'queue_buffer_size': 50
        }
        
        # Adjustment history for smoothing
        self.adjustment_history = deque(maxlen=10)
        self.last_adjustment_time = 0
        self.adjustment_count = 0
        self.adjustment_reset_time = time.time()
        
        # Target utilization levels
        self.target_cpu_utilization = 85.0
        self.target_gpu_utilization = 90.0
        self.target_memory_utilization = 75.0
        
        # State tracking
        self._adjustment_lock = threading.Lock()
        self._enabled = True
        
        # Start monitoring if not already started
        if not self.resource_monitor._monitoring:
            self.resource_monitor.start_monitoring()
    
    def _load_resource_limits(self) -> ResourceLimits:
        """Load resource limits from settings."""
        return ResourceLimits(
            min_concurrent_downloads=self.settings.min_concurrent_downloads,
            max_concurrent_downloads=self.settings.max_concurrent_downloads,
            min_batch_size=self.settings.min_batch_size,
            max_batch_size=self.settings.max_batch_size,
            min_concurrent_sites=self.settings.min_concurrent_sites,
            max_concurrent_sites=self.settings.max_concurrent_sites
        )
    
    def _load_adjustment_config(self) -> AdjustmentConfig:
        """Load adjustment configuration from settings."""
        return AdjustmentConfig(
            smoothing_factor=self.settings.smoothing_factor,
            adjustment_interval_s=self.settings.adjustment_interval_s,
            step_size=self.settings.adjustment_step_size,
            step_percent=self.settings.adjustment_step_percent,
            utilization_deadband=self.settings.utilization_deadband,
            warmup_period_s=self.settings.warmup_period_s,
            max_adjustments_per_minute=self.settings.max_adjustment_per_minute
        )
    
    def get_optimal_concurrency(self, operation_type: str, base_value: int) -> int:
        """
        Get optimal concurrency for a specific operation type.
        
        Args:
            operation_type: Type of operation (downloads, processing, sites, etc.)
            base_value: Base concurrency value
            
        Returns:
            Optimized concurrency value
        """
        if not self._enabled:
            return base_value
        
        with self._adjustment_lock:
            # Check if we're in warmup period
            if time.time() - self.resource_monitor._last_sample_time < self.config.warmup_period_s:
                return base_value
            
            # Get current resource status
            resource_summary = self.resource_monitor.get_resource_summary()
            bottleneck = self.resource_monitor.identify_bottleneck()
            
            # Calculate adjustment based on resource utilization
            adjustment = self._calculate_concurrency_adjustment(
                operation_type, base_value, resource_summary, bottleneck
            )
            
            # Apply adjustment with smoothing
            new_value = self._apply_smoothing(operation_type, adjustment)
            
            # Apply bounds
            bounded_value = self._apply_bounds(operation_type, new_value)
            
            # Update current parameters
            self.current_params[operation_type] = bounded_value
            
            return bounded_value
    
    def get_optimal_batch_size(self, operation_type: str, base_value: int) -> int:
        """
        Get optimal batch size for a specific operation type.
        
        Args:
            operation_type: Type of operation (processing, gpu_processing, storage)
            base_value: Base batch size value
            
        Returns:
            Optimized batch size value
        """
        if not self._enabled:
            return base_value
        
        with self._adjustment_lock:
            # Check if we're in warmup period
            if time.time() - self.resource_monitor._last_sample_time < self.config.warmup_period_s:
                return base_value
            
            # Get current resource status
            resource_summary = self.resource_monitor.get_resource_summary()
            memory_status = self.resource_monitor.get_memory_status()
            
            # Calculate adjustment based on resource utilization
            adjustment = self._calculate_batch_size_adjustment(
                operation_type, base_value, resource_summary, memory_status
            )
            
            # Apply adjustment with smoothing
            new_value = self._apply_smoothing(operation_type, adjustment)
            
            # Apply bounds
            bounded_value = self._apply_bounds(operation_type, new_value)
            
            # Update current parameters
            self.current_params[operation_type] = bounded_value
            
            return bounded_value
    
    def _calculate_concurrency_adjustment(
        self, 
        operation_type: str, 
        current_value: int, 
        resource_summary: Dict[str, Any],
        bottleneck: str
    ) -> int:
        """
        Calculate concurrency adjustment based on resource utilization.
        
        Uses conservative, small adjustments to prevent oscillation.
        """
        cpu_util = resource_summary.get('cpu_utilization', 0)
        gpu_util = resource_summary.get('gpu_utilization', 0)
        memory_util = resource_summary.get('memory_utilization', 0)
        io_util = resource_summary.get('io_utilization', 0)
        
        # Determine if we should increase or decrease
        should_increase = False
        should_decrease = False
        
        # Check for underutilization (increase concurrency)
        if operation_type in ['downloads', 'sites'] and cpu_util < self.target_cpu_utilization - self.config.utilization_deadband:
            should_increase = True
        elif operation_type in ['processing', 'gpu_processing'] and cpu_util < self.target_cpu_utilization - self.config.utilization_deadband:
            should_increase = True
        elif operation_type == 'gpu_processing' and gpu_util < self.target_gpu_utilization - self.config.utilization_deadband:
            should_increase = True
        
        # Check for overutilization (decrease concurrency)
        if memory_util > self.target_memory_utilization + self.config.utilization_deadband:
            should_decrease = True
        elif operation_type in ['downloads', 'sites'] and cpu_util > self.target_cpu_utilization + self.config.utilization_deadband:
            should_decrease = True
        elif operation_type in ['processing', 'gpu_processing'] and cpu_util > self.target_cpu_utilization + self.config.utilization_deadband:
            should_decrease = True
        
        # Apply small adjustments
        if should_increase:
            # Small additive increase
            return current_value + self.config.step_size
        elif should_decrease:
            # Small additive decrease
            return current_value - self.config.step_size
        else:
            # No change needed
            return current_value
    
    def _calculate_batch_size_adjustment(
        self,
        operation_type: str,
        current_value: int,
        resource_summary: Dict[str, Any],
        memory_status: Dict[str, Any]
    ) -> int:
        """
        Calculate batch size adjustment based on resource utilization.
        
        Uses conservative, small adjustments to prevent oscillation.
        """
        memory_pressure = memory_status.get('pressure_level', 'low')
        cpu_util = resource_summary.get('cpu_utilization', 0)
        gpu_util = resource_summary.get('gpu_utilization', 0)
        
        # Determine if we should increase or decrease
        should_increase = False
        should_decrease = False
        
        # Check for underutilization (increase batch size)
        if operation_type == 'gpu_processing' and gpu_util < self.target_gpu_utilization - self.config.utilization_deadband:
            should_increase = True
        elif operation_type == 'processing' and cpu_util < self.target_cpu_utilization - self.config.utilization_deadband:
            should_increase = True
        
        # Check for overutilization (decrease batch size)
        if memory_pressure in ['high', 'critical']:
            should_decrease = True
        elif operation_type == 'gpu_processing' and gpu_util > self.target_gpu_utilization + self.config.utilization_deadband:
            should_decrease = True
        elif operation_type == 'processing' and cpu_util > self.target_cpu_utilization + self.config.utilization_deadband:
            should_decrease = True
        
        # Apply small adjustments
        if should_increase:
            # Small multiplicative increase
            return int(current_value * (1 + self.config.step_percent / 100))
        elif should_decrease:
            # Small multiplicative decrease
            return int(current_value * (1 - self.config.step_percent / 100))
        else:
            # No change needed
            return current_value
    
    def _apply_smoothing(self, operation_type: str, new_value: int) -> int:
        """
        Apply exponential moving average smoothing to prevent oscillation.
        
        Args:
            operation_type: Type of operation
            new_value: New calculated value
            
        Returns:
            Smoothed value
        """
        current_value = self.current_params.get(operation_type, new_value)
        
        # Exponential moving average
        smoothed_value = (
            self.config.smoothing_factor * new_value +
            (1 - self.config.smoothing_factor) * current_value
        )
        
        return int(smoothed_value)
    
    def _apply_bounds(self, operation_type: str, value: int) -> int:
        """
        Apply bounds to prevent extreme values.
        
        Args:
            operation_type: Type of operation
            value: Value to bound
            
        Returns:
            Bounded value
        """
        # Define bounds for different operation types
        bounds_map = {
            'downloads': (self.limits.min_concurrent_downloads, self.limits.max_concurrent_downloads),
            'processing': (2, 32),  # CPU processing
            'gpu_processing': (self.limits.min_gpu_batch_size, self.limits.max_gpu_batch_size),
            'sites': (self.limits.min_concurrent_sites, self.limits.max_concurrent_sites),
            'storage': (self.limits.min_storage_batch_size, self.limits.max_storage_batch_size),
            'batch_size': (self.limits.min_batch_size, self.limits.max_batch_size)
        }
        
        min_val, max_val = bounds_map.get(operation_type, (1, 100))
        return max(min_val, min(max_val, value))
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits for adjustments.
        
        Returns:
            True if adjustment is allowed, False otherwise
        """
        current_time = time.time()
        
        # Check minimum interval
        if current_time - self.last_adjustment_time < self.config.adjustment_interval_s:
            return False
        
        # Check maximum adjustments per minute
        if current_time - self.adjustment_reset_time > 60:
            self.adjustment_count = 0
            self.adjustment_reset_time = current_time
        
        if self.adjustment_count >= self.config.max_adjustments_per_minute:
            return False
        
        # Update counters
        self.last_adjustment_time = current_time
        self.adjustment_count += 1
        
        return True
    
    def get_current_parameters(self) -> Dict[str, int]:
        """Get current resource parameters."""
        return self.current_params.copy()
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status and recommendations."""
        resource_summary = self.resource_monitor.get_resource_summary()
        
        return {
            'resource_utilization': resource_summary,
            'current_parameters': self.current_parameters(),
            'bottleneck': self.resource_monitor.identify_bottleneck(),
            'adjustment_enabled': self._enabled,
            'adjustment_config': {
                'smoothing_factor': self.config.smoothing_factor,
                'step_size': self.config.step_size,
                'utilization_deadband': self.config.utilization_deadband
            }
        }
    
    def enable_adjustments(self):
        """Enable dynamic resource adjustments."""
        self._enabled = True
        logger.info("Dynamic resource adjustments enabled")
    
    def disable_adjustments(self):
        """Disable dynamic resource adjustments."""
        self._enabled = False
        logger.info("Dynamic resource adjustments disabled")
    
    def reset_parameters(self):
        """Reset parameters to default values."""
        with self._adjustment_lock:
            self.current_params = {
                'concurrent_downloads': 10,
                'connection_pool_size': 20,
                'batch_size': 16,
                'concurrent_processing': 4,
                'gpu_batch_size': 32,
                'concurrent_sites': 3,
                'per_host_concurrency': 2,
                'storage_batch_size': 25,
                'concurrent_uploads': 5,
                'queue_buffer_size': 50
            }
            logger.info("Resource parameters reset to defaults")
    
    def cleanup(self):
        """Clean up resources."""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        logger.info("Adaptive resource manager cleaned up")


# Global instance
_resource_manager: Optional[AdaptiveResourceManager] = None


def get_adaptive_resource_manager() -> AdaptiveResourceManager:
    """Get the global adaptive resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = AdaptiveResourceManager()
    return _resource_manager


def cleanup_adaptive_resource_manager():
    """Clean up the global adaptive resource manager."""
    global _resource_manager
    if _resource_manager is not None:
        _resource_manager.cleanup()
        _resource_manager = None
