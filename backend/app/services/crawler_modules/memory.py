"""
Memory management module for the crawler service.

This module contains the MemoryMonitor class and related memory management
functionality for adaptive resource management during crawling operations.
"""

import logging
import psutil
from typing import Dict

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Memory monitoring with adaptive thresholds for system resource management."""
    
    def __init__(self):
        self.initial_memory = psutil.virtual_memory().percent
        self.peak_memory = self.initial_memory
        self.memory_history = []
        self.gc_triggered = False
        
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
        logger.info(f"Memory status: {status}")
        
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
