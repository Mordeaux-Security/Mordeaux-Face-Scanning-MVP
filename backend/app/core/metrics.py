import time
import statistics
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import logging
from threading import Lock
import json

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.request_times = deque(maxlen=max_samples)
        self.endpoint_metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.tenant_metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.error_counts = defaultdict(int)
        self.rate_limit_violations = defaultdict(int)
        self.lock = Lock()
        self.start_time = time.time()
    
    def record_request(self, endpoint: str, tenant_id: str, duration: float, 
                      status_code: int, request_id: str = None):
        """Record a request metric."""
        with self.lock:
            self.request_times.append(duration)
            self.endpoint_metrics[endpoint].append(duration)
            self.tenant_metrics[tenant_id].append(duration)
            
            if status_code >= 400:
                self.error_counts[endpoint] += 1
            
            # Log performance metrics
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "tenant_id": tenant_id,
                    "duration": duration,
                    "status_code": status_code
                }
            )
    
    def record_rate_limit_violation(self, tenant_id: str):
        """Record a rate limit violation."""
        with self.lock:
            self.rate_limit_violations[tenant_id] += 1
    
    def get_p95_latency(self) -> float:
        """Get P95 latency across all requests."""
        with self.lock:
            if not self.request_times:
                return 0.0
            sorted_times = sorted(self.request_times)
            p95_index = int(len(sorted_times) * 0.95)
            return sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
    
    def get_p99_latency(self) -> float:
        """Get P99 latency across all requests."""
        with self.lock:
            if not self.request_times:
                return 0.0
            sorted_times = sorted(self.request_times)
            p99_index = int(len(sorted_times) * 0.99)
            return sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
    
    def get_avg_latency(self) -> float:
        """Get average latency across all requests."""
        with self.lock:
            if not self.request_times:
                return 0.0
            return statistics.mean(self.request_times)
    
    def get_median_latency(self) -> float:
        """Get median latency across all requests."""
        with self.lock:
            if not self.request_times:
                return 0.0
            return statistics.median(self.request_times)
    
    def is_p95_threshold_exceeded(self) -> bool:
        """Check if P95 latency exceeds the configured threshold."""
        from .config import get_settings
        settings = get_settings()
        p95 = self.get_p95_latency()
        return p95 > settings.p95_latency_threshold_seconds
    
    def get_endpoint_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Get metrics for a specific endpoint."""
        with self.lock:
            times = list(self.endpoint_metrics[endpoint])
            if not times:
                return {
                    "request_count": 0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "p99_latency": 0.0,
                    "error_count": 0,
                    "error_rate": 0.0
                }
            
            sorted_times = sorted(times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            
            return {
                "request_count": len(times),
                "avg_latency": statistics.mean(times),
                "p95_latency": sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1],
                "p99_latency": sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1],
                "error_count": self.error_counts[endpoint],
                "error_rate": self.error_counts[endpoint] / len(times) if times else 0.0
            }
    
    def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get metrics for a specific tenant."""
        with self.lock:
            times = list(self.tenant_metrics[tenant_id])
            if not times:
                return {
                    "request_count": 0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "rate_limit_violations": 0
                }
            
            sorted_times = sorted(times)
            p95_index = int(len(sorted_times) * 0.95)
            
            return {
                "request_count": len(times),
                "avg_latency": statistics.mean(times),
                "p95_latency": sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1],
                "rate_limit_violations": self.rate_limit_violations[tenant_id]
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary."""
        with self.lock:
            total_requests = len(self.request_times)
            total_errors = sum(self.error_counts.values())
            total_rate_limit_violations = sum(self.rate_limit_violations.values())
            
            return {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self.start_time,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "total_rate_limit_violations": total_rate_limit_violations,
                "error_rate": total_errors / total_requests if total_requests > 0 else 0.0,
                "avg_latency": self.get_avg_latency(),
                "p95_latency": self.get_p95_latency(),
                "p99_latency": self.get_p99_latency(),
                "median_latency": self.get_median_latency(),
                "p95_threshold_exceeded": self.is_p95_threshold_exceeded(),
                "endpoints": {
                    endpoint: self.get_endpoint_metrics(endpoint)
                    for endpoint in self.endpoint_metrics.keys()
                },
                "tenants": {
                    tenant_id: self.get_tenant_metrics(tenant_id)
                    for tenant_id in self.tenant_metrics.keys()
                }
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self.lock:
            self.request_times.clear()
            self.endpoint_metrics.clear()
            self.tenant_metrics.clear()
            self.error_counts.clear()
            self.rate_limit_violations.clear()
            self.start_time = time.time()

# Global metrics instance
_metrics = None

def get_metrics() -> PerformanceMetrics:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PerformanceMetrics()
    return _metrics

def record_request_metrics(endpoint: str, tenant_id: str, duration: float, 
                          status_code: int, request_id: str = None):
    """Record request metrics."""
    metrics = get_metrics()
    metrics.record_request(endpoint, tenant_id, duration, status_code, request_id)

def record_rate_limit_violation(tenant_id: str):
    """Record rate limit violation."""
    metrics = get_metrics()
    metrics.record_rate_limit_violation(tenant_id)