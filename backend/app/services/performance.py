import asyncio
import time
import logging
from typing import Dict, Any, List, Optional


            # Use async storage operations



from concurrent.futures import ThreadPoolExecutor
from ..core.config import get_settings
from ..core.metrics import get_metrics
from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb_async
from ..services.vector import get_vector_client

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    def __init__(self):
        self.settings = get_settings()
        self.metrics = get_metrics()
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="performance")

    async def optimize_face_processing(self, content: bytes, tenant_id: str) -> Dict[str, Any]:
        """Optimize face processing for better performance."""
        start_time = time.time()

        try:
            # Run face processing operations in parallel
            face_service = get_face_service()

            # Use asyncio.gather for parallel execution
            phash_task = face_service.compute_phash_async(content)
            faces_task = face_service.detect_and_embed_async(content)

            phash, faces = await asyncio.gather(phash_task, faces_task)

            processing_time = time.time() - start_time

            # Log performance metrics
            logger.info(
                f"Face processing completed",
                extra={
                    "tenant_id": tenant_id,
                    "processing_time": processing_time,
                    "faces_detected": len(faces),
                    "phash_computed": bool(phash)
                }
            )

            return {
                "phash": phash,
                "faces": faces,
                "processing_time": processing_time,
                "optimized": True
            }

        except Exception as e:
            logger.error(f"Face processing optimization failed: {e}")
            raise

    async def optimize_storage_operations(self, image_bytes: bytes, tenant_id: str) -> Dict[str, Any]:
        """Optimize storage operations for better performance."""
        start_time = time.time()

        try:
            raw_key, raw_url, thumb_key, thumb_url = await save_raw_and_thumb_async(
                image_bytes, tenant_id
            )

            storage_time = time.time() - start_time

            logger.info(
                f"Storage operations completed",
                extra={
                    "tenant_id": tenant_id,
                    "storage_time": storage_time,
                    "raw_key": raw_key,
                    "thumb_key": thumb_key
                }
            )

            return {
                "raw_key": raw_key,
                "raw_url": raw_url,
                "thumb_key": thumb_key,
                "thumb_url": thumb_url,
                "storage_time": storage_time,
                "optimized": True
            }

        except Exception as e:
            logger.error(f"Storage operations optimization failed: {e}")
            raise

    async def optimize_vector_operations(self, items: List[Dict[str, Any]], tenant_id: str) -> Dict[str, Any]:
        """Optimize vector operations for better performance."""
        start_time = time.time()

        try:
            vector_client = get_vector_client()

            # Batch upsert operations
            vector_client.upsert_embeddings(items, tenant_id)

            vector_time = time.time() - start_time

            logger.info(
                f"Vector operations completed",
                extra={
                    "tenant_id": tenant_id,
                    "vector_time": vector_time,
                    "items_count": len(items)
                }
            )

            return {
                "items_upserted": len(items),
                "vector_time": vector_time,
                "optimized": True
            }

        except Exception as e:
            logger.error(f"Vector operations optimization failed: {e}")
            raise

    async def optimize_search_operations(self, embedding: List[float], tenant_id: str, topk: int = 10) -> Dict[str, Any]:
        """Optimize search operations for better performance."""
        start_time = time.time()

        try:
            vector_client = get_vector_client()

            # Perform similarity search
            results = vector_client.search_similar(embedding, tenant_id, topk)

            search_time = time.time() - start_time

            logger.info(
                f"Search operations completed",
                extra={
                    "tenant_id": tenant_id,
                    "search_time": search_time,
                    "results_count": len(results),
                    "topk": topk
                }
            )

            return {
                "results": results,
                "search_time": search_time,
                "results_count": len(results),
                "optimized": True
            }

        except Exception as e:
            logger.error(f"Search operations optimization failed: {e}")
            raise

    async def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations based on current metrics."""
        try:
            metrics_summary = self.metrics.get_metrics_summary()
            p95_latency = self.metrics.get_p95_latency()
            avg_latency = self.metrics.get_avg_latency()

            recommendations = []

            # Check P95 latency threshold
            if p95_latency > self.settings.p95_latency_threshold_seconds:
                recommendations.append({
                    "type": "latency",
                    "severity": "high",
                    "message": f"P95 latency ({p95_latency:.2f}s) exceeds threshold ({self.settings.p95_latency_threshold_seconds}s)",
                    "suggestion": "Consider optimizing face processing or vector operations"
                })

            # Check average latency
            if avg_latency > self.settings.p95_latency_threshold_seconds * 0.5:
                recommendations.append({
                    "type": "latency",
                    "severity": "medium",
                    "message": f"Average latency ({avg_latency:.2f}s) is high",
                    "suggestion": "Consider caching frequently accessed data"
                })

            # Check error rate
            error_rate = metrics_summary.get("error_rate", 0)
            if error_rate > 0.05:  # 5% error rate
                recommendations.append({
                    "type": "reliability",
                    "severity": "high",
                    "message": f"Error rate ({error_rate:.2%}) is high",
                    "suggestion": "Investigate and fix error sources"
                })

            # Check rate limit violations
            rate_limit_violations = metrics_summary.get("total_rate_limit_violations", 0)
            if rate_limit_violations > 0:
                recommendations.append({
                    "type": "rate_limiting",
                    "severity": "medium",
                    "message": f"Rate limit violations detected ({rate_limit_violations})",
                    "suggestion": "Consider adjusting rate limits or optimizing client behavior"
                })

            return {
                "timestamp": time.time(),
                "current_metrics": {
                    "p95_latency": p95_latency,
                    "avg_latency": avg_latency,
                    "error_rate": error_rate,
                    "rate_limit_violations": rate_limit_violations
                },
                "recommendations": recommendations,
                "recommendation_count": len(recommendations)
            }

        except Exception as e:
            logger.error(f"Failed to get performance recommendations: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "recommendations": [],
                "recommendation_count": 0
            }

    async def monitor_performance_thresholds(self) -> Dict[str, Any]:
        """Monitor performance thresholds and alert if exceeded."""
        try:
            p95_latency = self.metrics.get_p95_latency()
            threshold_exceeded = self.metrics.is_p95_threshold_exceeded()

            return {
                "timestamp": time.time(),
                "p95_latency": p95_latency,
                "threshold": self.settings.p95_latency_threshold_seconds,
                "threshold_exceeded": threshold_exceeded,
                "status": "warning" if threshold_exceeded else "ok"
            }

        except Exception as e:
            logger.error(f"Failed to monitor performance thresholds: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "status": "error"
            }

# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer
