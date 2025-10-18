import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from ..core.config import get_settings
from ..core.metrics import get_metrics
from ..services.health import get_health_service
from ..crawler.cache import get_cache_service
from ..services.webhook import get_webhook_service
from ..crawler.batch import get_batch_processor
import psycopg

logger = logging.getLogger(__name__)

class DashboardService:
    """Service for generating dashboard metrics and analytics."""
    
    def __init__(self):
        self.settings = get_settings()
        self.metrics = get_metrics()
        self.health_service = get_health_service()
        self.cache_service = get_cache_service()
        self.webhook_service = get_webhook_service()
        self.batch_processor = get_batch_processor()
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview metrics."""
        try:
            # Get health status
            health_data = await self.health_service.get_comprehensive_health(use_cache=True)
            
            # Get performance metrics
            performance_metrics = self.metrics.get_metrics_summary()
            
            # Get cache stats
            cache_stats = await self.cache_service.get_cache_stats()
            
            # Get webhook stats (aggregated across all tenants)
            webhook_stats = await self._get_aggregated_webhook_stats()
            
            # Get batch processing stats
            batch_stats = await self._get_batch_processing_stats()
            
            return {
                "timestamp": time.time(),
                "system_health": {
                    "overall_status": health_data["status"],
                    "healthy_services": health_data["summary"]["healthy_services"],
                    "total_services": health_data["summary"]["total_services"],
                    "unhealthy_services": health_data["summary"]["unhealthy_services"]
                },
                "performance": {
                    "p95_latency": performance_metrics["latency"]["p95"],
                    "p99_latency": performance_metrics["latency"]["p99"],
                    "avg_latency": performance_metrics["latency"]["avg"],
                    "total_requests": performance_metrics["requests"]["total"],
                    "error_rate": performance_metrics["requests"]["errors"] / max(performance_metrics["requests"]["total"], 1),
                    "threshold_exceeded": performance_metrics["latency"]["threshold_exceeded"]
                },
                "cache": {
                    "total_cached_items": cache_stats.get("cache_counts", {}).get("total_cached_items", 0),
                    "hit_rate": self._calculate_cache_hit_rate(cache_stats),
                    "memory_usage": cache_stats.get("redis_info", {}).get("used_memory_human", "N/A")
                },
                "webhooks": webhook_stats,
                "batch_processing": batch_stats,
                "environment": self.settings.environment
            }
        except Exception as e:
            logger.error(f"Error generating system overview: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance metrics for the specified time range."""
        try:
            metrics_summary = self.metrics.get_metrics_summary()
            
            # Get endpoint-specific metrics
            endpoint_metrics = {}
            for endpoint, data in metrics_summary.get("endpoints", {}).items():
                endpoint_metrics[endpoint] = {
                    "request_count": data["count"],
                    "p95_latency": data["p95_latency"],
                    "avg_latency": data["avg_latency"],
                    "error_count": data["error_count"],
                    "error_rate": data["error_count"] / max(data["count"], 1)
                }
            
            # Get tenant-specific metrics
            tenant_metrics = {}
            for tenant, data in metrics_summary.get("tenants", {}).items():
                tenant_metrics[tenant] = {
                    "request_count": data["count"],
                    "p95_latency": data["p95_latency"],
                    "rate_limit_violations": data["rate_limit_violations"]
                }
            
            return {
                "timestamp": time.time(),
                "time_range_hours": time_range_hours,
                "overall_metrics": {
                    "total_requests": metrics_summary["requests"]["total"],
                    "total_errors": metrics_summary["requests"]["errors"],
                    "rate_limit_violations": metrics_summary["requests"]["rate_limit_violations"],
                    "p95_latency": metrics_summary["latency"]["p95"],
                    "p99_latency": metrics_summary["latency"]["p99"],
                    "avg_latency": metrics_summary["latency"]["avg"],
                    "median_latency": metrics_summary["latency"]["median"]
                },
                "endpoint_metrics": endpoint_metrics,
                "tenant_metrics": tenant_metrics,
                "threshold_exceeded": metrics_summary["latency"]["threshold_exceeded"]
            }
        except Exception as e:
            logger.error(f"Error generating performance metrics: {e}")
            return {"error": str(e)}
    
    async def get_usage_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get usage analytics and trends."""
        try:
            # Get database analytics
            db_analytics = await self._get_database_analytics(time_range_hours)
            
            # Get cache analytics
            cache_analytics = await self._get_cache_analytics()
            
            # Get webhook analytics
            webhook_analytics = await self._get_webhook_analytics()
            
            # Get batch processing analytics
            batch_analytics = await self._get_batch_analytics()
            
            return {
                "timestamp": time.time(),
                "time_range_hours": time_range_hours,
                "database": db_analytics,
                "cache": cache_analytics,
                "webhooks": webhook_analytics,
                "batch_processing": batch_analytics
            }
        except Exception as e:
            logger.error(f"Error generating usage analytics: {e}")
            return {"error": str(e)}
    
    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get metrics specific to a tenant."""
        try:
            # Get tenant-specific performance metrics
            metrics_summary = self.metrics.get_metrics_summary()
            tenant_data = metrics_summary.get("tenants", {}).get(tenant_id, {})
            
            # Get tenant webhook stats
            webhook_stats = await self.webhook_service.get_webhook_stats(tenant_id)
            
            # Get tenant batch stats
            batch_stats = await self._get_tenant_batch_stats(tenant_id)
            
            # Get tenant cache stats (approximate)
            cache_stats = await self.cache_service.get_cache_stats()
            
            return {
                "tenant_id": tenant_id,
                "timestamp": time.time(),
                "performance": {
                    "request_count": tenant_data.get("count", 0),
                    "p95_latency": tenant_data.get("p95_latency"),
                    "rate_limit_violations": tenant_data.get("rate_limit_violations", 0)
                },
                "webhooks": webhook_stats,
                "batch_processing": batch_stats,
                "cache_usage": {
                    "estimated_items": self._estimate_tenant_cache_usage(tenant_id, cache_stats)
                }
            }
        except Exception as e:
            logger.error(f"Error generating tenant metrics for {tenant_id}: {e}")
            return {"error": str(e)}
    
    async def get_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive health dashboard data."""
        try:
            # Get detailed health data
            health_data = await self.health_service.get_comprehensive_health(use_cache=False)
            
            # Get individual service health
            service_health = {}
            for service_name, service_data in health_data.get("services", {}).items():
                service_health[service_name] = {
                    "status": service_data.get("status"),
                    "response_time_ms": service_data.get("response_time_ms"),
                    "details": {k: v for k, v in service_data.items() if k not in ["status", "response_time_ms"]}
                }
            
            return {
                "timestamp": time.time(),
                "overall_health": {
                    "status": health_data["status"],
                    "total_check_time_ms": health_data.get("total_check_time_ms"),
                    "summary": health_data.get("summary", {})
                },
                "services": service_health,
                "environment": self.settings.environment,
                "unhealthy_services": health_data.get("unhealthy_services", []),
                "degraded_services": health_data.get("degraded_services", [])
            }
        except Exception as e:
            logger.error(f"Error generating health dashboard: {e}")
            return {"error": str(e)}
    
    async def _get_aggregated_webhook_stats(self) -> Dict[str, Any]:
        """Get aggregated webhook statistics across all tenants."""
        try:
            # This is a simplified version - in a real system you'd aggregate from all tenants
            return {
                "total_endpoints": 0,  # Would be calculated from all tenants
                "total_deliveries": 0,
                "success_rate": 0.0,
                "avg_delivery_time_ms": 0.0
            }
        except Exception as e:
            logger.error(f"Error getting aggregated webhook stats: {e}")
            return {"error": str(e)}
    
    async def _get_batch_processing_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        try:
            all_batches = self.batch_processor.list_batches()
            
            total_batches = len(all_batches)
            active_batches = len([b for b in all_batches if b["status"] in ["created", "processing"]])
            completed_batches = len([b for b in all_batches if b["status"] == "completed"])
            failed_batches = len([b for b in all_batches if b["status"] == "failed"])
            
            total_images = sum(b["total_images"] for b in all_batches)
            successful_images = sum(b["successful_images"] for b in all_batches)
            failed_images = sum(b["failed_images"] for b in all_batches)
            
            return {
                "total_batches": total_batches,
                "active_batches": active_batches,
                "completed_batches": completed_batches,
                "failed_batches": failed_batches,
                "total_images_processed": total_images,
                "successful_images": successful_images,
                "failed_images": failed_images,
                "success_rate": successful_images / max(total_images, 1)
            }
        except Exception as e:
            logger.error(f"Error getting batch processing stats: {e}")
            return {"error": str(e)}
    
    async def _get_database_analytics(self, time_range_hours: int) -> Dict[str, Any]:
        """Get database analytics for the specified time range."""
        try:
            # Connect to database to get analytics
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Get audit log statistics
                        cutoff_time = time.time() - (time_range_hours * 3600)
                        
                        await cur.execute("""
                            SELECT 
                                COUNT(*) as total_requests,
                                COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_requests,
                                AVG(process_time) as avg_process_time,
                                COUNT(DISTINCT tenant_id) as unique_tenants
                            FROM audit_logs 
                            WHERE created_at >= %s
                        """, (cutoff_time,))
                        
                        audit_stats = await cur.fetchone()
                        
                        # Get search audit statistics
                        await cur.execute("""
                            SELECT 
                                operation_type,
                                COUNT(*) as count,
                                SUM(face_count) as total_faces,
                                SUM(result_count) as total_results
                            FROM search_audit_logs 
                            WHERE created_at >= %s
                            GROUP BY operation_type
                        """, (cutoff_time,))
                        
                        search_stats = await cur.fetchall()
                        
                        return {
                            "audit_logs": {
                                "total_requests": audit_stats[0] if audit_stats else 0,
                                "error_requests": audit_stats[1] if audit_stats else 0,
                                "avg_process_time": float(audit_stats[2]) if audit_stats and audit_stats[2] else 0.0,
                                "unique_tenants": audit_stats[3] if audit_stats else 0
                            },
                            "search_operations": {
                                row[0]: {
                                    "count": row[1],
                                    "total_faces": row[2],
                                    "total_results": row[3]
                                }
                                for row in search_stats
                            }
                        }
        except Exception as e:
            logger.error(f"Error getting database analytics: {e}")
            return {"error": str(e)}
    
    async def _get_cache_analytics(self) -> Dict[str, Any]:
        """Get cache analytics."""
        try:
            cache_stats = await self.cache_service.get_cache_stats()
            
            return {
                "total_items": cache_stats.get("cache_counts", {}).get("total_cached_items", 0),
                "embedding_cache": cache_stats.get("cache_counts", {}).get("embedding_cache", 0),
                "search_cache": cache_stats.get("cache_counts", {}).get("search_cache", 0),
                "phash_cache": cache_stats.get("cache_counts", {}).get("phash_cache", 0),
                "face_detection_cache": cache_stats.get("cache_counts", {}).get("face_detection_cache", 0),
                "memory_usage": cache_stats.get("redis_info", {}).get("used_memory_human", "N/A"),
                "hit_rate": self._calculate_cache_hit_rate(cache_stats)
            }
        except Exception as e:
            logger.error(f"Error getting cache analytics: {e}")
            return {"error": str(e)}
    
    async def _get_webhook_analytics(self) -> Dict[str, Any]:
        """Get webhook analytics."""
        try:
            # This would aggregate webhook stats from all tenants
            return {
                "total_endpoints": 0,
                "total_deliveries": 0,
                "success_rate": 0.0,
                "avg_delivery_time_ms": 0.0
            }
        except Exception as e:
            logger.error(f"Error getting webhook analytics: {e}")
            return {"error": str(e)}
    
    async def _get_batch_analytics(self) -> Dict[str, Any]:
        """Get batch processing analytics."""
        try:
            all_batches = self.batch_processor.list_batches()
            
            # Calculate trends over time
            now = time.time()
            last_24h = now - 86400
            last_7d = now - (7 * 86400)
            
            batches_24h = [b for b in all_batches if b["created_at"] >= last_24h]
            batches_7d = [b for b in all_batches if b["created_at"] >= last_7d]
            
            return {
                "total_batches": len(all_batches),
                "batches_24h": len(batches_24h),
                "batches_7d": len(batches_7d),
                "avg_batch_size": sum(b["total_images"] for b in all_batches) / max(len(all_batches), 1),
                "success_rate": sum(b["successful_images"] for b in all_batches) / max(sum(b["total_images"] for b in all_batches), 1)
            }
        except Exception as e:
            logger.error(f"Error getting batch analytics: {e}")
            return {"error": str(e)}
    
    async def _get_tenant_batch_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get batch processing stats for a specific tenant."""
        try:
            tenant_batches = self.batch_processor.list_batches(tenant_id=tenant_id)
            
            return {
                "total_batches": len(tenant_batches),
                "active_batches": len([b for b in tenant_batches if b["status"] in ["created", "processing"]]),
                "completed_batches": len([b for b in tenant_batches if b["status"] == "completed"]),
                "failed_batches": len([b for b in tenant_batches if b["status"] == "failed"]),
                "total_images": sum(b["total_images"] for b in tenant_batches),
                "successful_images": sum(b["successful_images"] for b in tenant_batches)
            }
        except Exception as e:
            logger.error(f"Error getting tenant batch stats: {e}")
            return {"error": str(e)}
    
    def _calculate_cache_hit_rate(self, cache_stats: Dict[str, Any]) -> float:
        """Calculate cache hit rate from Redis info."""
        try:
            redis_info = cache_stats.get("redis_info", {})
            hits = redis_info.get("keyspace_hits", 0)
            misses = redis_info.get("keyspace_misses", 0)
            
            if hits + misses == 0:
                return 0.0
            
            return hits / (hits + misses)
        except Exception:
            return 0.0
    
    def _estimate_tenant_cache_usage(self, tenant_id: str, cache_stats: Dict[str, Any]) -> int:
        """Estimate cache usage for a specific tenant."""
        try:
            # This is a rough estimate - in a real system you'd count actual keys
            total_items = cache_stats.get("cache_counts", {}).get("total_cached_items", 0)
            # Assume roughly equal distribution across tenants (not accurate but gives an idea)
            return total_items // 10  # Assuming ~10 tenants on average
        except Exception:
            return 0

# Global dashboard service instance
_dashboard_service = None

def get_dashboard_service() -> DashboardService:
    """Get dashboard service instance."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service
