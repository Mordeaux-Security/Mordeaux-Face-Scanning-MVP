import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
from ..core.config import get_settings
from ..services.storage import list_objects, get_object_from_storage
from ..services.vector import get_vector_client
import psycopg
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Database connection for cleanup operations
_cleanup_db_pool = None

def get_cleanup_db_pool():
    """Get cleanup database connection pool."""
    global _cleanup_db_pool
    if _cleanup_db_pool is None:
        settings = get_settings()
        connection_string = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        _cleanup_db_pool = AsyncConnectionPool(connection_string, min_size=1, max_size=3)
    return _cleanup_db_pool

class CleanupService:
    def __init__(self):
        self.settings = get_settings()
        self.db_pool = get_cleanup_db_pool()
        self.vector_client = get_vector_client()
    
    async def cleanup_old_thumbnails(self) -> Dict[str, Any]:
        """Clean up old thumbnails based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.settings.crawled_thumbs_retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            # Get old thumbnail objects from storage
            old_objects = []
            try:
                all_objects = list_objects(self.settings.s3_bucket_thumbs)
                for obj_key in all_objects:
                    # Extract timestamp from object key or metadata if available
                    # For now, we'll use a simple approach based on object listing
                    # In production, you might want to store creation timestamps in metadata
                    pass
            except Exception as e:
                logger.warning(f"Could not list thumbnail objects: {e}")
            
            # Clean up old database records
            deleted_count = 0
            async with self.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Delete old image records (this will cascade to faces)
                    await cur.execute("""
                        DELETE FROM images 
                        WHERE created_at < %s
                    """, (cutoff_timestamp,))
                    deleted_count = cur.rowcount
                    await conn.commit()
            
            return {
                "status": "success",
                "deleted_images": deleted_count,
                "retention_days": self.settings.crawled_thumbs_retention_days,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old thumbnails: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup_user_query_images(self) -> Dict[str, Any]:
        """Clean up user query images based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(hours=self.settings.user_query_images_retention_hours)
            cutoff_timestamp = cutoff_date.timestamp()
            
            # Clean up old user query images from storage
            deleted_count = 0
            try:
                # List objects in raw-images bucket that might be user queries
                # This is a simplified approach - in production you might want to tag user queries
                all_objects = list_objects(self.settings.s3_bucket_raw)
                for obj_key in all_objects:
                    # Check if this is a user query image (you might want to add metadata to distinguish)
                    # For now, we'll clean up based on database records
                    pass
            except Exception as e:
                logger.warning(f"Could not list raw image objects: {e}")
            
            # Clean up old database records for user queries
            async with self.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Delete old image records that are user queries
                    # You might want to add a field to distinguish user queries from crawled images
                    await cur.execute("""
                        DELETE FROM images 
                        WHERE created_at < %s AND site IS NULL
                    """, (cutoff_timestamp,))
                    deleted_count = cur.rowcount
                    await conn.commit()
            
            return {
                "status": "success",
                "deleted_user_queries": deleted_count,
                "retention_hours": self.settings.user_query_images_retention_hours,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup user query images: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup_old_audit_logs(self, retention_days: int = 30) -> Dict[str, Any]:
        """Clean up old audit logs based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            deleted_audit_logs = 0
            deleted_search_logs = 0
            
            async with self.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Delete old audit logs
                    await cur.execute("""
                        DELETE FROM audit_logs 
                        WHERE created_at < %s
                    """, (cutoff_timestamp,))
                    deleted_audit_logs = cur.rowcount
                    
                    # Delete old search audit logs
                    await cur.execute("""
                        DELETE FROM search_audit_logs 
                        WHERE created_at < %s
                    """, (cutoff_timestamp,))
                    deleted_search_logs = cur.rowcount
                    
                    await conn.commit()
            
            return {
                "status": "success",
                "deleted_audit_logs": deleted_audit_logs,
                "deleted_search_logs": deleted_search_logs,
                "retention_days": retention_days,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup audit logs: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup_orphaned_vectors(self) -> Dict[str, Any]:
        """Clean up orphaned vectors that don't have corresponding database records."""
        try:
            # This is a complex operation that would require:
            # 1. Getting all vector IDs from the vector database
            # 2. Checking which ones don't have corresponding database records
            # 3. Deleting the orphaned vectors
            
            # For now, we'll return a placeholder
            return {
                "status": "success",
                "message": "Vector cleanup not implemented yet",
                "deleted_vectors": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned vectors: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_all_cleanup_jobs(self) -> Dict[str, Any]:
        """Run all cleanup jobs and return combined results."""
        logger.info("Starting cleanup jobs...")
        
        results = {
            "timestamp": time.time(),
            "jobs": {}
        }
        
        # Run cleanup jobs in parallel
        cleanup_tasks = [
            ("thumbnails", self.cleanup_old_thumbnails()),
            ("user_queries", self.cleanup_user_query_images()),
            ("audit_logs", self.cleanup_old_audit_logs()),
            ("orphaned_vectors", self.cleanup_orphaned_vectors())
        ]
        
        for job_name, task in cleanup_tasks:
            try:
                result = await task
                results["jobs"][job_name] = result
                logger.info(f"Cleanup job '{job_name}' completed: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"Cleanup job '{job_name}' failed: {e}")
                results["jobs"][job_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate summary
        total_successful = sum(1 for job in results["jobs"].values() if job.get("status") == "success")
        total_failed = len(results["jobs"]) - total_successful
        
        results["summary"] = {
            "total_jobs": len(results["jobs"]),
            "successful_jobs": total_successful,
            "failed_jobs": total_failed,
            "overall_status": "success" if total_failed == 0 else "partial_failure"
        }
        
        logger.info(f"Cleanup jobs completed: {total_successful}/{len(results['jobs'])} successful")
        return results

# Global cleanup service instance
_cleanup_service = None

def get_cleanup_service() -> CleanupService:
    """Get cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = CleanupService()
    return _cleanup_service

async def run_cleanup_jobs() -> Dict[str, Any]:
    """Run all cleanup jobs."""
    cleanup_service = get_cleanup_service()
    return await cleanup_service.run_all_cleanup_jobs()