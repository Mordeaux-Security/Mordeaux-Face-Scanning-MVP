import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any


from ..services.cleanup import get_cleanup_service
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class CleanupScheduler:
    def __init__(self):
        self.settings = get_settings()
        self.cleanup_service = get_cleanup_service()
        self.running = False
        self.cleanup_interval_hours = 24  # Run cleanup daily

    async def start_scheduler(self):
        """Start the cleanup scheduler."""
        if self.running:
            logger.warning("Cleanup scheduler is already running")
            return

        self.running = True
        logger.info("Starting cleanup scheduler...")

        while self.running:
            try:
                await self.run_cleanup_cycle()
                # Wait for the next cleanup cycle
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
            except Exception as e:
                logger.error(f"Error in cleanup scheduler: {e}")
                # Wait a shorter time before retrying on error
                await asyncio.sleep(3600)  # 1 hour

    async def stop_scheduler(self):
        """Stop the cleanup scheduler."""
        self.running = False
        logger.info("Stopping cleanup scheduler...")

    async def run_cleanup_cycle(self):
        """Run a complete cleanup cycle."""
        logger.info("Starting cleanup cycle...")
        start_time = datetime.now()

        try:
            # Run all cleanup jobs
            results = await self.cleanup_service.run_all_cleanup_jobs()

            # Log summary
            summary = results.get("summary", {})
            logger.info(
                f"Cleanup cycle completed: {summary.get('successful_jobs', 0)}/{summary.get('total_jobs', 0)} jobs successful"
            )

            # Log individual job results
            for job_name, job_result in results.get("jobs", {}).items():
                status = job_result.get("status", "unknown")
                if status == "success":
                    logger.info(f"Cleanup job '{job_name}' completed successfully")
                else:
                    logger.error(f"Cleanup job '{job_name}' failed: {job_result.get('error', 'Unknown error')}")

            # Calculate cycle duration
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Cleanup cycle took {duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Cleanup cycle failed: {e}")
            raise

    async def run_immediate_cleanup(self, job_types: list = None) -> Dict[str, Any]:
        """Run cleanup jobs immediately (for manual triggers)."""
        logger.info(f"Running immediate cleanup for job types: {job_types or 'all'}")

        if job_types is None:
            # Run all cleanup jobs
            return await self.cleanup_service.run_all_cleanup_jobs()
        else:
            # Run specific cleanup jobs
            results = {"timestamp": datetime.now().timestamp(), "jobs": {}}

            for job_type in job_types:
                try:
                    if job_type == "thumbnails":
                        result = await self.cleanup_service.cleanup_old_thumbnails()
                    elif job_type == "user_queries":
                        result = await self.cleanup_service.cleanup_user_query_images()
                    elif job_type == "audit_logs":
                        result = await self.cleanup_service.cleanup_old_audit_logs()
                    elif job_type == "orphaned_vectors":
                        result = await self.cleanup_service.cleanup_orphaned_vectors()
                    else:
                        result = {"status": "error", "error": f"Unknown job type: {job_type}"}

                    results["jobs"][job_type] = result
                    logger.info(f"Immediate cleanup job '{job_type}' completed: {result.get('status', 'unknown')}")

                except Exception as e:
                    logger.error(f"Immediate cleanup job '{job_type}' failed: {e}")
                    results["jobs"][job_type] = {"status": "error", "error": str(e)}

            return results

# Global cleanup scheduler instance
_cleanup_scheduler = None

def get_cleanup_scheduler() -> CleanupScheduler:
    """Get cleanup scheduler instance."""
    global _cleanup_scheduler
    if _cleanup_scheduler is None:
        _cleanup_scheduler = CleanupScheduler()
    return _cleanup_scheduler

async def start_cleanup_scheduler():
    """Start the cleanup scheduler (called from main application)."""
    scheduler = get_cleanup_scheduler()
    await scheduler.start_scheduler()

async def stop_cleanup_scheduler():
    """Stop the cleanup scheduler."""
    scheduler = get_cleanup_scheduler()
    await scheduler.stop_scheduler()

async def run_immediate_cleanup(job_types: list = None) -> Dict[str, Any]:
    """Run cleanup jobs immediately."""
    scheduler = get_cleanup_scheduler()
    return await scheduler.run_immediate_cleanup(job_types)
