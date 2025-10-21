import asyncio
import time
import uuid
import aiohttp
from typing import Dict, List, Optional, Any
import logging


from ..core.config import get_settings
from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb_async
from ..services.vector import get_vector_client
from ..services.cache import get_cache_service
from ..services.webhook import get_webhook_service, WebhookEvent
from ..core.audit import get_audit_logger

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Service for processing batch face indexing operations."""

    def __init__(self):
        self.settings = get_settings()
        self.active_batches: Dict[str, Dict[str, Any]] = {}
        self.max_concurrent_downloads = 5
        self.max_concurrent_processing = 3

    async def create_batch_job(self, tenant_id: str, image_urls: List[str], metadata: Optional[dict] = None) -> str:
        """Create a new batch processing job."""
        batch_id = str(uuid.uuid4())

        batch_info = {
            "batch_id": batch_id,
            "tenant_id": tenant_id,
            "image_urls": image_urls,
            "metadata": metadata or {},
            "status": "created",
            "total_images": len(image_urls),
            "processed_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "errors": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }

        self.active_batches[batch_id] = batch_info
        logger.info(f"Created batch job {batch_id} for tenant {tenant_id} with {len(image_urls)} images")

        # Send webhook notification for batch creation
        webhook_service = get_webhook_service()
        webhook_event = WebhookEvent(
            event_type="batch.created",
            tenant_id=tenant_id,
            data={
                "batch_id": batch_id,
                "total_images": len(image_urls),
                "metadata": metadata or {}
            },
            request_id=f"batch_{batch_id}"
        )
        asyncio.create_task(webhook_service.send_webhook(webhook_event))

        return batch_id

    async def process_batch_job(self, batch_id: str) -> None:
        """Process a batch job asynchronously."""
        if batch_id not in self.active_batches:
            logger.error(f"Batch job {batch_id} not found")
            return

        batch_info = self.active_batches[batch_id]
        batch_info["status"] = "processing"
        batch_info["updated_at"] = time.time()

        logger.info(f"Starting batch processing for job {batch_id}")

        try:
            # Process images in batches to avoid overwhelming the system
            semaphore = asyncio.Semaphore(self.max_concurrent_processing)

            tasks = []
            for i, image_url in enumerate(batch_info["image_urls"]):
                task = self._process_single_image(batch_id, image_url, i, semaphore)
                tasks.append(task)

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Mark batch as completed
            batch_info["status"] = "completed"
            batch_info["updated_at"] = time.time()

            logger.info(f"Completed batch processing for job {batch_id}: {batch_info['successful_images']} successful, {batch_info['failed_images']} failed")

            # Send webhook notification for batch completion
            webhook_service = get_webhook_service()
            webhook_event = WebhookEvent(
                event_type="batch.completed",
                tenant_id=batch_info["tenant_id"],
                data={
                    "batch_id": batch_id,
                    "total_images": batch_info["total_images"],
                    "successful_images": batch_info["successful_images"],
                    "failed_images": batch_info["failed_images"],
                    "errors": batch_info["errors"]
                },
                request_id=f"batch_{batch_id}"
            )
            asyncio.create_task(webhook_service.send_webhook(webhook_event))

        except Exception as e:
            logger.error(f"Error processing batch job {batch_id}: {e}")
            batch_info["status"] = "failed"
            batch_info["updated_at"] = time.time()
            batch_info["errors"].append({
                "type": "batch_error",
                "message": str(e),
                "timestamp": time.time()
            })

            # Send webhook notification for batch failure
            webhook_service = get_webhook_service()
            webhook_event = WebhookEvent(
                event_type="batch.failed",
                tenant_id=batch_info["tenant_id"],
                data={
                    "batch_id": batch_id,
                    "error": str(e),
                    "errors": batch_info["errors"]
                },
                request_id=f"batch_{batch_id}"
            )
            asyncio.create_task(webhook_service.send_webhook(webhook_event))

    async def _process_single_image(self, batch_id: str, image_url: str, index: int, semaphore: asyncio.Semaphore) -> None:
        """Process a single image in the batch."""
        async with semaphore:
            batch_info = self.active_batches[batch_id]
            tenant_id = batch_info["tenant_id"]

            try:
                # Download image
                image_content = await self._download_image(image_url)
                if not image_content:
                    raise Exception(f"Failed to download image from {image_url}")

                # Process face detection and embedding
                face_service = get_face_service()
                cache_service = get_cache_service()

                # Check cache first
                cached_phash = await cache_service.get_cached_perceptual_hash(image_content, tenant_id)
                cached_faces = await cache_service.get_cached_face_detection(image_content, tenant_id)

                if cached_phash and cached_faces:
                    phash = cached_phash
                    faces = cached_faces
                else:
                    phash, faces = await asyncio.gather(
                        face_service.compute_phash_async(image_content),
                        face_service.detect_and_embed_async(image_content)
                    )

                    # Cache the results
                    await asyncio.gather(
                        cache_service.cache_perceptual_hash(image_content, tenant_id, phash),
                        cache_service.cache_face_detection(image_content, tenant_id, faces)
                    )

                if not faces:
                    logger.warning(f"No faces detected in image {index} from {image_url}")
                    batch_info["processed_images"] += 1
                    batch_info["updated_at"] = time.time()
                    return

                # Save images to storage
                raw_key, raw_url, thumb_key, thumb_url = await save_raw_and_thumb_async(image_content, tenant_id)

                # Prepare vector embeddings
                vec = get_vector_client()
                items = []
                for f in faces:
                    items.append({
                        "id": str(uuid.uuid4()),
                        "embedding": f["embedding"],
                        "metadata": {
                            "raw_key": raw_key,
                            "thumb_key": thumb_key,
                            "raw_url": raw_url,
                            "thumb_url": thumb_url,
                            "bbox": f["bbox"],
                            "det_score": f["det_score"],
                            "phash": phash,
                            "batch_id": batch_id,
                            "image_url": image_url,
                            "image_index": index,
                            **batch_info["metadata"]
                        },
                    })

                # Upsert to vector database
                if items:
                    vec.upsert_embeddings(items, tenant_id)

                # Log audit
                audit_logger = get_audit_logger()
                await audit_logger.log_search_operation(
                    tenant_id=tenant_id,
                    operation_type="batch_index",
                    face_count=len(faces),
                    result_count=len(items),
                    vector_backend="pinecone" if vec.using_pinecone() else "qdrant",
                    request_id=f"batch_{batch_id}_{index}"
                )

                # Update batch status
                batch_info["successful_images"] += 1
                batch_info["processed_images"] += 1
                batch_info["updated_at"] = time.time()

                logger.debug(f"Successfully processed image {index} from batch {batch_id}")

            except Exception as e:
                logger.error(f"Error processing image {index} from {image_url} in batch {batch_id}: {e}")

                # Update batch status with error
                batch_info["failed_images"] += 1
                batch_info["processed_images"] += 1
                batch_info["updated_at"] = time.time()
                batch_info["errors"].append({
                    "image_index": index,
                    "image_url": image_url,
                    "error": str(e),
                    "timestamp": time.time()
                })

    async def _download_image(self, image_url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"Failed to download image from {image_url}: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return None

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a batch job."""
        return self.active_batches.get(batch_id)

    def list_batches(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all batch jobs, optionally filtered by tenant."""
        batches = list(self.active_batches.values())
        if tenant_id:
            batches = [b for b in batches if b["tenant_id"] == tenant_id]
        return sorted(batches, key=lambda x: x["created_at"], reverse=True)

    def cleanup_old_batches(self, max_age_hours: int = 24) -> int:
        """Clean up old completed batch jobs."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        to_remove = []
        for batch_id, batch_info in self.active_batches.items():
            if (batch_info["status"] in ["completed", "failed"] and
                current_time - batch_info["updated_at"] > max_age_seconds):
                to_remove.append(batch_id)

        for batch_id in to_remove:
            del self.active_batches[batch_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old batch jobs")

        return len(to_remove)

# Global batch processor instance
_batch_processor = None

def get_batch_processor() -> BatchProcessor:
    """Get batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor
