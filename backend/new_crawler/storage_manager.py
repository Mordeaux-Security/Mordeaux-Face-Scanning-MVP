"""
Storage Manager for New Crawler System

Handles MinIO save operations for raw images and face thumbnails.
Provides content-addressed naming and metadata sidecar files.
"""

import asyncio
import io
import logging
import os
import time
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import hashlib
import json
from datetime import datetime

from .config import get_config
from .data_structures import ImageTask, FaceDetection, FaceResult

logger = logging.getLogger(__name__)

# Singleton pattern per process
_storage_manager_instance = None


class StorageManager:
    """Storage manager for MinIO operations."""
    
    def __init__(self):
        self.config = get_config()
        self._minio_client = None
        self._s3_client = None
        
    def _get_client(self):
        """Get MinIO/S3 client with connection pooling."""
        if self.config.s3_endpoint:
            # Using MinIO
            if self._minio_client is None:
                try:
                    from minio import Minio
                    from urllib3 import PoolManager
                    
                    # Create connection pool to prevent connection exhaustion
                    http_client = PoolManager(
                        maxsize=self.config.minio_max_pool_size,  # Max connections in pool
                        retries=3,
                        timeout=self.config.minio_pool_timeout,
                        block=False  # Don't block if pool is full
                    )
                    
                    self._minio_client = Minio(
                        self.config.s3_endpoint.replace('http://', '').replace('https://', ''),
                        access_key=self.config.s3_access_key,
                        secret_key=self.config.s3_secret_key,
                        secure=self.config.s3_use_ssl,
                        http_client=http_client  # Use connection pool
                    )
                    logger.debug(f"Created MinIO client with connection pooling (maxsize={self.config.minio_max_pool_size})")
                except ImportError:
                    logger.error("MinIO client not available. Install with: pip install minio")
                    raise
            return self._minio_client
        else:
            # Using AWS S3
            if self._s3_client is None:
                try:
                    import boto3
                    self._s3_client = boto3.client(
                        's3',
                        region_name=self.config.s3_region,
                        aws_access_key_id=self.config.s3_access_key,
                        aws_secret_access_key=self.config.s3_secret_key
                    )
                except ImportError:
                    logger.error("Boto3 not available. Install with: pip install boto3")
                    raise
            return self._s3_client
    
    def _compute_content_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(data).hexdigest()
    
    def _get_content_key(self, content_hash: str, extension: str = '.jpg') -> str:
        """Get content-addressed key for storage."""
        # Use default/{first2}/{hash}.jpg structure
        return f"default/{content_hash[:2]}/{content_hash}{extension}"
    
    def _create_metadata(self, image_task: ImageTask, face_result: FaceResult) -> Dict[str, Any]:
        """Create metadata for image."""
        return {
            'content_hash': self._compute_content_hash(open(image_task.temp_path, 'rb').read()),
            'phash': image_task.phash,
            'site_id': image_task.candidate.site_id,
            'page_url': image_task.candidate.page_url,
            'img_url': image_task.candidate.img_url,
            'selector_hint': image_task.candidate.selector_hint,
            'alt_text': image_task.candidate.alt_text,
            'width': image_task.candidate.width,
            'height': image_task.candidate.height,
            'file_size': image_task.file_size,
            'mime_type': image_task.mime_type,
            'faces_count': len(face_result.faces),
            'faces': [
                {
                    'bbox': face.bbox,
                    'quality': face.quality,
                    'age': face.age,
                    'gender': face.gender
                }
                for face in face_result.faces
            ],
            'processing_time_ms': face_result.processing_time_ms,
            'gpu_used': face_result.gpu_used,
            'created_at': datetime.now().isoformat(),
            'crawler_version': '1.0.0'
        }
    
    def save_raw_image(self, image_task: ImageTask) -> Tuple[Optional[str], Optional[str]]:
        """Save raw image to MinIO."""
        try:
            client = self._get_client()
            
            # Read image bytes from temp file
            with open(image_task.temp_path, 'rb') as f:
                image_data = f.read()
            
            # Generate key
            key = f"{image_task.phash}.jpg"
            
            # Save to MinIO with retry
            for attempt in range(3):
                try:
                    client.put_object(
                        bucket_name=self.config.s3_bucket_raw,
                        object_name=key,
                        data=io.BytesIO(image_data),  # Wrap bytes in BytesIO
                        length=len(image_data),
                        content_type=image_task.mime_type
                    )
                    logger.debug(f"[STORAGE] Raw image saved: {key}, size={len(image_data)}bytes")
                    return key, f"s3://{self.config.s3_bucket_raw}/{key}"
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1.0)
        except Exception as e:
            logger.error(f"Failed to save raw image: {e}")
            return None, None
    
    # TODO: ARCHITECTURE ISSUE - Mixed Storage Locations
    # Current structure:
    #   - raw-images/ bucket: {phash}.jpg + {phash}.meta.json (mixed)
    #   - thumbnails/ bucket: {face_hash}.jpg (separate)
    #
    # Problems:
    #   1. Raw and thumbnails in different buckets (management complexity)
    #   2. Metadata mixed with images
    #   3. Hard to find all files for one image
    #
    # Proposed structure (single bucket):
    #   images/
    #       {phash}/
    #           raw.jpg              # Original image
    #           meta.json            # Detection metadata
    #           faces/
    #               {face_hash}_0.jpg  # Cropped face 0
    #               {face_hash}_1.jpg  # Cropped face 1
    #
    # Benefits: Atomic operations, easier cleanup, clearer organization
    
    async def save_raw_image_async(self, image_task: ImageTask) -> Tuple[Optional[str], Optional[str]]:
        """Save raw image to MinIO (async wrapper around sync)."""
        try:
            # Use sync client in thread pool for async behavior
            return await asyncio.to_thread(self.save_raw_image, image_task)
        except Exception as e:
            logger.error(f"Failed to save raw image: {e}")
            return None, None
    
    def save_face_thumbnail(self, face_crop_data: bytes, face_detection: FaceDetection,  
                           base_hash: str, face_index: int) -> Tuple[Optional[str], Optional[str]]:
        """Save face thumbnail to MinIO."""
        try:
            # Compute content hash for face crop
            content_hash = self._compute_content_hash(face_crop_data)
            
            # Get storage key with corrected structure
            key = self._get_content_key(content_hash, '.jpg')  # Remove face_index from filename
            
            # Upload to MinIO thumbnails bucket
            client = self._get_client()
            
            if self.config.s3_endpoint:
                # MinIO - wrap bytes in BytesIO
                client.put_object(
                    bucket_name=self.config.s3_bucket_thumbs,
                    object_name=key,
                    data=io.BytesIO(face_crop_data),  # Wrap in BytesIO
                    length=len(face_crop_data),
                    content_type='image/jpeg'
                )
            else:
                # AWS S3
                client.put_object(
                    Bucket=self.config.s3_bucket_thumbs,
                    Key=key,
                    Body=face_crop_data,
                    ContentType='image/jpeg'
                )
            
            # Generate URL
            if self.config.s3_endpoint:
                url = f"{self.config.s3_endpoint}/{self.config.s3_bucket_thumbs}/{key}"
            else:
                url = f"https://{self.config.s3_bucket_thumbs}.s3.{self.config.s3_region}.amazonaws.com/{key}"
            
            logger.debug(f"Saved face thumbnail to {key}")
            return key, url
            
        except Exception as e:
            logger.error(f"Failed to save face thumbnail: {e}")
            return None, None
    
    async def save_face_thumbnail_async(self, face_crop_data: bytes, face_detection: FaceDetection, 
                                       base_hash: str, face_index: int) -> Tuple[Optional[str], Optional[str]]:
        """Save face thumbnail to MinIO (async wrapper around sync)."""
        try:
            # Use sync client in thread pool for async behavior
            return await asyncio.to_thread(
                self.save_face_thumbnail, face_crop_data, face_detection, base_hash, face_index
            )
        except Exception as e:
            logger.error(f"Failed to save face thumbnail: {e}")
            return None, None
    
    def save_metadata(self, image_task: ImageTask, face_result: FaceResult,  
                     raw_key: str, thumbnail_keys: List[str]) -> bool:
        """Save metadata sidecar file."""
        try:
            # Create metadata
            metadata = self._create_metadata(image_task, face_result)
            metadata['raw_image_key'] = raw_key
            metadata['thumbnail_keys'] = thumbnail_keys
            
            # Convert to JSON
            metadata_json = json.dumps(metadata, indent=2)
            metadata_data = metadata_json.encode('utf-8')
            
            # Get metadata key
            metadata_key = raw_key.replace('.jpg', '.meta.json')
            
            # Upload metadata
            client = self._get_client()
            
            if self.config.s3_endpoint:
                # MinIO - wrap bytes in BytesIO
                client.put_object(
                    bucket_name=self.config.s3_bucket_raw,
                    object_name=metadata_key,
                    data=io.BytesIO(metadata_data),  # FIX: Wrap in BytesIO
                    length=len(metadata_data),
                    content_type='application/json'
                )
            else:
                # AWS S3
                client.put_object(
                    Bucket=self.config.s3_bucket_raw,
                    Key=metadata_key,
                    Body=metadata_data,
                    ContentType='application/json'
                )
            
            logger.debug(f"Saved metadata to {metadata_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
    
    async def save_metadata_async(self, image_task: ImageTask, face_result: FaceResult, 
                                 raw_key: str, thumbnail_keys: List[str]) -> bool:
        """Save metadata sidecar file (async wrapper around sync)."""
        try:
            # Use sync client in thread pool for async behavior
            return await asyncio.to_thread(
                self.save_metadata, image_task, face_result, raw_key, thumbnail_keys
            )
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
    
    # TODO: ARCHITECTURE ISSUE - Compute in Storage Layer
    # Storage manager should NOT do image manipulation (PIL operations).
    # Cropping is compute-heavy and belongs in GPU processor worker.
    #
    # Current (WRONG):
    #   GPU Processor -> Storage -> PIL crop -> Save thumbnails
    #
    # Proposed (CLEAN):
    #   GPU Processor -> PIL crop -> Storage -> Save pre-cropped bytes
    #
    # This method should be moved to gpu_processor_worker.py or a dedicated
    # image_processing utility module. Storage should receive pre-cropped bytes.
    #
    # Impact: ~100-200ms saved per batch by doing crops during GPU wait time
    
    def crop_face_from_image(self, image_path: str, face_detection: FaceDetection, 
                            margin: float = None) -> Optional[bytes]:
        """Crop face from image."""
        try:
            from PIL import Image
            import os
            
            # DEBUG: Log inputs
            logger.info(f"[CROP DEBUG] Starting crop: path={image_path}, bbox={face_detection.bbox}")
            
            # Check file exists
            if not os.path.exists(image_path):
                logger.error(f"[CROP DEBUG] File not found: {image_path}")
                return None
            
            if margin is None:
                margin = self.config.face_margin
            
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Extract bounding box coordinates
                bbox = face_detection.bbox
                x1, y1, x2, y2 = bbox
                
                # DEBUG: Check if coordinates are already normalized (0-1 range)
                if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                    # Coordinates are in pixels, normalize them
                    logger.info(f"[CROP DEBUG] Normalizing pixel coordinates: {bbox} -> image size: {width}x{height}")
                    x1 = x1 / width
                    y1 = y1 / height
                    x2 = x2 / width
                    y2 = y2 / height
                    logger.info(f"[CROP DEBUG] Normalized coordinates: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                else:
                    logger.info(f"[CROP DEBUG] Using normalized coordinates: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                
                # Convert to pixel coordinates
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                
                # Add margin
                face_width = x2 - x1
                face_height = y2 - y1
                margin_x = int(face_width * margin)
                margin_y = int(face_height * margin)
                
                # Expand bounding box with margin
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(width, x2 + margin_x)
                y2 = min(height, y2 + margin_y)
                
                # Ensure valid bounds
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid face bounds: ({x1}, {y1}, {x2}, {y2})")
                    return None
                
                # Crop the face
                face_crop = img.crop((x1, y1, x2, y2))
                
                # Resize to thumbnail size
                face_crop.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                output = io.BytesIO()
                face_crop.save(output, format='JPEG', quality=85)
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"Failed to crop face from {image_path}: {e}", exc_info=True)
            return None
    
    async def crop_face_from_image_async(self, image_path: str, face_detection: FaceDetection, 
                                        margin: float = None) -> Optional[bytes]:
        """Crop face from image (async - runs PIL in thread pool)."""
        return await asyncio.to_thread(
            self.crop_face_from_image, image_path, face_detection, margin
        )
    
    def save_face_result(self, image_task: ImageTask, face_result: FaceResult) -> Tuple[FaceResult, Dict[str, int]]:
        """Save complete face result to storage."""
        try:
            save_counts = {'saved_raw': 0, 'saved_thumbs': 0}
            
            # Save raw image
            logger.debug(f"[STORAGE] Saving raw image: {image_task.phash[:8]}..., size={image_task.file_size}bytes")
            raw_key, raw_url = self.save_raw_image(image_task)
            if not raw_key:
                logger.error("Failed to save raw image")
                return face_result, save_counts
            
            # Update face result with raw image key
            face_result.raw_image_key = raw_key
            save_counts['saved_raw'] = 1
            logger.debug(f"[STORAGE] Raw image saved: {raw_key}")
            
            # Save face thumbnails
            thumbnail_keys = []
            crop_paths = []
            
            for i, face_detection in enumerate(face_result.faces):
                logger.debug(f"[STORAGE] Saving thumbnail {i+1}/{len(face_result.faces)}: {image_task.phash[:8]}...")
                # Crop face from image
                face_crop_data = self.crop_face_from_image(image_task.temp_path, face_detection)
                if face_crop_data:
                    # Save thumbnail
                    thumb_key, thumb_url = self.save_face_thumbnail(
                        face_crop_data, face_detection, raw_key, i
                    )
                    if thumb_key:
                        thumbnail_keys.append(thumb_key)
                        crop_paths.append(f"temp_crop_{i}.jpg")  # Temporary path for compatibility
                        save_counts['saved_thumbs'] += 1
                        logger.debug(f"[STORAGE] Thumbnail saved: {thumb_key}")
                    else:
                        logger.warning(f"[STORAGE] Failed to save thumbnail: {thumb_key}")
                else:
                    logger.warning(f"[STORAGE] Failed to crop face {i} from {image_task.phash[:8]}...")
            
            # Update face result with thumbnail keys
            face_result.thumbnail_keys = thumbnail_keys
            face_result.crop_paths = crop_paths
            
            # Save metadata
            logger.debug(f"[STORAGE] Saving metadata: {image_task.phash[:8]}...")
            self.save_metadata(image_task, face_result, raw_key, thumbnail_keys)
            
            logger.info(f"[STORAGE] Save complete: raw={raw_key}, thumbs={len(thumbnail_keys)}")
            return face_result, save_counts
            
        except Exception as e:
            logger.error(f"Failed to save face result: {e}")
            return face_result, save_counts
    
    # TODO: ARCHITECTURE ISSUE - Complex Storage API
    # This method does too much:
    #   1. Saves raw image (I/O)
    #   2. Crops faces (COMPUTE - wrong layer!)
    #   3. Saves thumbnails (I/O)
    #   4. Saves metadata (I/O)
    #
    # Should be simplified to:
    #   async def save_complete_result(
    #       raw_bytes: bytes,
    #       face_crops: List[bytes],  # Pre-cropped!
    #       metadata: Dict
    #   ) -> SaveResult
    #
    # All compute (cropping) happens in GPU processor before calling storage.
    # Storage becomes pure I/O operations.
    
    async def save_face_result_async(self, image_task: ImageTask, face_result: FaceResult) -> Tuple[FaceResult, Dict[str, int]]:
        """Save complete face result to storage (async with concurrent operations)."""
        try:
            save_counts = {'saved_raw': 0, 'saved_thumbs': 0}
            
            # Save raw image (async)
            logger.debug(f"[STORAGE] Saving raw image: {image_task.phash[:8]}..., size={image_task.file_size}bytes")
            raw_key, raw_url = await self.save_raw_image_async(image_task)
            if not raw_key:
                logger.error("Failed to save raw image")
                return face_result, save_counts
            
            face_result.raw_image_key = raw_key
            save_counts['saved_raw'] = 1
            logger.debug(f"[STORAGE] Raw image saved: {raw_key}")
            
            # Process all face thumbnails concurrently
            thumbnail_tasks = []
            for i, face_detection in enumerate(face_result.faces):
                logger.debug(f"[STORAGE] Preparing thumbnail {i+1}/{len(face_result.faces)}: {image_task.phash[:8]}...")
                # Create task for concurrent execution
                task = self._save_single_thumbnail(image_task, face_detection, raw_key, i)
                thumbnail_tasks.append(task)
            
            # Execute all thumbnail saves concurrently
            thumbnail_results = await asyncio.gather(*thumbnail_tasks, return_exceptions=True)
            
            # Collect successful saves
            thumbnail_keys = []
            crop_paths = []
            for i, result in enumerate(thumbnail_results):
                if isinstance(result, Exception):
                    logger.warning(f"[STORAGE] Failed to save thumbnail {i}: {result}")
                elif result:
                    thumb_key, _ = result
                    if thumb_key:
                        thumbnail_keys.append(thumb_key)
                        crop_paths.append(f"temp_crop_{i}.jpg")
                        save_counts['saved_thumbs'] += 1
                        logger.debug(f"[STORAGE] Thumbnail saved: {thumb_key}")
            
            face_result.thumbnail_keys = thumbnail_keys
            face_result.crop_paths = crop_paths
            
            # Save metadata (async)
            logger.debug(f"[STORAGE] Saving metadata: {image_task.phash[:8]}...")
            await self.save_metadata_async(image_task, face_result, raw_key, thumbnail_keys)
            
            logger.info(f"[STORAGE] Save complete: raw={raw_key}, thumbs={len(thumbnail_keys)}")
            return face_result, save_counts
            
        except Exception as e:
            logger.error(f"Failed to save face result: {e}")
            return face_result, save_counts

    async def _save_single_thumbnail(self, image_task: ImageTask, face_detection: FaceDetection, 
                                    raw_key: str, face_index: int) -> Optional[Tuple[str, str]]:
        """Save a single thumbnail (helper for concurrent execution)."""
        try:
            logger.info(f"[THUMB DEBUG] Starting thumbnail {face_index}: task={image_task.phash[:8]}, temp_path={image_task.temp_path}")
            
            # Crop face (runs PIL in thread pool)
            face_crop_data = await self.crop_face_from_image_async(image_task.temp_path, face_detection)
            if not face_crop_data:
                logger.warning(f"[STORAGE] Failed to crop face {face_index} from {image_task.phash[:8]}, temp_path={image_task.temp_path}")
                return None
            
            logger.info(f"[THUMB DEBUG] Crop succeeded, saving to MinIO: {len(face_crop_data)} bytes")
            
            # Save thumbnail (async)
            thumb_key, thumb_url = await self.save_face_thumbnail_async(
                face_crop_data, face_detection, raw_key, face_index
            )
            if not thumb_key:
                logger.warning(f"[STORAGE] Failed to save thumbnail {face_index} to MinIO")
                return None
            
            logger.info(f"[THUMB DEBUG] Thumbnail saved: {thumb_key}")
            return thumb_key, thumb_url
        except Exception as e:
            logger.error(f"[STORAGE] Error saving thumbnail {face_index}: {e}", exc_info=True)
            return None
    
    def skip_save(self, image_task: ImageTask, face_result: FaceResult, skip_reason: str) -> FaceResult:
        """Create FaceResult without saving to storage."""
        logger.info(f"[STORAGE] Skipping save: {skip_reason}")
        face_result.saved_to_raw = False
        face_result.saved_to_thumbs = False
        face_result.skip_reason = skip_reason
        return face_result
    
    def cleanup_temp_file(self, temp_path: str) -> bool:
        """Clean up temporary file."""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup temp file {temp_path}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform storage health check."""
        try:
            client = self._get_client()
            
            # Test bucket access
            if self.config.s3_endpoint:
                # MinIO
                buckets = [bucket.name for bucket in client.list_buckets()]
            else:
                # AWS S3
                response = client.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            required_buckets = [self.config.s3_bucket_raw, self.config.s3_bucket_thumbs]
            missing_buckets = [bucket for bucket in required_buckets if bucket not in buckets]
            
            return {
                'status': 'healthy' if not missing_buckets else 'unhealthy',
                'buckets': buckets,
                'required_buckets': required_buckets,
                'missing_buckets': missing_buckets,
                'using_minio': bool(self.config.s3_endpoint),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }




def get_storage_manager() -> StorageManager:
    """Get singleton storage manager instance."""
    global _storage_manager_instance
    if _storage_manager_instance is None:
        _storage_manager_instance = StorageManager()
    return _storage_manager_instance


def close_storage_manager():
    """Close singleton storage manager."""
    global _storage_manager_instance
    if _storage_manager_instance:
        _storage_manager_instance = None
