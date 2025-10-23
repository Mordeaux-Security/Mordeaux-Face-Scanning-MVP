"""
Storage facade module for the crawler service.

This module provides a clean interface to the storage service, abstracting away
the complexity of MinIO, Redis, and Postgres storage operations.
"""

import logging
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse

from .types import ImageInfo

logger = logging.getLogger(__name__)


class StorageFacade:
    """
    Storage facade providing a clean interface to storage operations.
    
    Abstracts away the complexity of MinIO, Redis, and Postgres storage,
    providing simple methods for saving images and thumbnails.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def save_image_with_metadata(
        self,
        image_bytes: bytes,
        image_info: ImageInfo,
        page_url: str,
        mime: str = "image/jpeg",
        filename: str = "image.jpg",
        bucket: str = "raw-images"
    ) -> Optional[Dict[str, Any]]:
        """
        Save an image with metadata to storage.
        
        Args:
            image_bytes: The image data to save
            image_info: Image information object
            page_url: URL of the page where image was found
            mime: MIME type of the image
            filename: Filename for the image
            bucket: Storage bucket name
            
        Returns:
            Dictionary with storage result or None if failed
        """
        try:
            # Import storage functions here to avoid circular imports
            from ..storage import save_image, _minio
            
            # Extract site for storage organization
            parsed_url = urlparse(page_url)
            site = parsed_url.netloc.replace('www.', '')  # Remove www prefix
            
            # Get MinIO client
            minio_client = _minio()
            
            # Save image with sidecar metadata
            result = await save_image(
                image_bytes=image_bytes,
                mime=mime,
                filename=filename,
                bucket=bucket,
                client=minio_client,
                site=site,
                page_url=page_url,
                source_video_url=image_info.video_url,
                source_image_url=image_info.url,
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving image {self._truncate_log_string(image_info.url)}: {e}")
            return None
    
    async def save_thumbnail(
        self,
        thumbnail_bytes: bytes,
        image_info: ImageInfo,
        page_url: str,
        mime: str = "image/jpeg",
        filename: str = "thumbnail.jpg",
        bucket: str = "thumbnails"
    ) -> Optional[Dict[str, Any]]:
        """
        Save a thumbnail with metadata to storage.
        
        Args:
            thumbnail_bytes: The thumbnail data to save
            image_info: Image information object
            page_url: URL of the page where image was found
            mime: MIME type of the thumbnail
            filename: Filename for the thumbnail
            bucket: Storage bucket name
            
        Returns:
            Dictionary with storage result or None if failed
        """
        return await self.save_image_with_metadata(
            image_bytes=thumbnail_bytes,
            image_info=image_info,
            page_url=page_url,
            mime=mime,
            filename=filename,
            bucket=bucket
        )
    
    async def save_raw_and_thumbnail(
        self,
        image_bytes: bytes,
        thumbnail_bytes: Optional[bytes],
        image_info: ImageInfo,
        page_url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Save both raw image and thumbnail to storage.
        
        Args:
            image_bytes: The raw image data
            thumbnail_bytes: The thumbnail data (optional)
            image_info: Image information object
            page_url: URL of the page where image was found
            
        Returns:
            Tuple of (raw_key, thumb_key) or (None, None) if failed
        """
        try:
            # Save raw image
            raw_result = await self.save_image_with_metadata(
                image_bytes=image_bytes,
                image_info=image_info,
                page_url=page_url,
                bucket="raw-images"
            )
            
            raw_key = None
            if raw_result and 'image_key' in raw_result:
                raw_key = raw_result['image_key']
            
            # Save thumbnail if available
            thumb_key = None
            if thumbnail_bytes is not None:
                thumb_result = await self.save_thumbnail(
                    thumbnail_bytes=thumbnail_bytes,
                    image_info=image_info,
                    page_url=page_url,
                    bucket="thumbnails"
                )
                
                if thumb_result and 'image_key' in thumb_result:
                    thumb_key = thumb_result['image_key']
            
            return raw_key, thumb_key
            
        except Exception as e:
            self.logger.error(f"Error saving raw and thumbnail for {self._truncate_log_string(image_info.url)}: {e}")
            return None, None
    
    def _truncate_log_string(self, text: str, max_length: int = 120) -> str:
        """
        Truncate long strings for logging with a hash suffix for identification.
        
        Args:
            text: String to truncate
            max_length: Maximum length before truncation
            
        Returns:
            Truncated string with hash suffix if truncated
        """
        if len(text) <= max_length:
            return text
        
        # Create a short hash of the original string for identification
        import hashlib
        hash_suffix = hashlib.md5(text.encode()).hexdigest()[:8]
        truncated = text[:max_length - len(hash_suffix) - 3]  # Reserve space for "..." and hash
        return f"{truncated}...{hash_suffix}"
