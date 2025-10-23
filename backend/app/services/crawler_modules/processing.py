"""
Image processing module for the crawler service.

This module contains face detection, image filtering, and processing logic
including face similarity checking and thumbnail generation.
"""

import asyncio
import logging
import concurrent.futures
import psutil
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from PIL import Image, ImageFile
import io

from .types import ImageInfo

# Import face service with error handling
try:
    from ..face import get_face_service
    from .. import face as face_module
    FACE_SERVICE_AVAILABLE = True
except ImportError:
    FACE_SERVICE_AVAILABLE = False
    get_face_service = None
    face_module = None

# Configure PIL for better performance
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """
    Comprehensive image processing service with face detection, filtering, and thumbnail generation.
    
    Consolidates all image processing logic from the crawler including:
    - Face detection with thread pool execution
    - Image enhancement for better face detection
    - Face quality filtering
    - Thumbnail creation with face cropping
    - Dimension validation
    - EXIF data stripping
    """
    
    def __init__(self, min_face_quality: float = 0.5, require_face: bool = True, 
                 crop_faces: bool = True, face_margin: float = 0.2,
                 min_dimension: int = 100):
        """
        Initialize the image processing service.
        
        Args:
            min_face_quality: Minimum detection score for face quality
            require_face: Whether to require at least one face
            crop_faces: Whether to crop and save only face regions
            face_margin: Margin around face as fraction of face size
            min_dimension: Minimum image dimension in pixels
        """
        self.min_face_quality = min_face_quality
        self.require_face = require_face
        self.crop_faces = crop_faces
        self.face_margin = face_margin
        self.min_dimension = min_dimension
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for CPU-intensive face detection
        self._face_detection_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(psutil.cpu_count() or 4, 8),  # Limit to CPU cores but cap at 8
            thread_name_prefix="face_detection"
        )
    
    async def process_single_image(self, image_info: ImageInfo, image_bytes: bytes, 
                                 index: int, total: int) -> Tuple[List, Optional[bytes], List[str]]:
        """
        Process a single image with face detection and filtering.
        
        Args:
            image_info: Image information object
            image_bytes: Raw image bytes
            index: Current image index
            total: Total number of images
            
        Returns:
            Tuple of (faces, thumbnail_bytes, errors)
        """
        try:
            self.logger.info(f"Processing image {index}/{total}: {self._truncate_log_string(image_info.url)}")
            
            # Check image dimensions - skip if any dimension is below minimum
            if not self.check_image_dimensions(image_bytes):
                self.logger.info(f"Image dimensions too small, skipping: {self._truncate_log_string(image_info.url)}")
                return [], None, ["Image dimensions below minimum threshold"]
            
            # Strip EXIF data for privacy
            image_bytes = self.strip_exif_data(image_bytes)
            
            # Use async face detection with thread pool
            faces = []
            thumbnail_bytes = None
            if self.require_face or self.crop_faces:
                t_detect_start = datetime.utcnow()
                faces, thumbnail_bytes = await self.async_face_detection(image_bytes, image_info)
                t_detect_ms = (datetime.utcnow() - t_detect_start).total_seconds() * 1000.0
                self.logger.debug(f"Face detection pipeline completed in {t_detect_ms:.1f} ms for {self._truncate_log_string(image_info.url)}")
                
                if self.require_face and not faces:
                    self.logger.info(f"No faces detected for {self._truncate_log_string(image_info.url)}, skipping.")
                    return [], None, []
            
            return faces, thumbnail_bytes, []
            
        except Exception as e:
            self.logger.error(f"Error processing image {self._truncate_log_string(image_info.url)}: {e}")
            return [], None, [str(e)]
    
    async def async_face_detection(self, image_bytes: bytes, image_info: ImageInfo) -> Tuple[List, Optional[bytes]]:
        """
        Run face detection in thread pool to avoid blocking async loop.
        
        Args:
            image_bytes: Raw image bytes
            image_info: Image information object
            
        Returns:
            Tuple of (faces_list, thumbnail_bytes)
        """
        def _run_face_detection():
            """Synchronous face detection function to run in thread pool."""
            try:
                if not FACE_SERVICE_AVAILABLE:
                    self.logger.warning("Face service not available, skipping face detection")
                    return [], None
                
                face_service = get_face_service()
                
                # Enhance image for better face detection
                enhanced_bytes, enhancement_scale = face_service.enhance_image_for_face_detection(image_bytes)
                faces = face_service.detect_and_embed(enhanced_bytes, enhancement_scale, min_size=0)
                
                # Face deduplication temporarily disabled for performance
                # TODO: Implement efficient face deduplication (e.g., using approximate nearest neighbors)
                # if faces and face_module:
                #     # Check each detected face for similarity
                #     unique_faces = []
                #     for face in faces:
                #         # Check if this face is similar to any previously seen faces
                #         is_similar, similarity_score, similar_face_id = face_module.check_face_similarity(
                #             face['embedding'], face
                #         )
                #         
                #         if is_similar:
                #             self.logger.info(f"Face similarity detected: {similarity_score:.3f} (similar to {similar_face_id})")
                #             # Skip this face as it's too similar to a previously seen face
                #             continue
                #         else:
                #             # This is a unique face, add it to the list
                #             unique_faces.append(face)
                #     
                #     faces = unique_faces
                
                # Create thumbnail if faces were detected and cropping is enabled
                thumbnail_bytes = None
                if self.crop_faces and faces:
                    # Use the first face for thumbnail creation
                    thumbnail_bytes = face_service.crop_face_and_create_thumbnail(
                        image_bytes, faces[0], self.face_margin
                    )
                elif self.crop_faces and not faces:
                    # Don't create thumbnail for images without faces when CROP_FACES=true
                    # Only crop faces that are actually detected
                    thumbnail_bytes = None
                # If CROP_FACES=false, no thumbnail is created regardless of face detection
                
                return faces, thumbnail_bytes
                
            except Exception as e:
                self.logger.error(f"Error in face detection thread: {e}")
                return [], None
        
        # Run face detection in thread pool
        loop = asyncio.get_event_loop()
        faces, thumbnail_bytes = await loop.run_in_executor(
            self._face_detection_thread_pool, _run_face_detection
        )
        
        return faces, thumbnail_bytes
    
    def should_process_image(self, faces: List[Dict]) -> bool:
        """
        Determine if an image should be processed based on face detection results.
        
        Args:
            faces: List of detected faces
            
        Returns:
            True if image should be processed, False otherwise
        """
        if self.require_face and not faces:
            return False
        
        if faces:
            # Check if any face meets quality requirements
            for face in faces:
                if face.get('det_score', 0) >= self.min_face_quality:
                    return True
            return False
        
        return True
    
    def filter_faces_by_quality(self, faces: List[Dict]) -> List[Dict]:
        """
        Filter faces by quality score.
        
        Args:
            faces: List of detected faces
            
        Returns:
            Filtered list of faces meeting quality requirements
        """
        return [face for face in faces if face.get('det_score', 0) >= self.min_face_quality]
    
    def cleanup(self):
        """Clean up thread pool resources."""
        try:
            if hasattr(self, '_face_detection_thread_pool'):
                self.logger.info("Shutting down image processing thread pool...")
                self._face_detection_thread_pool.shutdown(wait=True)
                self.logger.info("Image processing thread pool shutdown complete")
        except Exception as e:
            self.logger.warning(f"Error cleaning up image processing resources: {e}")
    
    def check_image_dimensions(self, image_bytes: bytes) -> bool:
        """
        Check if image meets minimum dimension requirements.
        
        Args:
            image_bytes: Image data
            
        Returns:
            True if image meets dimension requirements, False otherwise
        """
        try:
            # Open image to get dimensions
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            
            # Check if either dimension is below minimum
            if width < self.min_dimension or height < self.min_dimension:
                self.logger.info(f"Image dimensions {width}x{height} below minimum {self.min_dimension}px, skipping")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Error checking image dimensions: {e}")
            return False
    
    def strip_exif_data(self, image_bytes: bytes) -> bytes:
        """
        Strip EXIF data from image for privacy.
        
        Args:
            image_bytes: Original image bytes
            
        Returns:
            Image bytes with EXIF data removed
        """
        try:
            # Open image and remove EXIF
            image = Image.open(io.BytesIO(image_bytes))
            
            # Create new image without EXIF
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB for JPEG compatibility
                image = image.convert('RGB')
            
            # Save to bytes without EXIF
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95, optimize=True)
            return output.getvalue()
            
        except Exception as e:
            self.logger.warning(f"Failed to strip EXIF data: {e}. Using original image.")
            return image_bytes
    
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


# Legacy aliases for backward compatibility
ImageProcessor = ImageProcessingService
