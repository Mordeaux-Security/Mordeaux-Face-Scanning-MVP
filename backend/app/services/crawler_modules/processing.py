"""
Image processing module for the crawler service.

This module contains face detection, image filtering, and processing logic
including face similarity checking and thumbnail generation.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from PIL import Image
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

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing service with face detection and filtering capabilities."""
    
    def __init__(self, min_face_quality: float = 0.5, require_face: bool = True, 
                 crop_faces: bool = True, face_margin: float = 0.2):
        """
        Initialize the image processor.
        
        Args:
            min_face_quality: Minimum detection score for face quality
            require_face: Whether to require at least one face
            crop_faces: Whether to crop and save only face regions
            face_margin: Margin around face as fraction of face size
        """
        self.min_face_quality = min_face_quality
        self.require_face = require_face
        self.crop_faces = crop_faces
        self.face_margin = face_margin
        self.logger = logging.getLogger(__name__)
    
    async def process_single_image(self, image_info: ImageInfo, image_bytes: bytes, 
                                 index: int, total: int) -> Tuple[Optional[str], Optional[str], bool, List[str]]:
        """
        Process a single image with face detection and filtering.
        
        Args:
            image_info: Image information object
            image_bytes: Raw image bytes
            index: Current image index
            total: Total number of images
            
        Returns:
            Tuple of (raw_key, thumb_key, was_cached, errors)
        """
        try:
            self.logger.info(f"Processing image {index}/{total}: {self._truncate_log_string(image_info.url)}")
            
            # Check image dimensions - skip if any dimension is below 100px
            if not self._check_image_dimensions(image_bytes, min_dimension=100):
                self.logger.info(f"Image dimensions too small, skipping: {self._truncate_log_string(image_info.url)}")
                return None, None, False, ["Image dimensions below minimum threshold"]
            
            # Use async face detection with thread pool
            faces = []
            thumbnail_bytes = None
            if self.require_face or self.crop_faces:
                t_detect_start = datetime.utcnow()
                faces, thumbnail_bytes = await self._async_face_detection(image_bytes, image_info)
                t_detect_ms = (datetime.utcnow() - t_detect_start).total_seconds() * 1000.0
                self.logger.debug(f"Face detection pipeline completed in {t_detect_ms:.1f} ms for {self._truncate_log_string(image_info.url)}")
                
                if self.require_face and not faces:
                    self.logger.info(f"No faces detected for {self._truncate_log_string(image_info.url)}, skipping.")
                    return None, None, False, []
            
            # Process the image based on face detection results
            return await self._process_image_with_faces(image_info, image_bytes, faces, thumbnail_bytes)
            
        except Exception as e:
            self.logger.error(f"Error processing image {self._truncate_log_string(image_info.url)}: {e}")
            return None, None, False, [str(e)]
    
    async def _async_face_detection(self, image_bytes: bytes, image_info: ImageInfo) -> Tuple[List, Optional[bytes]]:
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
                    # Don't create thumbnail for images without faces
                    thumbnail_bytes = None
                
                return faces, thumbnail_bytes
                
            except Exception as e:
                self.logger.error(f"Error in face detection thread: {e}")
                return [], None
        
        # Run face detection in thread pool
        loop = asyncio.get_event_loop()
        faces, thumbnail_bytes = await loop.run_in_executor(None, _run_face_detection)
        
        return faces, thumbnail_bytes
    
    async def _process_image_with_faces(self, image_info: ImageInfo, image_bytes: bytes, 
                                      faces: List, thumbnail_bytes: Optional[bytes]) -> Tuple[Optional[str], Optional[str], bool, List[str]]:
        """
        Process image with detected faces and save to storage.
        
        Args:
            image_info: Image information object
            image_bytes: Raw image bytes
            faces: List of detected faces
            thumbnail_bytes: Thumbnail bytes if available
            
        Returns:
            Tuple of (raw_key, thumb_key, was_cached, errors)
        """
        try:
            # This method would typically integrate with storage service
            # For now, return placeholder values
            self.logger.info(f"Processing image with video URL: {image_info.video_url}")
            
            # In a real implementation, this would:
            # 1. Save the raw image to storage
            # 2. Save the thumbnail if available
            # 3. Return the storage keys
            
            return "raw_key_placeholder", "thumb_key_placeholder", False, []
            
        except Exception as e:
            self.logger.error(f"Error processing image with faces: {e}")
            return None, None, False, [str(e)]
    
    def _check_image_dimensions(self, image_bytes: bytes, min_dimension: int = 100) -> bool:
        """
        Check if image meets minimum dimension requirements.
        
        Args:
            image_bytes: Raw image bytes
            min_dimension: Minimum dimension in pixels
            
        Returns:
            True if image meets requirements, False otherwise
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                self.logger.info(f"Image dimensions {width}x{height} below minimum {min_dimension}px, skipping")
                return width >= min_dimension and height >= min_dimension
        except Exception as e:
            self.logger.debug(f"Error checking image dimensions: {e}")
            return False
    
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


class FaceSimilarityChecker:
    """Face similarity checking service for deduplication."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize face similarity checker.
        
        Args:
            similarity_threshold: Cosine similarity threshold for considering faces similar
        """
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def check_face_similarity(self, embedding: List[float], face_data: Dict) -> Tuple[bool, float, Optional[str]]:
        """
        Check if a face is similar to previously seen faces.
        
        Args:
            embedding: Face embedding vector
            face_data: Face data dictionary
            
        Returns:
            Tuple of (is_similar, similarity_score, similar_face_id)
        """
        try:
            # This would integrate with the face similarity service
            # For now, return placeholder values
            return False, 0.0, None
        except Exception as e:
            self.logger.error(f"Error checking face similarity: {e}")
            return False, 0.0, None


class ImageFilter:
    """Image filtering service for quality and content filtering."""
    
    def __init__(self, min_face_quality: float = 0.5, require_face: bool = True):
        """
        Initialize image filter.
        
        Args:
            min_face_quality: Minimum face detection quality score
            require_face: Whether to require at least one face
        """
        self.min_face_quality = min_face_quality
        self.require_face = require_face
        self.logger = logging.getLogger(__name__)
    
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
