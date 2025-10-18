"""
Album/Gallery Processing Module

This module provides functionality for processing image galleries and albums with:
- Album detection and metadata tracking
- Face deduplication within albums
- Enhanced thumbnail processing for all quality faces
- Video thumbnail face extraction
"""

import asyncio
import logging
import hashlib
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    from .crawler_settings import *
    from .face import get_face_service
except ImportError:
    # Fallback for direct imports
    from crawler_settings import *
    from face import get_face_service

logger = logging.getLogger(__name__)

@dataclass
class AlbumMetadata:
    """Metadata for an album/gallery."""
    album_id: str
    url: str
    title: Optional[str] = None
    image_count: int = 0
    processed_faces: int = 0
    unique_faces: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    face_embeddings: List[np.ndarray] = field(default_factory=list)
    face_metadata: List[Dict] = field(default_factory=list)
    # Enhanced metadata tracking
    origin_url: str = ""
    method_used: str = ""
    javascript_rendered: bool = False
    image_urls: List[str] = field(default_factory=list)
    processed_image_urls: List[str] = field(default_factory=list)
    unique_face_urls: List[str] = field(default_factory=list)

@dataclass
class FaceInfo:
    """Information about a detected face."""
    face_id: str
    embedding: np.ndarray
    bbox: List[float]
    det_score: float
    image_url: str
    image_index: int
    face_index: int
    thumbnail_bytes: Optional[bytes] = None
    # Enhanced metadata
    album_id: str = ""
    origin_url: str = ""
    method_used: str = ""
    javascript_rendered: bool = False
    is_unique: bool = True
    similarity_scores: List[float] = field(default_factory=list)

class AlbumProcessor:
    """
    Processes image albums/galleries with face deduplication and enhanced thumbnail logic.
    """
    
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.face_service = get_face_service()
        self._active_albums: Dict[str, AlbumMetadata] = {}
        self._face_cache: Dict[str, List[FaceInfo]] = {}  # album_id -> faces
        
    def generate_album_id(self, url: str) -> str:
        """Generate a unique album ID from URL."""
        return hashlib.sha256(f"{url}:{self.tenant_id}".encode()).hexdigest()[:16]
    
    def is_album_url(self, url: str, image_count: int) -> bool:
        """
        Determine if a URL represents an album/gallery based on image count and patterns.
        """
        if not ALBUM_DETECTION_ENABLED:
            return False
            
        # Check minimum image count
        if image_count < ALBUM_MIN_IMAGES:
            return False
            
        # Check for album/gallery patterns in URL
        album_patterns = [
            '/album/', '/gallery/', '/photos/', '/images/',
            '/collection/', '/portfolio/', '/showcase/',
            'album=', 'gallery=', 'photos=', 'images='
        ]
        
        url_lower = url.lower()
        has_album_pattern = any(pattern in url_lower for pattern in album_patterns)
        
        # Consider it an album if it has enough images or album patterns
        return has_album_pattern or image_count >= ALBUM_MIN_IMAGES * 2
    
    async def process_album_images(
        self, 
        images: List[Any], 
        url: str, 
        method_used: str,
        javascript_rendered: bool = False
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process images as an album with face deduplication and enhanced thumbnail logic.
        
        Args:
            images: List of ImageInfo objects
            url: Source URL
            method_used: Extraction method used
            javascript_rendered: Whether JavaScript rendering was used
            
        Returns:
            Tuple of (processed_images, album_metadata)
        """
        if not self.is_album_url(url, len(images)):
            logger.debug(f"URL {url} not detected as album (images: {len(images)})")
            return images, {}
            
        album_id = self.generate_album_id(url)
        logger.info(f"Processing album {album_id} with {len(images)} images from {url}")
        
        # Initialize album metadata with enhanced tracking
        album_metadata = AlbumMetadata(
            album_id=album_id,
            url=url,
            origin_url=url,
            method_used=method_used,
            javascript_rendered=javascript_rendered,
            image_count=len(images),
            image_urls=[getattr(img, 'url', str(img)) for img in images]
        )
        self._active_albums[album_id] = album_metadata
        
        # Process images with album-specific logic
        processed_images = []
        all_faces = []
        
        for i, image_info in enumerate(images):
            if i >= ALBUM_MAX_IMAGES:
                logger.warning(f"Album {album_id} truncated at {ALBUM_MAX_IMAGES} images")
                break
                
            # Add album metadata to image info
            if hasattr(image_info, 'album_metadata'):
                image_info.album_metadata = {
                    'album_id': album_id,
                    'origin_url': url,
                    'method_used': method_used,
                    'javascript_rendered': javascript_rendered,
                    'image_index': i
                }
            
            processed_images.append(image_info)
            album_metadata.processed_image_urls.append(getattr(image_info, 'url', str(image_info)))
        
        # Perform face deduplication within album
        if ALBUM_FACE_DEDUPLICATION and all_faces:
            unique_faces = await self._deduplicate_faces_in_album(all_faces, album_id)
            album_metadata.unique_faces = len(unique_faces)
            album_metadata.processed_faces = len(all_faces)
            
            # Update face metadata with enhanced tracking
            album_metadata.face_embeddings = [face.embedding for face in unique_faces]
            album_metadata.face_metadata = [
                {
                    'face_id': face.face_id,
                    'det_score': face.det_score,
                    'image_url': face.image_url,
                    'image_index': face.image_index,
                    'face_index': face.face_index,
                    'album_id': face.album_id,
                    'origin_url': face.origin_url,
                    'method_used': face.method_used,
                    'javascript_rendered': face.javascript_rendered,
                    'is_unique': face.is_unique,
                    'similarity_scores': face.similarity_scores
                }
                for face in unique_faces
            ]
            
            # Track unique face URLs
            album_metadata.unique_face_urls = [face.image_url for face in unique_faces]
            
            logger.info(f"Album {album_id}: {len(all_faces)} faces -> {len(unique_faces)} unique faces")
        
        logger.info(f"Album {album_id} processed {len(processed_images)} images from {len(images)} found")
        
        return processed_images, {
            'album_id': album_id,
            'is_album': True,
            'total_images': len(images),
            'processed_images': len(processed_images),
            'total_faces': len(all_faces),
            'unique_faces': album_metadata.unique_faces,
            'method_used': method_used,
            'javascript_rendered': javascript_rendered,
            'origin_url': url,
            'image_urls': album_metadata.image_urls,
            'processed_image_urls': album_metadata.processed_image_urls,
            'unique_face_urls': album_metadata.unique_face_urls
        }
    
    async def _extract_faces_from_image(self, image_info: Any, image_index: int) -> List[FaceInfo]:
        """
        Extract all quality faces from an image for album processing.
        """
        try:
            # This would need to be integrated with the existing image processing pipeline
            # For now, return empty list as placeholder
            # TODO: Integrate with existing face detection pipeline
            return []
        except Exception as e:
            logger.error(f"Error extracting faces from image {image_info.url}: {e}")
            return []
    
    async def _deduplicate_faces_in_album(self, faces: List[FaceInfo], album_id: str) -> List[FaceInfo]:
        """
        Deduplicate faces within an album using embedding similarity.
        Ensures only unique faces are saved while processing the whole album.
        """
        if not faces:
            return []
            
        unique_faces = []
        seen_embeddings = []
        
        # Sort faces by detection score (highest first) to prioritize better quality faces
        sorted_faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        
        for face in sorted_faces:
            is_duplicate = False
            similarity_scores = []
            
            # Check against already seen faces
            for i, seen_embedding in enumerate(seen_embeddings):
                # Calculate cosine similarity
                similarity = np.dot(face.embedding, seen_embedding) / (
                    np.linalg.norm(face.embedding) * np.linalg.norm(seen_embedding)
                )
                similarity_scores.append(similarity)
                
                # If similarity is above threshold, consider it a duplicate
                if similarity > (1.0 - FACE_DUP_DIST_ALBUM):
                    is_duplicate = True
                    logger.debug(f"Face {face.face_id} marked as duplicate (similarity: {similarity:.3f} > threshold: {1.0 - FACE_DUP_DIST_ALBUM:.3f})")
                    break
            
            # Update face metadata
            face.similarity_scores = similarity_scores
            face.is_unique = not is_duplicate
            face.album_id = album_id
            
            if not is_duplicate:
                unique_faces.append(face)
                seen_embeddings.append(face.embedding)
                logger.debug(f"Face {face.face_id} added as unique (max similarity: {max(similarity_scores) if similarity_scores else 0:.3f})")
            else:
                logger.debug(f"Face {face.face_id} rejected as duplicate")
        
        logger.info(f"Album {album_id} deduplication: {len(faces)} faces -> {len(unique_faces)} unique faces")
        return unique_faces
    
    async def process_video_thumbnails(self, video_info: Any) -> List[FaceInfo]:
        """
        Extract faces from video thumbnails.
        """
        if not VIDEO_THUMBNAIL_FACE_EXTRACTION:
            return []
            
        try:
            # TODO: Implement video thumbnail face extraction
            # This would involve:
            # 1. Extracting thumbnail from video
            # 2. Running face detection on thumbnail
            # 3. Filtering by quality threshold
            # 4. Creating face thumbnails
            return []
        except Exception as e:
            logger.error(f"Error processing video thumbnail: {e}")
            return []
    
    def get_album_metadata(self, album_id: str) -> Optional[AlbumMetadata]:
        """Get metadata for a specific album."""
        return self._active_albums.get(album_id)
    
    def cleanup_album(self, album_id: str):
        """Clean up album data from memory."""
        if album_id in self._active_albums:
            del self._active_albums[album_id]
        if album_id in self._face_cache:
            del self._face_cache[album_id]
        logger.debug(f"Cleaned up album {album_id}")

# Global album processor instance
_album_processor = None

def get_album_processor(tenant_id: str = "default") -> AlbumProcessor:
    """Get the global album processor instance."""
    global _album_processor
    if _album_processor is None:
        _album_processor = AlbumProcessor(tenant_id)
    return _album_processor

def close_album_processor():
    """Close the album processor and clean up resources."""
    global _album_processor
    if _album_processor:
        _album_processor._active_albums.clear()
        _album_processor._face_cache.clear()
        _album_processor = None
