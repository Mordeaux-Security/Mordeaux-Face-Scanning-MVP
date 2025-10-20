"""
Image Processing Pipeline

Handles image processing operations including face detection, enhancement,
filtering, and thumbnail generation. Supports both CPU and future GPU processing.
"""

import asyncio
import io
import logging
import tempfile
import time
import threading
import os
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageFile
import imagehash
import httpx

from .config import CrawlerConfig
from .memory import MemoryManager

# Image safety configuration
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# Global singleton for face detection model
_FACE_MODEL = None
_FACE_MODEL_LOCK = None


def get_face_model():
    """Get singleton face detection model instance.
    
    Returns:
        InsightFace FaceAnalysis instance
    """
    global _FACE_MODEL, _FACE_MODEL_LOCK
    
    if _FACE_MODEL_LOCK is None:
        import threading
        _FACE_MODEL_LOCK = threading.Lock()
    
    if _FACE_MODEL is None:
        with _FACE_MODEL_LOCK:
            if _FACE_MODEL is None:
                try:
                    from insightface.app import FaceAnalysis
                    logger.info("Loading face detection model (singleton)...")
                    _FACE_MODEL = FaceAnalysis(
                        name='buffalo_l',
                        providers=['CPUExecutionProvider']  # Future: GPU support
                    )
                    _FACE_MODEL.prepare(ctx_id=0, det_size=(640, 640))
                    logger.info("Face detection model loaded successfully (singleton)")
                except Exception as e:
                    logger.error(f"Failed to load face detection model: {e}")
                    raise
    
    return _FACE_MODEL


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Calculate cosine distance between two face embeddings.
    
    Args:
        a: First face embedding vector
        b: Second face embedding vector
        
    Returns:
        Cosine distance (1 - cosine_similarity), where 0 = identical, 1 = orthogonal
    """
    try:
        if not a or not b or len(a) != len(b):
            return 1.0  # Maximum distance for invalid embeddings
        
        # Calculate dot product
        dot = sum(x * y for x, y in zip(a, b))
        
        # Calculate magnitudes
        na = (sum(x * x for x in a)) ** 0.5
        nb = (sum(x * x for x in b)) ** 0.5
        
        # Avoid division by zero
        if na == 0 or nb == 0:
            return 1.0
        
        # Calculate cosine similarity and convert to distance
        cosine_sim = dot / (na * nb)
        return 1.0 - cosine_sim
        
    except Exception as e:
        logger.error(f"Error calculating cosine distance: {e}")
        return 1.0


@dataclass
class ProcessedImage:
    """Represents a processed image with metadata."""
    original_url: str
    image_data: bytes
    thumbnail_data: List[bytes]  # List of thumbnails (one per face)
    faces: List[Dict[str, Any]]
    perceptual_hash: str
    dimensions: Tuple[int, int]
    file_size: int
    processing_time: float
    enhancement_applied: bool = False


@dataclass
class ProcessingResult:
    """Result of image processing operation."""
    processed_images: List[ProcessedImage]
    total_processed: int
    faces_detected: int
    processing_time: float
    success: bool
    error: Optional[str] = None


class FaceDetector:
    """Face detection and processing using InsightFace singleton model."""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self._initialized = False
    
    def _get_model(self):
        """Get the singleton face detection model."""
        return get_face_model()
    
    
    def detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect faces in an image with proper scaling and upscaling fallback."""
        if not self._initialized:
            self._initialized = True  # Mark as initialized (model is loaded on first use)
        
        face_app = self._get_model()
        if face_app is None:
            return []
        
        # Try detection on original image first
        faces = self._detect_faces_with_scaling(image)
        
        # If no faces found and image is small, try upscaling
        if not faces and (image.width < 800 or image.height < 800):
            logger.info(f"No faces found in small image ({image.width}x{image.height}), trying upscaling")
            upscaled_image = self._upscale_image(image)
            if upscaled_image:
                # Detect faces on upscaled image
                upscaled_faces = self._detect_faces_with_scaling(upscaled_image, is_upscaled=True)
                
                # Scale bbox coordinates back to original image dimensions
                faces = self._scale_bboxes_to_original(upscaled_faces, image, upscaled_image)
        
        return faces
    
    def detect_and_embed_path(self, tmp_path: str) -> List[Dict[str, Any]]:
        """Detect faces from a temporary file path (for thread pool usage)."""
        try:
            # Load image from file path
            image = Image.open(tmp_path)
            
            # Apply EXIF orientation correction
            image = ImageOps.exif_transpose(image)
            
            # Convert to RGB if necessary
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            
            # Check minimum image size
            min_width, min_height = self.config.min_image_size
            if image.width < min_width or image.height < min_height:
                logger.debug(f"Image too small: {image.width}x{image.height} < {min_width}x{min_height}")
                return []
            
            # Detect faces
            return self.detect_faces(image)
            
        except Exception as e:
            logger.error(f"Error detecting faces from path {tmp_path}: {e}")
            return []
    
    def _upscale_image(self, image: Image.Image, target_size: int = 1024) -> Optional[Image.Image]:
        """Upscale image for better face detection on low-resolution images."""
        try:
            # Calculate upscale factor to make the larger dimension target_size
            max_dim = max(image.width, image.height)
            if max_dim >= target_size:
                return None  # Already large enough
            
            scale_factor = target_size / max_dim
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            logger.debug(f"Upscaling image from {image.width}x{image.height} to {new_width}x{new_height} (scale: {scale_factor:.2f})")
            
            # Use high-quality resampling for upscaling
            upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return upscaled
            
        except Exception as e:
            logger.error(f"Error upscaling image: {e}")
            return None
    
    def _scale_bboxes_to_original(self, upscaled_faces: List[Dict[str, Any]], original_image: Image.Image, upscaled_image: Image.Image) -> List[Dict[str, Any]]:
        """Scale bbox coordinates from upscaled image back to original image dimensions."""
        try:
            if not upscaled_faces:
                return []
            
            # Calculate scale factors
            scale_x = original_image.width / upscaled_image.width
            scale_y = original_image.height / upscaled_image.height
            
            logger.debug(f"Scaling bboxes: upscaled=({upscaled_image.width}x{upscaled_image.height}) -> original=({original_image.width}x{original_image.height}), scale=({scale_x:.3f}x{scale_y:.3f})")
            
            scaled_faces = []
            for face in upscaled_faces:
                bbox = face['bbox']
                x, y, w, h = bbox
                
                # Scale coordinates back to original image
                x_scaled = x * scale_x
                y_scaled = y * scale_y
                w_scaled = w * scale_x
                h_scaled = h * scale_y
                
                # Create new face dict with scaled coordinates
                scaled_face = face.copy()
                scaled_face['bbox'] = [float(x_scaled), float(y_scaled), float(w_scaled), float(h_scaled)]
                scaled_face['face_area'] = w_scaled * h_scaled
                scaled_face['face_ratio'] = scaled_face['face_area'] / (original_image.width * original_image.height)
                scaled_face['upscaled'] = True  # Mark as from upscaled detection
                
                scaled_faces.append(scaled_face)
            
            logger.info(f"Scaled {len(scaled_faces)} faces from upscaled image back to original dimensions")
            return scaled_faces
            
        except Exception as e:
            logger.error(f"Error scaling bboxes to original: {e}")
            return []
    
    def _detect_faces_with_scaling(self, image: Image.Image, is_upscaled: bool = False) -> List[Dict[str, Any]]:
        """Detect faces - InsightFace returns coordinates relative to original image."""
        try:
            # Store original image dimensions
            orig_w, orig_h = image.width, image.height
            
            # Convert PIL to numpy array (InsightFace requires numpy arrays, not file paths)
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Get singleton model and detect faces
            face_app = self._get_model()
            faces = face_app.get(img_array)
            
            # Filter faces by quality and size
            filtered_faces = []
            for face in faces:
                if (face.det_score >= self.config.face_min_quality and
                    face.bbox[2] - face.bbox[0] >= self.config.face_min_size and
                    face.bbox[3] - face.bbox[1] >= self.config.face_min_size):
                    
                    # InsightFace already returns coordinates relative to original image
                    bbox_det = face.bbox.tolist()
                    if isinstance(bbox_det, list) and len(bbox_det) == 4:
                        x, y, w, h = bbox_det
                        
                        # Ensure all elements are numbers
                        bbox_original = [float(x), float(y), float(w), float(h)]
                    else:
                        logger.error(f"Unexpected bbox format from InsightFace: {bbox_det}")
                        continue
                    
                    # Calculate face area using original coordinates
                    face_area = w * h
                    image_area = orig_w * orig_h
                    face_ratio = face_area / image_area
                    
                    filtered_faces.append({
                        'bbox': bbox_original,
                        'det_score': float(face.det_score),
                        'embedding': face.embedding.tolist(),
                        'face_area': face_area,
                        'face_ratio': face_ratio,
                        'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                        'upscaled': is_upscaled
                    })
            
            if is_upscaled:
                logger.info(f"Detected {len(filtered_faces)} faces after upscaling")
            else:
                logger.debug(f"Detected {len(filtered_faces)} faces")
            
            # Apply deduplication if we have multiple faces
            if len(filtered_faces) > 1:
                filtered_faces = self._deduplicate_faces(filtered_faces)
            
            if is_upscaled:
                logger.info(f"Detected {len(filtered_faces)} faces after upscaling and deduplication")
            else:
                logger.debug(f"Detected {len(filtered_faces)} faces after deduplication")
            
            return filtered_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _deduplicate_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate faces using cosine distance on embeddings.
        
        Args:
            faces: List of detected faces with embeddings
            
        Returns:
            List of unique faces after deduplication
        """
        try:
            if len(faces) <= 1:
                return faces
            
            # Use image-level deduplication threshold
            threshold = self.config.face_dup_dist_image
            
            unique_faces = []
            for face in faces:
                is_duplicate = False
                embedding = face.get('embedding', [])
                
                if not embedding:
                    # Keep faces without embeddings (shouldn't happen with InsightFace)
                    unique_faces.append(face)
                    continue
                
                # Check against all previously accepted faces
                for unique_face in unique_faces:
                    unique_embedding = unique_face.get('embedding', [])
                    if not unique_embedding:
                        continue
                    
                    # Calculate cosine distance
                    distance = cosine_distance(embedding, unique_embedding)
                    
                    if distance < threshold:
                        # This face is too similar to an existing one
                        is_duplicate = True
                        logger.debug(f"Removing duplicate face: distance={distance:.3f} < threshold={threshold}")
                        break
                
                if not is_duplicate:
                    unique_faces.append(face)
            
            removed_count = len(faces) - len(unique_faces)
            if removed_count > 0:
                logger.info(f"Face deduplication: removed {removed_count} duplicates, kept {len(unique_faces)} unique faces")
            
            return unique_faces
            
        except Exception as e:
            logger.error(f"Error in face deduplication: {e}")
            return faces  # Return original faces if deduplication fails
    
    def crop_face(self, image: Image.Image, bbox: List[float], margin: float = 0.2) -> Optional[Image.Image]:
        """Crop face from image with margin."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Add margin
            width = x2 - x1
            height = y2 - y1
            margin_x = width * margin
            margin_y = height * margin
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(image.width, x2 + margin_x)
            y2 = min(image.height, y2 + margin_y)
            
            # Crop image
            cropped = image.crop((x1, y1, x2, y2))
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            return None


class ImageProcessor:
    """
    Processes images for face detection, enhancement, and storage.
    
    Handles streaming processing with batched operations for efficiency
    and memory management.
    """
    
    def __init__(self, config: CrawlerConfig, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.face_detector = FaceDetector(config)
        
        # Small thread pool for face detection (one face model, small pool)
        self.face_thread_pool = ThreadPoolExecutor(
            max_workers=2,  # Small thread pool as specified
            thread_name_prefix="face_processing"
        )
        self.processing_thread_pool = ThreadPoolExecutor(
            max_workers=config.concurrent_processing,
            thread_name_prefix="image_processing"
        )
        
        # HTTP client for downloading images
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Processing statistics
        self.stats = {
            'images_processed': 0,
            'faces_detected': 0,
            'enhancements_applied': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_http_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize_http_client(self):
        """Initialize HTTP client for image downloading."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.timeout_seconds,
                    pool=self.config.timeout_seconds
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections
                ),
                follow_redirects=True,
                max_redirects=self.config.max_redirects
            )
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        
        self.face_thread_pool.shutdown(wait=True)
        self.processing_thread_pool.shutdown(wait=True)
    
    async def process_single_image(
        self, 
        image_url: str, 
        source_url: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ProcessedImage]:
        """
        Process a single image with face detection and enhancement.
        
        Args:
            image_url: URL of the image to process
            source_url: Source URL where the image was found
            context: Additional context information
            
        Returns:
            ProcessedImage object or None if processing failed
        """
        start_time = time.time()
        
        try:
            # Download image
            image_data = await self._download_image(image_url)
            if not image_data:
                return None
            
            # Load and validate image
            image = await self._load_image(image_data)
            if not image:
                return None
            
            # Check memory pressure
            if self.memory_manager.is_memory_pressured():
                await self.memory_manager.force_gc("image_processing")
            
            # Process image in thread pool
            loop = asyncio.get_event_loop()
            processed_image = await loop.run_in_executor(
                self.processing_thread_pool,
                self._process_image_sync,
                image,
                image_data,
                image_url,
                source_url,
                context or {}
            )
            
            processing_time = time.time() - start_time
            processed_image.processing_time = processing_time
            
            # Update statistics
            self.stats['images_processed'] += 1
            self.stats['faces_detected'] += len(processed_image.faces)
            self.stats['processing_time'] += processing_time
            
            if processed_image.enhancement_applied:
                self.stats['enhancements_applied'] += 1
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def process_batch(
        self, 
        image_urls: List[str], 
        source_url: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process a batch of images with concurrency control.
        
        Args:
            image_urls: List of image URLs to process
            source_url: Source URL where images were found
            context: Additional context information
            
        Returns:
            ProcessingResult with processed images and statistics
        """
        start_time = time.time()
        processed_images = []
        total_faces = 0
        
        # Process images with concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_downloads)
        
        async def process_with_semaphore(url: str) -> Optional[ProcessedImage]:
            async with semaphore:
                return await self.process_single_image(url, source_url, context)
        
        # Process all images
        tasks = [process_with_semaphore(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, ProcessedImage):
                processed_images.append(result)
                total_faces += len(result.faces)
            elif isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                self.stats['errors'] += 1
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            processed_images=processed_images,
            total_processed=len(processed_images),
            faces_detected=total_faces,
            processing_time=processing_time,
            success=True
        )
    
    async def process_images_streaming(
        self, 
        image_urls: List[str], 
        source_url: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[ProcessedImage]:
        """
        Stream processed images as they're completed.
        
        Args:
            image_urls: List of image URLs to process
            source_url: Source URL where images were found
            context: Additional context information
            
        Yields:
            ProcessedImage objects as they're processed
        """
        semaphore = asyncio.Semaphore(self.config.concurrent_downloads)
        
        async def process_with_semaphore(url: str) -> Optional[ProcessedImage]:
            async with semaphore:
                return await self.process_single_image(url, source_url, context)
        
        # Create tasks for all images
        tasks = [process_with_semaphore(url) for url in image_urls]
        
        # Process and yield results as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result:
                    yield result
            except Exception as e:
                logger.error(f"Error in streaming processing: {e}")
                self.stats['errors'] += 1
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL using streaming to temp file."""
        if not self.http_client:
            await self._initialize_http_client()
        
        temp_file = None
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
            temp_path = temp_file.name
            temp_file.close()  # Close the file handle, we'll use the path
            
            # Stream download to temp file
            async with self.http_client.stream('GET', url) as response:
                response.raise_for_status()
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.config.max_image_bytes:
                    logger.warning(f"Image too large: {url} ({content_length} bytes)")
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in self.config.allowed_content_types):
                    logger.warning(f"Unsupported content type: {url} ({content_type})")
                    return None
                
                # Stream to temp file with size checking
                total_bytes = 0
                with open(temp_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        
                        # Check size limit during download
                        if total_bytes > self.config.max_image_bytes:
                            logger.warning(f"Image too large during download: {url} ({total_bytes} bytes)")
                            return None
                        
                        f.write(chunk)
                
                # Read the temp file into memory
                with open(temp_path, 'rb') as f:
                    return f.read()
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error downloading {url}: {e.response.status_code}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Timeout downloading {url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.debug(f"Error cleaning up temp file {temp_path}: {e}")
    
    async def _load_image(self, image_data: bytes) -> Optional[Image.Image]:
        """Load and validate image from bytes with EXIF orientation correction."""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Apply EXIF orientation correction - CRITICAL for rotated images
            image = ImageOps.exif_transpose(image)
            logger.debug(f"Applied EXIF orientation correction to image")

            # Convert to RGB if necessary (RGBA -> RGB for JPEG compatibility)
            if image.mode == 'RGBA':
                # Create white background for RGBA images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')

            # Check image size
            if image.width * image.height > self.config.max_image_pixels:
                logger.warning(f"Image too large: {image.width}x{image.height}")
                return None
            
            # Check minimum image size
            min_width, min_height = self.config.min_image_size
            if image.width < min_width or image.height < min_height:
                logger.debug(f"Image too small: {image.width}x{image.height} < {min_width}x{min_height}")
                return None

            return image

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def _process_image_sync(
        self, 
        image: Image.Image, 
        image_data: bytes, 
        image_url: str, 
        source_url: str,
        context: Dict[str, Any]
    ) -> ProcessedImage:
        """Synchronous image processing (runs in thread pool)."""
        try:
            # Enhance image if needed
            enhanced_image, enhancement_applied = self._enhance_image(image)
            
            # Detect faces using thread pool with temp file
            faces = self._detect_faces_with_temp_file(enhanced_image)
            
            # Generate thumbnails for all faces if cropping is enabled
            thumbnail_data_list = []
            if faces and self.config.crop_faces:
                # Generate thumbnails from all cropped faces
                thumbnail_data_list = self._generate_face_thumbnails(enhanced_image, faces)
            
            # Calculate perceptual hash
            perceptual_hash = self._calculate_perceptual_hash(enhanced_image)
            
            # Convert enhanced image back to bytes
            enhanced_bytes = self._image_to_bytes(enhanced_image)
            
            return ProcessedImage(
                original_url=image_url,
                image_data=enhanced_bytes,
                thumbnail_data=thumbnail_data_list,  # Now a list of thumbnails
                faces=faces,
                perceptual_hash=perceptual_hash,
                dimensions=(enhanced_image.width, enhanced_image.height),
                file_size=len(enhanced_bytes),
                processing_time=0.0,  # Will be set by caller
                enhancement_applied=enhancement_applied
            )
            
        except Exception as e:
            logger.error(f"Error in sync image processing: {e}")
            raise
    
    def _detect_faces_with_temp_file(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect faces using temp file and thread pool."""
        temp_file = None
        try:
            # Save image to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            # Save image to temp file
            image.save(temp_path, 'JPEG', quality=95)
            
            # Use thread pool for face detection
            future = self.face_thread_pool.submit(self.face_detector.detect_and_embed_path, temp_path)
            faces = future.result(timeout=30)  # 30 second timeout
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection with temp file: {e}")
            return []
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.debug(f"Error cleaning up temp file {temp_path}: {e}")
    
    def _enhance_image(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Enhance image quality if needed."""
        enhancement_applied = False
        
        try:
            # Check if image needs enhancement
            if (image.width < self.config.image_enhancement_low_res_width or
                image.height < self.config.image_enhancement_low_res_height):
                
                # Apply contrast enhancement
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(self.config.image_enhancement_contrast)
                
                # Apply sharpness enhancement
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(self.config.image_enhancement_sharpness)
                
                enhancement_applied = True
                logger.debug(f"Applied image enhancement to {image.width}x{image.height} image")
            
            return image, enhancement_applied
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image, False
    
    def _generate_thumbnail(self, image: Image.Image) -> Optional[bytes]:
        """Generate thumbnail from image."""
        try:
            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail(self.config.thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            thumbnail_bytes = io.BytesIO()
            thumbnail.save(
                thumbnail_bytes, 
                format='JPEG', 
                quality=self.config.thumbnail_quality,
                optimize=True
            )
            
            return thumbnail_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return None
    
    def _generate_face_thumbnails(self, image: Image.Image, faces: List[Dict[str, Any]]) -> List[bytes]:
        """Generate thumbnails from all cropped faces."""
        thumbnails = []
        
        try:
            if not faces:
                return thumbnails
            
            logger.info(f"Generating thumbnails for {len(faces)} faces")
            
            # Generate thumbnail for each face
            for i, face in enumerate(faces):
                try:
                    thumbnail_data = self._generate_single_face_thumbnail(image, face, i)
                    if thumbnail_data:
                        thumbnails.append(thumbnail_data)
                        logger.info(f"Generated thumbnail {i+1}/{len(faces)}: {len(thumbnail_data)} bytes")
                except Exception as e:
                    logger.error(f"Error generating thumbnail for face {i}: {e}")
                    continue
            
            logger.info(f"Successfully generated {len(thumbnails)} thumbnails from {len(faces)} faces")
            return thumbnails
            
        except Exception as e:
            logger.error(f"Error generating face thumbnails: {e}")
            return thumbnails
    
    def _generate_single_face_thumbnail(self, image: Image.Image, face: Dict[str, Any], face_index: int) -> Optional[bytes]:
        """Generate thumbnail from a single face."""
        try:
            # Extract face region
            bbox = face['bbox']
            
            # Debug: Log the bbox format
            logger.debug(f"Face {face_index} bbox format: {type(bbox)} = {bbox}")
            
            # Handle bbox format from InsightFace: (x, y, w, h) -> (left, top, right, bottom)
            try:
                # bbox should now be a flat list [x, y, w, h] from our fix above
                if isinstance(bbox, list) and len(bbox) == 4:
                    x, y, w, h = bbox
                    logger.debug(f"Face {face_index} bbox: x={x}, y={y}, w={w}, h={h}")
                else:
                    logger.error(f"Face {face_index} unexpected bbox format: {bbox} (type: {type(bbox)})")
                    return None
                
                # Convert (x, y, w, h) to (left, top, right, bottom) with clamping
                try:
                    L, T, R, B = int(x), int(y), int(x + w), int(y + h)
                    logger.debug(f"Face {face_index} crop coordinates: L={L}, T={T}, R={R}, B={B}")
                except Exception as e:
                    logger.error(f"Face {face_index} error converting bbox to crop coordinates: {e}")
                    return None
                
                L = max(0, L)
                T = max(0, T) 
                R = min(image.width, R)
                B = min(image.height, B)
                
                # Validate crop bounds
                if L >= R or T >= B:
                    logger.error(f"Face {face_index} invalid crop bounds: L={L}, T={T}, R={R}, B={B}")
                    return None
                    
            except Exception as e:
                logger.error(f"Face {face_index} error processing bbox coordinates: {e}, bbox: {bbox}")
                return None
            
            # Add some padding around the face
            try:
                face_width = R - L
                face_height = B - T
                logger.debug(f"Face {face_index} dimensions: width={face_width}, height={face_height}")
                
                padding = max(face_width, face_height) * 0.2
                logger.debug(f"Face {face_index} calculated padding: {padding}")
                
                L_padded = max(0, int(L - padding))
                T_padded = max(0, int(T - padding))
                R_padded = min(image.width, int(R + padding))
                B_padded = min(image.height, int(B + padding))
                
                logger.debug(f"Face {face_index} padded coordinates: L={L_padded}, T={T_padded}, R={R_padded}, B={B_padded}")
                
            except Exception as e:
                logger.error(f"Face {face_index} error calculating padding: {e}")
                return None
            
            # Crop face region with padding
            try:
                logger.debug(f"Face {face_index} cropping image with coordinates: ({L_padded}, {T_padded}, {R_padded}, {B_padded})")
                face_crop = image.crop((L_padded, T_padded, R_padded, B_padded))
                logger.debug(f"Face {face_index} successfully cropped face region: {face_crop.size}")
            except Exception as e:
                logger.error(f"Face {face_index} error cropping face region: {e}")
                return None
            
            # Create thumbnail
            try:
                logger.debug(f"Face {face_index} creating thumbnail from cropped image: {face_crop.size}")
                face_crop.thumbnail(
                    self.config.thumbnail_size,
                    Image.Resampling.LANCZOS
                )
                logger.debug(f"Face {face_index} thumbnail created: {face_crop.size}")
            except Exception as e:
                logger.error(f"Face {face_index} error creating thumbnail: {e}")
                return None
            
            # Convert to bytes
            try:
                thumbnail_bytes = io.BytesIO()
                face_crop.save(
                    thumbnail_bytes, 
                    format='JPEG', 
                    quality=self.config.thumbnail_quality,
                    optimize=True
                )
                logger.debug(f"Face {face_index} thumbnail saved to bytes: {len(thumbnail_bytes.getvalue())} bytes")
                return thumbnail_bytes.getvalue()
            except Exception as e:
                logger.error(f"Face {face_index} error saving thumbnail to bytes: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Face {face_index} error generating face thumbnail: {e}")
            return None
    
    def _calculate_perceptual_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash for image."""
        try:
            # Convert to grayscale for hashing
            gray_image = image.convert('L')
            
            # Calculate perceptual hash
            phash = imagehash.phash(gray_image)
            
            return str(phash)
            
        except Exception as e:
            logger.error(f"Error calculating perceptual hash: {e}")
            return ""
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL image to bytes."""
        try:
            image_bytes = io.BytesIO()
            image.save(
                image_bytes,
                format='JPEG',
                quality=self.config.image_jpeg_quality,
                optimize=True
            )
            return image_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            return b""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'images_processed': 0,
            'faces_detected': 0,
            'enhancements_applied': 0,
            'processing_time': 0.0,
            'errors': 0
        }
