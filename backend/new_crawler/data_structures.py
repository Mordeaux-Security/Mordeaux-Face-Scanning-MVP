"""
Data Structures for New Crawler System

Clean Pydantic models for all queue messages and data flow.
Ensures type safety and consistent data structures throughout the system.
"""

import base64
import json
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, model_serializer, field_validator
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Status of a processing task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SiteTask(BaseModel):
    """Task for crawling a site."""
    url: str = Field(..., description="Site URL to crawl")
    site_id: str = Field(..., description="Unique site identifier")
    priority: int = Field(default=1, description="Processing priority (1=highest)")
    max_pages: int = Field(default=5, description="Maximum pages to crawl")
    max_images_per_site: int = Field(default=1000, description="Maximum images to save per site")
    use_3x3_mining: bool = Field(default=True, description="Enable 3x3 selector mining")
    pages_crawled: int = Field(default=0, description="Number of pages crawled")
    images_saved: int = Field(default=0, description="Number of images saved to MinIO")
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class CandidateImage(BaseModel):
    """Candidate image found during crawling."""
    page_url: str = Field(..., description="URL of the page containing the image")
    img_url: str = Field(..., description="Direct URL to the image")
    selector_hint: str = Field(..., description="CSS selector that found this image")
    site_id: str = Field(..., description="Site identifier")
    alt_text: Optional[str] = Field(None, description="Image alt text")
    width: Optional[int] = Field(None, description="Image width")
    height: Optional[int] = Field(None, description="Image height")
    discovered_at: datetime = Field(default_factory=datetime.now)

    # NEW: Add metadata from HTML parsing to skip HEAD requests
    content_type: Optional[str] = Field(None, description="Content type inferred from URL extension")
    estimated_size: Optional[int] = Field(None, description="Estimated file size from dimensions")
    has_srcset: bool = Field(False, description="Whether image has srcset attribute")

    @validator('img_url')
    def validate_img_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Image URL must start with http:// or https://')
        return v


class CandidatePost(BaseModel):
    """Candidate post found during crawling that may contain diabetes-related content."""
    page_url: str = Field(..., description="URL of the page containing the post")
    post_url: str = Field(..., description="Direct URL to the post")
    selector_hint: str = Field(..., description="CSS selector that found this post")
    site_id: str = Field(..., description="Site identifier")
    title: Optional[str] = Field(None, description="Post title")
    content: Optional[str] = Field(None, description="Post content text")
    author: Optional[str] = Field(None, description="Post author")
    date: Optional[datetime] = Field(None, description="Post date extracted from HTML")
    raw_html: Optional[str] = Field(None, description="Full HTML of the post element(s)")
    discovered_at: datetime = Field(default_factory=datetime.now)

    @validator('post_url')
    def validate_post_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Post URL must start with http:// or https://')
        return v


class ImageTask(BaseModel):
    """Task for processing an image."""
    temp_path: str = Field(..., description="Path to temporary image file")
    phash: str = Field(..., description="Perceptual hash of the image")
    candidate: CandidateImage = Field(..., description="Original candidate data")
    file_size: int = Field(..., description="Image file size in bytes")
    mime_type: str = Field(..., description="MIME type of the image")
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = Field(default=TaskStatus.PENDING)


class PostTask(BaseModel):
    """Task for processing a post for diabetes-related content."""
    candidate: CandidatePost = Field(..., description="Original candidate data")
    content_hash: str = Field(..., description="Hash of the post content")
    has_keywords: bool = Field(default=False, description="Whether post contains diabetes keywords")
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = Field(default=TaskStatus.PENDING)


class PostMetadata(BaseModel):
    """Metadata for a diabetes-related post to be stored in MinIO."""
    title: Optional[str] = Field(None, description="Post title")
    content: Optional[str] = Field(None, description="Post content text")
    author: Optional[str] = Field(None, description="Post author")
    url: str = Field(..., description="Post URL")
    date: Optional[datetime] = Field(None, description="Post date")
    site_id: str = Field(..., description="Site identifier")
    content_hash: str = Field(..., description="Hash of the post content")
    discovered_at: datetime = Field(..., description="When the post was discovered")
    page_url: str = Field(..., description="URL of the page containing the post")


class ImageMetadata(BaseModel):
    """Metadata for image processing (JSON serializable, binary data stored separately)."""
    phash: str = Field(..., description="Perceptual hash of the image")
    candidate: CandidateImage = Field(..., description="Original candidate data")
    file_size: int = Field(..., description="Image file size in bytes")
    mime_type: str = Field(..., description="MIME type of the image")
    redis_binary_key: str = Field(..., description="Redis key for binary image data")
    created_at: datetime = Field(default_factory=datetime.now)


# TODO: ARCHITECTURE ISSUE - Duplicate Data Structures
# FaceDetection is defined in THREE places:
#   1. new_crawler/data_structures.py (this file)
#   2. gpu_worker/worker.py (GPU service)
#   3. app/services/gpu_client.py (legacy service)
#
# All three are identical but manually kept in sync.
# Should create common/models.py with single definition:
#
#   from common.models import FaceDetection
#
# Instead of three separate implementations.
# Risk: If one changes, others break silently.

class FaceDetection(BaseModel):
    """Face detection result."""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    landmarks: List[List[float]] = Field(default_factory=list, description="Facial landmarks")
    embedding: Optional[List[float]] = Field(None, description="Face embedding vector")
    quality: float = Field(..., description="Detection quality score")
    age: Optional[int] = Field(None, description="Estimated age")
    gender: Optional[str] = Field(None, description="Estimated gender")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError('Bounding box must have exactly 4 coordinates')
        return v


class FaceResult(BaseModel):
    """Result of face processing for an image."""
    image_task: ImageTask = Field(..., description="Original image task")
    faces: List[FaceDetection] = Field(default_factory=list, description="Detected faces")
    crop_paths: List[str] = Field(default_factory=list, description="Paths to cropped face images")
    raw_image_key: Optional[str] = Field(None, description="MinIO key for raw image")
    thumbnail_keys: List[str] = Field(default_factory=list, description="MinIO keys for face thumbnails")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    gpu_used: bool = Field(..., description="Whether GPU was used for processing")
    saved_to_raw: bool = Field(default=False, description="Whether raw image was saved")
    saved_to_thumbs: bool = Field(default=False, description="Whether thumbnails were saved")
    skip_reason: Optional[str] = Field(None, description="Reason for skipping save")
    created_at: datetime = Field(default_factory=datetime.now)


class StorageTask(BaseModel):
    """Task for storage operations - pre-cropped faces ready for I/O."""
    image_task: ImageTask = Field(..., description="Original image task")
    face_result: FaceResult = Field(..., description="Face detection result with faces but not yet saved")
    face_crops: List[bytes] = Field(default_factory=list, description="Pre-cropped face images as bytes")
    batch_start_time: float = Field(..., description="Batch processing start time for timing")
    created_at: datetime = Field(default_factory=datetime.now)
    
    def model_dump(self, **kwargs):
        """Override model_dump to base64-encode face_crops for JSON serialization."""
        data = super().model_dump(**kwargs)
        # Convert bytes to base64 strings for JSON serialization
        if 'face_crops' in data and data['face_crops']:
            data['face_crops'] = [base64.b64encode(crop).decode('utf-8') if isinstance(crop, bytes) else crop 
                                   for crop in data['face_crops']]
        return data
    
    def model_dump_json(self, **kwargs):
        """Override model_dump_json to ensure face_crops are base64-encoded."""
        # First dump to dict with base64 encoding
        data = self.model_dump(**kwargs)
        # Then convert to JSON string
        return json.dumps(data, default=str)
    
    @field_validator('face_crops', mode='before')
    @classmethod
    def decode_face_crops(cls, v):
        """Decode base64 strings back to bytes if needed."""
        if not v:
            return []
        if isinstance(v, list) and len(v) > 0:
            # Check if first element is a string (base64-encoded)
            if isinstance(v[0], str):
                # Already base64-encoded strings, decode them
                return [base64.b64decode(crop.encode('utf-8')) for crop in v]
            elif isinstance(v[0], bytes):
                # Already bytes, return as-is
                return v
        return v


class ProcessingStats(BaseModel):
    """Statistics for site processing."""
    site_id: str = Field(..., description="Site identifier")
    site_url: str = Field(..., description="Site URL")
    pages_fetched: int = Field(default=0, description="Number of pages fetched")
    pages_crawled: int = Field(default=0, description="Number of pages crawled")
    images_found: int = Field(default=0, description="Number of images found")
    images_processed: int = Field(default=0, description="Number of images processed")
    images_saved_raw: int = Field(default=0, description="Number of images saved to raw-images bucket")
    images_saved_thumbs: int = Field(default=0, description="Number of thumbnails saved to thumbnails bucket")
    images_skipped_limit: int = Field(default=0, description="Number of images skipped due to per-site limit")
    images_cached: int = Field(default=0, description="Number of images skipped (cached)")
    faces_detected: int = Field(default=0, description="Number of faces detected")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None, description="Processing end time")
    total_time_seconds: Optional[float] = Field(None, description="Total processing time")
    
    # NEW: Add timing fields for throughput calculation
    extraction_start_time: Optional[datetime] = Field(None, description="Extraction start time")
    extraction_end_time: Optional[datetime] = Field(None, description="Extraction end time")
    gpu_processing_start_time: Optional[datetime] = Field(None, description="GPU processing start time")
    gpu_processing_end_time: Optional[datetime] = Field(None, description="GPU processing end time")
    storage_start_time: Optional[datetime] = Field(None, description="Storage start time")
    storage_end_time: Optional[datetime] = Field(None, description="Storage end time")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.images_found == 0:
            return 0.0
        return (self.images_processed + self.images_cached) / self.images_found
    
    @property
    def images_per_second(self) -> float:
        """Calculate overall images/second throughput."""
        if self.total_time_seconds and self.total_time_seconds > 0:
            return self.images_processed / self.total_time_seconds
        return 0.0
    
    @property
    def extraction_images_per_second(self) -> float:
        """Calculate extraction throughput."""
        if self.extraction_start_time and self.extraction_end_time:
            duration = (self.extraction_end_time - self.extraction_start_time).total_seconds()
            if duration > 0:
                return self.images_processed / duration
        return 0.0
    
    @property
    def gpu_images_per_second(self) -> float:
        """Calculate GPU processing throughput."""
        if self.gpu_processing_start_time and self.gpu_processing_end_time:
            duration = (self.gpu_processing_end_time - self.gpu_processing_start_time).total_seconds()
            if duration > 0:
                return self.images_processed / duration
        return 0.0


class BatchRequest(BaseModel):
    """Request for GPU worker batch processing."""
    image_tasks: List[Union[ImageTask, ImageMetadata]] = Field(..., description="List of image tasks or metadata to process")
    min_face_quality: float = Field(default=0.5, description="Minimum face quality threshold")
    require_face: bool = Field(default=False, description="Whether to require at least one face")
    crop_faces: bool = Field(default=True, description="Whether to crop face regions")
    face_margin: float = Field(default=0.2, description="Margin around face as fraction of face size")
    batch_id: str = Field(..., description="Unique batch identifier")
    created_at: datetime = Field(default_factory=datetime.now)


class BatchResponse(BaseModel):
    """Response from GPU worker batch processing."""
    batch_id: str = Field(..., description="Batch identifier")
    results: List[List[FaceDetection]] = Field(..., description="Face detection results for each image")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    gpu_used: bool = Field(..., description="Whether GPU was used for processing")
    worker_id: Optional[str] = Field(None, description="GPU worker identifier")
    created_at: datetime = Field(default_factory=datetime.now)


class QueueMetrics(BaseModel):
    """Metrics for queue monitoring."""
    queue_name: str = Field(..., description="Queue name")
    depth: int = Field(..., description="Current queue depth")
    max_depth: int = Field(..., description="Maximum queue depth")
    utilization_percent: float = Field(..., description="Queue utilization percentage")
    last_updated: datetime = Field(default_factory=datetime.now)


class SystemMetrics(BaseModel):
    """Overall system metrics."""
    active_crawlers: int = Field(default=0, description="Number of active crawler workers")
    active_extractors: int = Field(default=0, description="Number of active extractor workers")
    active_gpu_processors: int = Field(default=0, description="Number of active GPU processors")
    queue_metrics: List[QueueMetrics] = Field(default_factory=list, description="Queue metrics")
    total_sites_processed: int = Field(default=0, description="Total sites processed")
    total_images_processed: int = Field(default=0, description="Total images processed")
    total_faces_detected: int = Field(default=0, description="Total faces detected")
    gpu_worker_available: bool = Field(default=False, description="GPU worker availability")
    last_updated: datetime = Field(default_factory=datetime.now)


class CrawlResults(BaseModel):
    """Final results of a crawl operation."""
    sites: List[ProcessingStats] = Field(..., description="Statistics for each site")
    system_metrics: SystemMetrics = Field(..., description="Overall system metrics")
    total_time_seconds: float = Field(..., description="Total crawl time")
    success_rate: float = Field(..., description="Overall success rate")
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def total_images_found(self) -> int:
        """Total images found across all sites."""
        return sum(site.images_found for site in self.sites)
    
    @property
    def total_images_processed(self) -> int:
        """Total images processed across all sites."""
        return sum(site.images_processed for site in self.sites)
    
    @property
    def total_faces_detected(self) -> int:
        """Total faces detected across all sites."""
        return sum(site.faces_detected for site in self.sites)
    
    @property
    def overall_images_per_second(self) -> float:
        """Calculate overall images/second across all sites."""
        total_images = self.total_images_processed
        if self.total_time_seconds > 0:
            return total_images / self.total_time_seconds
        return 0.0
