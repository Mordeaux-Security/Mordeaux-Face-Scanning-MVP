"""Pydantic schemas for Mordeaux services."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BaseEvent(BaseModel):
    """Base event model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BoundingBox(BaseModel):
    """Bounding box model."""
    x: float
    y: float
    width: float
    height: float


class FaceExtracted(BaseModel):
    """Extracted face model."""
    face_id: str = Field(..., description="Unique face identifier")
    bbox: BoundingBox
    quality: float = Field(..., ge=0, le=1, description="Face quality score")
    aligned_s3_key: str = Field(..., description="S3 key for aligned face image")


class NewContentEvent(BaseEvent):
    """New content event."""
    type: str = Field(default="NEW_CONTENT")
    content_id: str = Field(..., description="Unique content identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    source_id: str = Field(..., description="Source identifier")
    s3_key_raw: str = Field(..., description="S3 key for raw content")
    url: str = Field(..., description="Source URL")
    fetch_ts: datetime = Field(..., description="Fetch timestamp")


class FacesExtractedEvent(BaseEvent):
    """Faces extracted event."""
    type: str = Field(default="FACES_EXTRACTED")
    content_id: str = Field(..., description="Content identifier")
    faces: List[FaceExtracted] = Field(..., description="List of extracted faces")


class IndexedEvent(BaseEvent):
    """Indexed event."""
    type: str = Field(default="INDEXED")
    content_id: str = Field(..., description="Content identifier")
    embedding_ids: List[str] = Field(..., description="List of embedding identifiers")
    index_ns: str = Field(..., description="Index namespace")
    ts: datetime = Field(..., description="Index timestamp")


# Union type for all events
EventSchema = NewContentEvent | FacesExtractedEvent | IndexedEvent


class SearchByVectorRequest(BaseModel):
    """Search by vector request."""
    vector: List[float] = Field(..., min_length=512, max_length=512, description="512-dimensional vector")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class SearchResult(BaseModel):
    """Search result."""
    content_id: str = Field(..., description="Content identifier")
    face_id: str = Field(..., description="Face identifier")
    score: float = Field(..., ge=0, le=1, description="Similarity score")
    thumb_s3_key: str = Field(..., description="Thumbnail S3 key")
    cluster_id: Optional[str] = Field(None, description="Cluster identifier")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
