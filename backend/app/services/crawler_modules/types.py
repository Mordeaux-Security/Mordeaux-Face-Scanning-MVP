"""
Type definitions and dataclasses for the crawler service.

This module contains all the data structures used by the crawler,
including result objects, image information, and type hints.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    url: str
    images_found: int
    raw_images_saved: int
    thumbnails_saved: int
    pages_crawled: int
    saved_raw_keys: List[str]
    saved_thumbnail_keys: List[str]
    errors: List[str]
    targeting_method: str
    cache_hits: int = 0
    cache_misses: int = 0
    redis_hits: int = 0
    postgres_hits: int = 0
    tenant_id: str = "default"
    early_exit_count: int = 0
    total_duration_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    checked_urls: List[str] = field(default_factory=list)  # URLs that were checked during crawl


@dataclass
class ImageInfo:
    """Information about a discovered image."""
    url: str
    alt_text: str
    title: str
    width: Optional[int]
    height: Optional[int]
    video_url: Optional[str] = None  # URL of the video this thumbnail links to


# Type aliases for better code readability
ImageUrl = str
PageUrl = str
TenantId = str
SiteDomain = str

# Common type hints used throughout the crawler
ImageBytes = bytes
HtmlContent = str
Selector = str
CacheKey = str
StorageKey = str
