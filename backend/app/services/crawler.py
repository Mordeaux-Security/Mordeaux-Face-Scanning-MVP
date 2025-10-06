"""
Enhanced Image Crawler Service

Integrated crawler service that uses the existing MinIO storage service
and provides multiple targeting strategies for finding specific types of images.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from .storage import save_raw_and_thumb, save_raw_image_only, save_thumbnail_only, BUCKET_RAW, BUCKET_THUMBS
from .face import get_face_service
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


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


@dataclass
class ImageInfo:
    """Information about a discovered image."""
    url: str
    alt_text: str
    title: str
    width: Optional[int]
    height: Optional[int]


class EnhancedImageCrawler:
    """
    Enhanced image crawler integrated with MinIO storage service.
    """
    
    def __init__(
        self,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_extensions: Optional[Set[str]] = None,
        timeout: int = 30,
        min_face_quality: float = 0.5,
        require_face: bool = True,
        crop_faces: bool = True,
        face_margin: float = 0.2,
        max_total_images: int = 50,
        max_pages: int = 20,
        same_domain_only: bool = True,
        save_both: bool = False
    ):
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or {
            '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'
        }
        self.timeout = timeout
        self.min_face_quality = min_face_quality  # Minimum detection score for face quality
        self.require_face = require_face  # Whether to require at least one face
        self.crop_faces = crop_faces  # Whether to crop and save only face regions
        self.face_margin = face_margin  # Margin around face as fraction of face size
        self.max_total_images = max_total_images  # Maximum total images to collect
        self.max_pages = max_pages  # Maximum pages to crawl
        self.same_domain_only = same_domain_only  # Only crawl same domain
        self.save_both = save_both  # Whether to save both original and cropped images
        self.session: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    async def fetch_page(self, url: str) -> Tuple[Optional[str], List[str]]:
        """Fetch a web page and return its content and any errors."""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")
            
        errors = []
        
        try:
            logger.info(f"Fetching page: {url}")
            response = await self.session.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                errors.append(f"Content type '{content_type}' is not HTML")
                return None, errors
                
            return response.text, errors
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error fetching {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
        except Exception as e:
            error_msg = f"Unexpected error fetching {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
    
    def extract_images_by_method(self, html_content: str, base_url: str, method: str = "smart") -> Tuple[List[ImageInfo], str]:
        """
        Extract images using different targeting methods.
        
        Args:
            html_content: The HTML content to parse
            base_url: Base URL for resolving relative URLs
            method: Targeting method ('smart', 'data-mediumthumb', 'js-videoThumb', 'size', 'phimage', 'latestThumb', 'all')
            
        Returns:
            Tuple of (images_list, method_used)
        """
        images = []
        method_used = method
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            if method == "smart":
                # Try multiple methods in order of preference
                methods_to_try = [
                    ("data-mediumthumb", self._extract_by_data_mediumthumb),
                    ("js-videoThumb", self._extract_by_js_video_thumb),
                    ("size", self._extract_by_size),
                    ("phimage", self._extract_by_phimage),
                    ("latestThumb", self._extract_by_latest_thumb),
                    ("all", self._extract_all_images)
                ]
                
                for method_name, extract_func in methods_to_try:
                    images = extract_func(soup, base_url)
                    if images:
                        method_used = method_name
                        logger.info(f"Smart method selected: {method_name} (found {len(images)} images)")
                        break
                        
            elif method == "data-mediumthumb":
                images = self._extract_by_data_mediumthumb(soup, base_url)
            elif method == "js-videoThumb":
                images = self._extract_by_js_video_thumb(soup, base_url)
            elif method == "size":
                images = self._extract_by_size(soup, base_url)
            elif method == "phimage":
                images = self._extract_by_phimage(soup, base_url)
            elif method == "latestThumb":
                images = self._extract_by_latest_thumb(soup, base_url)
            elif method == "all":
                images = self._extract_all_images(soup, base_url)
            else:
                logger.warning(f"Unknown method '{method}', falling back to 'all'")
                images = self._extract_all_images(soup, base_url)
                method_used = "all"
            
            logger.info(f"Found {len(images)} images using method: {method_used}")
                
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            
        return images, method_used
    
    def _extract_by_data_mediumthumb(self, soup, base_url: str) -> List[ImageInfo]:
        """Extract images with data-mediumthumb attribute (video thumbnails)."""
        imgs = soup.find_all('img', attrs={'data-mediumthumb': True})
        return self._process_img_tags(imgs, base_url)
    
    def _extract_by_js_video_thumb(self, soup, base_url: str) -> List[ImageInfo]:
        """Extract images with js-videoThumb class."""
        imgs = soup.find_all('img', class_='js-videoThumb')
        return self._process_img_tags(imgs, base_url)
    
    def _extract_by_size(self, soup, base_url: str) -> List[ImageInfo]:
        """Extract images with video thumbnail dimensions (320x180)."""
        imgs = soup.find_all('img', attrs={'width': '320', 'height': '180'})
        return self._process_img_tags(imgs, base_url)
    
    def _extract_by_phimage(self, soup, base_url: str) -> List[ImageInfo]:
        """Extract images inside .phimage divs."""
        phimage_divs = soup.find_all('div', class_='phimage')
        imgs = []
        for div in phimage_divs:
            imgs.extend(div.find_all('img'))
        return self._process_img_tags(imgs, base_url)
    
    def _extract_by_latest_thumb(self, soup, base_url: str) -> List[ImageInfo]:
        """Extract images inside links with latestThumb class."""
        links = soup.find_all('a', class_='latestThumb')
        imgs = []
        for link in links:
            imgs.extend(link.find_all('img'))
        return self._process_img_tags(imgs, base_url)
    
    def _extract_all_images(self, soup, base_url: str) -> List[ImageInfo]:
        """Extract all img tags."""
        imgs = soup.find_all('img')
        return self._process_img_tags(imgs, base_url)
    
    def _process_img_tags(self, img_tags, base_url: str) -> List[ImageInfo]:
        """Process a list of img tags and return ImageInfo objects."""
        images = []
        
        for img in img_tags:
            src = img.get('src')
            if not src:
                continue
                
            # Resolve relative URLs
            absolute_url = urljoin(base_url, src)
            
            # Extract image metadata
            alt_text = img.get('alt', '')
            title = img.get('title', '')
            width = img.get('width')
            height = img.get('height')
            
            # Convert width/height to integers if possible
            try:
                width = int(width) if width else None
            except (ValueError, TypeError):
                width = None
                
            try:
                height = int(height) if height else None
            except (ValueError, TypeError):
                height = None
            
            images.append(ImageInfo(
                url=absolute_url,
                alt_text=alt_text,
                title=title,
                width=width,
                height=height
            ))
        
        return images
    
    async def download_image(self, image_info: ImageInfo) -> Tuple[Optional[bytes], List[str]]:
        """Download an image from its URL."""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")
            
        errors = []
        
        try:
            logger.info(f"Downloading image: {image_info.url}")
            
            # Check file extension
            parsed_url = urlparse(image_info.url)
            path = parsed_url.path.lower()
            if not any(path.endswith(ext) for ext in self.allowed_extensions):
                # Try to get content type from HEAD request
                head_response = await self.session.head(image_info.url)
                content_type = head_response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                    errors.append(f"File extension not in allowed list: {path}")
                    return None, errors
            
            # Download the image
            response = await self.session.get(image_info.url)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                errors.append(f"File too large: {content_length} bytes")
                return None, errors
                
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                errors.append(f"Content type not an image: {content_type}")
                return None, errors
            
            return response.content, errors
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error downloading {image_info.url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
        except Exception as e:
            error_msg = f"Unexpected error downloading {image_info.url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
    
    def extract_page_urls(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract URLs from HTML content for further crawling.
        
        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative links
            
        Returns:
            List of URLs found on the page
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            urls = set()
            
            # Extract links from <a> tags
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    parsed_url = urlparse(absolute_url)
                    
                    # Filter URLs
                    if self._is_valid_page_url(absolute_url, base_url):
                        urls.add(absolute_url)
            
            # Also look for pagination patterns (common on adult sites)
            pagination_selectors = [
                'a[href*="page="]',
                'a[href*="p="]', 
                'a[href*="/page/"]',
                'a[href*="/p/"]',
                '.pagination a',
                '.pager a',
                '.page-nav a',
                'a[class*="page"]',
                'a[class*="next"]',
                'a[class*="more"]'
            ]
            
            for selector in pagination_selectors:
                for link in soup.select(selector):
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(base_url, href)
                        if self._is_valid_page_url(absolute_url, base_url):
                            urls.add(absolute_url)
            
            return list(urls)
            
        except Exception as e:
            logger.error(f"Error extracting page URLs: {str(e)}")
            return []
    
    def _is_valid_page_url(self, url: str, base_url: str) -> bool:
        """
        Check if a URL is valid for crawling.
        
        Args:
            url: URL to check
            base_url: Base URL for domain comparison
            
        Returns:
            True if URL is valid for crawling
        """
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            
            # Must have scheme and netloc
            if not parsed_url.scheme or not parsed_url.netloc:
                return False
            
            # Check domain restriction
            if self.same_domain_only and parsed_url.netloc != parsed_base.netloc:
                return False
            
            # Skip non-HTTP protocols
            if parsed_url.scheme not in ['http', 'https']:
                return False
            
            # Skip common non-page URLs
            skip_patterns = [
                'javascript:', 'mailto:', 'tel:', '#',
                '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg',
                '.pdf', '.doc', '.docx', '.zip', '.rar',
                'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com'
            ]
            
            url_lower = url.lower()
            for pattern in skip_patterns:
                if pattern in url_lower:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {str(e)}")
            return False
    
    def check_face_quality(self, image_bytes: bytes) -> Tuple[bool, str, dict]:
        """
        Check if the image contains high-quality faces suitable for recognition.
        
        Args:
            image_bytes: The image data to analyze
            
        Returns:
            Tuple of (is_high_quality, reason, best_face)
        """
        try:
            face_service = get_face_service()
            faces = face_service.detect_and_embed(image_bytes)
            
            if not faces:
                if self.require_face:
                    return False, "No faces detected", {}
                else:
                    return True, "No faces required", {}
            
            # Check each face for quality
            high_quality_faces = []
            for face in faces:
                det_score = face.get('det_score', 0.0)
                if det_score >= self.min_face_quality:
                    high_quality_faces.append(face)
            
            if not high_quality_faces:
                return False, f"No faces meet quality threshold (min: {self.min_face_quality})", {}
            
            # Find the best face (highest quality score and largest size)
            best_face = None
            best_score = 0
            for face in high_quality_faces:
                bbox = face.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Skip faces that are too small (less than 50x50 pixels)
                if face_width < 50 or face_height < 50:
                    continue
                
                # Score based on detection confidence and face size
                face_score = face.get('det_score', 0.0) + (face_width * face_height) / 100000.0
                if face_score > best_score:
                    best_score = face_score
                    best_face = face
            
            if best_face:
                return True, f"Found {len(high_quality_faces)} high-quality faces", best_face
            else:
                return False, "All faces are too small for reliable recognition", {}
            
        except Exception as e:
            logger.error(f"Error checking face quality: {str(e)}")
            if self.require_face:
                return False, f"Face quality check failed: {str(e)}", {}
            else:
                return True, f"Face quality check failed, but not required: {str(e)}", {}

    async def save_image_to_storage(self, image_bytes: bytes, image_info: ImageInfo) -> Tuple[Optional[str], Optional[str]]:
        """
        Save image bytes to MinIO storage with correct logic:
        1. Always save to raw-images bucket
        2. Check face quality and save cropped face to thumbnails bucket if high quality
        
        Returns:
            Tuple of (raw_key, thumbnail_key) - either can be None
        """
        try:
            # STEP 1: Always save raw image to raw-images bucket
            raw_key, raw_url = save_raw_image_only(image_bytes)
            logger.info(f"Saved raw image to MinIO - Raw: {raw_key}")
            
            # STEP 2: Check face quality for potential thumbnail processing
            is_high_quality, reason, best_face = self.check_face_quality(image_bytes)
            
            thumbnail_key = None
            if is_high_quality and self.crop_faces and best_face:
                # STEP 3: Crop and save high-quality face to thumbnails bucket
                face_service = get_face_service()
                bbox = best_face.get('bbox', [0, 0, 0, 0])
                cropped_image_bytes = face_service.crop_face_from_image(image_bytes, bbox, self.face_margin)
                
                # Save cropped face to thumbnails bucket only
                thumbnail_key, thumbnail_url = save_thumbnail_only(cropped_image_bytes)
                
                # Log face details
                x1, y1, x2, y2 = bbox
                face_width = int(x2 - x1)
                face_height = int(y2 - y1)
                det_score = best_face.get('det_score', 0.0)
                
                logger.info(f"Saved cropped face to MinIO - Thumbnail: {thumbnail_key} "
                           f"(Face: {face_width}x{face_height}, Score: {det_score:.3f}, {reason})")
            else:
                logger.info(f"No thumbnail saved - Quality check: {reason}")
            
            return raw_key, thumbnail_key
            
        except Exception as e:
            logger.error(f"Error saving image to MinIO storage: {str(e)}")
            return None, None
    
    async def crawl_page(self, url: str, method: str = "smart") -> CrawlResult:
        """Crawl a single page for images using the specified method."""
        logger.info(f"Starting crawl of: {url} using method: {method} (min_face_quality: {self.min_face_quality}, require_face: {self.require_face})")
        
        saved_raw_keys = []
        saved_thumbnail_keys = []
        all_errors = []
        
        # Fetch the page
        html_content, fetch_errors = await self.fetch_page(url)
        all_errors.extend(fetch_errors)
        
        if not html_content:
            logger.error("Failed to fetch page content")
            return CrawlResult(
                url=url,
                images_found=0,
                raw_images_saved=0,
                thumbnails_saved=0,
                pages_crawled=0,
                saved_raw_keys=[],
                saved_thumbnail_keys=[],
                errors=all_errors,
                targeting_method="failed"
            )
        
        # Extract images using specified method
        images, method_used = self.extract_images_by_method(html_content, url, method)
        
        if not images:
            logger.warning("No images found on the page")
            return CrawlResult(
                url=url,
                images_found=0,
                raw_images_saved=0,
                thumbnails_saved=0,
                pages_crawled=0,
                saved_raw_keys=[],
                saved_thumbnail_keys=[],
                errors=all_errors,
                targeting_method=method_used
            )
        
        # Process ALL images found (no artificial per-page limits)
        images_to_process = images
        logger.info(f"Processing all {len(images_to_process)} images found")
        
        for i, image_info in enumerate(images_to_process, 1):
            logger.info(f"Processing image {i}/{len(images_to_process)}: {image_info.url}")
            
            # Download the image
            image_bytes, download_errors = await self.download_image(image_info)
            all_errors.extend(download_errors)
            
            if not image_bytes:
                logger.warning(f"Failed to download image: {image_info.url}")
                continue
            
            # Save to MinIO storage (always saves raw, conditionally saves thumbnail)
            raw_key, thumbnail_key = await self.save_image_to_storage(image_bytes, image_info)
            
            if raw_key:
                saved_raw_keys.append(raw_key)
                logger.info(f"Successfully saved raw image {i}/{len(images_to_process)}")
                
                if thumbnail_key:
                    saved_thumbnail_keys.append(thumbnail_key)
                    logger.info(f"Successfully saved thumbnail {i}/{len(images_to_process)}")
            else:
                logger.warning(f"Failed to save image {i}/{len(images_to_process)}")
        
        result = CrawlResult(
            url=url,
            images_found=len(images),
            raw_images_saved=len(saved_raw_keys),
            thumbnails_saved=len(saved_thumbnail_keys),
            pages_crawled=1,
            saved_raw_keys=saved_raw_keys,
            saved_thumbnail_keys=saved_thumbnail_keys,
            errors=all_errors,
            targeting_method=method_used
        )
        
        logger.info(f"Crawl completed - Found: {result.images_found}, Raw Images Saved: {result.raw_images_saved}, Thumbnails Saved: {result.thumbnails_saved}")
        return result

    async def crawl_site(self, start_url: str, method: str = "smart") -> CrawlResult:
        """
        Crawl multiple pages on a site to collect images up to max_total_images.
        
        Args:
            start_url: Starting URL for crawling
            method: Targeting method for image extraction
            
        Returns:
            CrawlResult with aggregated statistics
        """
        logger.info(f"Starting site crawl from: {start_url} (max_images: {self.max_total_images}, max_pages: {self.max_pages})")
        
        # Initialize crawling state
        visited_urls = set()
        urls_to_visit = [start_url]
        all_saved_raw_keys = []
        all_saved_thumbnail_keys = []
        all_errors = []
        total_images_found = 0
        total_raw_saved = 0
        total_thumbnails_saved = 0
        pages_crawled = 0
        
        while urls_to_visit and len(all_saved_thumbnail_keys) < self.max_total_images and pages_crawled < self.max_pages:
            # Get next URL to visit
            current_url = urls_to_visit.pop(0)
            
            # Skip if already visited
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            pages_crawled += 1
            
            logger.info(f"Crawling page {pages_crawled}/{self.max_pages}: {current_url}")
            logger.info(f"Thumbnails collected so far: {len(all_saved_thumbnail_keys)}/{self.max_total_images}")
            
            try:
                # Crawl the current page
                page_result = await self.crawl_page(current_url, method)
                
                # Accumulate results
                total_images_found += page_result.images_found
                total_raw_saved += page_result.raw_images_saved
                total_thumbnails_saved += page_result.thumbnails_saved
                all_saved_raw_keys.extend(page_result.saved_raw_keys)
                all_saved_thumbnail_keys.extend(page_result.saved_thumbnail_keys)
                all_errors.extend(page_result.errors)
                
                logger.info(f"Page {pages_crawled} results: Found {page_result.images_found}, Raw Saved {page_result.raw_images_saved}, Thumbnails Saved {page_result.thumbnails_saved}")
                
                # If we haven't reached the thumbnail limit, discover new URLs
                if len(all_saved_thumbnail_keys) < self.max_total_images and len(urls_to_visit) < self.max_pages * 2:
                    # Fetch page content for URL discovery
                    html_content, fetch_errors = await self.fetch_page(current_url)
                    all_errors.extend(fetch_errors)
                    
                    if html_content:
                        # Extract new URLs
                        new_urls = self.extract_page_urls(html_content, current_url)
                        
                        # Add new URLs to visit queue (prioritize unseen URLs)
                        for url in new_urls:
                            if url not in visited_urls and url not in urls_to_visit:
                                urls_to_visit.append(url)
                        
                        logger.info(f"Found {len(new_urls)} new URLs to explore")
                
                # Check if we've reached our goals
                if len(all_saved_thumbnail_keys) >= self.max_total_images:
                    logger.info(f"Reached target of {self.max_total_images} thumbnails, stopping crawl")
                    break
                    
            except Exception as e:
                error_msg = f"Error crawling page {current_url}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                continue
        
        # Create aggregated result
        result = CrawlResult(
            url=start_url,
            images_found=total_images_found,
            raw_images_saved=total_raw_saved,
            thumbnails_saved=total_thumbnails_saved,
            pages_crawled=pages_crawled,
            saved_raw_keys=all_saved_raw_keys,
            saved_thumbnail_keys=all_saved_thumbnail_keys,
            errors=all_errors,
            targeting_method=method
        )
        
        logger.info(f"Site crawl completed - Pages: {pages_crawled}, Found: {total_images_found}, Raw Images Saved: {total_raw_saved}, Thumbnails Saved: {total_thumbnails_saved}")
        return result


# Convenience function for easy integration
async def crawl_images_from_url(
    url: str, 
    method: str = "smart"
) -> CrawlResult:
    """
    Convenience function to crawl images from a URL.
    
    Args:
        url: The URL to crawl
        method: Targeting method ('smart', 'data-mediumthumb', 'js-videoThumb', etc.)
        
    Returns:
        CrawlResult with crawl statistics
    """
    async with EnhancedImageCrawler() as crawler:
        return await crawler.crawl_page(url, method)
