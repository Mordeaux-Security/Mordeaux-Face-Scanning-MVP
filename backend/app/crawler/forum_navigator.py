"""
Forum Navigation Module

Provides generic forum detection and navigation capabilities for multiple
forum platforms (Discourse, phpBB, vBulletin, XenForo).
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .config import CrawlerConfig
from .http_service import HTTPService, get_http_service

logger = logging.getLogger(__name__)


@dataclass
class ForumStructure:
    """Detected forum structure information."""
    platform: str
    thread_selector: str
    post_selector: str
    image_selector: str
    confidence: float


@dataclass
class ForumThread:
    """Represents a forum thread."""
    url: str
    title: str
    post_count: int
    last_activity: Optional[str] = None


class ForumNavigator:
    """
    Generic forum navigation and extraction.
    
    Supports multiple forum platforms with automatic detection
    and platform-specific extraction patterns.
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.http_service = None
        
        # Forum platform patterns
        self.forum_patterns = {
            'discourse': {
                'thread_selector': 'a.topic-link, a[href*="/t/"]',
                'post_selector': 'div.post, article.post, div.topic-body',
                'image_selector': 'div.post img, article.post img, div.topic-body img',
                'indicators': ['discourse', 'topic-link', '/t/']
            },
            'phpbb': {
                'thread_selector': 'a.topictitle, a[href*="viewtopic"]',
                'post_selector': 'div.post, .postbody',
                'image_selector': 'div.post img, .postbody img',
                'indicators': ['phpbb', 'topictitle', 'viewtopic']
            },
            'vbulletin': {
                'thread_selector': 'a[href*="showthread"]',
                'post_selector': 'div.post, .postbit',
                'image_selector': 'div.post img, .postbit img',
                'indicators': ['vbulletin', 'showthread', 'postbit']
            },
            'xenforo': {
                'thread_selector': 'a[href*="threads/"]',
                'post_selector': 'article.message, .message-content',
                'image_selector': 'article.message img, .message-content img',
                'indicators': ['xenforo', 'threads/', 'message']
            }
        }
    
    async def detect_forum_structure(self, html: str, url: str) -> Optional[ForumStructure]:
        """Detect forum platform and structure."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Check for platform indicators
            for platform, patterns in self.forum_patterns.items():
                confidence = 0.0
                
                # Check for platform-specific indicators
                for indicator in patterns['indicators']:
                    if indicator in html.lower():
                        confidence += 0.3
                
                # Check for thread links
                thread_links = soup.select(patterns['thread_selector'])
                if thread_links:
                    confidence += 0.4
                
                # Check for post containers
                post_containers = soup.select(patterns['post_selector'])
                if post_containers:
                    confidence += 0.3
                
                if confidence >= 0.5:
                    logger.info(f"Detected {platform} forum with confidence {confidence:.2f}")
                    return ForumStructure(
                        platform=platform,
                        thread_selector=patterns['thread_selector'],
                        post_selector=patterns['post_selector'],
                        image_selector=patterns['image_selector'],
                        confidence=confidence
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting forum structure: {e}")
            return None
    
    async def extract_forum_threads(self, html: str, base_url: str, structure: ForumStructure) -> List[ForumThread]:
        """Extract forum thread links."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            threads = []
            
            # Find thread links
            thread_links = soup.select(structure.thread_selector)
            
            for link in thread_links[:self.config.forum_max_threads]:
                try:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Get full URL
                    thread_url = urljoin(base_url, href)
                    
                    # Extract title
                    title = link.get_text(strip=True)
                    if not title:
                        continue
                    
                    # Try to get post count from nearby elements
                    post_count = 0
                    try:
                        # Look for post count in parent or siblings
                        parent = link.parent
                        if parent:
                            post_text = parent.get_text()
                            import re
                            post_match = re.search(r'(\d+)\s*(?:posts?|replies?)', post_text, re.IGNORECASE)
                            if post_match:
                                post_count = int(post_match.group(1))
                    except:
                        pass
                    
                    threads.append(ForumThread(
                        url=thread_url,
                        title=title,
                        post_count=post_count
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error processing thread link: {e}")
                    continue
            
            logger.info(f"Found {len(threads)} forum threads")
            return threads
            
        except Exception as e:
            logger.error(f"Error extracting forum threads: {e}")
            return []
    
    async def extract_forum_post_images(self, thread_url: str, structure: ForumStructure) -> List[str]:
        """Extract images from a forum thread."""
        try:
            # Initialize HTTP service if needed
            if self.http_service is None:
                self.http_service = await get_http_service(self.config)
            
            # Fetch thread page
            response = await self.http_service.get(thread_url, as_text=True)
            if not response:
                return []
            
            soup = BeautifulSoup(response, 'html.parser')
            image_urls = []
            
            # Find post containers
            posts = soup.select(structure.post_selector)
            
            for post in posts[:self.config.forum_max_posts_per_thread]:
                # Find images in this post
                images = post.select('img')
                
                for img in images:
                    src = img.get('src') or img.get('data-src')
                    if src:
                        # Convert to absolute URL
                        full_url = urljoin(thread_url, src)
                        image_urls.append(full_url)
            
            logger.info(f"Found {len(image_urls)} images in thread {thread_url}")
            return image_urls
            
        except Exception as e:
            logger.error(f"Error extracting images from thread {thread_url}: {e}")
            return []
    
    async def navigate_forum(self, url: str) -> List[str]:
        """Navigate forum and extract all images."""
        try:
            # Initialize HTTP service if needed
            if self.http_service is None:
                self.http_service = await get_http_service(self.config)
            
            # Fetch main page
            response = await self.http_service.get(url, as_text=True)
            if not response:
                return []
            
            # Detect forum structure
            structure = await self.detect_forum_structure(response, url)
            if not structure:
                logger.warning("Could not detect forum structure")
                return []
            
            # Extract threads
            threads = await self.extract_forum_threads(response, url, structure)
            if not threads:
                logger.warning("No forum threads found")
                return []
            
            # Extract images from threads
            all_images = []
            for thread in threads:
                try:
                    thread_images = await self.extract_forum_post_images(thread.url, structure)
                    all_images.extend(thread_images)
                    
                    # Stop if we have enough images
                    if len(all_images) >= self.config.max_images:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing thread {thread.url}: {e}")
                    continue
            
            logger.info(f"Total forum images extracted: {len(all_images)}")
            return all_images
            
        except Exception as e:
            logger.error(f"Error navigating forum: {e}")
            return []
