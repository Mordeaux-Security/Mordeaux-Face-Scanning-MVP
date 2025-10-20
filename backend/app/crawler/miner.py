"""
Selector Mining Module

Consolidates selector_miner functionality for automatic CSS selector discovery
and site recipe generation. Provides intelligent selector mining with validation.
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import CrawlerConfig

logger = logging.getLogger(__name__)


@dataclass
class CandidateSelector:
    """Represents a candidate CSS selector with metadata."""
    selector: str
    description: str
    evidence: Dict[str, float]
    sample_urls: List[str]
    repetition_count: int
    score: float = 0.0
    kind: str = "video_grid"


@dataclass
class MiningResult:
    """Result of selector mining operation."""
    candidates: List[CandidateSelector]
    status: str  # "OK", "NO_THUMBS_STATIC", "EXTRA_ONLY"
    stats: Dict[str, int]
    mining_time: float
    success: bool
    error: Optional[str] = None


class SelectorMiner:
    """
    Mines CSS selectors for image extraction from web pages.
    
    Analyzes HTML structure to discover patterns and generate site-specific
    recipes for image extraction with validation and scoring.
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Mining configuration
        self.max_candidates = 15  # Increased for better selector discovery
        self.max_samples_per_candidate = 5  # More samples for validation
        self.max_bytes = config.max_image_bytes
        self.timeout_seconds = config.timeout_seconds
        
        # URL validation
        self.host_deny = {
            "doubleclick.net", "exoclick.com", "s.magsrv.com", "afcdn.net",
            "google-analytics.com", "googletagmanager.com", "facebook.com", "twitter.com"
        }
        
        # Gallery detection patterns
        self.gallery_patterns = [
            r'/gallery/', r'/album/', r'/photos/', r'/images/',
            r'/collection/', r'/portfolio/', r'/showcase/', r'/media/',
            r'/pictures/', r'/pics/', r'album=', r'gallery=', r'photos=', r'images='
        ]
        
        # Forum detection patterns
        self.forum_patterns = [
            r'/forum/', r'/forums/', r'/thread/', r'/threads/',
            r'/topic/', r'/topics/', r'/post/', r'/posts/',
            r'/discussion/', r'/discussions/', r'/board/', r'/boards/',
            r'/community/', r'/message/', r'/messages/', r'forum=', r'thread='
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'high': ['high-res', 'highres', 'hd', 'full', 'original', 'large', 'xl', 'xxl'],
            'medium': ['medium', 'med', 'standard', 'normal', 'regular'],
            'low': ['thumb', 'thumbnail', 'small', 'mini', 'preview', 'low-res', 'lowres']
        }
        
        # Mining statistics
        self.stats = {
            'sites_mined': 0,
            'selectors_found': 0,
            'validated_selectors': 0,
            'mining_time': 0.0,
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
        """Initialize HTTP client for mining operations."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.timeout_seconds,
                    pool=self.config.timeout_seconds
                ),
                follow_redirects=True,
                max_redirects=self.config.max_redirects,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
    
    async def _cleanup(self):
        """Cleanup HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def mine_site(self, url: str) -> MiningResult:
        """
        Mine selectors for a site.
        
        Args:
            url: URL to mine selectors for
            
        Returns:
            MiningResult with discovered selectors and metadata
        """
        start_time = time.time()
        
        try:
            # Fetch page content
            page_content = await self._fetch_page(url)
            if not page_content:
                return MiningResult(
                    candidates=[],
                    status="NO_THUMBS_STATIC",
                    stats={},
                    mining_time=time.time() - start_time,
                    success=False,
                    error="Failed to fetch page content"
                )
            
            # Parse HTML
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Detect site type
            is_forum, forum_confidence = self._detect_forum_site(url, soup)
            is_gallery = self._detect_gallery_site(url, soup)
            
            # Mine selectors based on site type
            if is_forum:
                candidates = await self._mine_forum_selectors(soup, url)
            elif is_gallery:
                candidates = await self._mine_gallery_selectors(soup, url)
            else:
                candidates = await self._mine_general_selectors(soup, url)
            
            # Validate candidates
            validated_candidates = await self._validate_candidates(candidates)
            
            # Determine status
            if len(validated_candidates) > 0:
                status = "OK"
            else:
                status = "NO_THUMBS_STATIC"
            
            mining_time = time.time() - start_time
            
            # Update statistics
            self.stats['sites_mined'] += 1
            self.stats['selectors_found'] += len(candidates)
            self.stats['validated_selectors'] += len(validated_candidates)
            self.stats['mining_time'] += mining_time
            
            logger.info(f"Mined {len(validated_candidates)} selectors for {url} in {mining_time:.2f}s")
            
            return MiningResult(
                candidates=validated_candidates,
                status=status,
                stats={
                    'total_candidates': len(candidates),
                    'validated_candidates': len(validated_candidates),
                    'is_forum': is_forum,
                    'is_gallery': is_gallery
                },
                mining_time=mining_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error mining selectors for {url}: {e}")
            self.stats['errors'] += 1
            return MiningResult(
                candidates=[],
                status="NO_THUMBS_STATIC",
                stats={},
                mining_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with retry logic."""
        if not self.http_client:
            await self._initialize_http_client()
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.http_client.get(url)
                response.raise_for_status()
                return response.text
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Page not found: {url}")
                    return None
                elif e.response.status_code >= 500:
                    logger.warning(f"Server error {e.response.status_code} for {url}, attempt {attempt + 1}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                else:
                    logger.warning(f"HTTP error {e.response.status_code} for {url}")
                    return None
                    
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(f"Network error for {url}, attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None
        
        logger.error(f"Failed to fetch {url} after {self.config.max_retries} attempts")
        return None
    
    def _detect_forum_site(self, url: str, soup: BeautifulSoup) -> Tuple[bool, float]:
        """Detect if a site is a forum."""
        confidence = 0.0
        
        # Check URL patterns
        url_lower = url.lower()
        for pattern in self.forum_patterns:
            if re.search(pattern, url_lower):
                confidence += 0.3
                break
        
        # Check HTML structure indicators
        html_text = soup.get_text().lower()
        forum_indicators = [
            'forum', 'thread', 'topic', 'post', 'discussion', 'board', 'community',
            'message', 'reply', 'comment', 'author', 'member', 'user', 'avatar'
        ]
        
        for indicator in forum_indicators:
            if indicator in html_text:
                confidence += 0.05
        
        # Check for forum-specific elements
        forum_selectors = [
            '.forum', '.thread', '.topic', '.post', '.discussion', '.board',
            '.message', '.reply', '.comment', '.thread-list', '.topic-list'
        ]
        
        for selector in forum_selectors:
            if soup.select(selector):
                confidence += 0.1
        
        is_forum = confidence >= 0.3
        return is_forum, min(confidence, 1.0)
    
    def _detect_gallery_site(self, url: str, soup: BeautifulSoup) -> bool:
        """Detect if a site is a gallery/album site."""
        # Check URL patterns
        url_lower = url.lower()
        for pattern in self.gallery_patterns:
            if re.search(pattern, url_lower):
                return True
        
        # Check HTML structure
        gallery_selectors = [
            '.gallery', '.album', '.photos', '.images', '.collection',
            '.portfolio', '.showcase', '.media-grid', '.photo-grid'
        ]
        
        for selector in gallery_selectors:
            if soup.select(selector):
                return True
        
        return False
    
    async def _mine_general_selectors(self, soup: BeautifulSoup, base_url: str) -> List[CandidateSelector]:
        """Mine selectors for general sites."""
        candidates = []
        
        # Common container pattern keywords - look for partial matches
        container_keywords = [
            'thumb', 'thumbnail', 'video', 'item', 'gallery', 'media', 'content', 
            'post', 'block', 'grid', 'card', 'tile', 'photo', 'image', 'picture', 
            'pic', 'img', 'slide', 'slider', 'carousel', 'banner', 'ad', 'advertisement',
            'preview', 'cover', 'featured', 'main', 'hero', 'header', 'avatar', 
            'profile', 'user', 'member', 'author', 'album', 'collection', 'portfolio',
            'showcase', 'feet', 'face', 'head', 'body', 'leg', 'foot'
        ]
        
        # Find all elements with classes that contain our keywords
        all_elements = soup.find_all(attrs={'class': True})
        keyword_matches = {}
        
        for element in all_elements:
            classes = element.get('class', [])
            for cls in classes:
                cls_lower = cls.lower()
                for keyword in container_keywords:
                    if keyword in cls_lower:
                        if keyword not in keyword_matches:
                            keyword_matches[keyword] = []
                        keyword_matches[keyword].append(element)
        
        # Process matches for each keyword
        for keyword, elements in keyword_matches.items():
            if len(elements) >= 3:  # Minimum threshold
                # Group elements by their full class name for better selectors
                class_groups = {}
                for element in elements:
                    classes = ' '.join(element.get('class', []))
                    if classes not in class_groups:
                        class_groups[classes] = []
                    class_groups[classes].append(element)
                
                # Create selectors for each class group
                for classes, group_elements in class_groups.items():
                    if len(group_elements) >= 2:  # At least 2 elements with same classes
                        # Find images within these elements
                        all_images = []
                        for element in group_elements:
                            images = element.find_all(['img', 'source', 'video'])
                            all_images.extend(images)
                        
                        if all_images:
                            # Create selector from the class
                            selector = f".{'.'.join(classes.split())}"
                            
                            candidates.append(CandidateSelector(
                                selector=selector,
                                description=f"Keyword '{keyword}': {len(group_elements)} containers with {selector}",
                                evidence={'repeats': len(group_elements), 'keyword': keyword},
                                sample_urls=self._extract_sample_urls(all_images, base_url),
                                repetition_count=len(group_elements),
                                score=self._score_selector(selector, group_elements, soup),
                                kind="video_grid"
                            ))
        
        return candidates
    
    async def _mine_forum_selectors(self, soup: BeautifulSoup, base_url: str) -> List[CandidateSelector]:
        """Mine selectors for forum sites."""
        candidates = []
        
        # Forum-specific patterns
        forum_patterns = [
            '.post img', '.message img', '.thread img', '.topic img',
            '.reply img', '.comment img', '.attachment img', '.user-avatar img'
        ]
        
        for pattern in forum_patterns:
            elements = soup.select(pattern)
            if len(elements) >= 2:
                selector = pattern
                candidates.append(CandidateSelector(
                    selector=selector,
                    description=f"Forum: {len(elements)} images in {pattern}",
                    evidence={'repeats': len(elements), 'forum_specific': True},
                    sample_urls=self._extract_sample_urls(elements, base_url),
                    repetition_count=len(elements),
                    score=self._score_selector(selector, elements, soup),
                    kind="forum_content"
                ))
        
        return candidates
    
    async def _mine_gallery_selectors(self, soup: BeautifulSoup, base_url: str) -> List[CandidateSelector]:
        """Mine selectors for gallery/album sites."""
        candidates = []
        
        # Gallery-specific patterns
        gallery_patterns = [
            '.gallery img', '.album img', '.photos img', '.images img',
            '.collection img', '.portfolio img', '.showcase img',
            '.media-grid img', '.photo-grid img', '.image-grid img',
            '.gallery-item img', '.photo-item img', '.image-item img',
            '.album-item img', '.collection-item img', '.portfolio-item img'
        ]
        
        for pattern in gallery_patterns:
            elements = soup.select(pattern)
            if len(elements) >= 3:
                selector = pattern
                candidates.append(CandidateSelector(
                    selector=selector,
                    description=f"Gallery: {len(elements)} images in {pattern}",
                    evidence={'repeats': len(elements), 'gallery_specific': True},
                    sample_urls=self._extract_sample_urls(elements, base_url),
                    repetition_count=len(elements),
                    score=self._score_selector(selector, elements, soup),
                    kind="gallery_images"
                ))
        
        return candidates
    
    def _generate_selector(self, element) -> Optional[str]:
        """Generate CSS selector for an element."""
        try:
            if not element or not element.name:
                return None
            
            # Start with tag name
            selector = element.name
            
            # Add class if present and not random
            if element.get('class'):
                classes = element.get('class')
                # Find non-random classes
                non_random_classes = []
                for cls in classes:
                    if len(cls) > 2 and not self._is_random_class(cls):
                        non_random_classes.append(cls)
                
                if non_random_classes:
                    selector += f'.{non_random_classes[0]}'
            
            return selector
            
        except Exception as e:
            logger.debug(f"Error generating selector: {e}")
            return None
    
    def _is_random_class(self, cls: str) -> bool:
        """Check if a class name appears to be random/generated."""
        if len(cls) < 3:
            return True
        
        random_patterns = [
            r'^[a-f0-9]{8,}$',  # Hex strings
            r'^[0-9]+$',        # Pure numbers
            r'[A-Z]{3,}',       # Multiple caps
            r'_[a-f0-9]{6,}_',  # Underscore hex patterns
            r'^[a-z]{1,2}[0-9]{4,}$',  # Short letters + long numbers
            r'^[0-9]{4,}[a-z]{1,2}$',  # Long numbers + short letters
        ]
        
        return any(re.search(pattern, cls) for pattern in random_patterns)
    
    def _extract_sample_urls(self, elements: List, base_url: str) -> List[str]:
        """Extract sample URLs from elements."""
        urls = []
        seen_urls = set()
        
        for element in elements:
            url = self._resolve_image_url(element, base_url)
            if url and url not in seen_urls and len(urls) < 12:
                urls.append(url)
                seen_urls.add(url)
        
        return urls
    
    def _resolve_image_url(self, element, base_url: str) -> Optional[str]:
        """Resolve image URL from element."""
        attrs_priority = ['data-src', 'data-srcset', 'srcset', 'src']
        
        for attr in attrs_priority:
            value = element.get(attr)
            if value:
                if attr == 'srcset':
                    url = self._parse_srcset(value)
                else:
                    url = value
                
                if url:
                    return urljoin(base_url, url)
        
        return None
    
    def _parse_srcset(self, srcset: str) -> Optional[str]:
        """Parse srcset and return highest resolution URL."""
        try:
            candidates = []
            for candidate in srcset.split(','):
                candidate = candidate.strip()
                if not candidate:
                    continue
                
                parts = candidate.split()
                if len(parts) < 2:
                    continue
                
                url = parts[0]
                descriptor = parts[1]
                
                # Parse descriptor
                width = 0
                if descriptor.endswith('x'):
                    try:
                        density = float(descriptor[:-1])
                        width = int(320 * density)
                    except ValueError:
                        continue
                elif descriptor.endswith('w'):
                    try:
                        width = int(descriptor[:-1])
                    except ValueError:
                        continue
                else:
                    width = 320
                
                candidates.append((url, width))
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
                
        except Exception as e:
            logger.debug(f"Error parsing srcset: {e}")
        
        return None
    
    def _score_selector(self, selector: str, elements: List, soup: BeautifulSoup) -> float:
        """Advanced scoring algorithm for selectors based on multiple criteria."""
        try:
            score = 0.0
            
            # 1. Element count scoring (logarithmic scale)
            element_count = len(elements)
            if element_count >= 3:
                score += min(1.0, 0.3 + (element_count - 3) * 0.05)
            else:
                score += element_count * 0.1
            
            # 2. Selector specificity scoring
            specificity_score = self._calculate_specificity(selector)
            score += specificity_score * 0.2
            
            # 3. Pattern recognition scoring
            pattern_score = self._score_patterns(selector)
            score += pattern_score * 0.25
            
            # 4. Content analysis scoring
            content_score = self._analyze_content_quality(elements, soup)
            score += content_score * 0.15
            
            # 5. URL validation scoring
            url_score = self._validate_extracted_urls(elements)
            score += url_score * 0.1
            
            # 6. Consistency scoring
            consistency_score = self._check_selector_consistency(selector, elements)
            score += consistency_score * 0.1
            
            # 7. Penalty for overly complex selectors
            if selector.count('.') > 4 or selector.count('#') > 2:
                score -= 0.2
            
            # 8. Bonus for semantic selectors
            semantic_bonus = self._calculate_semantic_bonus(selector)
            score += semantic_bonus
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error scoring selector {selector}: {e}")
            return 0.0
    
    def _calculate_specificity(self, selector: str) -> float:
        """Calculate CSS selector specificity score."""
        try:
            # Count different types of selectors
            id_count = selector.count('#')
            class_count = selector.count('.')
            tag_count = len([s for s in selector.split() if s and not s.startswith(('.', '#', '['))])
            attr_count = selector.count('[')
            
            # Calculate specificity (higher is more specific)
            specificity = (id_count * 100) + (class_count * 10) + tag_count + (attr_count * 5)
            
            # Normalize to 0-1 scale (prefer moderate specificity)
            if specificity == 0:
                return 0.0
            elif specificity <= 20:
                return 0.8  # Good specificity
            elif specificity <= 50:
                return 0.6  # Moderate specificity
            else:
                return 0.3  # Too specific
            
        except Exception:
            return 0.5
    
    def _score_patterns(self, selector: str) -> float:
        """Score selector based on recognized patterns."""
        score = 0.0
        selector_lower = selector.lower()
        
        # High-value patterns
        high_value_patterns = {
            'thumb': 0.4, 'thumbnail': 0.4, 'gallery': 0.3, 'album': 0.3,
            'photo': 0.3, 'image': 0.2, 'media': 0.2, 'content': 0.2,
            'item': 0.2, 'card': 0.2, 'tile': 0.2, 'grid': 0.2
        }
        
        for pattern, value in high_value_patterns.items():
            if pattern in selector_lower:
                score += value
        
        # Medium-value patterns
        medium_value_patterns = {
            'post': 0.1, 'entry': 0.1, 'article': 0.1, 'section': 0.1,
            'container': 0.1, 'wrapper': 0.1, 'box': 0.1
        }
        
        for pattern, value in medium_value_patterns.items():
            if pattern in selector_lower:
                score += value
        
        # Penalty patterns
        penalty_patterns = {
            'ad': -0.3, 'advertisement': -0.3, 'banner': -0.2, 'sidebar': -0.2,
            'footer': -0.2, 'header': -0.1, 'nav': -0.1, 'menu': -0.1
        }
        
        for pattern, penalty in penalty_patterns.items():
            if pattern in selector_lower:
                score += penalty
        
        return max(0.0, min(1.0, score))
    
    def _analyze_content_quality(self, elements: List, soup: BeautifulSoup) -> float:
        """Analyze the quality of content in elements."""
        try:
            if not elements:
                return 0.0
            
            score = 0.0
            total_images = 0
            valid_images = 0
            
            for element in elements:
                # Count images within elements
                images = element.find_all(['img', 'source', 'video'])
                total_images += len(images)
                
                for img in images:
                    # Check for valid image attributes
                    if any(attr in img.attrs for attr in ['src', 'data-src', 'data-srcset']):
                        valid_images += 1
                        
                        # Check image dimensions
                        width = img.get('width') or img.get('data-width')
                        height = img.get('height') or img.get('data-height')
                        
                        if width and height:
                            try:
                                w, h = int(width), int(height)
                                if w >= 100 and h >= 100:  # Minimum size
                                    score += 0.1
                            except (ValueError, TypeError):
                                pass
            
            # Ratio of valid images
            if total_images > 0:
                valid_ratio = valid_images / total_images
                score += valid_ratio * 0.5
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error analyzing content quality: {e}")
            return 0.0
    
    def _validate_extracted_urls(self, elements: List) -> float:
        """Validate URLs extracted from elements."""
        try:
            if not elements:
                return 0.0
            
            total_urls = 0
            valid_urls = 0
            
            for element in elements:
                images = element.find_all(['img', 'source', 'video'])
                for img in images:
                    for attr in ['src', 'data-src', 'data-srcset']:
                        url = img.get(attr)
                        if url:
                            total_urls += 1
                            if self._is_valid_image_url(url):
                                valid_urls += 1
            
            if total_urls == 0:
                return 0.0
            
            return valid_urls / total_urls
            
        except Exception as e:
            logger.error(f"Error validating URLs: {e}")
            return 0.0
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            
            # Must have valid scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Must have hostname
            if not parsed.hostname:
                return False
            
            # Check for image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
            path_lower = parsed.path.lower()
            
            if any(path_lower.endswith(ext) for ext in image_extensions):
                return True
            
            # Check for image indicators in path
            image_indicators = ['image', 'img', 'photo', 'pic', 'thumb', 'gallery']
            if any(indicator in path_lower for indicator in image_indicators):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_selector_consistency(self, selector: str, elements: List) -> float:
        """Check consistency of selector results."""
        try:
            if len(elements) < 2:
                return 0.5  # Neutral for single elements
            
            # Check if elements have similar structure
            structures = []
            for element in elements:
                structure = self._get_element_structure(element)
                structures.append(structure)
            
            # Calculate consistency
            if len(set(structures)) == 1:
                return 1.0  # Perfect consistency
            elif len(set(structures)) <= len(structures) * 0.7:
                return 0.7  # Good consistency
            else:
                return 0.3  # Poor consistency
            
        except Exception as e:
            logger.error(f"Error checking selector consistency: {e}")
            return 0.5
    
    def _get_element_structure(self, element) -> str:
        """Get a simplified structure representation of an element."""
        try:
            # Get tag name and key attributes
            tag = element.name or 'unknown'
            classes = ' '.join(sorted(element.get('class', [])))
            id_attr = element.get('id', '')
            
            return f"{tag}:{classes}:{id_attr}"
        except Exception:
            return "unknown"
    
    def _calculate_semantic_bonus(self, selector: str) -> float:
        """Calculate bonus for semantically meaningful selectors."""
        try:
            bonus = 0.0
            selector_lower = selector.lower()
            
            # Semantic patterns that indicate good selectors
            semantic_patterns = {
                'gallery-item': 0.2, 'photo-item': 0.2, 'image-item': 0.2,
                'thumb-item': 0.2, 'media-item': 0.15, 'content-item': 0.15,
                'post-image': 0.15, 'article-image': 0.15, 'entry-image': 0.15
            }
            
            for pattern, value in semantic_patterns.items():
                if pattern in selector_lower:
                    bonus += value
            
            return min(0.3, bonus)  # Cap at 0.3
            
        except Exception:
            return 0.0
    
    async def _validate_candidates(self, candidates: List[CandidateSelector]) -> List[CandidateSelector]:
        """Validate candidates by checking their sample URLs."""
        validated_candidates = []
        
        for candidate in candidates:
            valid_urls = []
            
            # Validate sample URLs
            for url in candidate.sample_urls[:self.max_samples_per_candidate]:
                if await self._validate_image_url(url):
                    valid_urls.append(url)
            
            # Only keep candidates with at least one valid URL
            if valid_urls:
                candidate.sample_urls = valid_urls
                candidate.repetition_count = len(valid_urls)
                validated_candidates.append(candidate)
        
        # Sort by score and limit
        validated_candidates.sort(key=lambda x: x.score, reverse=True)
        return validated_candidates[:self.max_candidates]
    
    async def _validate_image_url(self, url: str) -> bool:
        """Validate image URL."""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme.lower() not in {'http', 'https'}:
                return False
            
            # Check host
            if parsed.netloc.lower() in {h.lower() for h in self.host_deny}:
                return False
            
            # Check for SVG files
            if parsed.path.lower().endswith('.svg'):
                return False
            
            # Quick HEAD request to validate
            if self.http_client:
                try:
                    response = await self.http_client.head(url, timeout=5.0)
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        return content_type.startswith('image/')
                except:
                    pass
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating URL {url}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mining statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset mining statistics."""
        self.stats = {
            'sites_mined': 0,
            'selectors_found': 0,
            'validated_selectors': 0,
            'mining_time': 0.0,
            'errors': 0
        }
