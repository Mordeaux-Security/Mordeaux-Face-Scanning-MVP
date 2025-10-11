"""
Selector Miner Core - Phase 2

A deterministic selector-miner that analyzes HTML content to generate
candidate CSS selectors for image extraction with evidence-based scoring.
"""

import re
import logging
import asyncio
import os
import yaml
import json
from typing import List, Dict, Set, Tuple, Optional, NamedTuple, Union
from urllib.parse import urljoin, urlparse
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from .redirect_utils import create_safe_client, head_with_redirects

# Optional Playwright import for JavaScript rendering
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

logger = logging.getLogger(__name__)

# Default maximum image size (10MB)
DEFAULT_MAX_BYTES = int(os.getenv("MINER_MAX_IMAGE_BYTES", 10 * 1024 * 1024))


@dataclass
class CandidateSelector:
    """Represents a candidate CSS selector with metadata and evidence."""
    selector: str
    description: str
    evidence: Dict[str, float]
    sample_urls: List[str]
    repetition_count: int
    score: float = 0.0


@dataclass
class ImageNode:
    """Represents an image node found in HTML."""
    tag: Tag
    url: str
    selector_path: str
    attributes: Dict[str, str]
    context: Dict[str, any]


@dataclass
class ValidationResult:
    """Result of URL validation."""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RecipeCandidate:
    """A candidate recipe for a domain."""
    domain: str
    selectors: List[Dict[str, str]]
    attributes_priority: List[str]
    extra_sources: List[str]
    method: str
    confidence: float
    sample_urls: List[str]
    validation_results: List[ValidationResult]


class SelectorMiner:
    """
    Deterministic selector miner for extracting image selectors from HTML.
    
    Analyzes HTML content to generate stable, minimal CSS selectors that
    target image elements with high confidence based on evidence patterns.
    Includes validation loop and YAML recipe generation.
    """
    
    # Duration regex pattern for video content detection
    DURATION_PATTERN = re.compile(r'\b\d{1,2}:\d{2}\b')
    
    # Video-related URL patterns
    VIDEO_URL_PATTERNS = [
        re.compile(r'/(video|watch|v|id)/', re.IGNORECASE),
        re.compile(r'embed/', re.IGNORECASE),
        re.compile(r'player/', re.IGNORECASE)
    ]
    
    # Positive class name hints for image content
    POSITIVE_CLASS_HINTS = {
        'thumb', 'thumbnail', 'preview', 'poster', 'image', 'img', 
        'pic', 'picture', 'cover', 'artwork', 'screenshot', 'still',
        'frame', 'snapshot', 'gallery', 'media', 'visual', 'display'
    }
    
    # Negative class name hints (likely not target images)
    NEGATIVE_CLASS_HINTS = {
        'icon', 'logo', 'avatar', 'profile', 'social', 'share', 'button',
        'nav', 'menu', 'header', 'footer', 'sidebar', 'banner', 'ad',
        'sponsor', 'sponsored', 'advertisement', 'promo', 'promotion'
    }
    
    def __init__(self, base_url: str = "", max_bytes: Optional[int] = None):
        self.base_url = base_url
        self.image_nodes: List[ImageNode] = []
        self.candidate_selectors: List[CandidateSelector] = []
        
        # Anti-malware configuration
        self.MALICIOUS_SCHEMES = {'javascript', 'data', 'file', 'ftp'}
        self.SAFE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        self.MAX_CONTENT_LENGTH = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
        
    def mine_selectors(self, html_content: str) -> List[CandidateSelector]:
        """
        Main entry point for selector mining.
        
        Args:
            html_content: Raw HTML content to analyze
            
        Returns:
            List of scored candidate selectors
        """
        logger.info("Starting selector mining process")
        
        # Parse HTML and extract image nodes
        soup = BeautifulSoup(html_content, 'html.parser')
        self.image_nodes = self._extract_image_nodes(soup)
        
        if not self.image_nodes:
            logger.warning("No image nodes found in HTML content")
            return []
            
        logger.info(f"Found {len(self.image_nodes)} image nodes")
        
        # Generate candidate selectors
        self.candidate_selectors = self._generate_candidate_selectors()
        
        # Score candidates
        self._score_candidates()
        
        # Filter and rank candidates
        top_candidates = self._filter_and_rank_candidates()
        
        logger.info(f"Generated {len(top_candidates)} top candidate selectors")
        return top_candidates
    
    def _extract_image_nodes(self, soup: BeautifulSoup) -> List[ImageNode]:
        """Extract all image-related nodes from the HTML."""
        image_nodes = []
        
        # Find all image elements
        for img in soup.find_all(['img', 'source', 'video']):
            node_data = self._analyze_image_node(img)
            if node_data:
                image_nodes.append(node_data)
        
        # Find elements with background-image CSS
        for element in soup.find_all(style=True):
            if 'background-image' in str(element.get('style', '')):
                node_data = self._analyze_background_image_node(element)
                if node_data:
                    image_nodes.append(node_data)
        
        # Find meta og:image tags
        for meta in soup.find_all('meta', property='og:image'):
            node_data = self._analyze_meta_image_node(meta)
            if node_data:
                image_nodes.append(node_data)
        
        # Find link tags with image_src
        for link in soup.find_all('link', rel='image_src'):
            node_data = self._analyze_link_image_node(link)
            if node_data:
                image_nodes.append(node_data)
        
        # Find JSON-LD script blocks with thumbnailUrl and image
        for script in soup.find_all('script', type='application/ld+json'):
            json_nodes = self._analyze_json_ld_node(script, self.base_url)
            image_nodes.extend(json_nodes)
        
        return image_nodes
    
    def _analyze_image_node(self, tag: Tag) -> Optional[ImageNode]:
        """Analyze a single image node and extract relevant information."""
        # Extract image URL
        url = self._extract_image_url(tag)
        if not url or not self._is_valid_image_url(url):
            return None
        
        # Generate selector path
        selector_path = self._generate_selector_path(tag)
        
        # Extract attributes
        attributes = dict(tag.attrs) if tag.attrs else {}
        
        # Analyze context
        context = self._analyze_node_context(tag)
        
        return ImageNode(
            tag=tag,
            url=url,
            selector_path=selector_path,
            attributes=attributes,
            context=context
        )
    
    def _extract_image_url(self, tag: Tag) -> Optional[str]:
        """Extract image URL from various attributes, including srcset parsing."""
        # Check for srcset first (highest priority for responsive images)
        srcset_url = self._extract_srcset_url(tag)
        if srcset_url:
            return srcset_url
        
        # Check other URL attributes
        url_attributes = ['src', 'data-src', 'data-lazy-src', 'data-original', 
                         'data-large', 'data-medium', 'data-thumb', 'poster']
        
        for attr in url_attributes:
            if tag.get(attr):
                url = tag.get(attr)
                if url.startswith('//'):
                    url = 'https:' + url
                elif url.startswith('/'):
                    url = urljoin(self.base_url, url)
                return url
        
        return None
    
    def _extract_srcset_url(self, tag: Tag) -> Optional[str]:
        """Extract the largest candidate URL from srcset attribute."""
        srcset = tag.get('srcset')
        if not srcset:
            return None
        
        try:
            # Parse srcset: "url1 1x, url2 2x, url3 320w, url4 640w"
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
                
                # Normalize URL
                if url.startswith('//'):
                    url = 'https:' + url
                elif url.startswith('/'):
                    url = urljoin(self.base_url, url)
                
                # Parse descriptor (1x, 2x, 320w, etc.)
                width = 0
                if descriptor.endswith('x'):
                    # Density descriptor (1x, 2x)
                    try:
                        density = float(descriptor[:-1])
                        # For density descriptors, assume base width of 320px
                        width = int(320 * density)
                    except ValueError:
                        continue
                elif descriptor.endswith('w'):
                    # Width descriptor (320w, 640w)
                    try:
                        width = int(descriptor[:-1])
                    except ValueError:
                        continue
                else:
                    # Default descriptor (1x)
                    width = 320
                
                candidates.append((url, width))
            
            if not candidates:
                return None
            
            # Return URL with largest width
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
            
        except Exception as e:
            logger.debug(f"Failed to parse srcset: {e}")
            return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL (basic validation)."""
        if not url or len(url) < 3:
            return False
            
        # Check for malicious schemes
        parsed = urlparse(url)
        if parsed.scheme in {'javascript', 'data', 'file', 'ftp'}:
            return False
            
        # Explicitly reject SVG URLs
        path_lower = parsed.path.lower()
        if path_lower.endswith('.svg'):
            return False
            
        # Check for image extensions or common image patterns
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        
        if any(path_lower.endswith(ext) for ext in image_extensions):
            return True
            
        # Check for common image URL patterns
        image_patterns = ['/image/', '/img/', '/photo/', '/picture/', '/thumb/']
        if any(pattern in path_lower for pattern in image_patterns):
            return True
            
        # Allow relative URLs that look like images
        if not parsed.scheme and not parsed.netloc and '/' in url:
            return True
            
        return False  # Be more strict about URL validation
    
    def _generate_selector_path(self, tag: Tag) -> str:
        """Generate a minimal, stable CSS selector for the tag."""
        path_parts = []
        current = tag
        
        # Build path from element up to root (max 4 levels)
        level = 0
        while current and current.name and level < 4:
            # Skip document and html elements to avoid overly long selectors
            if current.name in ['[document]', 'html']:
                current = current.parent
                continue
                
            selector_part = current.name
            
            # Prefer classes over IDs for stability
            if current.get('class'):
                classes = [cls for cls in current.get('class') 
                          if not self._is_random_token(cls)]
                if classes:
                    # Use most stable class (avoid random tokens)
                    stable_class = self._find_most_stable_class(classes)
                    selector_part += f'.{stable_class}'
            
            path_parts.insert(0, selector_part)
            current = current.parent
            level += 1
            
            # Stop if we hit a stable container
            if current and current.get('id') and self._is_stable_id(current.get('id')):
                break
        
        return ' > '.join(path_parts)
    
    def _is_random_token(self, token: str) -> bool:
        """Check if a token appears to be randomly generated."""
        if len(token) < 3:
            return False
            
        # Check for random-looking patterns
        random_patterns = [
            r'^[a-f0-9]{8,}$',  # Hex strings
            r'^[0-9]+$',        # Pure numbers
            r'[A-Z]{3,}',       # Multiple caps (often random)
            r'_[a-f0-9]{6,}_',  # Underscore hex patterns
        ]
        
        for pattern in random_patterns:
            if re.search(pattern, token):
                return True
                
        return False
    
    def _find_most_stable_class(self, classes: List[str]) -> str:
        """Find the most stable class from a list (avoiding random tokens)."""
        stable_classes = [cls for cls in classes if not self._is_random_token(cls)]
        if stable_classes:
            return stable_classes[0]  # Return first stable class
        return classes[0]  # Fallback to first class
    
    def _is_stable_id(self, id_value: str) -> bool:
        """Check if an ID appears to be stable (not randomly generated)."""
        if not id_value or len(id_value) < 3:
            return False
            
        # IDs that look stable (contain meaningful words)
        stable_indicators = ['main', 'content', 'header', 'footer', 'sidebar', 'nav']
        id_lower = id_value.lower()
        
        return any(indicator in id_lower for indicator in stable_indicators)
    
    def _analyze_node_context(self, tag: Tag) -> Dict[str, any]:
        """Analyze the context around an image node for evidence."""
        context = {
            'duration_text': False,
            'video_url_ancestor': False,
            'positive_class_hints': 0,
            'negative_class_hints': 0,
            'srcset_richness': 0,
            'nearby_links': []
        }
        
        # Check for duration text in nearby elements
        context['duration_text'] = self._has_duration_text_nearby(tag)
        
        # Check for video-related ancestor links
        context['video_url_ancestor'] = self._has_video_url_ancestor(tag)
        
        # Count positive/negative class hints
        all_classes = self._get_all_classes_in_hierarchy(tag)
        for class_name in all_classes:
            if any(hint in class_name.lower() for hint in self.POSITIVE_CLASS_HINTS):
                context['positive_class_hints'] += 1
            if any(hint in class_name.lower() for hint in self.NEGATIVE_CLASS_HINTS):
                context['negative_class_hints'] += 1
        
        # Check srcset richness
        if tag.get('srcset'):
            context['srcset_richness'] = len(tag.get('srcset', '').split(','))
        
        # Find nearby links
        context['nearby_links'] = self._find_nearby_links(tag)
        
        return context
    
    def _has_duration_text_nearby(self, tag: Tag) -> bool:
        """Check if duration text (MM:SS) is present in nearby elements."""
        # Check siblings and parent elements
        for element in [tag] + list(tag.parents)[:3]:
            # Check element's own text content
            if element.string and self.DURATION_PATTERN.search(element.string):
                return True
            
            # Check all text content in element
            if element.get_text() and self.DURATION_PATTERN.search(element.get_text()):
                return True
            
            # Check siblings
            if element.parent:
                for sibling in element.parent.children:
                    if hasattr(sibling, 'get_text'):
                        if sibling.get_text() and self.DURATION_PATTERN.search(sibling.get_text()):
                            return True
                    elif isinstance(sibling, NavigableString):
                        if self.DURATION_PATTERN.search(str(sibling)):
                            return True
        
        return False
    
    def _has_video_url_ancestor(self, tag: Tag) -> bool:
        """Check if tag has video-related URL in ancestor links."""
        for ancestor in tag.parents:
            if ancestor.name == 'a' and ancestor.get('href'):
                href = ancestor.get('href')
                for pattern in self.VIDEO_URL_PATTERNS:
                    if pattern.search(href):
                        return True
        return False
    
    def _get_all_classes_in_hierarchy(self, tag: Tag) -> List[str]:
        """Get all class names in the tag's hierarchy."""
        classes = []
        current = tag
        level = 0
        
        while current and level < 3:  # Check up to 3 levels
            if current.get('class'):
                classes.extend(current.get('class'))
            current = current.parent
            level += 1
            
        return classes
    
    def _find_nearby_links(self, tag: Tag) -> List[str]:
        """Find nearby link URLs for context analysis."""
        links = []
        
        # Check parent and sibling links
        for element in [tag] + list(tag.parents)[:2]:
            if element.parent:
                for sibling in element.parent.find_all('a', href=True):
                    if sibling != element:
                        links.append(sibling.get('href'))
        
        return links[:5]  # Limit to 5 nearby links
    
    def _analyze_background_image_node(self, element: Tag) -> Optional[ImageNode]:
        """Analyze elements with background-image CSS."""
        style = element.get('style', '')
        # Extract background-image URL from CSS
        bg_match = re.search(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
        if not bg_match:
            return None
            
        url = bg_match.group(1)
        
        # Normalize URL (handle relative URLs)
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            url = urljoin(self.base_url, url)
        
        if not self._is_valid_image_url(url):
            return None
            
        selector_path = self._generate_selector_path(element)
        attributes = dict(element.attrs) if element.attrs else {}
        context = self._analyze_node_context(element)
        
        return ImageNode(
            tag=element,
            url=url,
            selector_path=selector_path,
            attributes=attributes,
            context=context
        )
    
    def _analyze_meta_image_node(self, meta_tag: Tag) -> Optional[ImageNode]:
        """Analyze meta og:image tags."""
        content = meta_tag.get('content')
        if not content or not self._is_valid_image_url(content):
            return None
            
        selector_path = self._generate_selector_path(meta_tag)
        attributes = dict(meta_tag.attrs) if meta_tag.attrs else {}
        context = self._analyze_node_context(meta_tag)
        
        return ImageNode(
            tag=meta_tag,
            url=content,
            selector_path=selector_path,
            attributes=attributes,
            context=context
        )
    
    def _analyze_link_image_node(self, link_tag: Tag) -> Optional[ImageNode]:
        """Analyze link tags with image_src."""
        href = link_tag.get('href')
        if not href or not self._is_valid_image_url(href):
            return None
            
        selector_path = self._generate_selector_path(link_tag)
        attributes = dict(link_tag.attrs) if link_tag.attrs else {}
        context = self._analyze_node_context(link_tag)
        
        return ImageNode(
            tag=link_tag,
            url=href,
            selector_path=selector_path,
            attributes=attributes,
            context=context
        )
    
    def _analyze_json_ld_node(self, script_tag: Tag, base_url: str) -> List[ImageNode]:
        """Analyze JSON-LD script blocks for thumbnailUrl and image fields."""
        json_nodes = []
        
        try:
            # Get the JSON content from the script tag
            json_text = script_tag.string
            if not json_text:
                return json_nodes
            
            # Parse the JSON content
            json_data = json.loads(json_text.strip())
            
            # Handle both single objects and arrays of objects
            json_objects = json_data if isinstance(json_data, list) else [json_data]
            
            for i, obj in enumerate(json_objects):
                if not isinstance(obj, dict):
                    continue
                
                # Extract thumbnailUrl and image fields
                image_urls = self._extract_json_ld_image_urls(obj, base_url)
                
                for j, (url, field_name) in enumerate(image_urls):
                    if not self._is_valid_image_url(url):
                        continue
                    
                    # Create synthetic selector path for JSON-LD
                    selector_path = f"script[type='application/ld+json']::jsonpath($.{field_name})"
                    
                    # Create synthetic attributes
                    attributes = {
                        'type': 'application/ld+json',
                        'field': field_name,
                        'json_index': i
                    }
                    
                    # Create context
                    context = self._analyze_node_context(script_tag)
                    context['json_ld_field'] = field_name
                    context['json_ld_index'] = i
                    
                    # Create ImageNode
                    node = ImageNode(
                        tag=script_tag,
                        url=url,
                        selector_path=selector_path,
                        attributes=attributes,
                        context=context
                    )
                    json_nodes.append(node)
                    
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse JSON-LD: {e}")
        
        return json_nodes
    
    def _extract_json_ld_image_urls(self, json_obj: Dict, base_url: str) -> List[Tuple[str, str]]:
        """Extract image URLs from JSON-LD object, returning (url, field_name) tuples."""
        image_urls = []
        
        # Extract thumbnailUrl
        if 'thumbnailUrl' in json_obj:
            thumbnail_urls = self._normalize_json_ld_url_field(json_obj['thumbnailUrl'], base_url)
            for url in thumbnail_urls:
                image_urls.append((url, 'thumbnailUrl'))
        
        # Extract image field
        if 'image' in json_obj:
            image_urls_found = self._normalize_json_ld_url_field(json_obj['image'], base_url)
            for url in image_urls_found:
                image_urls.append((url, 'image'))
        
        return image_urls
    
    def _normalize_json_ld_url_field(self, field_value: Union[str, List, Dict], base_url: str) -> List[str]:
        """Normalize JSON-LD URL field that can be string, list, or object with url property."""
        urls = []
        
        if isinstance(field_value, str):
            # Simple string URL
            absolute_url = urljoin(base_url, field_value)
            urls.append(absolute_url)
            
        elif isinstance(field_value, list):
            # Array of URLs or objects
            for item in field_value:
                if isinstance(item, str):
                    absolute_url = urljoin(base_url, item)
                    urls.append(absolute_url)
                elif isinstance(item, dict) and 'url' in item:
                    absolute_url = urljoin(base_url, item['url'])
                    urls.append(absolute_url)
                    
        elif isinstance(field_value, dict):
            # Single object with url property
            if 'url' in field_value:
                absolute_url = urljoin(base_url, field_value['url'])
                urls.append(absolute_url)
        
        return urls
    
    def _generate_candidate_selectors(self) -> List[CandidateSelector]:
        """Generate candidate selectors based on image node analysis."""
        candidates = []
        selector_groups = defaultdict(list)
        
        # Group nodes by their selector paths
        for node in self.image_nodes:
            selector_groups[node.selector_path].append(node)
        
        # Create candidates for each unique selector
        for selector, nodes in selector_groups.items():
            candidate = self._create_candidate_selector(selector, nodes)
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _create_candidate_selector(self, selector: str, nodes: List[ImageNode]) -> Optional[CandidateSelector]:
        """Create a candidate selector with evidence from nodes."""
        if not nodes:
            return None
        
        # Calculate evidence
        evidence = self._calculate_evidence(nodes)
        
        # Generate description
        description = self._generate_description(selector, nodes)
        
        # Extract sample URLs (deduplicated, max 12)
        sample_urls = []
        seen_urls = set()
        for node in nodes:
            if node.url not in seen_urls and len(sample_urls) < 12:
                sample_urls.append(node.url)
                seen_urls.add(node.url)
        
        return CandidateSelector(
            selector=selector,
            description=description,
            evidence=evidence,
            sample_urls=sample_urls,
            repetition_count=len(nodes)
        )
    
    def _calculate_evidence(self, nodes: List[ImageNode]) -> Dict[str, float]:
        """Calculate evidence scores for a group of nodes."""
        evidence = {
            'repetition_score': 0.0,
            'duration_score': 0.0,
            'video_url_score': 0.0,
            'class_hint_score': 0.0,
            'srcset_score': 0.0,
            'url_quality_score': 0.0
        }
        
        if not nodes:
            return evidence
        
        # Repetition score (higher is better, up to 1.0)
        evidence['repetition_score'] = min(len(nodes) / 10.0, 1.0)
        
        # Duration text score
        duration_count = sum(1 for node in nodes if node.context['duration_text'])
        evidence['duration_score'] = duration_count / len(nodes)
        
        # Video URL score
        video_count = sum(1 for node in nodes if node.context['video_url_ancestor'])
        evidence['video_url_score'] = video_count / len(nodes)
        
        # Class hint score (positive - negative)
        total_positive = sum(node.context['positive_class_hints'] for node in nodes)
        total_negative = sum(node.context['negative_class_hints'] for node in nodes)
        evidence['class_hint_score'] = max(0, (total_positive - total_negative) / len(nodes) / 2.0)
        
        # Srcset richness score
        srcset_count = sum(1 for node in nodes if node.context['srcset_richness'] > 1)
        evidence['srcset_score'] = srcset_count / len(nodes)
        
        # URL quality score (basic heuristic)
        quality_count = sum(1 for node in nodes if self._assess_url_quality(node.url))
        evidence['url_quality_score'] = quality_count / len(nodes)
        
        return evidence
    
    def _assess_url_quality(self, url: str) -> bool:
        """Assess if URL appears to be high-quality image URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check for quality indicators
        quality_indicators = [
            'thumb', 'thumbnail', 'preview', 'medium', 'large', 'high',
            'hd', '720', '1080', '1920', '2048'
        ]
        
        # Also consider image extensions as quality indicators
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        
        return (any(indicator in path for indicator in quality_indicators) or
                any(path.endswith(ext) for ext in image_extensions))
    
    def _generate_description(self, selector: str, nodes: List[ImageNode]) -> str:
        """Generate a human-readable description for the selector."""
        node_count = len(nodes)
        
        # Analyze common attributes
        common_attrs = self._find_common_attributes(nodes)
        
        desc_parts = [f"{node_count} images"]
        
        if common_attrs:
            desc_parts.append(f"with {', '.join(common_attrs)}")
        
        # Add context hints
        if any(node.context['duration_text'] for node in nodes):
            desc_parts.append("near duration text")
        
        if any(node.context['video_url_ancestor'] for node in nodes):
            desc_parts.append("in video links")
        
        return f"{selector} - {' '.join(desc_parts)}"
    
    def _find_common_attributes(self, nodes: List[ImageNode]) -> List[str]:
        """Find common attributes across nodes."""
        attr_counts = defaultdict(int)
        
        for node in nodes:
            for attr, value in node.attributes.items():
                if attr not in ['class', 'id']:  # Skip these as they're in selector
                    attr_counts[f"{attr}={value}"] += 1
        
        # Return attributes present in at least 30% of nodes
        threshold = len(nodes) * 0.3
        return [attr for attr, count in attr_counts.items() if count >= threshold]
    
    def _score_candidates(self):
        """Score all candidate selectors based on evidence."""
        for candidate in self.candidate_selectors:
            candidate.score = self._calculate_candidate_score(candidate)
    
    def _calculate_candidate_score(self, candidate: CandidateSelector) -> float:
        """Calculate final score for a candidate selector."""
        evidence = candidate.evidence
        
        # Weighted scoring
        weights = {
            'repetition_score': 0.3,    # Most important - how many images found
            'duration_score': 0.2,      # Video content indicator
            'video_url_score': 0.15,    # Video context
            'class_hint_score': 0.15,   # Semantic class names
            'srcset_score': 0.1,        # Responsive images
            'url_quality_score': 0.1    # URL quality
        }
        
        score = sum(evidence[key] * weight for key, weight in weights.items())
        
        # Bonus for higher repetition count
        if candidate.repetition_count > 5:
            score += 0.1
        
        # Penalty for very short selectors (might be too broad)
        if len(candidate.selector.split()) == 1:
            score -= 0.1
            
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _filter_and_rank_candidates(self) -> List[CandidateSelector]:
        """Filter and rank candidates by score."""
        # Filter out very low-scoring candidates (lowered threshold)
        filtered = [c for c in self.candidate_selectors if c.score > 0.05]
        
        # Sort by score (descending)
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to top candidates
        return filtered[:10]
    
    async def validate_candidate_urls(self, candidate: CandidateSelector, max_samples: int = 3) -> List[ValidationResult]:
        """
        Validate sample URLs for a candidate selector using HEAD then GET requests.
        
        Args:
            candidate: Candidate selector to validate
            max_samples: Maximum number of sample URLs to validate
            
        Returns:
            List of validation results
        """
        validation_results = []
        sample_urls = candidate.sample_urls[:max_samples]
        
        async with create_safe_client(timeout=10.0) as client:
            for url in sample_urls:
                result = await self._validate_single_url(client, url)
                validation_results.append(result)
                
        return validation_results
    
    async def _validate_single_url(self, client: httpx.AsyncClient, url: str) -> ValidationResult:
        """
        Validate a single URL with anti-malware guards.
        
        Args:
            client: HTTP client instance
            url: URL to validate
            
        Returns:
            Validation result
        """
        try:
            # Phase-0 guards: Check URL scheme and basic safety
            if not self._is_safe_url(url):
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error="Unsafe URL scheme or suspicious pattern"
                )
            
            # HEAD request first (lightweight) with redirect handling
            try:
                head_response, reason = await head_with_redirects(url, client, max_hops=3)
                
                if head_response is None:
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        error=reason
                    )
                
                # Check content type
                content_type = head_response.headers.get('content-type', '').lower()
                if 'image/svg+xml' in content_type:
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=head_response.status_code,
                        content_type=content_type,
                        error="SKIP_SVG"
                    )
                if not self._is_valid_image_content_type(content_type):
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=head_response.status_code,
                        content_type=content_type,
                        error="Invalid content type"
                    )
                
                # Check content length
                content_length = head_response.headers.get('content-length')
                if content_length and int(content_length) > self.MAX_CONTENT_LENGTH:
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=head_response.status_code,
                        content_type=content_type,
                        error="SIZE_CAP"
                    )
                
                # If HEAD succeeds, try GET for first few bytes with redirect handling
                if head_response.status_code == 200:
                    from .redirect_utils import fetch_with_redirects
                    get_response, get_reason = await fetch_with_redirects(url, client, max_hops=3, method="GET")
                    
                    if get_response is None:
                        return ValidationResult(
                            url=url,
                            is_valid=False,
                            error=f"GET failed: {get_reason}"
                        )
                    
                    if get_response.status_code in [200, 206]:  # 206 = Partial Content
                        # Check actual content length if available
                        actual_length = get_response.headers.get('content-length')
                        if actual_length and int(actual_length) > self.MAX_CONTENT_LENGTH:
                            return ValidationResult(
                                url=url,
                                is_valid=False,
                                status_code=get_response.status_code,
                                content_type=content_type,
                                error="SIZE_CAP"
                            )
                        
                        # Check downloaded content length
                        content_bytes = get_response.content
                        if len(content_bytes) > self.MAX_CONTENT_LENGTH:
                            return ValidationResult(
                                url=url,
                                is_valid=False,
                                status_code=get_response.status_code,
                                content_type=content_type,
                                error="SIZE_CAP"
                            )
                        
                        return ValidationResult(
                            url=url,
                            is_valid=True,
                            status_code=get_response.status_code,
                            content_type=content_type
                        )
                
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    status_code=head_response.status_code,
                    content_type=content_type,
                    error=f"HTTP {head_response.status_code}"
                )
                
            except httpx.TimeoutException:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error="Request timeout"
                )
            except httpx.HTTPStatusError as e:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    status_code=e.response.status_code,
                    error=f"HTTP {e.response.status_code}"
                )
            except Exception as e:
                return ValidationResult(
                    url=url,
                    is_valid=False,
                    error=f"Request failed: {str(e)}"
                )
                
        except Exception as e:
            return ValidationResult(
                url=url,
                is_valid=False,
                error=f"Validation error: {str(e)}"
            )
    
    def _is_safe_url(self, url: str) -> bool:
        """Check if URL passes basic safety checks."""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme.lower() in self.MALICIOUS_SCHEMES:
                return False
            
            # Check for suspicious patterns
            suspicious_patterns = [
                'javascript:', 'data:', 'vbscript:', 'onload=', 'onerror=',
                'eval(', 'expression(', 'url(', 'import('
            ]
            
            url_lower = url.lower()
            for pattern in suspicious_patterns:
                if pattern in url_lower:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _is_valid_image_content_type(self, content_type: str) -> bool:
        """Check if content type indicates a valid image."""
        if not content_type:
            return False
            
        # Explicitly reject SVG
        if 'image/svg+xml' in content_type.lower():
            return False
            
        # Valid image MIME types
        valid_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
            'image/webp', 'image/bmp'
        ]
        
        return any(valid_type in content_type for valid_type in valid_types)
    
    def _infer_attributes_priority(self, nodes: List[ImageNode]) -> List[str]:
        """
        Infer attributes_priority from observed attributes in image nodes.
        
        Args:
            nodes: List of image nodes to analyze
            
        Returns:
            Prioritized list of attribute names
        """
        attribute_counts = defaultdict(int)
        attribute_quality = defaultdict(float)
        
        # Count attribute occurrences and assess quality
        for node in nodes:
            for attr, value in node.attributes.items():
                # Handle different value types from BeautifulSoup
                if isinstance(value, list):
                    value_str = ' '.join(str(v) for v in value if v)
                else:
                    value_str = str(value) if value else ''
                
                if value_str and value_str.strip():  # Non-empty values
                    attribute_counts[attr] += 1
                    
                    # Assess quality based on value characteristics
                    quality_score = 0.0
                    
                    # Longer values often contain more meaningful content
                    if len(value_str) > 10:
                        quality_score += 0.3
                    if len(value_str) > 50:
                        quality_score += 0.2
                    
                    # Check for common meaningful patterns
                    if re.search(r'[a-zA-Z]{3,}', value_str):  # Contains words
                        quality_score += 0.2
                    
                    # Avoid random-looking values
                    if not re.search(r'^[a-f0-9]{8,}$', value_str):  # Not pure hex
                        quality_score += 0.1
                    
                    # Avoid pure numbers (likely IDs)
                    if not re.search(r'^\d+$', value_str):
                        quality_score += 0.1
                    
                    # Bonus for common semantic attributes
                    semantic_bonus = {
                        'alt': 0.5,
                        'title': 0.4,
                        'data-title': 0.3,
                        'data-alt': 0.3,
                        'aria-label': 0.3
                    }
                    
                    if attr in semantic_bonus:
                        quality_score += semantic_bonus[attr]
                    
                    attribute_quality[attr] = max(attribute_quality[attr], quality_score)
        
        # Combine frequency and quality for ranking
        scored_attributes = []
        for attr in attribute_counts:
            if attribute_counts[attr] >= len(nodes) * 0.2:  # Present in at least 20% of nodes
                score = attribute_counts[attr] * (1.0 + attribute_quality[attr])
                scored_attributes.append((attr, score))
        
        # Sort by score and return attribute names
        scored_attributes.sort(key=lambda x: x[1], reverse=True)
        return [attr for attr, _ in scored_attributes[:10]]  # Top 10 attributes
    
    def _extract_extra_sources(self, nodes: List[ImageNode]) -> List[str]:
        """
        Extract additional image sources from nodes.
        
        Args:
            nodes: List of image nodes to analyze
            
        Returns:
            List of additional source attributes found
        """
        source_attributes = set()
        
        for node in nodes:
            for attr in node.attributes:
                # Look for data-* attributes that might contain image URLs
                if attr.startswith('data-') and any(
                    keyword in attr.lower() 
                    for keyword in ['src', 'image', 'thumb', 'lazy', 'original']
                ):
                    source_attributes.add(attr)
        
        # Common extra sources to check
        common_sources = [
            'data-src', 'data-lazy-src', 'data-original', 'data-large',
            'data-medium', 'data-thumb', 'data-srcset'
        ]
        
        # Add common sources that were found
        for source in common_sources:
            if any(source in node.attributes for node in nodes):
                source_attributes.add(source)
        
        return list(source_attributes)
    
    async def propose_recipe(self, domain: str, html_content: str, max_candidates: int = 3) -> Optional[RecipeCandidate]:
        """
        Propose a complete recipe for a domain based on HTML analysis and validation.
        
        Args:
            domain: Domain name for the recipe
            html_content: HTML content to analyze
            max_candidates: Maximum number of candidates to validate
            
        Returns:
            Recipe candidate if validation succeeds, None otherwise
        """
        logger.info(f"Proposing recipe for domain: {domain}")
        
        # Mine selectors from HTML
        candidates = self.mine_selectors(html_content)
        
        if not candidates:
            logger.warning(f"No candidates found for domain: {domain}")
            return None
        
        # Take top candidates for validation
        top_candidates = candidates[:max_candidates]
        
        # Validate each candidate
        validated_candidates = []
        for candidate in top_candidates:
            logger.info(f"Validating candidate: {candidate.selector}")
            validation_results = await self.validate_candidate_urls(candidate, max_samples=3)
            candidate.validation_results = validation_results
            
            # Check if candidate passes validation (≥2 URLs succeed)
            successful_validations = sum(1 for result in validation_results if result.is_valid)
            
            if successful_validations >= 2:
                logger.info(f"Candidate passed validation: {candidate.selector} ({successful_validations}/3 URLs valid)")
                validated_candidates.append((candidate, successful_validations))
            else:
                logger.warning(f"Candidate failed validation: {candidate.selector} ({successful_validations}/3 URLs valid)")
        
        if not validated_candidates:
            logger.warning(f"No candidates passed validation for domain: {domain}")
            return None
        
        # Use the best validated candidate
        best_candidate, success_count = max(validated_candidates, key=lambda x: x[1])
        
        # Infer attributes priority from all image nodes
        attributes_priority = self._infer_attributes_priority(self.image_nodes)
        
        # Extract extra sources
        extra_sources = self._extract_extra_sources(self.image_nodes)
        
        # Create recipe structure
        selectors = []
        for candidate in candidates[:5]:  # Top 5 candidates
            selectors.append({
                'selector': candidate.selector,
                'description': candidate.description
            })
        
        # Calculate overall confidence
        confidence = (best_candidate.score * 0.7 + (success_count / 3.0) * 0.3)
        
        return RecipeCandidate(
            domain=domain,
            selectors=selectors,
            attributes_priority=attributes_priority,
            extra_sources=extra_sources,
            method='smart',
            confidence=confidence,
            sample_urls=best_candidate.sample_urls,
            validation_results=best_candidate.validation_results
        )
    
    def generate_yaml_recipe(self, recipe: RecipeCandidate) -> str:
        """
        Generate YAML block for a recipe candidate.
        
        Args:
            recipe: Recipe candidate to convert to YAML
            
        Returns:
            YAML string for the recipe
        """
        yaml_data = {
            recipe.domain: {
                'selectors': recipe.selectors,
                'attributes_priority': recipe.attributes_priority,
                'extra_sources': recipe.extra_sources,
                'method': recipe.method
            }
        }
        
        return yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def merge_recipe_to_yaml_file(self, recipe: RecipeCandidate, yaml_file_path: str) -> bool:
        """
        Merge a recipe into an existing YAML file or create a new one.
        
        Args:
            recipe: Recipe candidate to merge
            yaml_file_path: Path to the YAML file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing recipes
            existing_recipes = {'defaults': {}, 'sites': {}}
            if Path(yaml_file_path).exists():
                with open(yaml_file_path, 'r', encoding='utf-8') as file:
                    existing_recipes = yaml.safe_load(file) or existing_recipes
            
            # Ensure required sections exist
            if 'sites' not in existing_recipes:
                existing_recipes['sites'] = {}
            
            # Add or update the recipe
            existing_recipes['sites'][recipe.domain] = {
                'selectors': recipe.selectors,
                'attributes_priority': recipe.attributes_priority,
                'extra_sources': recipe.extra_sources,
                'method': recipe.method
            }
            
            # Write back to file
            with open(yaml_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(existing_recipes, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Successfully merged recipe for {recipe.domain} into {yaml_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge recipe: {e}")
            return False

    def save_accepted_selectors_to_yaml(self, domain: str, accepted_selectors: List[Dict], 
                                       yaml_file_path: str) -> bool:
        """
        Save accepted selectors back to YAML file.
        
        Args:
            domain: Domain name
            accepted_selectors: List of accepted selector dictionaries
            yaml_file_path: Path to the YAML file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing recipes
            existing_recipes = {'defaults': {}, 'sites': {}}
            if Path(yaml_file_path).exists():
                with open(yaml_file_path, 'r', encoding='utf-8') as file:
                    existing_recipes = yaml.safe_load(file) or existing_recipes
            
            # Ensure required sections exist
            if 'sites' not in existing_recipes:
                existing_recipes['sites'] = {}
            
            # Get or create domain recipe
            if domain not in existing_recipes['sites']:
                existing_recipes['sites'][domain] = {
                    'selectors': [],
                    'attributes_priority': [],
                    'extra_sources': [],
                    'method': 'smart'
                }
            
            # Update selectors with accepted ones
            existing_recipes['sites'][domain]['selectors'] = accepted_selectors
            
            # Infer attributes_priority and extra_sources from accepted selectors if not present
            if not existing_recipes['sites'][domain]['attributes_priority']:
                existing_recipes['sites'][domain]['attributes_priority'] = self._infer_attributes_priority(self.image_nodes)
            
            if not existing_recipes['sites'][domain]['extra_sources']:
                existing_recipes['sites'][domain]['extra_sources'] = self._extract_extra_sources(self.image_nodes)
            
            # Write back to file
            with open(yaml_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(existing_recipes, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Successfully saved {len(accepted_selectors)} accepted selectors for {domain} to {yaml_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save accepted selectors: {e}")
            return False

    async def render_with_javascript(self, url: str, timeout: int = 3000) -> Optional[str]:
        """
        Render a page with JavaScript using Playwright to capture dynamically injected content.
        
        Args:
            url: URL to render
            timeout: Timeout in milliseconds (default: 3000ms = 3 seconds)
            
        Returns:
            Rendered HTML content or None if Playwright not available or error occurs
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available - skipping JavaScript rendering")
            return None
        
        try:
            async with async_playwright() as p:
                # Launch browser in headless mode
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    # Block images to speed up rendering
                    java_script_enabled=True,
                    images=False,  # Block images for faster rendering
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                
                page = await context.new_page()
                
                # Set timeout
                page.set_default_timeout(timeout)
                
                logger.info(f"Rendering JavaScript for URL: {url}")
                
                # Navigate to the page
                await page.goto(url, wait_until='domcontentloaded')
                
                # Wait for potential dynamic content (2-3 seconds as specified)
                await page.wait_for_timeout(timeout)
                
                # Get the rendered HTML
                html_content = await page.content()
                
                await browser.close()
                
                logger.info(f"JavaScript rendering completed for {url} ({len(html_content)} chars)")
                return html_content
                
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            return None

    async def mine_selectors_with_js_fallback(self, html_content: str, url: str = None, min_candidates: int = 3) -> List[CandidateSelector]:
        """
        Mine selectors with JavaScript fallback if static mining finds too few candidates.
        
        Args:
            html_content: Initial HTML content
            url: URL to render with JavaScript if needed
            min_candidates: Minimum number of candidates to proceed without JS fallback
            
        Returns:
            List of candidate selectors (from static or JS-rendered content)
        """
        # First try static mining
        logger.info("Starting static HTML mining")
        candidates = self.mine_selectors(html_content)
        
        # If we have enough candidates, return them
        if len(candidates) >= min_candidates:
            logger.info(f"Static mining found {len(candidates)} candidates (≥{min_candidates}), skipping JS fallback")
            return candidates
        
        # If we have a URL and Playwright is available, try JavaScript rendering
        if url and PLAYWRIGHT_AVAILABLE:
            logger.info(f"Static mining found only {len(candidates)} candidates (<{min_candidates}), trying JavaScript fallback")
            
            # Render with JavaScript
            js_html = await self.render_with_javascript(url)
            
            if js_html:
                # Try mining again with JS-rendered content
                logger.info("Mining JavaScript-rendered HTML")
                js_candidates = self.mine_selectors(js_html)
                
                if len(js_candidates) > len(candidates):
                    logger.info(f"JavaScript fallback found {len(js_candidates)} candidates (vs {len(candidates)} static)")
                    return js_candidates
                else:
                    logger.info(f"JavaScript fallback found {len(js_candidates)} candidates (same or fewer than static)")
                    return candidates
            else:
                logger.warning("JavaScript rendering failed, returning static results")
                return candidates
        else:
            if not url:
                logger.info("No URL provided for JavaScript fallback")
            if not PLAYWRIGHT_AVAILABLE:
                logger.info("Playwright not available for JavaScript fallback")
            
            logger.info(f"Returning static results: {len(candidates)} candidates")
            return candidates


def mine_selectors_for_url(html_content: str, base_url: str = "") -> List[CandidateSelector]:
    """
    Convenience function to mine selectors from HTML content.
    
    Args:
        html_content: Raw HTML content
        base_url: Base URL for resolving relative links
        
    Returns:
        List of top candidate selectors
    """
    miner = SelectorMiner(base_url)
    return miner.mine_selectors(html_content)


async def mine_selectors_with_js_fallback(html_content: str, url: str = None, base_url: str = "", min_candidates: int = 3, max_bytes: Optional[int] = None) -> List[CandidateSelector]:
    """
    Convenience function to mine selectors with JavaScript fallback.
    
    Args:
        html_content: Initial HTML content
        url: URL to render with JavaScript if needed
        base_url: Base URL for resolving relative links
        min_candidates: Minimum number of candidates to proceed without JS fallback
        max_bytes: Maximum image size in bytes
        
    Returns:
        List of candidate selectors (from static or JS-rendered content)
    """
    miner = SelectorMiner(base_url, max_bytes=max_bytes)
    return await miner.mine_selectors_with_js_fallback(html_content, url, min_candidates)


async def propose_recipe_for_domain(domain: str, html_content: str, base_url: str = "", max_bytes: Optional[int] = None) -> Optional[RecipeCandidate]:
    """
    Convenience function to propose a recipe for a domain.
    
    Args:
        domain: Domain name for the recipe
        html_content: Raw HTML content
        base_url: Base URL for resolving relative links
        max_bytes: Maximum image size in bytes
        
    Returns:
        Recipe candidate if validation succeeds, None otherwise
    """
    miner = SelectorMiner(base_url, max_bytes=max_bytes)
    return await miner.propose_recipe(domain, html_content)


async def main():
    """Demo of the selector miner with validation and YAML emission."""
    # Example usage
    sample_html = """
    <html>
    <body>
        <div class="video-thumbnails">
            <a href="/video/123">
                <img src="/thumb/123.jpg" class="thumbnail" alt="Video 1" title="Amazing Video">
                <span class="duration">2:30</span>
            </a>
            <a href="/video/456">
                <img src="/thumb/456.jpg" class="thumbnail" alt="Video 2" title="Great Content">
                <span class="duration">1:45</span>
            </a>
            <a href="/video/789">
                <img src="/thumb/789.jpg" class="thumbnail" alt="Video 3" title="Best Video">
                <span class="duration">3:15</span>
            </a>
        </div>
    </body>
    </html>
    """
    
    print("=== Selector Mining Demo ===")
    candidates = mine_selectors_for_url(sample_html, "https://example.com")
    
    print("Top candidate selectors:")
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate.selector}")
        print(f"   Score: {candidate.score:.3f}")
        print(f"   Description: {candidate.description}")
        print(f"   Samples: {len(candidate.sample_urls)} URLs")
        print()
    
    print("\n=== Recipe Proposal Demo ===")
    print("Note: This demo uses mock URLs that won't actually validate.")
    print("In real usage, these would be actual image URLs that can be validated.")
    
    # Create a miner instance
    miner = SelectorMiner("https://example.com")
    
    # Mine selectors first
    miner.mine_selectors(sample_html)
    
    # Show what the recipe would look like (without validation)
    if miner.image_nodes:
        attributes_priority = miner._infer_attributes_priority(miner.image_nodes)
        extra_sources = miner._extract_extra_sources(miner.image_nodes)
        
        print(f"Attributes Priority: {attributes_priority}")
        print(f"Extra Sources: {extra_sources}")
        
        # Generate a sample recipe structure
        sample_recipe = RecipeCandidate(
            domain="example.com",
            selectors=[
                {
                    'selector': candidate.selector,
                    'description': candidate.description
                } for candidate in candidates[:3]
            ],
            attributes_priority=attributes_priority,
            extra_sources=extra_sources,
            method='smart',
            confidence=0.8,
            sample_urls=[],
            validation_results=[]
        )
        
        print("\nGenerated YAML Recipe:")
        yaml_output = miner.generate_yaml_recipe(sample_recipe)
        print(yaml_output)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
