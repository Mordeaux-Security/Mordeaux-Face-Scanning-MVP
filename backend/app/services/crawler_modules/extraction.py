"""
Image extraction module for the crawler service.

This module contains all image extraction logic including CSS selector patterns,
site recipe integration, and various extraction methods for different website structures.
"""

import re
import json
import logging
from typing import List, Optional, Tuple, Set, Dict
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from .types import ImageInfo
from ...config.site_recipes import get_recipe_for_url

logger = logging.getLogger(__name__)


def extract_style_bg_url(style_attr: str, base_url: str) -> Optional[str]:
    """
    Extract background image URL from CSS style attribute.
    
    Args:
        style_attr: The style attribute value
        base_url: Base URL for resolving relative URLs
        
    Returns:
        Extracted background image URL or None
    """
    if not style_attr:
        return None
    
    try:
        # Look for background-image: url(...) patterns
        patterns = [
            r'background-image:\s*url\(["\']?([^"\']+)["\']?\)',
            r'background:\s*[^;]*url\(["\']?([^"\']+)["\']?\)',
            r'background-image:\s*url\(([^)]+)\)',
            r'background:\s*[^;]*url\(([^)]+)\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, style_attr, re.IGNORECASE)
            if match:
                url = match.group(1).strip()
                if url and not url.startswith('data:'):
                    return urljoin(base_url, url)
        
        return None
    except Exception as e:
        logger.debug(f"Error extracting background URL from style: {e}")
        return None


def extract_jsonld_thumbnails(html_content: str, base_url: str) -> List[str]:
    """
    Extract thumbnail URLs from JSON-LD structured data.
    
    Args:
        html_content: The HTML content to parse
        base_url: Base URL for resolving relative URLs
        
    Returns:
        List of extracted thumbnail URLs
    """
    thumbnails = []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all script tags with type="application/ld+json"
        json_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                
                # Handle both single objects and arrays
                if isinstance(data, list):
                    for item in data:
                        thumbnails.extend(_extract_thumbnails_from_jsonld(item, base_url))
                else:
                    thumbnails.extend(_extract_thumbnails_from_jsonld(data, base_url))
                    
            except (json.JSONDecodeError, AttributeError):
                continue
                
    except Exception as e:
        logger.debug(f"Error extracting JSON-LD thumbnails: {e}")
    
    return thumbnails


def _extract_thumbnails_from_jsonld(data: Dict, base_url: str) -> List[str]:
    """Extract thumbnail URLs from JSON-LD data structure."""
    thumbnails = []
    
    # Common thumbnail fields in JSON-LD
    thumbnail_fields = ['thumbnail', 'thumbnailUrl', 'image', 'url']
    
    for field in thumbnail_fields:
        if field in data:
            value = data[field]
            if isinstance(value, str):
                thumbnails.append(urljoin(base_url, value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        thumbnails.append(urljoin(base_url, item))
                    elif isinstance(item, dict) and 'url' in item:
                        thumbnails.append(urljoin(base_url, item['url']))
    
    return thumbnails


class ImageExtractor:
    """Image extraction service with support for multiple extraction methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_images_by_method(self, html_content: str, base_url: str, method: str = "smart") -> Tuple[List[ImageInfo], str]:
        """
        Extract images using configurable targeting methods with site recipe support.
        
        Supports flexible CSS selector patterns for different website structures.
        Now integrates with site recipes for per-domain customization.
        
        Args:
            html_content: The HTML content to parse
            base_url: Base URL for resolving relative URLs
            method: Targeting method ('smart', 'data-mediumthumb', 'js-videoThumb', etc.)
            
        Returns:
            Tuple of (images_list, method_used)
        """
        images = []
        method_used = method
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get site recipe for this URL
            recipe = get_recipe_for_url(base_url)
            recipe_method = recipe.get("method", method)
            
            # Use recipe method if different from requested method
            if recipe_method != method and method == "smart":
                self.logger.info(f"Site recipe overrides method '{method}' with '{recipe_method}' for {base_url}")
                method = recipe_method
                method_used = recipe_method
            
            if method == "smart" or method == recipe_method:
                # Use site recipe selectors if available, otherwise fall back to built-in patterns
                recipe_selectors = recipe.get("selectors")
                
                if recipe_selectors:
                    self.logger.debug(f"Using site recipe selectors for {base_url}: {len(recipe_selectors)} selectors")
                    images = self._extract_with_selectors(soup, base_url, recipe_selectors)
                    if images:
                        method_used = f"recipe-{recipe_method}"
                        self.logger.info(f"Site recipe method '{recipe_method}' found {len(images)} images")
                        # Debug: Show first few image URLs
                        for i, img in enumerate(images[:3]):
                            self.logger.debug(f"DEBUG: Recipe image {i+1}: {img.url}")
                    else:
                        self.logger.debug(f"Site recipe found no images, falling back to built-in patterns")
                        # Fall back to built-in patterns
                        images, method_used = self._extract_with_builtin_patterns(soup, base_url, "smart")
                else:
                    # No recipe selectors, use built-in patterns
                    images, method_used = self._extract_with_builtin_patterns(soup, base_url, "smart")
            else:
                # Use specific method
                images, method_used = self._extract_with_builtin_patterns(soup, base_url, method)
            
            # Add additional sources if we have few images
            if len(images) < 5:
                additional_images = self._extract_additional_sources(soup, base_url, set(img.url for img in images))
                images.extend(additional_images)
            
        except Exception as e:
            self.logger.error(f"Error extracting images: {e}")
            images = []
        
        return images, method_used
    
    def _extract_with_builtin_patterns(self, soup, base_url: str, method: str) -> Tuple[List[ImageInfo], str]:
        """
        Extract images using built-in patterns (fallback when no recipe is available).
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            method: Targeting method
            
        Returns:
            Tuple of (images_list, method_used)
        """
        images = []
        method_used = method
        
        if method == "smart":
            # Try multiple extraction patterns in order of preference
            patterns = [
                # PornHub-style patterns
                ("data-mediumthumb", [{"selector": "img[data-mediumthumb]", "description": "data-mediumthumb attribute"}]),
                ("js-videoThumb", [{"selector": "img.js-videoThumb", "description": "js-videoThumb class"}]),
                ("phimage", [{"selector": ".phimage img", "description": "images in .phimage containers"}]),
                ("latestThumb", [{"selector": "a.latestThumb img", "description": "images in .latestThumb links"}]),
                
                # Common video thumbnail patterns
                ("video-thumb", [
                    {"selector": "img[data-video-thumb]", "description": "data-video-thumb attribute"},
                    {"selector": ".video-thumb img", "description": ".video-thumb container images"},
                    {"selector": ".thumbnail img", "description": ".thumbnail container images"},
                    {"selector": ".thumb img", "description": ".thumb container images"}
                ]),
                
                # General image patterns
                ("general", [
                    {"selector": "img", "description": "all images"},
                    {"selector": "picture img", "description": "picture element images"},
                    {"selector": "figure img", "description": "figure element images"}
                ])
            ]
            
            for pattern_name, selectors in patterns:
                try:
                    pattern_images = self._extract_with_selectors(soup, base_url, selectors)
                    if pattern_images:
                        images = pattern_images
                        method_used = f"builtin-{pattern_name}"
                        self.logger.debug(f"Built-in pattern '{pattern_name}' found {len(images)} images")
                        break
                except Exception as e:
                    self.logger.debug(f"Error with pattern '{pattern_name}': {e}")
                    continue
        else:
            # Use specific method
            if method == "data-mediumthumb":
                selectors = [{"selector": "img[data-mediumthumb]", "description": "data-mediumthumb attribute"}]
            elif method == "js-videoThumb":
                selectors = [{"selector": "img.js-videoThumb", "description": "js-videoThumb class"}]
            elif method == "phimage":
                selectors = [{"selector": ".phimage img", "description": "images in .phimage containers"}]
            elif method == "latestThumb":
                selectors = [{"selector": "a.latestThumb img", "description": "images in .latestThumb links"}]
            else:
                # Fallback to general images
                selectors = [{"selector": "img", "description": "all images"}]
            
            images = self._extract_with_selectors(soup, base_url, selectors)
            method_used = f"builtin-{method}"
        
        return images, method_used
    
    def _extract_with_selectors(self, soup, base_url: str, selectors: List[Dict]) -> List[ImageInfo]:
        """
        Extract images using CSS selectors with flexible matching and expanded sources.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            selectors: List of selector dictionaries
            
        Returns:
            List of ImageInfo objects
        """
        images = []
        seen_urls = set()
        
        for selector_config in selectors:
            selector = selector_config["selector"]
            description = selector_config.get("description", selector)
            
            try:
                elements = soup.select(selector)
                self.logger.debug(f"Selector '{selector}' ({description}) found {len(elements)} elements")
                
                for element in elements:
                    try:
                        # Extract all possible image URLs from this element
                        img_urls = self._extract_img_urls(element, base_url)
                        
                        for img_url in img_urls:
                            if img_url and img_url not in seen_urls:
                                seen_urls.add(img_url)
                                
                                # Extract video URL from context
                                video_url = self._extract_video_url_from_context(element, base_url)
                                
                                # Create ImageInfo object
                                img_info = ImageInfo(
                                    url=img_url,
                                    alt_text=element.get('alt', ''),
                                    title=element.get('title', ''),
                                    width=self._parse_dimension(element.get('width')),
                                    height=self._parse_dimension(element.get('height')),
                                    video_url=video_url
                                )
                                
                                images.append(img_info)
                                
                    except Exception as e:
                        self.logger.debug(f"Error processing element: {e}")
                        continue
                        
            except Exception as e:
                self.logger.debug(f"Error with selector '{selector}': {e}")
                continue
        
        return images
    
    def _extract_img_urls(self, img_tag, base_url: str) -> List[str]:
        """
        Extract all possible image URLs from an img tag including srcset.
        
        Args:
            img_tag: BeautifulSoup img element
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of image URLs
        """
        urls = []
        
        # Primary src attribute
        if img_tag.get('src'):
            urls.append(urljoin(base_url, img_tag['src']))
        
        # Srcset attribute
        if img_tag.get('srcset'):
            srcset_urls = self._parse_srcset(img_tag['srcset'], base_url)
            urls.extend(srcset_urls)
        
        # Data attributes (common in lazy loading)
        data_attrs = ['data-src', 'data-lazy-src', 'data-original', 'data-medium', 'data-large']
        for attr in data_attrs:
            if img_tag.get(attr):
                urls.append(urljoin(base_url, img_tag[attr]))
        
        return urls
    
    def _parse_srcset(self, srcset: str, base_url: str) -> List[str]:
        """Parse srcset attribute and return list of URLs."""
        urls = []
        try:
            for src in srcset.split(','):
                url = src.strip().split()[0]
                if url:
                    urls.append(urljoin(base_url, url))
        except Exception as e:
            self.logger.debug(f"Error parsing srcset: {e}")
        return urls
    
    def _parse_dimension(self, value) -> Optional[int]:
        """Parse dimension value to integer."""
        if not value:
            return None
        try:
            return int(str(value).replace('px', ''))
        except (ValueError, TypeError):
            return None
    
    def _extract_additional_sources(self, soup, base_url: str, seen_urls: Set[str]) -> List[ImageInfo]:
        """
        Extract images from additional sources beyond CSS selectors.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            seen_urls: Set of already seen URLs
            
        Returns:
            List of additional ImageInfo objects
        """
        additional_images = []
        
        # Extract from style attributes
        for element in soup.find_all(attrs={'style': True}):
            style_url = extract_style_bg_url(element.get('style', ''), base_url)
            if style_url and style_url not in seen_urls:
                seen_urls.add(style_url)
                additional_images.append(ImageInfo(
                    url=style_url,
                    alt_text='',
                    title='',
                    width=None,
                    height=None
                ))
        
        # Extract from JSON-LD
        jsonld_urls = extract_jsonld_thumbnails(str(soup), base_url)
        for url in jsonld_urls:
            if url not in seen_urls:
                seen_urls.add(url)
                additional_images.append(ImageInfo(
                    url=url,
                    alt_text='',
                    title='',
                    width=None,
                    height=None
                ))
        
        return additional_images
    
    def _extract_video_url_from_context(self, img_tag, base_url: str) -> Optional[str]:
        """
        Extract video URL from the context around an image tag.
        Looks for video URLs in parent <a> tags, data attributes, and common patterns.
        
        Args:
            img_tag: BeautifulSoup img element
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Video URL if found, None otherwise
        """
        try:
            # Check parent <a> tag
            parent_link = img_tag.find_parent('a')
            if parent_link and parent_link.get('href'):
                href = parent_link['href']
                absolute_url = urljoin(base_url, href)
                
                # Check if it's a video URL
                if self._is_video_url(absolute_url):
                    return absolute_url
            
            # Check data attributes
            for attr in ['data-video-url', 'data-video', 'data-src']:
                if img_tag.get(attr):
                    video_url = urljoin(base_url, img_tag[attr])
                    if self._is_video_url(video_url):
                        return video_url
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting video URL: {e}")
            return None
    
    def _is_video_url(self, url: str) -> bool:
        """Check if URL appears to be a video URL."""
        if not url:
            return False
        
        # Common video file extensions
        video_extensions = ['.mp4', '.webm', '.ogg', '.avi', '.mov', '.wmv', '.flv', '.mkv']
        url_lower = url.lower()
        
        # Check file extension
        for ext in video_extensions:
            if ext in url_lower:
                return True
        
        # Check for video-related patterns in URL
        video_patterns = ['/video/', '/watch/', '/play/', '/stream/']
        for pattern in video_patterns:
            if pattern in url_lower:
                return True
        
        return False
