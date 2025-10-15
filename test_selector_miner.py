#!/usr/bin/env python3
"""
Simple selector miner test for Wikifeet site
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_html(url):
    """Fetch HTML content from URL"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

def find_image_selectors(html_content, base_url):
    """Find potential image selectors in HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all images
    images = soup.find_all('img')
    logger.info(f"Found {len(images)} img tags")
    
    # Find images with different attributes
    selectors = []
    
    # Common image patterns
    patterns = [
        ('img[src*="thumb"]', 'Images with "thumb" in src'),
        ('img[class*="thumb"]', 'Images with "thumb" in class'),
        ('img[src*="photo"]', 'Images with "photo" in src'),
        ('img[class*="photo"]', 'Images with "photo" in class'),
        ('img[src*="image"]', 'Images with "image" in src'),
        ('img[class*="image"]', 'Images with "image" in class'),
        ('img[src*="pic"]', 'Images with "pic" in src'),
        ('img[class*="pic"]', 'Images with "pic" in class'),
        ('img[data-src]', 'Images with data-src attribute'),
        ('img[data-lazy-src]', 'Images with data-lazy-src attribute'),
        ('.gallery img', 'Images inside gallery containers'),
        ('.photo img', 'Images inside photo containers'),
        ('.image img', 'Images inside image containers'),
        ('.thumb img', 'Images inside thumb containers'),
        ('img', 'All images')
    ]
    
    for selector, description in patterns:
        try:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Selector '{selector}': {len(elements)} matches - {description}")
                selectors.append({
                    'selector': selector,
                    'count': len(elements),
                    'description': description,
                    'sample_urls': [urljoin(base_url, img.get('src', '')) for img in elements[:3] if img.get('src')]
                })
        except Exception as e:
            logger.warning(f"Error with selector '{selector}': {e}")
    
    # Look for specific attributes and classes
    unique_classes = set()
    unique_src_patterns = set()
    
    for img in images:
        if img.get('class'):
            for cls in img['class']:
                if any(keyword in cls.lower() for keyword in ['thumb', 'photo', 'image', 'pic', 'gallery']):
                    unique_classes.add(cls)
        
        src = img.get('src', '')
        if src:
            # Look for common patterns in URLs
            if any(pattern in src.lower() for pattern in ['thumb', 'photo', 'image', 'pic']):
                # Extract the pattern
                for pattern in ['thumb', 'photo', 'image', 'pic']:
                    if pattern in src.lower():
                        unique_src_patterns.add(f"img[src*='{pattern}']")
                        break
    
    logger.info(f"Unique classes found: {unique_classes}")
    logger.info(f"Unique src patterns: {unique_src_patterns}")
    
    return selectors

async def main():
    url = "https://wikifeet.com/Leah_Martin"
    
    logger.info(f"Fetching HTML from: {url}")
    html = await fetch_html(url)
    
    if not html:
        logger.error("Failed to fetch HTML")
        return
    
    logger.info(f"HTML content length: {len(html)} characters")
    
    # Find selectors
    selectors = find_image_selectors(html, url)
    
    logger.info("\n" + "="*50)
    logger.info("SELECTOR ANALYSIS RESULTS")
    logger.info("="*50)
    
    for sel in selectors:
        logger.info(f"Selector: {sel['selector']}")
        logger.info(f"  Count: {sel['count']}")
        logger.info(f"  Description: {sel['description']}")
        if sel['sample_urls']:
            logger.info(f"  Sample URLs:")
            for sample_url in sel['sample_urls']:
                logger.info(f"    - {sample_url}")
        logger.info("")

if __name__ == "__main__":
    asyncio.run(main())
