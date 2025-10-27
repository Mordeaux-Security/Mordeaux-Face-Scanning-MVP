#!/usr/bin/env python3
"""
Test script to render wikifeet.com and save HTML for analysis
"""
import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from new_crawler.http_utils import get_http_utils

async def test_wikifeet():
    """Test wikifeet.com rendering and save HTML"""
    http_utils = get_http_utils()
    
    try:
        print("Rendering wikifeet.com...")
        html = await http_utils.render_page("https://wikifeet.com")
        
        print(f"Rendered HTML length: {len(html)} characters")
        
        # Save HTML to file for analysis
        with open("wikifeet_rendered.html", "w", encoding="utf-8") as f:
            f.write(html)
        
        print("HTML saved to wikifeet_rendered.html")
        
        # Look for image-related patterns
        import re
        
        # Find all img tags
        img_tags = re.findall(r'<img[^>]*>', html, re.IGNORECASE)
        print(f"\nFound {len(img_tags)} <img> tags")
        
        # Find all picture tags
        picture_tags = re.findall(r'<picture[^>]*>.*?</picture>', html, re.IGNORECASE | re.DOTALL)
        print(f"Found {len(picture_tags)} <picture> tags")
        
        # Find all source tags
        source_tags = re.findall(r'<source[^>]*>', html, re.IGNORECASE)
        print(f"Found {len(source_tags)} <source> tags")
        
        # Show first few img tags
        print("\nFirst 5 img tags:")
        for i, tag in enumerate(img_tags[:5]):
            print(f"{i+1}: {tag}")
        
        # Look for data attributes
        data_attrs = re.findall(r'data-[a-zA-Z-]+="[^"]*"', html, re.IGNORECASE)
        print(f"\nFound {len(data_attrs)} data attributes")
        
        # Show unique data attributes
        unique_data_attrs = set()
        for attr in data_attrs:
            attr_name = attr.split('=')[0].lower()
            unique_data_attrs.add(attr_name)
        
        print("Unique data attributes:")
        for attr in sorted(unique_data_attrs):
            print(f"  {attr}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_wikifeet())
