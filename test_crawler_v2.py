#!/usr/bin/env python3
"""
Test script for the v2 crawler
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.crawler_v2 import crawl_images_v2

async def main():
    # Test on a safe site
    url = "https://httpbin.org/html"
    
    print(f"Testing v2 crawler on: {url}")
    
    try:
        result = await crawl_images_v2(
            url,
            max_pages_to_visit=1,
            max_total_images=5
        )
        
        print('\n' + '='*60)
        print('CRAWL RESULTS:')
        print('='*60)
        print(f'URL: {result.url}')
        print(f'Images found: {result.images_found}')
        print(f'Raw images saved: {result.raw_images_saved}')
        print(f'Face crops saved: {result.face_crops_saved}')
        print(f'Upscaling factors: {result.upscaling_factors}')
        print(f'Errors: {len(result.errors)}')
        
        if result.errors:
            print('\nErrors encountered:')
            for error in result.errors:
                print(f'  - {error}')
        
        print('='*60)
        
    except Exception as e:
        print(f"Error during crawling: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
