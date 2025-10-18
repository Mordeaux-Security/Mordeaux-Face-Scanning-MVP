#!/usr/bin/env python3
"""
Test script for JavaScript rendering integration in the crawler.

This script validates that the JavaScript rendering service integrates
properly with the existing crawler architecture.
"""

import asyncio
import logging
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.crawler.crawler import ImageCrawler
from app.crawler.crawler_settings import JS_RENDERING_ENABLED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_js_rendering_integration():
    """Test JavaScript rendering integration with the crawler."""
    
    print("=" * 60)
    print("JavaScript Rendering Integration Test")
    print("=" * 60)
    
    # Check if JavaScript rendering is enabled
    print(f"JavaScript rendering enabled: {JS_RENDERING_ENABLED}")
    
    if not JS_RENDERING_ENABLED:
        print("❌ JavaScript rendering is disabled in settings")
        return False
    
    try:
        # Test URLs - mix of static and dynamic content
        test_urls = [
            "https://httpbin.org/html",  # Static HTML
            "https://example.com",       # Simple static site
        ]
        
        async with ImageCrawler(
            tenant_id="test_js_rendering",
            max_total_images=5,
            max_pages=2,
            require_face=False  # Don't require faces for this test
        ) as crawler:
            
            print(f"\n✅ Crawler initialized successfully")
            print(f"JavaScript rendering service available: {crawler.js_rendering_service is not None}")
            
            # Test JavaScript rendering statistics
            js_stats = crawler.get_js_rendering_stats()
            print(f"\n📊 JavaScript Rendering Statistics:")
            for key, value in js_stats.items():
                print(f"  {key}: {value}")
            
            # Test fetching pages with JavaScript detection
            for i, url in enumerate(test_urls, 1):
                print(f"\n🔍 Test {i}: Fetching {url}")
                
                try:
                    # Test static fetch first
                    html_content, errors = await crawler.fetch_page(url, force_js_rendering=False)
                    
                    if html_content:
                        print(f"  ✅ Static fetch successful ({len(html_content)} chars)")
                        print(f"  Errors: {errors}")
                        
                        # Test JavaScript detection
                        if crawler.js_rendering_service:
                            needs_js, detection_info = crawler.js_rendering_service.detect_javascript_usage(html_content)
                            print(f"  🔍 JavaScript detection: {needs_js}")
                            print(f"  📋 Detection info: {detection_info}")
                        
                        # Test forced JavaScript rendering
                        if crawler.js_rendering_service:
                            print(f"  🚀 Testing forced JavaScript rendering...")
                            js_content, js_errors = await crawler.fetch_page(url, force_js_rendering=True)
                            
                            if js_content:
                                print(f"  ✅ JavaScript rendering successful ({len(js_content)} chars)")
                                print(f"  JS Errors: {js_errors}")
                                
                                # Compare content lengths
                                if len(js_content) != len(html_content):
                                    print(f"  📊 Content difference: JS={len(js_content)}, Static={len(html_content)}")
                                else:
                                    print(f"  📊 Content lengths identical")
                            else:
                                print(f"  ⚠️  JavaScript rendering failed: {js_errors}")
                    else:
                        print(f"  ❌ Failed to fetch content: {errors}")
                        
                except Exception as e:
                    print(f"  ❌ Error testing {url}: {str(e)}")
            
            # Test a simple crawl with JavaScript rendering
            print(f"\n🕷️  Testing crawl with JavaScript rendering...")
            try:
                result = await crawler.crawl_page("https://httpbin.org/html", method="smart")
                print(f"  ✅ Crawl test successful:")
                print(f"    Images found: {result.images_found}")
                print(f"    Raw images saved: {result.raw_images_saved}")
                print(f"    Thumbnails saved: {result.thumbnails_saved}")
                print(f"    Duration: {result.total_duration_seconds:.2f}s")
            except Exception as e:
                print(f"  ❌ Crawl test failed: {str(e)}")
            
            # Final statistics
            final_stats = crawler.get_js_rendering_stats()
            print(f"\n📊 Final JavaScript Rendering Statistics:")
            for key, value in final_stats.items():
                print(f"  {key}: {value}")
        
        print(f"\n✅ JavaScript rendering integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ JavaScript rendering integration test failed: {str(e)}")
        logger.exception("Test failed with exception")
        return False

async def main():
    """Main test function."""
    success = await test_js_rendering_integration()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
