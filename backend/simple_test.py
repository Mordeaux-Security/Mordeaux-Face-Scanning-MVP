#!/usr/bin/env python3
"""
Simple test script for the refactored crawler architecture.
Tests basic functionality without requiring external services.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, '/app')

from app.crawler.config import CrawlerConfig
from app.crawler.http_service import HTTPService, get_http_service
from app.crawler.js_rendering_service import JSRenderingService, get_js_rendering_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_http_service():
    """Test HTTP service functionality."""
    logger.info("Testing HTTP service...")
    
    config = CrawlerConfig.from_env()
    http_service = await get_http_service(config)
    
    # Test a simple HTTP request
    test_url = "https://httpbin.org/get"
    content, status_code = await http_service.get(test_url, as_text=True)
    
    if content and status_code == "200":
        logger.info("✅ HTTP service test passed")
        return True
    else:
        logger.error(f"❌ HTTP service test failed: {status_code}")
        return False


async def test_js_rendering_service():
    """Test JS rendering service functionality."""
    logger.info("Testing JS rendering service...")
    
    config = CrawlerConfig.from_env()
    js_service = await get_js_rendering_service(config)
    
    # Test JS detection
    test_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <div id="app">Loading...</div>
            <script>console.log('Hello World');</script>
        </body>
    </html>
    """
    
    needs_js, detection_info = js_service.detect_javascript_usage(test_html)
    
    if needs_js and detection_info.get('script_count', 0) > 0:
        logger.info("✅ JS rendering service test passed")
        return True
    else:
        logger.error(f"❌ JS rendering service test failed: {detection_info}")
        return False


async def test_crawler_components():
    """Test individual crawler components."""
    logger.info("Testing crawler components...")
    
    try:
        from app.crawler.extractor import ImageExtractor
        from app.crawler.miner import SelectorMiner
        from app.crawler.processor import ImageProcessor
        from app.crawler.storage import StorageManager
        from app.crawler.memory import MemoryManager
        
        config = CrawlerConfig.from_env()
        
        # Test component initialization
        extractor = ImageExtractor(config)
        miner = SelectorMiner(config)
        memory_manager = MemoryManager(config)
        processor = ImageProcessor(config, memory_manager)
        storage = StorageManager(config)
        
        logger.info("✅ All crawler components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Crawler components test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting refactored crawler architecture test")
    start_time = time.time()
    
    tests = [
        ("HTTP Service", test_http_service),
        ("JS Rendering Service", test_js_rendering_service),
        ("Crawler Components", test_crawler_components),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info(f"Time elapsed: {elapsed_time:.2f}s")
    
    if passed == total:
        logger.info("🎉 All tests passed! Refactored crawler is working correctly.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
