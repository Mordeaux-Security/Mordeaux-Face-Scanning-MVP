"""
Test script to verify refactored crawler functionality.

This script tests the new architecture to ensure all functionality
is preserved and working correctly.
"""

import asyncio
import logging
import sys
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import new architecture
from .config import CrawlerConfig, get_config
from .core import CrawlerEngine, CrawlResult
from .memory import MemoryManager
from .extractor import ImageExtractor
from .processor import ImageProcessor
from .storage import StorageManager
from .miner import SelectorMiner


async def test_configuration():
    """Test configuration system."""
    logger.info("Testing configuration system...")
    
    # Test default configuration
    config = get_config()
    assert config.max_images > 0
    assert config.concurrent_downloads > 0
    assert config.timeout_seconds > 0
    logger.info("✓ Configuration system working")
    
    # Test environment variable loading
    import os
    os.environ['CRAWLER_MAX_IMAGES'] = '100'
    os.environ['CRAWLER_CONCURRENT_DOWNLOADS'] = '16'
    
    config_env = CrawlerConfig.from_env()
    assert config_env.max_images == 100
    assert config_env.concurrent_downloads == 16
    logger.info("✓ Environment variable configuration working")


async def test_memory_management():
    """Test memory management system."""
    logger.info("Testing memory management...")
    
    config = get_config()
    memory_manager = MemoryManager(config)
    
    # Test memory pressure detection
    is_pressured = memory_manager.is_memory_pressured()
    assert isinstance(is_pressured, bool)
    
    # Test memory statistics
    stats = memory_manager.get_memory_stats()
    assert hasattr(stats, 'system_memory_percent')
    assert hasattr(stats, 'process_memory_mb')
    
    # Test forced GC
    gc_result = await memory_manager.force_gc("test")
    assert 'collected' in gc_result
    assert 'gc_time' in gc_result
    
    await memory_manager.cleanup()
    logger.info("✓ Memory management working")


async def test_extractor():
    """Test image extractor."""
    logger.info("Testing image extractor...")
    
    config = get_config()
    
    async with ImageExtractor(config) as extractor:
        # Test with a simple recipe
        recipe = {
            'selectors': [
                {'kind': 'video_grid', 'css': 'img'}
            ],
            'attributes_priority': ['src', 'data-src'],
            'extra_sources': []
        }
        
        # Test URL validation
        valid_url = "https://example.com/image.jpg"
        invalid_url = "javascript:alert('xss')"
        
        assert extractor._validate_url(valid_url) == True
        assert extractor._validate_url(invalid_url) == False
        
        logger.info("✓ Image extractor working")


async def test_processor():
    """Test image processor."""
    logger.info("Testing image processor...")
    
    config = get_config()
    memory_manager = MemoryManager(config)
    
    async with ImageProcessor(config, memory_manager) as processor:
        # Test statistics
        stats = processor.get_statistics()
        assert 'images_processed' in stats
        assert 'faces_detected' in stats
        
        # Test reset statistics
        processor.reset_statistics()
        stats_after_reset = processor.get_statistics()
        assert stats_after_reset['images_processed'] == 0
        
        logger.info("✓ Image processor working")
    
    await memory_manager.cleanup()


async def test_storage():
    """Test storage manager."""
    logger.info("Testing storage manager...")
    
    config = get_config()
    storage = StorageManager(config)
    
    # Test statistics
    stats = storage.get_statistics()
    assert 'images_stored' in stats
    assert 'thumbnails_stored' in stats
    
    # Test recipe management
    default_recipe = storage._get_default_recipe()
    assert 'selectors' in default_recipe
    assert 'attributes_priority' in default_recipe
    
    # Test recipe retrieval
    recipe = storage.get_recipe_for_url("https://example.com")
    # Should return None for unknown site
    assert recipe is None
    
    logger.info("✓ Storage manager working")


async def test_miner():
    """Test selector miner."""
    logger.info("Testing selector miner...")
    
    config = get_config()
    
    async with SelectorMiner(config) as miner:
        # Test statistics
        stats = miner.get_statistics()
        assert 'sites_mined' in stats
        assert 'selectors_found' in stats
        
        # Test URL validation
        valid_url = "https://example.com"
        assert await miner._validate_image_url(valid_url) == True
        
        # Test selector generation
        from bs4 import BeautifulSoup
        html = '<div class="gallery"><img src="test.jpg" class="thumb"></div>'
        soup = BeautifulSoup(html, 'html.parser')
        element = soup.find('img')
        
        selector = miner._generate_selector(element)
        assert selector is not None
        assert 'img' in selector
        
        logger.info("✓ Selector miner working")


async def test_core_engine():
    """Test core crawler engine."""
    logger.info("Testing core crawler engine...")
    
    config = get_config()
    
    async with CrawlerEngine(config) as crawler:
        # Test statistics
        stats = crawler.get_statistics()
        assert hasattr(stats, 'total_sites')
        assert hasattr(stats, 'successful_sites')
        assert hasattr(stats, 'failed_sites')
        
        # Test health check
        health = await crawler.health_check()
        assert 'overall_healthy' in health
        assert 'components' in health
        
        # Test domain extraction
        domain = crawler._extract_domain("https://example.com/path")
        assert domain == "example.com"
        
        logger.info("✓ Core crawler engine working")


async def test_integration():
    """Test integration between components."""
    logger.info("Testing component integration...")
    
    config = get_config()
    
    # Test that all components can be initialized together
    memory_manager = MemoryManager(config)
    extractor = ImageExtractor(config)
    processor = ImageProcessor(config, memory_manager)
    storage = StorageManager(config)
    miner = SelectorMiner(config)
    
    # Test that they can be used together
    async with extractor, processor, storage, miner:
        # Test recipe generation
        recipe = storage._get_default_recipe()
        assert recipe is not None
        
        # Test URL validation
        valid_url = "https://example.com/image.jpg"
        assert extractor._validate_url(valid_url) == True
        
        # Test memory management
        is_pressured = memory_manager.is_memory_pressured()
        assert isinstance(is_pressured, bool)
        
        logger.info("✓ Component integration working")
    
    await memory_manager.cleanup()


async def test_backward_compatibility():
    """Test backward compatibility with old interface."""
    logger.info("Testing backward compatibility...")
    
    # Test that old class names still work
    from . import ImageCrawler, ListCrawler
    
    assert ImageCrawler == CrawlerEngine
    assert ListCrawler == CrawlerEngine
    
    # Test convenience functions
    from . import crawl_site, crawl_list, mine_selectors
    
    assert callable(crawl_site)
    assert callable(crawl_list)
    assert callable(mine_selectors)
    
    logger.info("✓ Backward compatibility working")


async def run_all_tests():
    """Run all tests."""
    logger.info("Starting refactored crawler tests...")
    
    try:
        await test_configuration()
        await test_memory_management()
        await test_extractor()
        await test_processor()
        await test_storage()
        await test_miner()
        await test_core_engine()
        await test_integration()
        await test_backward_compatibility()
        
        logger.info("🎉 All tests passed! Refactored architecture is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
