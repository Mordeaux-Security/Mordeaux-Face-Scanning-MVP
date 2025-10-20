"""
Crawler Module - Refactored Architecture

Unified crawler system with clean separation of concerns, streaming processing,
and comprehensive resource management. All functionality consolidated under
a single module with simplified interfaces.

Main Components:
- CrawlerEngine: Main orchestration engine
- CrawlerConfig: Unified configuration system
- ImageExtractor: Image URL extraction
- ImageProcessor: Image processing pipeline
- StorageManager: Storage operations
- MemoryManager: Memory management and GC
- SelectorMiner: CSS selector mining
"""

import logging

# Core exports
from .core import CrawlerEngine, CrawlResult, CrawlStats
from .config import CrawlerConfig, get_config, set_config, reset_config

# Component exports
from .extractor import ImageExtractor, ExtractedImage, ExtractionResult
from .processor import ImageProcessor, ProcessedImage, ProcessingResult, FaceDetector
from .storage import StorageManager, StorageResult, StorageMetadata
from .memory import MemoryManager, MemoryStats, get_memory_manager, reset_memory_manager
from .miner import SelectorMiner, MiningResult, CandidateSelector

# Version information
__version__ = "2.0.0"
__author__ = "Mordeaux Face Scanning MVP"
__description__ = "Refactored crawler system with clean architecture"

# Main entry points
__all__ = [
    # Core
    "CrawlerEngine",
    "CrawlResult", 
    "CrawlStats",
    
    # Configuration
    "CrawlerConfig",
    "get_config",
    "set_config", 
    "reset_config",
    
    # Components
    "ImageExtractor",
    "ExtractedImage",
    "ExtractionResult",
    "ImageProcessor", 
    "ProcessedImage",
    "ProcessingResult",
    "FaceDetector",
    "StorageManager",
    "StorageResult",
    "StorageMetadata",
    "MemoryManager",
    "MemoryStats",
    "get_memory_manager",
    "reset_memory_manager",
    "SelectorMiner",
    "MiningResult",
    "CandidateSelector",
    
    # Version
    "__version__",
    "__author__",
    "__description__"
]


# Convenience functions for common operations
from typing import List

async def crawl_site(url: str, config: CrawlerConfig = None) -> CrawlResult:
    """
    Convenience function to crawl a single site.
    
    Args:
        url: URL to crawl
        config: Optional configuration (uses default if not provided)
        
    Returns:
        CrawlResult with crawling statistics
    """
    config = config or get_config()
    
    async with CrawlerEngine(config) as crawler:
        return await crawler.crawl_site(url)


async def crawl_list(urls: List[str], config: CrawlerConfig = None, max_concurrent: int = 2) -> List[CrawlResult]:
    """
    Convenience function to crawl multiple sites.
    
    Args:
        urls: List of URLs to crawl
        config: Optional configuration (uses default if not provided)
        max_concurrent: Maximum number of concurrent crawls
        
    Returns:
        List of CrawlResult objects
    """
    config = config or get_config()
    
    async with CrawlerEngine(config) as crawler:
        return await crawler.crawl_list(urls, max_concurrent)


async def mine_selectors(url: str, config: CrawlerConfig = None) -> MiningResult:
    """
    Convenience function to mine selectors for a site.
    
    Args:
        url: URL to mine selectors for
        config: Optional configuration (uses default if not provided)
        
    Returns:
        MiningResult with discovered selectors
    """
    config = config or get_config()
    
    async with SelectorMiner(config) as miner:
        return await miner.mine_site(url)


# Backward compatibility aliases (for existing code)
# These maintain compatibility with the old architecture
ImageCrawler = CrawlerEngine  # Main crawler class
ListCrawler = CrawlerEngine   # List crawling functionality

# Legacy imports for backward compatibility
try:
    from . import crawler_settings as _legacy_settings
    # Export legacy constants for backward compatibility
    for attr_name in dir(_legacy_settings):
        if not attr_name.startswith('_'):
            globals()[attr_name] = getattr(_legacy_settings, attr_name)
except ImportError:
    # Legacy settings not available, that's fine
    pass


# Module initialization
def _initialize_module():
    """Initialize module with default configuration."""
    try:
        # Set up default configuration
        config = get_config()
        logger = logging.getLogger(__name__)
        logger.info(f"Crawler module initialized with version {__version__}")
        logger.debug(f"Default configuration loaded: {config.max_images} max images, {config.concurrent_downloads} concurrent downloads")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to initialize crawler module: {e}")


# Auto-initialize on import
_initialize_module()