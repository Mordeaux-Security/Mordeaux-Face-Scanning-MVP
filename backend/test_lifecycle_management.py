#!/usr/bin/env python3
"""
Test script for Phase 2.2 lifecycle management.
Verifies clean shutdowns and no memory leaks across repeated cycles.
"""

import asyncio
import logging
import psutil
import os
import time
import gc
from backend.app.services.crawler import ImageCrawler
from backend.app.services.face import close_face_service
from backend.app.services.storage import get_storage_cleanup_function
from backend.app.services.cache import close_cache_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_open_files():
    """Get count of open file descriptors."""
    try:
        process = psutil.Process()
        return len(process.open_files())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

def get_connections():
    """Get count of open network connections."""
    try:
        process = psutil.Process()
        return len(process.connections())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

async def test_crawler_lifecycle():
    """Test crawler lifecycle management."""
    logger.info("=== Testing Crawler Lifecycle Management ===")
    
    # Test data
    test_url = "https://example.com"  # Simple test URL
    
    # Record initial state
    initial_memory = get_memory_usage()
    initial_files = get_open_files()
    initial_connections = get_connections()
    
    logger.info(f"Initial state - Memory: {initial_memory:.1f}MB, Files: {initial_files}, Connections: {initial_connections}")
    
    # Test multiple crawler cycles
    memory_peaks = []
    
    for cycle in range(5):
        logger.info(f"\n--- Crawler Cycle {cycle + 1} ---")
        
        # Record state before crawler
        before_memory = get_memory_usage()
        before_files = get_open_files()
        before_connections = get_connections()
        
        logger.info(f"Before crawler - Memory: {before_memory:.1f}MB, Files: {before_files}, Connections: {before_connections}")
        
        try:
            # Create and use crawler
            async with ImageCrawler(
                tenant_id="test_tenant",
                max_total_images=5,  # Small limit for testing
                max_pages=1,
                require_face=False,  # Disable face detection for faster testing
                crop_faces=False
            ) as crawler:
                # Simulate a small crawl (this will likely fail, but that's OK for testing lifecycle)
                try:
                    result = await crawler.crawl_page(test_url)
                    logger.info(f"Crawl result: {result.images_found} images found")
                except Exception as e:
                    logger.info(f"Crawl failed as expected: {e}")
                
                # Record peak memory during crawler
                peak_memory = get_memory_usage()
                memory_peaks.append(peak_memory)
                logger.info(f"Peak memory during crawler: {peak_memory:.1f}MB")
        
        except Exception as e:
            logger.warning(f"Crawler cycle {cycle + 1} failed: {e}")
        
        # Record state after crawler cleanup
        after_memory = get_memory_usage()
        after_files = get_open_files()
        after_connections = get_connections()
        
        logger.info(f"After crawler - Memory: {after_memory:.1f}MB, Files: {after_files}, Connections: {after_connections}")
        
        # Force garbage collection between cycles
        gc.collect()
        time.sleep(1)  # Brief pause between cycles
    
    # Test manual service cleanup
    logger.info("\n--- Testing Manual Service Cleanup ---")
    
    try:
        # Test individual service cleanup
        logger.info("Testing face service cleanup...")
        close_face_service()
        
        logger.info("Testing storage service cleanup...")
        storage_cleanup = get_storage_cleanup_function()
        storage_cleanup()
        
        logger.info("Testing cache service cleanup...")
        close_cache_service()
        
        logger.info("All service cleanup functions executed successfully")
    except Exception as e:
        logger.warning(f"Service cleanup failed: {e}")
    
    # Final state
    final_memory = get_memory_usage()
    final_files = get_open_files()
    final_connections = get_connections()
    
    logger.info(f"\nFinal state - Memory: {final_memory:.1f}MB, Files: {final_files}, Connections: {final_connections}")
    
    # Analysis
    logger.info("\n=== Lifecycle Management Analysis ===")
    
    # Check for file descriptor leaks
    file_leak = final_files > initial_files
    logger.info(f"File descriptor leak: {'YES' if file_leak else 'NO'} (Initial: {initial_files}, Final: {final_files})")
    
    # Check for connection leaks
    connection_leak = final_connections > initial_connections
    logger.info(f"Connection leak: {'YES' if connection_leak else 'NO'} (Initial: {initial_connections}, Final: {final_connections})")
    
    # Check for memory stability
    memory_variance = max(memory_peaks) - min(memory_peaks) if memory_peaks else 0
    memory_stable = memory_variance < 50  # Less than 50MB variance is considered stable
    
    logger.info(f"Memory stability: {'STABLE' if memory_stable else 'UNSTABLE'}")
    logger.info(f"Memory variance: {memory_variance:.1f}MB (Min: {min(memory_peaks):.1f}MB, Max: {max(memory_peaks):.1f}MB)")
    
    # Acceptance criteria check
    logger.info("\n=== Acceptance Criteria Check ===")
    
    criteria_1 = not file_leak and not connection_leak
    criteria_2 = memory_stable
    
    logger.info(f"✓ Criteria 1 (No open file descriptors/sockets): {'PASS' if criteria_1 else 'FAIL'}")
    logger.info(f"✓ Criteria 2 (Flat peak RSS over time): {'PASS' if criteria_2 else 'FAIL'}")
    
    if criteria_1 and criteria_2:
        logger.info("🎉 All acceptance criteria PASSED!")
        return True
    else:
        logger.warning("❌ Some acceptance criteria FAILED!")
        return False

async def test_service_cleanup():
    """Test individual service cleanup functions."""
    logger.info("\n=== Testing Individual Service Cleanup ===")
    
    try:
        # Test face service cleanup
        logger.info("Testing face service cleanup...")
        close_face_service()
        logger.info("✓ Face service cleanup successful")
        
        # Test storage service cleanup
        logger.info("Testing storage service cleanup...")
        storage_cleanup = get_storage_cleanup_function()
        storage_cleanup()
        logger.info("✓ Storage service cleanup successful")
        
        # Test cache service cleanup
        logger.info("Testing cache service cleanup...")
        close_cache_service()
        logger.info("✓ Cache service cleanup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Service cleanup test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting Phase 2.2 Lifecycle Management Tests...")
    
    # Test individual service cleanup
    service_test_passed = await test_service_cleanup()
    
    # Test crawler lifecycle
    lifecycle_test_passed = await test_crawler_lifecycle()
    
    # Overall result
    if service_test_passed and lifecycle_test_passed:
        logger.info("\n🎉 All lifecycle management tests PASSED!")
        return True
    else:
        logger.warning("\n❌ Some lifecycle management tests FAILED!")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
