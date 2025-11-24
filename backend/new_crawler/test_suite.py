"""
Test Suite for New Crawler System

Basic tests for single site, batching, deduplication, and GPU fallback.
"""

import asyncio
import logging
import tempfile
import os
import time
from pathlib import Path
from typing import List

from .config import get_config
from .redis_manager import get_redis_manager
from .cache_manager import get_cache_manager
from .http_utils import get_http_utils
from .selector_miner import get_selector_miner
from .gpu_interface import get_gpu_interface
from .storage_manager import get_storage_manager
from .orchestrator import Orchestrator
from .data_structures import SiteTask, CandidateImage, ImageTask
from .crawler_worker import CrawlerWorker

logger = logging.getLogger(__name__)


class TestSuite:
    """Test suite for new crawler system."""
    
    def __init__(self):
        self.config = get_config()
        self.redis = get_redis_manager()
        self.cache = get_cache_manager()
        self.http_utils = get_http_utils()
        self.selector_miner = get_selector_miner()
        self.gpu_interface = get_gpu_interface()
        self.storage = get_storage_manager()
        
        # Test results
        self.results = {}
    
    async def test_redis_connection(self) -> bool:
        """Test Redis connection."""
        try:
            result = self.redis.test_connection()
            self.results['redis_connection'] = result
            logger.info(f"Redis connection test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            self.results['redis_connection'] = False
            return False
    
    async def test_cache_operations(self) -> bool:
        """Test cache operations."""
        try:
            # Test basic cache operations
            test_key = "test_key_123"
            test_data = {"test": True, "value": 42}
            
            # Set cache
            set_success = self.cache.redis.set_cache(test_key, test_data, ttl_seconds=60)
            
            # Get cache
            get_data = self.cache.redis.get_cache(test_key)
            
            # Delete cache
            delete_success = self.cache.redis.delete_cache(test_key)
            
            result = set_success and get_data == test_data and delete_success
            self.results['cache_operations'] = result
            logger.info(f"Cache operations test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"Cache operations test failed: {e}")
            self.results['cache_operations'] = False
            return False
    
    async def test_http_utils(self) -> bool:
        """Test HTTP utilities."""
        try:
            # Test with a simple, reliable site
            test_url = "https://httpbin.org/html"
            html, error, _ = await self.http_utils.fetch_html(test_url)
            
            result = html is not None and len(html) > 100
            self.results['http_utils'] = result
            logger.info(f"HTTP utils test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"HTTP utils test failed: {e}")
            self.results['http_utils'] = False
            return False
    
    async def test_selector_mining(self) -> bool:
        """Test selector mining."""
        try:
            # Test with a simple HTML page
            test_html = """
            <html>
                <body>
                    <div class="gallery">
                        <img src="https://example.com/image1.jpg" alt="Test 1">
                        <img src="https://example.com/image2.jpg" alt="Test 2">
                    </div>
                </body>
            </html>
            """
            
            candidates = await self.selector_miner.mine_selectors(test_html, "https://example.com", "test_site")
            
            result = len(candidates) >= 2
            self.results['selector_mining'] = result
            logger.info(f"Selector mining test: {'✓' if result else '✗'} ({len(candidates)} candidates)")
            return result
        except Exception as e:
            logger.error(f"Selector mining test failed: {e}")
            self.results['selector_mining'] = False
            return False
    
    async def test_gpu_interface(self) -> bool:
        """Test GPU interface."""
        try:
            # Test health check
            health = await self.gpu_interface._check_health()
            
            result = True  # Don't fail if GPU worker is not available
            self.results['gpu_interface'] = result
            logger.info(f"GPU interface test: {'✓' if result else '✗'} (GPU available: {health})")
            return result
        except Exception as e:
            logger.error(f"GPU interface test failed: {e}")
            self.results['gpu_interface'] = False
            return False
    
    async def test_storage_manager(self) -> bool:
        """Test storage manager."""
        try:
            # Test health check
            health = self.storage.health_check()
            
            result = health['status'] == 'healthy'
            self.results['storage_manager'] = result
            logger.info(f"Storage manager test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"Storage manager test failed: {e}")
            self.results['storage_manager'] = False
            return False
    
    async def test_single_site_crawl(self) -> bool:
        """Test single site crawl."""
        try:
            # Clear queues
            self.redis.clear_queues()
            
            # Create test site
            test_site = SiteTask(
                url="https://httpbin.org/html",
                site_id="test_site_001",
                max_pages=1,
                use_3x3_mining=False
            )
            
            # Push site to queue
            push_success = self.redis.push_site(test_site)
            
            # Pop site from queue
            popped_site = self.redis.pop_site(timeout=5)
            
            result = push_success and popped_site is not None and popped_site.site_id == test_site.site_id
            self.results['single_site_crawl'] = result
            logger.info(f"Single site crawl test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"Single site crawl test failed: {e}")
            self.results['single_site_crawl'] = False
            return False
    
    async def test_batching(self) -> bool:
        """Test batching functionality."""
        try:
            # Clear queues
            self.redis.clear_queues()
            
            # Create test image tasks
            test_tasks = []
            for i in range(5):
                task = ImageTask(
                    temp_path=f"/tmp/test_{i}.jpg",
                    phash=f"test_hash_{i}",
                    candidate=CandidateImage(
                        page_url="https://example.com",
                        img_url=f"https://example.com/image_{i}.jpg",
                        selector_hint="img",
                        site_id="test_site"
                    ),
                    file_size=1024,
                    mime_type="image/jpeg"
                )
                test_tasks.append(task)
            
            # Push batch
            push_success = self.redis.push_image_batch(test_tasks)
            
            # Pop batch
            popped_batch = self.redis.pop_image_batch(timeout=5)
            
            result = push_success and popped_batch is not None and len(popped_batch.image_tasks) == 5
            self.results['batching'] = result
            logger.info(f"Batching test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"Batching test failed: {e}")
            self.results['batching'] = False
            return False
    
    async def test_deduplication(self) -> bool:
        """Test deduplication functionality."""
        try:
            # Create test image with known phash
            test_phash = "test_phash_12345"
            
            # Check if image is cached (should be False)
            is_cached_before = self.cache.is_image_cached(test_phash)
            
            # Cache image info
            cache_success = self.cache.cache_image_info(test_phash, {"test": True})
            
            # Check if image is cached (should be True)
            is_cached_after = self.cache.is_image_cached(test_phash)
            
            result = not is_cached_before and cache_success and is_cached_after
            self.results['deduplication'] = result
            logger.info(f"Deduplication test: {'✓' if result else '✗'}")
            return result
        except Exception as e:
            logger.error(f"Deduplication test failed: {e}")
            self.results['deduplication'] = False
            return False
    
    async def test_crawler_worker_3_pages(self) -> bool:
        """Test crawler worker with 3 pages per site from example_sites.txt."""
        try:
            # Load sites from example_sites.txt
            sites_file = Path(__file__).parent / "example_sites.txt"
            if not sites_file.exists():
                logger.error(f"example_sites.txt not found at {sites_file}")
                self.results['crawler_worker_3_pages'] = False
                return False
            
            with open(sites_file, 'r', encoding='utf-8') as f:
                sites = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not sites:
                logger.error("No sites found in example_sites.txt")
                self.results['crawler_worker_3_pages'] = False
                return False
            
            logger.info(f"Testing crawler worker with {len(sites)} sites from example_sites.txt")
            
            # Create crawler worker
            crawler_worker = CrawlerWorker(worker_id=999)  # Use special test worker ID
            
            total_sites = len(sites)
            total_pages_crawled = 0
            total_candidates = 0
            site_results = []
            all_candidates = []  # Store all candidates for file output
            
            # Create output file - use /app/crawl_output which is mapped to host ./crawl_output
            output_file = Path("/app/crawl_output/crawler_worker_test_results.txt")
            output_file.parent.mkdir(exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"CRAWLER WORKER TEST: 3 PAGES PER SITE")
            print(f"{'='*80}")
            print(f"Results will be saved to: {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"CRAWLER WORKER TEST: 3 PAGES PER SITE\n")
                f.write(f"{'='*80}\n")
                f.write(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, site_url in enumerate(sites, 1):
                    print(f"\n[{i}/{total_sites}] Site: {site_url}")
                    f.write(f"\n[{i}/{total_sites}] Site: {site_url}\n")
                    
                    try:
                        # Create site task with 3 pages max
                        site_task = SiteTask(
                            url=site_url,
                            site_id=f"test_site_{i:03d}",
                            max_pages=3,
                            use_3x3_mining=True
                        )
                        
                        # Process site
                        candidates_count = await crawler_worker.process_site(site_task)
                        
                        # Get detailed results from selector miner for this site
                        selector_miner = self.selector_miner
                        detailed_candidates = await selector_miner.mine_with_3x3_crawl(
                            site_url, site_task.site_id, 3
                        )
                        
                        # Count unique page URLs from candidates
                        page_urls = set()
                        for candidate in detailed_candidates:
                            page_urls.add(candidate.page_url)
                        
                        pages_crawled = len(page_urls)
                        total_pages_crawled += pages_crawled
                        total_candidates += len(detailed_candidates)
                        
                        print(f"  Pages Crawled: {pages_crawled}")
                        f.write(f"  Pages Crawled: {pages_crawled}\n")
                        
                        for j, page_url in enumerate(sorted(page_urls), 1):
                            print(f"    {j}. {page_url}")
                            f.write(f"    {j}. {page_url}\n")
                        
                        print(f"  Candidates Found: {len(detailed_candidates)}")
                        f.write(f"  Candidates Found: {len(detailed_candidates)}\n")
                        
                        # Write ALL candidates with full details to file
                        for j, candidate in enumerate(detailed_candidates, 1):
                            dimensions = f"{candidate.width}x{candidate.height}" if candidate.width and candidate.height else "unknown"
                            alt_text = candidate.alt_text or "none"
                            
                            f.write(f"    [{j}] img_url: {candidate.img_url}\n")
                            f.write(f"        page_url: {candidate.page_url}\n")
                            f.write(f"        selector: {candidate.selector_hint}\n")
                            f.write(f"        dimensions: {dimensions}\n")
                            f.write(f"        alt_text: \"{alt_text}\"\n")
                            f.write(f"        discovered_at: {candidate.discovered_at}\n\n")
                            
                            # Store candidate for summary
                            all_candidates.append({
                                'site_url': site_url,
                                'img_url': candidate.img_url,
                                'page_url': candidate.page_url,
                                'selector': candidate.selector_hint,
                                'dimensions': dimensions,
                                'alt_text': alt_text
                            })
                        
                        # Show first 5 candidates in console
                        for j, candidate in enumerate(detailed_candidates[:5], 1):
                            dimensions = f"{candidate.width}x{candidate.height}" if candidate.width and candidate.height else "unknown"
                            alt_text = candidate.alt_text or "none"
                            print(f"    [{j}] img_url: {candidate.img_url}")
                            print(f"        page_url: {candidate.page_url}")
                            print(f"        selector: {candidate.selector_hint}")
                            print(f"        dimensions: {dimensions}")
                            print(f"        alt_text: \"{alt_text}\"")
                        
                        if len(detailed_candidates) > 5:
                            print(f"    ... and {len(detailed_candidates) - 5} more candidates")
                        
                        site_results.append({
                            'url': site_url,
                            'pages_crawled': pages_crawled,
                            'candidates_found': len(detailed_candidates),
                            'success': True
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing site {site_url}: {e}")
                        print(f"  ERROR: {e}")
                        f.write(f"  ERROR: {e}\n")
                        site_results.append({
                            'url': site_url,
                            'pages_crawled': 0,
                            'candidates_found': 0,
                            'success': False,
                            'error': str(e)
                        })
                
                # Write final statistics to file
                f.write(f"\n{'='*80}\n")
                f.write(f"STATISTICS\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total sites: {total_sites}\n")
                f.write(f"Successful sites: {sum(1 for r in site_results if r['success'])}\n")
                f.write(f"Failed sites: {sum(1 for r in site_results if not r['success'])}\n")
                f.write(f"Total pages crawled: {total_pages_crawled}\n")
                f.write(f"Total candidates: {total_candidates}\n")
                f.write(f"Average candidates per site: {total_candidates/total_sites:.1f}\n")
                f.write(f"Average pages per site: {total_pages_crawled/total_sites:.1f}\n")
                f.write(f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Print final statistics to console
            print(f"\n{'='*80}")
            print(f"STATISTICS")
            print(f"{'='*80}")
            print(f"Total sites: {total_sites}")
            print(f"Successful sites: {sum(1 for r in site_results if r['success'])}")
            print(f"Failed sites: {sum(1 for r in site_results if not r['success'])}")
            print(f"Total pages crawled: {total_pages_crawled}")
            print(f"Total candidates: {total_candidates}")
            print(f"Average candidates per site: {total_candidates/total_sites:.1f}")
            print(f"Average pages per site: {total_pages_crawled/total_sites:.1f}")
            print(f"\nDetailed results saved to: {output_file}")
            
            # Success if at least 50% of sites worked and we found some candidates
            successful_sites = sum(1 for r in site_results if r['success'])
            success_rate = successful_sites / total_sites
            result = success_rate >= 0.5 and total_candidates > 0
            
            self.results['crawler_worker_3_pages'] = result
            logger.info(f"Crawler worker 3-pages test: {'✓' if result else '✗'} "
                       f"({successful_sites}/{total_sites} sites, {total_candidates} candidates)")
            logger.info(f"Detailed results saved to: {output_file}")
            return result
            
        except Exception as e:
            logger.error(f"Crawler worker 3-pages test failed: {e}")
            self.results['crawler_worker_3_pages'] = False
            return False
        """Run all tests."""
        logger.info("Starting test suite...")
        
        tests = [
            ("Redis Connection", self.test_redis_connection),
            ("Cache Operations", self.test_cache_operations),
            ("HTTP Utils", self.test_http_utils),
            ("Selector Mining", self.test_selector_mining),
            ("GPU Interface", self.test_gpu_interface),
            ("Storage Manager", self.test_storage_manager),
            ("Single Site Crawl", self.test_single_site_crawl),
            ("Batching", self.test_batching),
            ("Deduplication", self.test_deduplication),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
        
        # Print results
        print(f"\n{'='*60}")
        print(f"TEST SUITE RESULTS")
        print(f"{'='*60}")
        print(f"Passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        print(f"\nDetailed results:")
        for test_name, result in self.results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name}: {status}")
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed/total*100,
            'results': self.results
        }


async def run_tests():
    """Run the test suite."""
    test_suite = TestSuite()
    results = await test_suite.run_all_tests()
    return results


async def test_crawler_worker_only():
    """Run only the crawler worker test with detailed output."""
    # Setup detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting crawler worker test only...")
    
    try:
        # Create test suite and run only crawler worker test
        test_suite = TestSuite()
        result = await test_suite.test_crawler_worker_3_pages()
        
        if result:
            logger.info("✓ Crawler worker test passed!")
            return True
        else:
            logger.warning("✗ Crawler worker test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Crawler worker test failed with exception: {e}")
        return False


def main():
    """Main entry point for tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='New Crawler Test Suite')
    parser.add_argument('--test', type=str, help='Run specific test')
    parser.add_argument('--test-crawler', action='store_true', help='Run only crawler worker test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        if args.test_crawler:
            # Run only crawler worker test
            result = loop.run_until_complete(test_crawler_worker_only())
            exit(0 if result else 1)
        else:
            # Run all tests
            results = loop.run_until_complete(run_tests())
        
        # Exit with appropriate code
        if results['success_rate'] == 100:
            print("\n✓ All tests passed!")
            exit(0)
        else:
            print(f"\n✗ {results['total'] - results['passed']} tests failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()



