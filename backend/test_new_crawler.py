#!/usr/bin/env python3
"""
Real-world test script for the new refactored crawler architecture.

Tests the new crawler against real sites from the sites.txt file using Docker
to avoid dependency issues. Follows the testing criteria from the existing
crawl test output.
"""

import asyncio
import logging
import os
import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, '/app')

from app.crawler import CrawlerEngine, CrawlerConfig, crawl_list

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrawlTestRunner:
    """Test runner for the new crawler architecture."""
    
    def __init__(self):
        self.test_results = []
        self.critical_failures = []
        self.start_time = time.time()
        
    def load_test_sites(self, sites_file: str = "/app/sites.txt") -> List[str]:
        """Load test sites from file."""
        sites = []
        
        if not os.path.exists(sites_file):
            logger.error(f"Sites file not found: {sites_file}")
            return sites
        
        try:
            with open(sites_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Basic URL validation
                    if line.startswith(('http://', 'https://')):
                        sites.append(line)
                    else:
                        logger.warning(f"Invalid URL on line {line_num}: {line}")
        
        except Exception as e:
            logger.error(f"Error reading sites file {sites_file}: {e}")
        
        logger.info(f"Loaded {len(sites)} test sites")
        return sites
    
    def analyze_crawl_result(self, result, config: CrawlerConfig) -> Dict[str, Any]:
        """Analyze crawl result and identify critical failures."""
        analysis = {
            'success': result.success,
            'critical_failures': [],
            'warnings': [],
            'metrics': {
                'images_found': result.images_found,
                'images_processed': result.images_processed,
                'faces_detected': result.faces_detected,
                'images_saved': result.images_saved,
                'thumbnails_saved': result.thumbnails_saved,
                'duration': result.duration_seconds
            }
        }
        
        # Critical failure criteria based on existing test output
        if not result.success:
            analysis['critical_failures'].append("CRITICAL FAIL: Crawl failed")
        
        if result.mining_attempted and not result.mining_success:
            analysis['critical_failures'].append("CRITICAL FAIL: Selector mining failed")
        
        # Only expect thumbnails if faces were detected and cropping is enabled
        if (result.images_found > 0 and result.thumbnails_saved == 0 and 
            result.faces_detected > 0 and config.crop_faces):
            analysis['critical_failures'].append("CRITICAL FAIL: No thumbnails saved despite finding faces")
        
        if result.images_found > 0 and result.images_processed == 0:
            analysis['critical_failures'].append("CRITICAL FAIL: No images processed despite finding images")
        
        if result.images_found > 0 and result.faces_detected == 0 and result.images_saved == 0:
            analysis['critical_failures'].append("CRITICAL FAIL: No faces detected and no images saved")
        
        # Performance warnings
        if result.duration_seconds > 60:
            analysis['warnings'].append(f"Slow crawl: {result.duration_seconds:.2f}s")
        
        if result.images_found > 0 and result.images_processed / result.images_found < 0.5:
            analysis['warnings'].append("Low processing rate: <50% of found images processed")
        
        return analysis
    
    async def test_single_site(self, url: str, config: CrawlerConfig) -> Dict[str, Any]:
        """Test crawling a single site."""
        logger.info(f"Testing site: {url}")
        
        start_time = time.time()
        
        try:
            async with CrawlerEngine(config) as crawler:
                result = await crawler.crawl_site(url)
                
                # Analyze result
                analysis = self.analyze_crawl_result(result, config)
                
                # Log results
                if analysis['critical_failures']:
                    logger.error(f"❌ {url}: {' | '.join(analysis['critical_failures'])}")
                    self.critical_failures.extend(analysis['critical_failures'])
                elif analysis['warnings']:
                    logger.warning(f"⚠️  {url}: {' | '.join(analysis['warnings'])}")
                else:
                    logger.info(f"✅ {url}: OK - {result.images_saved} images saved, {result.faces_detected} faces detected")
                
                # Store test result
                test_result = {
                    'url': url,
                    'domain': result.domain,
                    'success': result.success,
                    'analysis': analysis,
                    'crawl_result': result,
                    'test_duration': time.time() - start_time
                }
                
                self.test_results.append(test_result)
                return test_result
                
        except Exception as e:
            logger.error(f"❌ {url}: Test failed with exception: {e}")
            
            test_result = {
                'url': url,
                'domain': url.split('/')[2] if '/' in url else url,
                'success': False,
                'analysis': {
                    'success': False,
                    'critical_failures': [f"CRITICAL FAIL: Test exception: {str(e)}"],
                    'warnings': [],
                    'metrics': {}
                },
                'crawl_result': None,
                'test_duration': time.time() - start_time,
                'exception': str(e)
            }
            
            self.test_results.append(test_result)
            self.critical_failures.append(f"Test exception for {url}: {str(e)}")
            return test_result
    
    async def test_multiple_sites(self, urls: List[str], config: CrawlerConfig, max_concurrent: int = 2) -> List[Dict[str, Any]]:
        """Test crawling multiple sites concurrently."""
        logger.info(f"Testing {len(urls)} sites with max_concurrent={max_concurrent}")
        
        async with CrawlerEngine(config) as crawler:
            results = await crawler.crawl_list(urls, max_concurrent)
            
            # Analyze all results
            for result in results:
                analysis = self.analyze_crawl_result(result, config)
                
                if analysis['critical_failures']:
                    logger.error(f"❌ {result.url}: {' | '.join(analysis['critical_failures'])}")
                    self.critical_failures.extend(analysis['critical_failures'])
                elif analysis['warnings']:
                    logger.warning(f"⚠️  {result.url}: {' | '.join(analysis['warnings'])}")
                else:
                    logger.info(f"✅ {result.url}: OK - {result.images_saved} images saved, {result.faces_detected} faces detected")
                
                test_result = {
                    'url': result.url,
                    'domain': result.domain,
                    'success': result.success,
                    'analysis': analysis,
                    'crawl_result': result,
                    'test_duration': 0.0  # Not available for batch results
                }
                
                self.test_results.append(test_result)
            
            return self.test_results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        # Calculate statistics
        total_sites = len(self.test_results)
        successful_sites = sum(1 for r in self.test_results if r['success'])
        failed_sites = total_sites - successful_sites
        
        total_images_found = sum(r['crawl_result'].images_found for r in self.test_results if r['crawl_result'])
        total_images_saved = sum(r['crawl_result'].images_saved for r in self.test_results if r['crawl_result'])
        total_faces_detected = sum(r['crawl_result'].faces_detected for r in self.test_results if r['crawl_result'])
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("NEW CRAWLER ARCHITECTURE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Duration: {total_duration:.2f} seconds")
        report.append(f"Total Sites Tested: {total_sites}")
        report.append(f"Successful Crawls: {successful_sites}")
        report.append(f"Failed Crawls: {failed_sites}")
        report.append(f"Success Rate: {(successful_sites/total_sites*100):.1f}%" if total_sites > 0 else "N/A")
        report.append("")
        
        report.append("OVERALL METRICS:")
        report.append(f"Total Images Found: {total_images_found}")
        report.append(f"Total Images Saved: {total_images_saved}")
        report.append(f"Total Faces Detected: {total_faces_detected}")
        report.append(f"Images per Second: {total_images_saved/total_duration:.2f}" if total_duration > 0 else "N/A")
        report.append("")
        
        # Critical failures
        if self.critical_failures:
            report.append("CRITICAL FAILURES:")
            for failure in set(self.critical_failures):
                count = self.critical_failures.count(failure)
                report.append(f"  - {failure} ({count} occurrences)")
            report.append("")
        
        # Per-site results
        report.append("PER-SITE DETAILED RESULTS:")
        report.append("-" * 80)
        
        for result in self.test_results:
            report.append(f"Domain: {result['domain']}")
            report.append(f"URL: {result['url']}")
            report.append(f"Crawl Success: {result['success']}")
            
            if result['crawl_result']:
                cr = result['crawl_result']
                report.append(f"Images Found: {cr.images_found}")
                report.append(f"Images Processed: {cr.images_processed}")
                report.append(f"Images Saved: {cr.images_saved}")
                report.append(f"Thumbnails Saved: {cr.thumbnails_saved}")
                report.append(f"Faces Detected: {cr.faces_detected}")
                report.append(f"Duration: {cr.duration_seconds:.2f} seconds")
                
                if cr.mining_attempted:
                    mining_status = "SUCCESS" if cr.mining_success else "CRITICAL FAIL"
                    report.append(f"Selector Mining: Attempted - {mining_status}")
                else:
                    report.append("Selector Mining: Skipped (using existing recipe)")
                
                # Status
                if result['analysis']['critical_failures']:
                    report.append(f"Status: {' | '.join(result['analysis']['critical_failures'])}")
                elif result['analysis']['warnings']:
                    report.append(f"Status: WARNING - {' | '.join(result['analysis']['warnings'])}")
                else:
                    report.append("Status: OK")
            else:
                report.append("Status: CRITICAL FAIL: Test exception")
            
            report.append("-" * 50)
            report.append("")
        
        # Summary
        report.append("TEST SUMMARY:")
        if self.critical_failures:
            report.append(f"❌ TEST FAILED: {len(set(self.critical_failures))} critical failure types")
            report.append("The new crawler architecture has critical issues that need to be addressed.")
        else:
            report.append("✅ TEST PASSED: No critical failures detected")
            report.append("The new crawler architecture is working correctly.")
        
        return "\n".join(report)


async def main():
    """Main test function."""
    logger.info("Starting new crawler architecture test")
    
    # Create test configuration
    config = CrawlerConfig(
        max_pages=10,  # Increased pages for better selector mining
        max_images=50,  # Increased images for better testing
        require_faces=False,  # Don't require faces to see more results
        crop_faces=True,
        concurrent_downloads=6,  # Increased for better performance
        timeout_seconds=45,  # Increased timeout for complex sites
        memory_pressure_threshold=0.8,  # Higher threshold for testing
        batch_size=20,  # Increased batch size
        gc_frequency=50,
        face_threads=3,  # Increased for better performance
        list_crawl_auto_selector_mining=True,
        list_crawl_skip_existing_recipes=False,  # Force mining for testing
        list_crawl_max_pages_per_site=8,  # More pages per site
        list_crawl_max_images_per_site=40  # More images per site
    )
    
    # Initialize test runner
    test_runner = CrawlTestRunner()
    
    # Load test sites
    sites = test_runner.load_test_sites()
    if not sites:
        logger.error("No test sites loaded. Exiting.")
        return 1
    
    # Test more sites for better coverage
    test_sites = sites[:5]  # Test first 5 sites
    logger.info(f"Testing {len(test_sites)} sites: {test_sites}")
    
    # Run tests
    try:
        # Test sites individually for detailed analysis
        for site in test_sites:
            await test_runner.test_single_site(site, config)
        
        # Generate and save report
        report = test_runner.generate_test_report()
        
        # Save report to file
        report_file = f"/app/crawl_test_output_new_{int(time.time())}.log"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print report
        print(report)
        
        # Return exit code based on critical failures
        return 1 if test_runner.critical_failures else 0
        
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
