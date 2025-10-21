#!/usr/bin/env python3
"""
Multi-Site Crawl Testing Script

Comprehensive test script for the refactored crawler architecture.
Tests individual sites and full multi-site crawls with detailed error categorization,
progress tracking, and actionable debugging information.
"""

import asyncio
import argparse
import logging
import os
import sys
import time
import json
import psutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the app directory to the path
sys.path.insert(0, '/app')

from app.crawler import CrawlerEngine, CrawlerConfig, crawl_list
from app.crawler.core import CrawlResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SiteTestResult:
    """Result of testing a single site."""
    url: str
    domain: str
    success: bool
    images_found: int
    images_processed: int
    faces_detected: int
    images_saved: int
    thumbnails_saved: int
    duration_seconds: float
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    mining_success: bool = False
    http_errors: int = 0
    processing_errors: int = 0
    storage_errors: int = 0
    memory_peak_mb: float = 0.0
    retry_count: int = 0


@dataclass
class MultiSiteTestResult:
    """Result of testing multiple sites."""
    total_sites: int
    successful_sites: int
    failed_sites: int
    total_images_found: int
    total_images_processed: int
    total_faces_detected: int
    total_images_saved: int
    total_thumbnails_saved: int
    total_duration_seconds: float
    site_results: List[SiteTestResult]
    critical_failures: List[str]
    performance_issues: List[str]
    recommendations: List[str]


class MultiSiteTestRunner:
    """Test runner for multi-site crawling."""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        self.memory_monitor = []
        
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
    
    async def monitor_memory(self):
        """Monitor memory usage during testing."""
        while True:
            try:
                memory = psutil.virtual_memory()
                self.memory_monitor.append({
                    'timestamp': time.time(),
                    'percent': memory.percent,
                    'available_mb': memory.available / (1024 * 1024),
                    'used_mb': memory.used / (1024 * 1024)
                })
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
    
    def get_memory_peak(self) -> float:
        """Get peak memory usage in MB."""
        if not self.memory_monitor:
            return 0.0
        return max(entry['used_mb'] for entry in self.memory_monitor)
    
    def categorize_error(self, error: Exception) -> Tuple[str, str]:
        """Categorize error for debugging."""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ['timeout', 'connection', 'network', 'http']):
            return "HTTP", f"Network/HTTP error: {error}"
        elif any(keyword in error_str for keyword in ['mining', 'selector', 'recipe']):
            return "MINING", f"Selector mining error: {error}"
        elif any(keyword in error_str for keyword in ['extract', 'parse', 'html', 'url']):
            return "EXTRACTION", f"Image extraction error: {error}"
        elif any(keyword in error_str for keyword in ['process', 'face', 'detect', 'enhance']):
            return "PROCESSING", f"Image processing error: {error}"
        elif any(keyword in error_str for keyword in ['storage', 'minio', 's3', 'save']):
            return "STORAGE", f"Storage error: {error}"
        else:
            return "UNKNOWN", f"Unknown error: {error}"
    
    async def test_single_site(self, url: str, config: CrawlerConfig, timeout: int = 300) -> SiteTestResult:
        """Test a single site with detailed error tracking."""
        domain = url.split('/')[2]
        start_time = time.time()
        retry_count = 0
        max_retries = 3
        
        logger.info(f"Testing site: {url}")
        
        while retry_count <= max_retries:
            try:
                # Create crawler engine
                engine = CrawlerEngine(config)
                
                # Run crawl with timeout
                result = await asyncio.wait_for(
                    engine.crawl_site(url),
                    timeout=timeout
                )
                
                # Calculate metrics
                duration = time.time() - start_time
                memory_peak = self.get_memory_peak()
                
                # Determine success
                success = (
                    result.success and 
                    result.images_found >= 10 and  # Minimum threshold
                    result.images_processed > 0
                )
                
                # Count errors by category
                http_errors = 0
                processing_errors = 0
                storage_errors = 0
                
                if result.error:
                    error_category, error_message = self.categorize_error(Exception(result.error))
                    if error_category == "HTTP":
                        http_errors = 1
                    elif error_category == "PROCESSING":
                        processing_errors = 1
                    elif error_category == "STORAGE":
                        storage_errors = 1
                
                return SiteTestResult(
                    url=url,
                    domain=domain,
                    success=success,
                    images_found=result.images_found,
                    images_processed=result.images_processed,
                    faces_detected=result.faces_detected,
                    images_saved=result.images_saved,
                    thumbnails_saved=result.thumbnails_saved,
                    duration_seconds=duration,
                    error_category=error_category if result.error else None,
                    error_message=result.error,
                    mining_success=result.mining_success,
                    http_errors=http_errors,
                    processing_errors=processing_errors,
                    storage_errors=storage_errors,
                    memory_peak_mb=memory_peak,
                    retry_count=retry_count
                )
                
            except asyncio.TimeoutError:
                retry_count += 1
                logger.warning(f"Timeout for {url} (attempt {retry_count}/{max_retries + 1})")
                if retry_count <= max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                else:
                    return SiteTestResult(
                        url=url,
                        domain=domain,
                        success=False,
                        images_found=0,
                        images_processed=0,
                        faces_detected=0,
                        images_saved=0,
                        thumbnails_saved=0,
                        duration_seconds=time.time() - start_time,
                        error_category="TIMEOUT",
                        error_message=f"Site timed out after {timeout}s",
                        retry_count=retry_count
                    )
            
            except Exception as e:
                retry_count += 1
                error_category, error_message = self.categorize_error(e)
                logger.error(f"Error testing {url} (attempt {retry_count}/{max_retries + 1}): {e}")
                
                if retry_count <= max_retries:
                    await asyncio.sleep(2 ** retry_count)
                    continue
                else:
                    return SiteTestResult(
                        url=url,
                        domain=domain,
                        success=False,
                        images_found=0,
                        images_processed=0,
                        faces_detected=0,
                        images_saved=0,
                        thumbnails_saved=0,
                        duration_seconds=time.time() - start_time,
                        error_category=error_category,
                        error_message=error_message,
                        retry_count=retry_count
                    )
    
    async def test_all_sites(self, sites: List[str], config: CrawlerConfig) -> MultiSiteTestResult:
        """Test all sites and generate comprehensive report."""
        logger.info(f"Starting multi-site test with {len(sites)} sites")
        
        # Start memory monitoring
        memory_task = asyncio.create_task(self.monitor_memory())
        
        try:
            site_results = []
            successful_sites = 0
            failed_sites = 0
            total_images_found = 0
            total_images_processed = 0
            total_faces_detected = 0
            total_images_saved = 0
            total_thumbnails_saved = 0
            critical_failures = []
            performance_issues = []
            recommendations = []
            
            for i, site in enumerate(sites, 1):
                logger.info(f"Testing site {i}/{len(sites)}: {site}")
                
                result = await self.test_single_site(site, config)
                site_results.append(result)
                
                # Update totals
                if result.success:
                    successful_sites += 1
                else:
                    failed_sites += 1
                
                total_images_found += result.images_found
                total_images_processed += result.images_processed
                total_faces_detected += result.faces_detected
                total_images_saved += result.images_saved
                total_thumbnails_saved += result.thumbnails_saved
                
                # Analyze results
                if not result.success:
                    if result.error_category == "HTTP":
                        critical_failures.append(f"{site}: HTTP connectivity issues")
                    elif result.error_category == "MINING":
                        critical_failures.append(f"{site}: Selector mining failed")
                    elif result.error_category == "EXTRACTION":
                        critical_failures.append(f"{site}: Image extraction failed")
                    elif result.error_category == "PROCESSING":
                        critical_failures.append(f"{site}: Image processing failed")
                    elif result.error_category == "STORAGE":
                        critical_failures.append(f"{site}: Storage operations failed")
                
                if result.images_found < 10:
                    performance_issues.append(f"{site}: Low image count ({result.images_found})")
                
                if result.duration_seconds > 120:
                    performance_issues.append(f"{site}: Slow performance ({result.duration_seconds:.1f}s)")
                
                if result.memory_peak_mb > 1000:
                    performance_issues.append(f"{site}: High memory usage ({result.memory_peak_mb:.1f}MB)")
                
                # Progress update
                logger.info(f"Site {i}/{len(sites)} complete: {result.images_found} images, {result.duration_seconds:.1f}s")
            
            # Generate recommendations
            if failed_sites > 0:
                recommendations.append("Review failed sites and fix critical issues")
            
            if total_images_found < len(sites) * 50:
                recommendations.append("Consider adjusting extraction patterns for better image discovery")
            
            if any(r.duration_seconds > 120 for r in site_results):
                recommendations.append("Optimize performance for slow sites")
            
            total_duration = time.time() - self.start_time
            
            return MultiSiteTestResult(
                total_sites=len(sites),
                successful_sites=successful_sites,
                failed_sites=failed_sites,
                total_images_found=total_images_found,
                total_images_processed=total_images_processed,
                total_faces_detected=total_faces_detected,
                total_images_saved=total_images_saved,
                total_thumbnails_saved=total_thumbnails_saved,
                total_duration_seconds=total_duration,
                site_results=site_results,
                critical_failures=critical_failures,
                performance_issues=performance_issues,
                recommendations=recommendations
            )
        
        finally:
            memory_task.cancel()
            try:
                await memory_task
            except asyncio.CancelledError:
                pass
    
    def generate_report(self, result: MultiSiteTestResult) -> str:
        """Generate detailed test report."""
        report = []
        report.append("# Multi-Site Crawl Test Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total Sites: {result.total_sites}")
        report.append(f"- Successful: {result.successful_sites}")
        report.append(f"- Failed: {result.failed_sites}")
        report.append(f"- Success Rate: {result.successful_sites/result.total_sites*100:.1f}%")
        report.append(f"- Total Images Found: {result.total_images_found}")
        report.append(f"- Total Images Processed: {result.total_images_processed}")
        report.append(f"- Total Faces Detected: {result.total_faces_detected}")
        report.append(f"- Total Images Saved: {result.total_images_saved}")
        report.append(f"- Total Thumbnails Saved: {result.total_thumbnails_saved}")
        report.append(f"- Total Duration: {result.total_duration_seconds:.1f}s")
        report.append("")
        
        # Per-site results
        report.append("## Per-Site Results")
        for site_result in result.site_results:
            status = "✅ PASS" if site_result.success else "❌ FAIL"
            report.append(f"### {site_result.domain}")
            report.append(f"- Status: {status}")
            report.append(f"- Images Found: {site_result.images_found}")
            report.append(f"- Images Processed: {site_result.images_processed}")
            report.append(f"- Faces Detected: {site_result.faces_detected}")
            report.append(f"- Images Saved: {site_result.images_saved}")
            report.append(f"- Thumbnails Saved: {site_result.thumbnails_saved}")
            report.append(f"- Duration: {site_result.duration_seconds:.1f}s")
            report.append(f"- Memory Peak: {site_result.memory_peak_mb:.1f}MB")
            if site_result.error_message:
                report.append(f"- Error: {site_result.error_message}")
            report.append("")
        
        # Critical failures
        if result.critical_failures:
            report.append("## Critical Failures")
            for failure in result.critical_failures:
                report.append(f"- {failure}")
            report.append("")
        
        # Performance issues
        if result.performance_issues:
            report.append("## Performance Issues")
            for issue in result.performance_issues:
                report.append(f"- {issue}")
            report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("## Recommendations")
            for rec in result.recommendations:
                report.append(f"- {rec}")
            report.append("")
        
        return "\n".join(report)


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Multi-site crawl testing')
    parser.add_argument('--site', help='Test specific site')
    parser.add_argument('--all-sites', action='store_true', help='Test all sites')
    parser.add_argument('--sites-file', default='/app/sites.txt', help='Sites file path')
    parser.add_argument('--timeout', type=int, default=300, help='Per-site timeout in seconds')
    parser.add_argument('--output', help='Output report file')
    
    args = parser.parse_args()
    
    if not args.site and not args.all_sites:
        parser.error("Must specify either --site or --all-sites")
    
    # Load configuration
    config = CrawlerConfig.from_env()
    logger.info(f"Loaded configuration: max_images={config.max_images}, max_pages={config.max_pages}")
    
    # Create test runner
    runner = MultiSiteTestRunner()
    
    if args.site:
        # Test single site
        logger.info(f"Testing single site: {args.site}")
        result = await runner.test_single_site(args.site, config, args.timeout)
        
        # Print result
        print(f"\nSite Test Result:")
        print(f"URL: {result.url}")
        print(f"Success: {result.success}")
        print(f"Images Found: {result.images_found}")
        print(f"Images Processed: {result.images_processed}")
        print(f"Faces Detected: {result.faces_detected}")
        print(f"Images Saved: {result.images_saved}")
        print(f"Thumbnails Saved: {result.thumbnails_saved}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")
    
    else:
        # Test all sites
        sites = runner.load_test_sites(args.sites_file)
        if not sites:
            logger.error("No sites to test")
            return 1
        
        result = await runner.test_all_sites(sites, config)
        
        # Generate and save report
        report = runner.generate_report(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")
        else:
            print(report)
        
        # Print summary
        print(f"\nTest Complete!")
        print(f"Successful: {result.successful_sites}/{result.total_sites}")
        print(f"Total Images: {result.total_images_found}")
        print(f"Total Duration: {result.total_duration_seconds:.1f}s")
        
        return 0 if result.failed_sites == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
