#!/usr/bin/env python3
"""
List Crawler CLI - Command line interface for crawling multiple sites from a text file.

Usage:
    python -m backend.app.crawler.crawl_list [OPTIONS]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.crawler.list_crawler import ListCrawler
from app.crawler.crawler_settings import (
    LIST_CRAWL_DEFAULT_SITES_FILE,
    LIST_CRAWL_MAX_PAGES_PER_SITE,
    LIST_CRAWL_MAX_IMAGES_PER_SITE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('list_crawl.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Crawl multiple sites from a text file using selector miner and crawler integration'
    )
    
    parser.add_argument(
        '--sites-file', '-f',
        default=LIST_CRAWL_DEFAULT_SITES_FILE,
        help=f'Path to text file containing list of sites (default: {LIST_CRAWL_DEFAULT_SITES_FILE})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='list_crawl_results',
        help='Directory to store crawling results (default: list_crawl_results)'
    )
    
    parser.add_argument(
        '--max-concurrent', '-c',
        type=int,
        default=2,
        help='Maximum number of sites to crawl concurrently (default: 2)'
    )
    
    parser.add_argument(
        '--max-pages-per-site', '-p',
        type=int,
        default=LIST_CRAWL_MAX_PAGES_PER_SITE,
        help=f'Maximum pages to crawl per site (default: {LIST_CRAWL_MAX_PAGES_PER_SITE})'
    )
    
    parser.add_argument(
        '--max-images-per-site', '-i',
        type=int,
        default=LIST_CRAWL_MAX_IMAGES_PER_SITE,
        help=f'Maximum images to save per site (default: {LIST_CRAWL_MAX_IMAGES_PER_SITE})'
    )
    
    parser.add_argument(
        '--require-face',
        action='store_true',
        help='Require face detection for saved images'
    )
    
    parser.add_argument(
        '--no-selector-mining',
        action='store_true',
        help='Skip automatic selector mining for new sites'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate sites file
    if not Path(args.sites_file).exists():
        logger.error(f"Sites file not found: {args.sites_file}")
        logger.info("Create a text file with one URL per line, for example:")
        logger.info("https://example1.com")
        logger.info("https://example2.com")
        logger.info("# This is a comment")
        sys.exit(1)
    
    # Update settings based on arguments
    from app.crawler import crawler_settings
    crawler_settings.LIST_CRAWL_MAX_PAGES_PER_SITE = args.max_pages_per_site
    crawler_settings.LIST_CRAWL_MAX_IMAGES_PER_SITE = args.max_images_per_site
    crawler_settings.LIST_CRAWL_AUTO_SELECTOR_MINING = not args.no_selector_mining
    
    logger.info(f"Starting crawl: {args.sites_file} -> {args.output_dir}")
    
    # Run the list crawler
    async def run_crawler():
        crawler = ListCrawler(
            sites_file=args.sites_file,
            output_dir=args.output_dir
        )
        
        try:
            results = await crawler.crawl_list(max_concurrent=args.max_concurrent)
            
            if 'error' in results:
                logger.error(f"Crawl failed: {results['error']}")
                return 1
            
            # Calculate actual statistics from results
            results_list = results['results']
            actual_stats = {
                'sites_processed': sum(1 for r in results_list if r['success']),
                'sites_failed': sum(1 for r in results_list if not r['success']),
                'total_images_found': sum(r.get('images_found', 0) for r in results_list),
                'total_images_saved': sum(r.get('images_saved', 0) for r in results_list),
                'total_thumbnails_saved': sum(r.get('thumbnails_saved', 0) for r in results_list),
                'total_images_processed': sum(r.get('images_saved', 0) + r.get('thumbnails_saved', 0) for r in results_list),
                'total_duration_seconds': sum(r.get('total_duration_seconds', 0) for r in results_list),
                'selector_mining_attempts': sum(1 for r in results_list if r.get('mining_attempted', False)),
                'selector_mining_successes': sum(1 for r in results_list if r.get('mining_success', False))
            }
            
            # Print summary
            logger.info(f"SUMMARY: {actual_stats['sites_processed']}/{results['total_sites']} sites successful")
            logger.info(f"Images found: {actual_stats['total_images_found']}, Images processed: {actual_stats['total_images_processed']}")
            logger.info(f"Images saved: {actual_stats['total_images_saved']}, Thumbnails saved: {actual_stats['total_thumbnails_saved']}")
            
            # Calculate and display images per second
            if actual_stats['total_duration_seconds'] > 0:
                images_per_second = actual_stats['total_images_processed'] / actual_stats['total_duration_seconds']
                logger.info(f"Images processed per second: {images_per_second:.2f}")
            
            # Count critical failures
            critical_failures = 0
            for result in results_list:
                if result.get('thumbnails_saved', 0) == 0 and result.get('images_found', 0) > 0:
                    critical_failures += 1
                if result.get('mining_attempted', False) and not result.get('mining_success', False):
                    critical_failures += 1
            
            if critical_failures > 0:
                logger.warning(f"CRITICAL FAILURES: {critical_failures} sites require immediate attention")
            
            return 0 if actual_stats['sites_failed'] == 0 else 1
            
        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error during crawl: {e}")
            return 1
    
    # Run the async crawler
    try:
        exit_code = asyncio.run(run_crawler())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
