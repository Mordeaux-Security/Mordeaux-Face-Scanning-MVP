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
    
    logger.info(f"Starting list crawl with settings:")
    logger.info(f"  Sites file: {args.sites_file}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Max concurrent: {args.max_concurrent}")
    logger.info(f"  Max pages per site: {args.max_pages_per_site}")
    logger.info(f"  Max images per site: {args.max_images_per_site}")
    logger.info(f"  Require face: {args.require_face}")
    logger.info(f"  Auto selector mining: {not args.no_selector_mining}")
    
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
                'total_images_saved': sum(r.get('images_saved', 0) for r in results_list),
                'total_thumbnails_saved': sum(r.get('thumbnails_saved', 0) for r in results_list),
                'selector_mining_attempts': sum(1 for r in results_list if r.get('mining_attempted', False)),
                'selector_mining_successes': sum(1 for r in results_list if r.get('mining_success', False))
            }
            
            # Print summary
            logger.info("\n" + "="*50)
            logger.info("CRAWL SUMMARY")
            logger.info("="*50)
            logger.info(f"Total sites processed: {results['total_sites']}")
            logger.info(f"Successful crawls: {actual_stats['sites_processed']}")
            logger.info(f"Failed crawls: {actual_stats['sites_failed']}")
            logger.info(f"Total images saved: {actual_stats['total_images_saved']}")
            logger.info(f"Total thumbnails saved: {actual_stats['total_thumbnails_saved']}")
            logger.info(f"Selector mining attempts: {actual_stats['selector_mining_attempts']}")
            logger.info(f"Selector mining successes: {actual_stats['selector_mining_successes']}")
            
            # Print per-site summary
            logger.info("\n" + "="*50)
            logger.info("PER-SITE RESULTS")
            logger.info("="*50)
            for result in results_list:
                logger.info(f"✅ {result['domain']}: {result.get('images_saved', 0)} images, {result.get('thumbnails_saved', 0)} thumbnails")
                if result.get('mining_attempted', False):
                    status = "✅ SUCCESS" if result.get('mining_success', False) else "❌ FAILED"
                    logger.info(f"   Selector Mining: {status}")
                else:
                    logger.info(f"   Selector Mining: Skipped (existing recipe)")
            logger.info("="*50)
            
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
