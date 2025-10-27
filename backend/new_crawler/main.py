"""
Main Entry Point for New Crawler System

Provides command-line interface and main entry point for the new crawler system.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

from .config import get_config, validate_configuration
from .orchestrator import Orchestrator
from .redis_manager import get_redis_manager
from .cache_manager import get_cache_manager
from .storage_manager import get_storage_manager
from .gpu_interface import get_gpu_interface

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "info"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('new_crawler.log')
        ]
    )


def load_sites_from_file(file_path: str) -> List[str]:
    """Load sites from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sites = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(sites)} sites from {file_path}")
        return sites
    except Exception as e:
        logger.error(f"Failed to load sites from {file_path}: {e}")
        return []


def load_sites_from_args(sites_arg: List[str]) -> List[str]:
    """Load sites from command line arguments."""
    sites = []
    for site in sites_arg:
        if site.startswith('http://') or site.startswith('https://'):
            sites.append(site)
        else:
            logger.warning(f"Skipping invalid site URL: {site}")
    return sites


async def health_check():
    """Perform system health check."""
    logger.info("Performing system health check...")
    
    config = get_config()
    
    # Check Redis
    redis_manager = get_redis_manager()
    redis_healthy = redis_manager.test_connection()
    logger.info(f"Redis: {'✓' if redis_healthy else '✗'}")
    
    # Check cache manager
    cache_manager = get_cache_manager()
    cache_health = cache_manager.health_check()
    logger.info(f"Cache: {'✓' if cache_health['status'] == 'healthy' else '✗'}")
    
    # Check storage
    storage_manager = get_storage_manager()
    storage_health = storage_manager.health_check()
    logger.info(f"Storage: {'✓' if storage_health['status'] == 'healthy' else '✗'}")
    
    # Check GPU worker
    gpu_interface = get_gpu_interface()
    gpu_healthy = await gpu_interface._check_health()
    logger.info(f"GPU Worker: {'✓' if gpu_healthy else '✗'}")
    
    # Overall health
    all_healthy = redis_healthy and cache_health['status'] == 'healthy' and storage_health['status'] == 'healthy'
    
    if all_healthy:
        logger.info("✓ All systems healthy")
    else:
        logger.warning("✗ Some systems unhealthy")
    
    return all_healthy


async def run_crawl(sites: List[str], config_file: str = None):
    """Run the crawl process."""
    logger.info(f"Starting crawl of {len(sites)} sites")
    
    # Validate configuration
    if not validate_configuration():
        logger.warning("Configuration validation failed, but continuing...")
    
    # Perform health check
    if not await health_check():
        logger.warning("Health check failed, but continuing...")
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    try:
        # Run crawl
        results = await orchestrator.crawl_sites(sites)
        
        # Print results
        print_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        raise
    finally:
        orchestrator.stop()


def print_results(results):
    """Print crawl results."""
    print(f"\n{'='*60}")
    print(f"NEW CRAWLER RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {results.total_time_seconds:.1f}s")
    print(f"Success rate: {results.success_rate:.1f}%")
    print(f"Sites processed: {len(results.sites)}")
    print(f"Total images found: {results.total_images_found}")
    print(f"Total images processed: {sum(site.images_processed for site in results.sites)}")
    print(f"Total faces detected: {results.total_faces_detected}")
    print(f"Raw images saved: {sum(site.raw_images_saved for site in results.sites)}")
    print(f"Thumbnails saved: {sum(site.thumbnails_saved for site in results.sites)}")
    
    print(f"\n{'='*60}")
    print(f"PER-SITE RESULTS")
    print(f"{'='*60}")
    
    for site_stats in results.sites:
        print(f"\n{site_stats.site_url}:")
        print(f"  Pages crawled: {site_stats.pages_crawled}")
        print(f"  Images found: {site_stats.images_found}")
        print(f"  Images processed: {site_stats.images_processed}")
        print(f"  Images cached: {site_stats.images_cached}")
        print(f"  Faces detected: {site_stats.faces_detected}")
        print(f"  Raw images saved: {site_stats.raw_images_saved}")
        print(f"  Thumbnails saved: {site_stats.thumbnails_saved}")
        print(f"  Success rate: {site_stats.success_rate:.1f}%")
        print(f"  Processing time: {site_stats.total_time_seconds:.1f}s")
        
        if site_stats.errors:
            print(f"  Errors: {len(site_stats.errors)}")
            for error in site_stats.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
    
    print(f"\n{'='*60}")
    print(f"SYSTEM METRICS")
    print(f"{'='*60}")
    
    metrics = results.system_metrics
    print(f"Active crawlers: {metrics.active_crawlers}")
    print(f"Active extractors: {metrics.active_extractors}")
    print(f"Active GPU processors: {metrics.active_gpu_processors}")
    print(f"GPU worker available: {metrics.gpu_worker_available}")
    
    if metrics.queue_metrics:
        print(f"\nQueue metrics:")
        for qm in metrics.queue_metrics:
            print(f"  {qm.queue_name}: {qm.depth}/{qm.max_depth} ({qm.utilization_percent:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='New Crawler System - Clean multiprocess crawler with GPU worker integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl sites from file
  python -m backend.new_crawler.main --sites-file sites.txt
  
  # Crawl specific sites
  python -m backend.new_crawler.main --sites https://example1.com https://example2.com
  
  # Health check only
  python -m backend.new_crawler.main --health-check
  
  # With custom config
  python -m backend.new_crawler.main --sites-file sites.txt --config-file custom.env
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--sites-file', type=str, help='File containing sites to crawl (one per line)')
    input_group.add_argument('--sites', nargs='+', help='Sites to crawl (space-separated URLs)')
    
    # Configuration
    parser.add_argument('--config-file', type=str, help='Configuration file (.env format)')
    parser.add_argument('--log-level', type=str, default='info', 
                       choices=['debug', 'info', 'warning', 'error'], help='Log level')
    
    # Operations
    parser.add_argument('--health-check', action='store_true', help='Perform health check only')
    parser.add_argument('--validate-config', action='store_true', help='Validate configuration only')
    
    # Worker configuration
    parser.add_argument('--num-crawlers', type=int, help='Number of crawler workers')
    parser.add_argument('--num-extractors', type=int, help='Number of extractor workers')
    parser.add_argument('--num-gpu-processors', type=int, help='Number of GPU processor workers')
    parser.add_argument('--batch-size', type=int, help='Batch size for GPU processing')
    
    # Crawling limits
    parser.add_argument('--max-pages-per-site', type=int, help='Maximum pages to crawl per site')
    parser.add_argument('--max-images-per-site', type=int, help='Maximum images to save per site')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = get_config()
    
    # Override config with command line arguments
    if args.num_crawlers:
        config.num_crawlers = args.num_crawlers
    if args.num_extractors:
        config.num_extractors = args.num_extractors
    if args.num_gpu_processors:
        config.num_gpu_processors = args.num_gpu_processors
    if args.batch_size:
        config.nc_batch_size = args.batch_size
    if args.max_pages_per_site:
        config.nc_max_pages_per_site = args.max_pages_per_site
    if args.max_images_per_site:
        config.nc_max_images_per_site = args.max_images_per_site
    
    # Log configuration
    config.log_configuration()
    
    try:
        # Handle different operations
        if args.health_check:
            # Health check only
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            healthy = loop.run_until_complete(health_check())
            sys.exit(0 if healthy else 1)
        
        elif args.validate_config:
            # Validate configuration only
            valid = validate_configuration()
            if valid:
                logger.info("✓ Configuration is valid")
            else:
                logger.warning("✗ Configuration validation failed")
            sys.exit(0 if valid else 1)
        
        else:
            # Load sites
            sites = []
            if args.sites_file:
                sites = load_sites_from_file(args.sites_file)
            elif args.sites:
                sites = load_sites_from_args(args.sites)
            else:
                logger.error("No sites provided. Use --sites-file or --sites")
                parser.print_help()
                sys.exit(1)
            
            if not sites:
                logger.error("No valid sites to crawl")
                sys.exit(1)
            
            # Run crawl
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(run_crawl(sites, args.config_file))
                logger.info("Crawl completed successfully")
            except KeyboardInterrupt:
                logger.info("Crawl interrupted by user")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Crawl failed: {e}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
