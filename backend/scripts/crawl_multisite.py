#!/usr/bin/env python3
"""
Multisite Image Crawler Script

This script provides multisite crawling capabilities using the refactored architecture
with smart HTML→JS fallback and concurrent site processing.

Features:
- Crawl multiple sites concurrently
- Smart HTML-first with JS fallback
- Site list from file or command line
- Comprehensive reporting
- Performance metrics

Usage examples:
  # Crawl sites from command line
  python scripts/crawl_multisite.py --sites "https://site1.com,https://site2.com,https://site3.com"

  # Crawl sites from file
  python scripts/crawl_multisite.py --sites-file sites.txt

  # Crawl with custom settings
  python scripts/crawl_multisite.py --sites "https://site1.com,https://site2.com" --max-images-per-site 20 --concurrent-sites 3
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.crawler import ImageCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def parse_sites(sites_input: str) -> List[str]:
    """Parse sites from comma-separated string."""
    sites = [site.strip() for site in sites_input.split(',') if site.strip()]
    return sites


def load_sites_from_file(file_path: str) -> List[str]:
    """Load sites from file, one URL per line."""
    sites = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                sites.append(line)
    return sites


def main():
    parser = argparse.ArgumentParser(
        description='Multisite Image Crawler - Crawl multiple sites concurrently with smart HTML→JS fallback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl sites from command line
  python scripts/crawl_multisite.py --sites "https://wikifeet.com,https://candidteens.net,https://forum.candidgirls.io"

  # Crawl sites from file
  python scripts/crawl_multisite.py --sites-file sites.txt

  # Crawl with custom settings
  python scripts/crawl_multisite.py --sites "https://site1.com,https://site2.com" --max-images-per-site 20 --concurrent-sites 3

  # Crawl with face detection
  python scripts/crawl_multisite.py --sites "https://site1.com" --require-face --crop-faces --min-face-quality 0.7

Site list file format (one URL per line):
  https://wikifeet.com
  https://candidteens.net
  https://forum.candidgirls.io
  # This is a comment
        """
    )
    
    # Site input options (mutually exclusive)
    site_group = parser.add_mutually_exclusive_group(required=True)
    site_group.add_argument('--sites', 
                           help='Comma-separated list of URLs to crawl')
    site_group.add_argument('--sites-file', 
                           help='File containing URLs to crawl (one per line)')
    
    # Crawling parameters
    parser.add_argument('--method', default='smart',
                       help='Targeting method (default: smart)')
    parser.add_argument('--max-images-per-site', type=int, default=20,
                       help='Maximum images to collect per site (default: 20)')
    parser.add_argument('--max-pages-per-site', type=int, default=5,
                       help='Maximum pages to crawl per site (default: 5)')
    parser.add_argument('--concurrent-sites', type=int, default=3,
                       help='Maximum number of sites to crawl concurrently (default: 3)')
    
    # Face detection parameters
    parser.add_argument('--min-face-quality', type=float, default=0.5,
                       help='Minimum face detection quality score (default: 0.5)')
    parser.add_argument('--require-face', action='store_true', default=False,
                       help='Require at least one face in images (default: False)')
    parser.add_argument('--crop-faces', action='store_true', default=True,
                       help='Crop and save face regions as thumbnails (default: True)')
    parser.add_argument('--face-margin', type=float, default=0.2,
                       help='Margin around face as fraction of face size (default: 0.2)')
    
    # System parameters
    parser.add_argument('--tenant-id', default='multisite',
                       help='Tenant ID for multi-tenancy support (default: multisite)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--max-concurrent-images', type=int, default=10,
                       help='Maximum number of images to process concurrently per site (default: 10)')
    parser.add_argument('--batch-size', type=int, default=25,
                       help='Batch size for operations (default: 25)')
    parser.add_argument('--use-3x3-mining', action='store_true', default=False,
                       help='Enable 3x3 depth mining for better selector discovery (default: False)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output, show only summary')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Parse sites
    if args.sites:
        sites = parse_sites(args.sites)
    else:
        sites = load_sites_from_file(args.sites_file)
    
    if not sites:
        print("Error: No valid sites provided")
        sys.exit(1)
    
    print(f"Starting multisite crawl of {len(sites)} sites...")
    print(f"Sites: {', '.join(sites[:3])}{'...' if len(sites) > 3 else ''}")
    
    async def run_multisite_crawler():
        crawler_config = {
            'tenant_id': args.tenant_id,
            'timeout': args.timeout,
            'min_face_quality': args.min_face_quality,
            'require_face': args.require_face,
            'crop_faces': args.crop_faces,
            'face_margin': args.face_margin,
            'max_total_images': args.max_images_per_site,
            'max_pages': args.max_pages_per_site,
            'max_concurrent_images': args.max_concurrent_images,
            'batch_size': args.batch_size,
            'use_3x3_mining': args.use_3x3_mining,
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with ImageCrawler(**crawler_config) as crawler:
            results = await crawler.crawl_site_list(
                urls=sites,
                method=args.method,
                max_images_per_site=args.max_images_per_site,
                concurrent_sites=args.concurrent_sites
            )
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Print results
        if not args.quiet:
            print("\n" + "="*80)
            print("MULTISITE CRAWL RESULTS")
            print("="*80)
        
        # Individual site results
        total_images = 0
        total_saved = 0
        total_pages = 0
        total_cache_hits = 0
        total_cache_misses = 0
        successful_sites = 0
        total_errors = 0
        
        for i, result in enumerate(results, 1):
            status = "✅" if result.images_found > 0 else "❌"
            
            if not args.quiet:
                print(f"\n{status} SITE {i}: {result.url}")
                print(f"  📊 Images found: {result.images_found}")
                print(f"  💾 Raw images saved: {result.raw_images_saved}")
                print(f"  🖼️  Thumbnails saved: {result.thumbnails_saved}")
                print(f"  📄 Pages crawled: {result.pages_crawled}")
                print(f"  🎯 Cache performance: {result.cache_hits} hits, {result.cache_misses} misses")
                print(f"  🔧 Extraction method: {result.targeting_method}")
                
                if result.errors:
                    print(f"  ⚠️  Errors ({len(result.errors)}):")
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"    - {error}")
            
            total_images += result.images_found
            total_saved += result.raw_images_saved
            total_pages += result.pages_crawled
            total_cache_hits += result.cache_hits
            total_cache_misses += result.cache_misses
            total_errors += len(result.errors)
            
            if result.images_found > 0:
                successful_sites += 1
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"📊 Total sites: {len(sites)}")
        print(f"✅ Successful sites: {successful_sites}")
        print(f"❌ Failed sites: {len(sites) - successful_sites}")
        print(f"📊 Total images processed: {total_images}")
        print(f"💾 Total raw images saved: {total_saved}")
        print(f"📄 Total pages crawled: {total_pages}")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        
        if total_time > 0:
            print(f"🚀 Average processing rate: {total_images / total_time:.2f} img/s")
            print(f"📄 Average page rate: {total_pages / total_time:.2f} pages/s")
        
        cache_total = total_cache_hits + total_cache_misses
        if cache_total > 0:
            cache_hit_rate = (total_cache_hits / cache_total) * 100
            print(f"🎯 Overall cache hit rate: {cache_hit_rate:.1f}%")
        
        print(f"⚠️  Total errors: {total_errors}")
        
        # Performance assessment
        if successful_sites == len(sites):
            print(f"\n🎉 ALL SITES SUCCESSFUL! Perfect multisite crawl!")
        elif successful_sites > len(sites) / 2:
            print(f"\n✅ Good success rate: {successful_sites}/{len(sites)} sites working")
        else:
            print(f"\n⚠️  Low success rate: {successful_sites}/{len(sites)} sites working")
        
        print("="*80)
        
        # Return exit code based on success
        if successful_sites == 0:
            print("\n❌ Exit code 3: No sites were successfully crawled")
            return 3
        elif successful_sites < len(sites) / 2:
            print(f"\n⚠️  Exit code 2: Only {successful_sites}/{len(sites)} sites successful (less than 50%)")
            return 2
        elif (len(sites) - successful_sites) > 0:
            failed_sites = len(sites) - successful_sites
            print(f"\n⚠️  Exit code 1: All sites crawled but {failed_sites} had failures")
            return 1
        else:
            print("\n✅ Exit code 0: All sites successfully crawled")
            return 0
    
    # Run the multisite crawler
    exit_code = asyncio.run(run_multisite_crawler())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
