#!/usr/bin/env python3
"""
Single Site Image Crawler Script

This script provides single site crawling capabilities using the refactored architecture
with smart HTML→JS fallback.

Features:
- Crawl a single site with configurable parameters
- Smart HTML-first with JS fallback
- Face detection and cropping
- Comprehensive reporting
- Performance metrics

Usage examples:
  # Basic crawl
  python scripts/crawl_images.py https://example.com

  # Crawl with face detection
  python scripts/crawl_images.py https://example.com --require-face --crop-faces --min-face-quality 0.7

  # Crawl with custom settings
  python scripts/crawl_images.py https://example.com --max-images 50 --max-pages 10 --method smart
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.crawler import ImageCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Single Site Image Crawler - Crawl a single site with smart HTML→JS fallback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic crawl
  python scripts/crawl_images.py https://example.com

  # Crawl with face detection
  python scripts/crawl_images.py https://example.com --require-face --crop-faces --min-face-quality 0.7

  # Crawl with custom settings
  python scripts/crawl_images.py https://example.com --max-images 50 --max-pages 10 --method smart

  # Crawl with 3x3 mining
  python scripts/crawl_images.py https://example.com --use-3x3-mining --max-images 100
        """
    )
    
    # Required arguments
    parser.add_argument('url', help='URL to crawl')
    
    # Crawling parameters
    parser.add_argument('--method', default='smart',
                       help='Targeting method (default: smart)')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum images to collect (default: 50)')
    parser.add_argument('--max-pages', type=int, default=20,
                       help='Maximum pages to crawl (default: 20)')
    parser.add_argument('--mode', default='single', choices=['single', 'site'],
                       help='Crawl mode: single page or entire site (default: single)')
    
    # Face detection parameters
    parser.add_argument('--min-face-quality', type=float, default=0.5,
                       help='Minimum face detection quality score (default: 0.5)')
    parser.add_argument('--require-face', action='store_true', default=False,
                       help='Require at least one face in images (default: False)')
    parser.add_argument('--no-require-face', dest='require_face', action='store_false',
                       help='Do not require faces in images')
    parser.add_argument('--crop-faces', action='store_true', default=True,
                       help='Crop and save face regions as thumbnails (default: True)')
    parser.add_argument('--no-crop-faces', dest='crop_faces', action='store_false',
                       help='Do not crop faces')
    parser.add_argument('--face-margin', type=float, default=0.2,
                       help='Margin around face as fraction of face size (default: 0.2)')
    
    # System parameters
    parser.add_argument('--tenant-id', default='default',
                       help='Tenant ID for multi-tenancy support (default: default)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--max-concurrent-images', type=int, default=10,
                       help='Maximum number of images to process concurrently (default: 10)')
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
    
    print(f"Starting crawl of: {args.url}")
    print(f"Method: {args.method}, Max images: {args.max_images}, Max pages: {args.max_pages}")
    
    async def run_single_site_crawler():
        crawler_config = {
            'tenant_id': args.tenant_id,
            'timeout': args.timeout,
            'min_face_quality': args.min_face_quality,
            'require_face': args.require_face,
            'crop_faces': args.crop_faces,
            'face_margin': args.face_margin,
            'max_total_images': args.max_images,
            'max_pages': args.max_pages,
            'max_concurrent_images': args.max_concurrent_images,
            'batch_size': args.batch_size,
            'use_3x3_mining': args.use_3x3_mining,
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with ImageCrawler(**crawler_config) as crawler:
            if args.mode == 'single':
                result = await crawler.crawl_page(args.url, method=args.method)
            else:
                result = await crawler.crawl_site(args.url, method=args.method)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Print results
        if not args.quiet:
            print("\n" + "="*80)
            print("CRAWL RESULTS")
            print("="*80)
            print(f"🌐 URL: {result.url}")
            print(f"📊 Images found: {result.images_found}")
            print(f"💾 Raw images saved: {result.raw_images_saved}")
            print(f"🖼️  Thumbnails saved: {result.thumbnails_saved}")
            print(f"📄 Pages crawled: {result.pages_crawled}")
            print(f"🎯 Cache performance: {result.cache_hits} hits, {result.cache_misses} misses")
            print(f"🔧 Extraction method: {result.targeting_method}")
            print(f"⏱️  Processing time: {total_time:.2f}s")
            
            if total_time > 0:
                print(f"🚀 Processing rate: {result.images_found / total_time:.2f} img/s")
            
            cache_total = result.cache_hits + result.cache_misses
            if cache_total > 0:
                cache_hit_rate = (result.cache_hits / cache_total) * 100
                print(f"🎯 Cache hit rate: {cache_hit_rate:.1f}%")
            
            if result.errors:
                print(f"⚠️  Errors ({len(result.errors)}):")
                for error in result.errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        if result.images_found > 0:
            print("✅ Crawl completed successfully")
            print(f"📊 Total images processed: {result.images_found}")
            print(f"💾 Raw images saved: {result.raw_images_saved}")
            print(f"🖼️  Thumbnails saved: {result.thumbnails_saved}")
            print(f"📄 Pages crawled: {result.pages_crawled}")
            print(f"⏱️  Total time: {total_time:.2f}s")
            return 0
        else:
            print("❌ No images found")
            if result.errors:
                print(f"⚠️  Errors encountered: {len(result.errors)}")
            return 1
    
    # Run the single site crawler
    exit_code = asyncio.run(run_single_site_crawler())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
