#!/usr/bin/env python3
"""
Enhanced Image Crawler Script v2

This script combines the advanced features from basic_crawler1.1 with the commercial 
readiness features from main branch.

Key improvements:
- Multi-tenancy support with tenant_id parameter
- Advanced caching with similarity detection
- Enhanced face detection with image enhancement
- Flexible CSS selector patterns
- Concurrent processing with semaphores
- Audit logging and compliance features
- Improved error handling and logging

Usage examples:
  # Basic crawl with default tenant
  python scripts/crawl_images_v2.py https://www.pornhub.com

  # Crawl with specific tenant and method
  python scripts/crawl_images_v2.py https://www.pornhub.com --tenant-id tenant_123 --method js-videoThumb

  # Multi-page crawl with custom settings
  python scripts/crawl_images_v2.py https://www.pornhub.com --tenant-id tenant_123 --mode site --max-images 100 --max-pages 10

  # Target by size with enhanced face detection
  python scripts/crawl_images_v2.py https://www.pornhub.com --tenant-id tenant_123 --method size-320x180 --max-images 50

Available targeting methods:
  smart           - Automatically picks the best method (default)
  data-mediumthumb - Target images with data-mediumthumb attribute
  js-videoThumb   - Target images with js-videoThumb class
  phimage         - Target images inside .phimage divs
  latestThumb     - Target images inside links with latestThumb class
  video-thumb     - Target common video thumbnail patterns
  size-320x180    - Target images with 320x180 dimensions
  size-640x360    - Target images with 640x360 dimensions
  size-1280x720   - Target images with 1280x720 dimensions
  all-images      - Target all images on the page
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.crawler import EnhancedImageCrawlerV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Image Crawler v2 - Combines advanced features with commercial readiness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic crawl with default tenant
  python scripts/crawl_images_v2.py https://www.pornhub.com

  # Crawl with specific tenant and method
  python scripts/crawl_images_v2.py https://www.pornhub.com --tenant-id tenant_123 --method js-videoThumb

  # Multi-page crawl with custom settings
  python scripts/crawl_images_v2.py https://www.pornhub.com --tenant-id tenant_123 --mode site --max-images 100 --max-pages 10

  # Target by size with enhanced face detection
  python scripts/crawl_images_v2.py https://www.pornhub.com --tenant-id tenant_123 --method size-320x180 --max-images 50

Available targeting methods:
  smart           - Automatically picks the best method (default)
  data-mediumthumb - Target images with data-mediumthumb attribute
  js-videoThumb   - Target images with js-videoThumb class
  phimage         - Target images inside .phimage divs
  latestThumb     - Target images inside links with latestThumb class
  video-thumb     - Target common video thumbnail patterns
  size-320x180    - Target images with 320x180 dimensions
  size-640x360    - Target images with 640x360 dimensions
  size-1280x720   - Target images with 1280x720 dimensions
  all-images      - Target all images on the page
        """
    )
    
    parser.add_argument('url', help='URL to crawl for images')
    parser.add_argument('--tenant-id', default='default', 
                       help='Tenant ID for multi-tenancy support (default: default)')
    parser.add_argument('--max-size', type=int, default=10485760, 
                       help='Maximum file size in bytes (default: 10MB)')
    parser.add_argument('--method', default='smart', 
                       choices=['smart', 'data-mediumthumb', 'js-videoThumb', 'phimage', 'latestThumb', 'video-thumb', 'size-320x180', 'size-640x360', 'size-1280x720', 'all-images'],
                       help='Targeting method (default: smart)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--min-face-quality', type=float, default=0.5,
                       help='Minimum face detection quality score (default: 0.5)')
    parser.add_argument('--require-face', action='store_true', default=True,
                       help='Require at least one face in images (default: True)')
    parser.add_argument('--no-require-face', dest='require_face', action='store_false',
                       help='Disable face requirement')
    parser.add_argument('--crop-faces', action='store_true', default=True,
                       help='Crop and save face regions as thumbnails (default: True)')
    parser.add_argument('--no-crop-faces', dest='crop_faces', action='store_false',
                       help='Disable face cropping')
    parser.add_argument('--face-margin', type=float, default=0.2,
                       help='Margin around face as fraction of face size (default: 0.2)')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum total images to collect (default: 50)')
    parser.add_argument('--max-pages', type=int, default=20,
                       help='Maximum pages to crawl (default: 20)')
    parser.add_argument('--mode', choices=['single', 'site'], default='single',
                       help='Crawling mode: single page or multi-page site crawling (default: single)')
    parser.add_argument('--cross-domain', action='store_true', default=False,
                       help='Allow crawling across different domains (default: same domain only)')
    parser.add_argument('--similarity-threshold', type=int, default=5,
                       help='Hamming distance threshold for content similarity (0-64, default: 5)')
    parser.add_argument('--max-concurrent-images', type=int, default=10,
                       help='Maximum number of images to process concurrently (default: 10)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for operations (default: 50)')
    parser.add_argument('--enable-audit-logging', action='store_true', default=True,
                       help='Enable audit logging for compliance (default: True)')
    parser.add_argument('--disable-audit-logging', dest='enable_audit_logging', action='store_false',
                       help='Disable audit logging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_crawler():
        crawler_config = {
            'tenant_id': args.tenant_id,
            'max_file_size': args.max_size,
            'timeout': args.timeout,
            'min_face_quality': args.min_face_quality,
            'require_face': args.require_face,
            'crop_faces': args.crop_faces,
            'face_margin': args.face_margin,
            'max_total_images': args.max_images,
            'max_pages': args.max_pages,
            'same_domain_only': not args.cross_domain,
            'similarity_threshold': args.similarity_threshold,
            'max_concurrent_images': args.max_concurrent_images,
            'batch_size': args.batch_size,
            'enable_audit_logging': args.enable_audit_logging,
        }
        
        async with EnhancedImageCrawlerV2(**crawler_config) as crawler:
            if args.mode == 'single':
                result = await crawler.crawl_page(args.url, args.method)
            else:
                result = await crawler.crawl_site(args.url, args.method)
        
        # Print results
        print("="*70)
        print("CRAWL RESULTS")
        print("="*70)
        print(f"URL: {result.url}")
        print(f"Tenant ID: {result.tenant_id}")
        print(f"Targeting method: {result.targeting_method}")
        print(f"Mode: {args.mode}")
        print(f"Images found: {result.images_found}")
        print(f"Raw images saved: {result.raw_images_saved}")
        print(f"Thumbnails saved: {result.thumbnails_saved}")
        print(f"Pages crawled: {result.pages_crawled}")
        print(f"Max total images: {args.max_images}")
        print(f"Max pages: {args.max_pages}")
        print(f"Same domain only: {not args.cross_domain}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        print(f"Max concurrent images: {args.max_concurrent_images}")
        print(f"Batch size: {args.batch_size}")
        print(f"Audit logging: {'Enabled' if args.enable_audit_logging else 'Disabled'}")
        print(f"Storage: MinIO (raw-images & thumbnails buckets)")
        print(f"Cache hits: {result.cache_hits}")
        print(f"Cache misses: {result.cache_misses}")
        if result.cache_hits + result.cache_misses > 0:
            hit_rate = (result.cache_hits / (result.cache_hits + result.cache_misses)) * 100
            print(f"Cache hit rate: {hit_rate:.1f}%")
        print(f"Saved raw image keys:")
        for key in result.saved_raw_keys:
            print(f"  - {key}")
        print(f"Saved thumbnail keys:")
        for key in result.saved_thumbnail_keys:
            print(f"  - {key}")
        
        if result.errors:
            print(f"\nErrors encountered:")
            for error in result.errors:
                print(f"  - {error}")
        
        print("="*70)
        
        # Return exit code based on success
        if result.errors and len(result.errors) > 5:  # More than 5 errors
            return 1
        elif result.raw_images_saved == 0:  # No images saved
            return 2
        else:
            return 0
    
    # Run the crawler
    exit_code = asyncio.run(run_crawler())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
