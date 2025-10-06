#!/usr/bin/env python3
"""
Image Crawler CLI Script

Command-line interface for the enhanced image crawler integrated with MinIO storage.
This script can be run from within the Docker container or locally with proper environment setup.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.crawler import EnhancedImageCrawler, crawl_images_from_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the integrated crawler."""
    parser = argparse.ArgumentParser(
        description='Enhanced image crawler integrated with MinIO storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart crawl (automatically picks best method)
  python scripts/crawl_images.py https://example.com --max-images 5

  # Target video thumbnails specifically
  python scripts/crawl_images.py https://example.com --method data-mediumthumb --max-images 10

  # Target by JavaScript class
  python scripts/crawl_images.py https://example.com --method js-videoThumb --max-images 5

  # Target by size (320x180 thumbnails)
  python scripts/crawl_images.py https://example.com --method size --max-images 5

Available targeting methods:
  smart           - Automatically picks the best method (default)
  data-mediumthumb - Target images with data-mediumthumb attribute
  js-videoThumb   - Target images with js-videoThumb class
  size            - Target images with 320x180 dimensions
  phimage         - Target images inside .phimage divs
  latestThumb     - Target images inside links with latestThumb class
  all             - Target all images on the page
        """
    )
    
    parser.add_argument('url', help='URL to crawl for images')
    parser.add_argument('--max-size', type=int, default=10485760, 
                       help='Maximum file size in bytes (default: 10MB)')
    parser.add_argument('--method', default='smart', 
                       choices=['smart', 'data-mediumthumb', 'js-videoThumb', 'size', 'phimage', 'latestThumb', 'all'],
                       help='Targeting method (default: smart)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--min-face-quality', type=float, default=0.5,
                       help='Minimum face detection quality score (0.0-1.0, default: 0.5)')
    parser.add_argument('--require-face', action='store_true', default=True,
                       help='Require at least one face in the image (default: True)')
    parser.add_argument('--no-require-face', dest='require_face', action='store_false',
                       help='Allow images without faces')
    parser.add_argument('--crop-faces', action='store_true', default=True,
                       help='Crop and save only face regions (default: True)')
    parser.add_argument('--no-crop-faces', dest='crop_faces', action='store_false',
                       help='Save full images instead of cropping faces')
    parser.add_argument('--face-margin', type=float, default=0.2,
                       help='Margin around face as fraction of face size (default: 0.2 = 20 percent)')
    parser.add_argument('--max-total-images', type=int, default=50,
                       help='Maximum total images to collect across all pages (default: 50)')
    parser.add_argument('--max-pages', type=int, default=20,
                       help='Maximum pages to crawl (default: 20)')
    parser.add_argument('--crawl-mode', choices=['single', 'site'], default='single',
                       help='Crawling mode: single page or multi-page site crawling (default: single)')
    parser.add_argument('--cross-domain', action='store_true', default=False,
                       help='Allow crawling across different domains (default: same domain only)')
    parser.add_argument('--save-both', action='store_true', default=False,
                       help='Save both original and cropped images for comparison (default: False)')
    
    args = parser.parse_args()
    
    # Validate URL
    from urllib.parse import urlparse
    parsed_url = urlparse(args.url)
    if not parsed_url.scheme or not parsed_url.netloc:
        logger.error("Invalid URL provided")
        return 1
    
    try:
        # Create crawler and run
        crawler_config = {
            'max_file_size': args.max_size,
            'timeout': args.timeout,
            'min_face_quality': args.min_face_quality,
            'require_face': args.require_face,
            'crop_faces': args.crop_faces,
            'face_margin': args.face_margin,
            'max_total_images': args.max_total_images,
            'max_pages': args.max_pages,
            'same_domain_only': not args.cross_domain,
            'save_both': args.save_both
        }
        
        async with EnhancedImageCrawler(**crawler_config) as crawler:
            if args.crawl_mode == 'site':
                result = await crawler.crawl_site(args.url, args.method)
            else:
                result = await crawler.crawl_page(args.url, args.method)
            
            # Print results
            print("\n" + "="*70)
            print("ENHANCED IMAGE CRAWLER RESULTS")
            print("="*70)
            print(f"URL: {result.url}")
            print(f"Targeting method: {result.targeting_method}")
            print(f"Crawl mode: {args.crawl_mode}")
            print(f"Pages crawled: {result.pages_crawled}")
            print(f"Images found: {result.images_found}")
            print(f"Raw images saved: {result.raw_images_saved}")
            print(f"Thumbnails saved: {result.thumbnails_saved}")
            print(f"Face quality threshold: {args.min_face_quality}")
            print(f"Require face: {args.require_face}")
            print(f"Crop faces: {args.crop_faces}")
            print(f"Face margin: {args.face_margin}")
            print(f"Max total images: {args.max_total_images}")
            print(f"Max pages: {args.max_pages}")
            print(f"Same domain only: {not args.cross_domain}")
            print(f"Save both original and cropped: {args.save_both}")
            print(f"Storage: MinIO (raw-images & thumbnails buckets)")
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
            
            # Return appropriate exit code
            return 0 if result.thumbnails_saved > 0 else 1
            
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
