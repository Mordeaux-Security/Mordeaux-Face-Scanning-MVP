#!/usr/bin/env python3
"""
Multiprocessing Multisite Image Crawler Script

Crawls multiple sites using separate worker processes for crawling and extraction.
Uses the native Windows GPU worker service for all face detection processing.

Architecture:
- 5 Crawling Workers: Fetch HTML and extract image URLs
- 1 Extraction Worker: Download images and add to batch queue  
- 1 Batch Processor: Collect batches and send to GPU worker
- Native Windows GPU Worker: Process face detection (via HTTP)

Requirements:
- Native Windows GPU worker must be running at localhost:8765
- For Docker: GPU_WORKER_URL=http://host.docker.internal:8765
- Redis server for inter-process communication

Usage examples:
  # Crawl sites from file with default settings
  python scripts/crawl_multisite_multiprocess.py --sites-file sites.txt

  # Crawl with custom worker counts
  python scripts/crawl_multisite_multiprocess.py --sites-file sites.txt --num-crawlers 5 --num-extractors 1 --num-batch-processors 1

  # Crawl with custom batch size
  python scripts/crawl_multisite_multiprocess.py --sites-file sites.txt --batch-size 128
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.multiprocess_crawler import MultiprocessCrawler
from app.services.gpu_health import validate_gpu_worker_startup, print_gpu_worker_troubleshooting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_sites_from_file(file_path: str):
    """Load sites from file, one URL per line."""
    sites = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sites.append(line)
    return sites


def main():
    parser = argparse.ArgumentParser(
        description='Multiprocessing multisite image crawler'
    )
    
    parser.add_argument(
        '--sites',
        type=str,
        help='Comma-separated list of URLs to crawl'
    )
    
    parser.add_argument(
        '--sites-file',
        type=str,
        help='File containing URLs to crawl (one per line)'
    )
    
    parser.add_argument(
        '--num-crawlers',
        type=int,
        default=5,
        help='Number of crawling worker processes (default: 5)'
    )
    
    parser.add_argument(
        '--num-extractors',
        type=int,
        default=1,
        help='Number of extraction worker processes (default: 1)'
    )
    
    parser.add_argument(
        '--num-batch-processors',
        type=int,
        default=1,
        help='Number of batch processing workers (default: 1)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for GPU processing (default: 64)'
    )
    
    parser.add_argument(
        '--use-3x3-mining',
        action='store_true',
        help='Enable 3x3 mining for better selector discovery'
    )
    
    parser.add_argument(
        '--redis-url',
        type=str,
        help='Redis connection URL'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to crawl per site (default: unlimited)'
    )
    
    parser.add_argument(
        '--max-images-per-site',
        type=int,
        default=None,
        help='Maximum number of images to process per site (default: unlimited)'
    )
    
    args = parser.parse_args()
    
    # Parse sites
    if args.sites:
        sites = [s.strip() for s in args.sites.split(',') if s.strip()]
    elif args.sites_file:
        sites = load_sites_from_file(args.sites_file)
    else:
        print("Error: Must provide --sites or --sites-file")
        sys.exit(1)
    
    if not sites:
        print("Error: No valid sites provided")
        sys.exit(1)
    
    print(f"Starting multiprocessing crawl of {len(sites)} sites...")
    print(f"Crawlers: {args.num_crawlers}, Extractors: {args.num_extractors}")
    print(f"Batch Processors: {args.num_batch_processors}")
    print(f"Batch size: {args.batch_size}")
    print("Note: GPU processing handled by native Windows GPU worker service")
    
    # Validate GPU worker is ready
    print("\n🔍 Validating GPU worker...")
    if not validate_gpu_worker_startup():
        print("❌ GPU worker validation failed!")
        print_gpu_worker_troubleshooting()
        sys.exit(1)
    
    print("✅ GPU worker validation successful!")
    
    # Create crawler
    crawler = MultiprocessCrawler(
        redis_url=args.redis_url,
        num_crawlers=args.num_crawlers,
        num_extractors=args.num_extractors,
        num_batch_processors=args.num_batch_processors,
        batch_size=args.batch_size,
        use_3x3_mining=args.use_3x3_mining,
        max_pages=args.max_pages,
        max_images_per_site=args.max_images_per_site
    )
    
    # Run crawl
    try:
        results = crawler.crawl_sites(sites)
        
        # Print detailed results summary
        print("\n" + "="*80)
        print("MULTIPROCESS CRAWL RESULTS")
        print("="*80)
        
        # Individual site results
        total_images = 0
        total_processed = 0
        total_pages = 0
        total_raw_saved = 0
        total_thumbnails_saved = 0
        total_cache_hits = 0
        total_cache_misses = 0
        successful_sites = 0
        total_errors = 0
        
        for i, result in enumerate(results.sites, 1):
            status = "✅" if result.images_found > 0 else "❌"
            
            print(f"\n{status} SITE {i}: {result.url}")
            print(f"  📊 Images found: {result.images_found}")
            print(f"  🔄 Images processed: {result.images_processed}")
            print(f"  💾 Raw images saved: {result.raw_images_saved}")
            print(f"  🖼️  Thumbnails saved: {result.thumbnails_saved}")
            print(f"  📄 Pages crawled: {result.pages_crawled}")
            print(f"  ⏱️  Processing time: {result.processing_time:.2f}s")
            if result.targeting_method:
                print(f"  🎯 Targeting method: {result.targeting_method}")
            
            if result.errors:
                print(f"  ⚠️  Errors ({len(result.errors)}):")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
            
            total_images += result.images_found
            total_processed += result.images_processed
            total_pages += result.pages_crawled
            total_raw_saved += result.raw_images_saved
            total_thumbnails_saved += result.thumbnails_saved
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
        print(f"📊 Total images found: {total_images}")
        print(f"🔄 Total images processed: {total_processed}")
        print(f"💾 Total raw images saved: {total_raw_saved}")
        print(f"🖼️  Total thumbnails saved: {total_thumbnails_saved}")
        print(f"📄 Total pages crawled: {total_pages}")
        print(f"⏱️  Total processing time: {results.total_time:.2f}s")
        
        if results.total_time > 0:
            print(f"🚀 Average processing rate: {total_images / results.total_time:.2f} img/s")
            print(f"📄 Average page rate: {total_pages / results.total_time:.2f} pages/s")
        
        # Storage stats
        if total_cache_hits > 0 or total_cache_misses > 0:
            print(f"\n💾 STORAGE STATS")
            print(f"  🎯 Cache hits: {total_cache_hits}")
            print(f"  ❌ Cache misses: {total_cache_misses}")
            if total_cache_hits + total_cache_misses > 0:
                hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses) * 100
                print(f"  📈 Cache hit rate: {hit_rate:.1f}%")
        
        # Batch processing stats
        if results.batch_stats:
            print(f"\n🔧 BATCH PROCESSING STATS")
            print(f"  📦 Batches sent: {results.batch_stats.get('batches_sent', 0)}")
            print(f"  🖼️  Images processed: {results.batch_stats.get('images_processed', 0)}")
            print(f"  💾 Raw images saved to MinIO: {results.batch_stats.get('raw_images_saved', 0)}")
            print(f"  🖼️  Face thumbnails saved to MinIO: {results.batch_stats.get('thumbnails_saved', 0)}")
            print(f"  ⚡ Average batch size: {results.batch_stats.get('avg_batch_size', 0):.1f}")
        
        print(f"⚠️  Total errors: {total_errors}")
        
        # Performance assessment
        if successful_sites == len(sites):
            print(f"\n🎉 ALL SITES SUCCESSFUL! Perfect multiprocess crawl!")
        elif successful_sites > len(sites) / 2:
            print(f"\n✅ Good success rate: {successful_sites}/{len(sites)} sites working")
        else:
            print(f"\n⚠️  Low success rate: {successful_sites}/{len(sites)} sites working")
        
        print("="*80)
        
        # Return exit code based on success
        if successful_sites == 0:
            print("\n❌ Exit code 3: No sites were successfully crawled")
            sys.exit(3)
        elif successful_sites < len(sites) / 2:
            print(f"\n⚠️  Exit code 2: Only {successful_sites}/{len(sites)} sites successful (less than 50%)")
            sys.exit(2)
        elif (len(sites) - successful_sites) > 0:
            failed_sites = len(sites) - successful_sites
            print(f"\n⚠️  Exit code 1: All sites crawled but {failed_sites} had failures")
            sys.exit(1)
        else:
            print("\n✅ Exit code 0: All sites successfully crawled")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nCrawl interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during crawl: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        crawler.stop_workers()


if __name__ == '__main__':
    main()
