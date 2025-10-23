#!/usr/bin/env python3
"""
Site List Crawler Script

A simple script to crawl multiple sites using the refactored crawler architecture.
This demonstrates the new modular architecture with clean separation of concerns.
"""

import asyncio
import sys
import time
from typing import List
from backend.app.services.crawler import ImageCrawler


async def crawl_sites(
    urls: List[str],
    max_images_per_site: int = 10,
    concurrent_sites: int = 3,
    require_face: bool = False,
    crop_faces: bool = False
):
    """
    Crawl multiple sites and display results.
    
    Args:
        urls: List of URLs to crawl
        max_images_per_site: Maximum images to process per site
        concurrent_sites: Maximum number of sites to crawl concurrently
        require_face: Whether to require face detection
        crop_faces: Whether to crop faces for thumbnails
    """
    print(f"🚀 Starting crawl of {len(urls)} sites...")
    print(f"📊 Max images per site: {max_images_per_site}")
    print(f"⚡ Concurrent sites: {concurrent_sites}")
    print(f"👤 Face detection: {'Required' if require_face else 'Optional'}")
    print(f"✂️  Face cropping: {'Enabled' if crop_faces else 'Disabled'}")
    print("-" * 60)
    
    start_time = time.time()
    
    async with ImageCrawler(
        max_total_images=max_images_per_site,
        max_pages=1,
        require_face=require_face,
        crop_faces=crop_faces
    ) as crawler:
        try:
            results = await crawler.crawl_site_list(
                urls=urls,
                method='smart',
                max_images_per_site=max_images_per_site,
                concurrent_sites=concurrent_sites
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Display results
            print(f"\n🎉 Crawl completed in {total_time:.2f}s!")
            print(f"\n📊 RESULTS SUMMARY:")
            print("-" * 60)
            
            total_images = 0
            total_saved = 0
            total_cache_hits = 0
            total_cache_misses = 0
            successful_sites = 0
            
            for i, result in enumerate(results, 1):
                status = "✅" if result.images_found > 0 else "❌"
                print(f"\n  {status} Site {i}: {result.url}")
                print(f"    📊 Images found: {result.images_found}")
                print(f"    💾 Raw saved: {result.raw_images_saved}")
                print(f"    🖼️  Thumbnails saved: {result.thumbnails_saved}")
                print(f"    🎯 Cache: {result.cache_hits} hits, {result.cache_misses} misses")
                print(f"    🔧 Method: {result.targeting_method}")
                
                if result.errors:
                    print(f"    ⚠️  Errors: {len(result.errors)}")
                    for error in result.errors[:2]:  # Show first 2 errors
                        print(f"      - {error}")
                
                total_images += result.images_found
                total_saved += result.raw_images_saved
                total_cache_hits += result.cache_hits
                total_cache_misses += result.cache_misses
                
                if result.images_found > 0:
                    successful_sites += 1
            
            print(f"\n🏆 FINAL SUMMARY:")
            print("-" * 60)
            print(f"📊 Total images processed: {total_images}")
            print(f"💾 Total images saved: {total_saved}")
            print(f"✅ Successful sites: {successful_sites}/{len(urls)}")
            print(f"⏱️  Total time: {total_time:.2f}s")
            if total_time > 0:
                print(f"🚀 Average rate: {total_images / total_time:.2f} img/s")
            
            cache_total = total_cache_hits + total_cache_misses
            if cache_total > 0:
                cache_hit_rate = (total_cache_hits / cache_total) * 100
                print(f"🎯 Cache hit rate: {cache_hit_rate:.1f}%")
            
            print(f"\n🎉 Site list crawling completed successfully!")
            print(f"🏗️  Refactored architecture demonstrated!")
                    
        except Exception as e:
            print(f"❌ Error during crawl: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the site crawler."""
    # Default sites to crawl
    default_sites = [
        "https://wikifeet.com",
        "https://candidteens.net", 
        "https://forum.candidgirls.io"
    ]
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python crawl_sites.py [url1] [url2] [url3] ...")
            print("\nExample:")
            print("  python crawl_sites.py https://example1.com https://example2.com")
            print("\nDefault sites will be used if no URLs provided.")
            return
        else:
            urls = sys.argv[1:]
    else:
        urls = default_sites
        print("No URLs provided, using default sites...")
    
    # Run the crawler
    asyncio.run(crawl_sites(
        urls=urls,
        max_images_per_site=10,
        concurrent_sites=2,
        require_face=False,
        crop_faces=False
    ))


if __name__ == "__main__":
    main()
