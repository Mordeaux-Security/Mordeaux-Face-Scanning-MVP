"""
List Crawler - Automated crawling of multiple sites using selector miner and crawler integration.

This module provides functionality to crawl a list of sites from a text file,
automatically running the selector miner for new sites and then crawling them.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

from app.crawler.crawler_settings import (
    LIST_CRAWL_DEFAULT_SITES_FILE,
    LIST_CRAWL_MAX_PAGES_PER_SITE,
    LIST_CRAWL_MAX_IMAGES_PER_SITE,
    LIST_CRAWL_AUTO_SELECTOR_MINING,
    LIST_CRAWL_SKIP_EXISTING_RECIPES
)
from app.crawler.crawler import ImageCrawler
from app.selector_miner.selector_miner import mine_page, Limits, emit_recipe_yaml_block
from app.selector_miner.site_recipes import load_site_recipes, save_site_recipes
import yaml
import httpx
import logging

logger = logging.getLogger(__name__)


class ListCrawler:
    """Crawler for processing multiple sites from a text file."""
    
    def __init__(self, sites_file: str = None, output_dir: str = "list_crawl_results"):
        """
        Initialize the list crawler.
        
        Args:
            sites_file: Path to text file containing list of sites (one per line)
            output_dir: Directory to store crawling results
        """
        self.sites_file = sites_file or LIST_CRAWL_DEFAULT_SITES_FILE
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load existing site recipes
        self.site_recipes = load_site_recipes()
        
        # Statistics
        self.stats = {
            'sites_processed': 0,
            'sites_skipped': 0,
            'sites_failed': 0,
            'total_images_saved': 0,
            'total_thumbnails_saved': 0,
            'selector_mining_attempts': 0,
            'selector_mining_successes': 0
        }
    
    def parse_sites_file(self) -> List[str]:
        """
        Parse the sites file and return a list of valid URLs.
        
        Returns:
            List of valid URLs to crawl
        """
        sites = []
        
        if not os.path.exists(self.sites_file):
            logger.error(f"Sites file not found: {self.sites_file}")
            return sites
        
        try:
            with open(self.sites_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Basic URL validation
                    if self._is_valid_url(line):
                        sites.append(line)
                    else:
                        logger.warning(f"Invalid URL on line {line_num}: {line}")
        
        except Exception as e:
            logger.error(f"Error reading sites file {self.sites_file}: {e}")
        
        logger.info(f"Parsed {len(sites)} valid URLs from {self.sites_file}")
        return sites
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc.lower()
        except:
            return url.lower()
    
    def _site_has_recipe(self, domain: str) -> bool:
        """Check if a site already has a recipe."""
        return domain in self.site_recipes.get('sites', {})
    
    async def mine_site_selectors(self, url: str) -> bool:
        """
        Run selector miner on a site to generate selectors.
        
        Args:
            url: URL to mine selectors for
            
        Returns:
            True if successful, False otherwise
        """
        domain = self._extract_domain(url)
        logger.info(f"Mining selectors for {domain} ({url})")
        
        try:
            limits = Limits(
                max_candidates=20,
                max_samples_per_candidate=5,
                max_bytes=10*1024*1024,
                timeout_seconds=15
            )
            
            async with httpx.AsyncClient() as client:
                result = await mine_page(url, None, use_js=False, client=client, limits=limits)
                
                if result.candidates:
                    logger.info(f"Found {len(result.candidates)} selector candidates for {domain}")
                    
                    # Generate recipe
                    recipe = emit_recipe_yaml_block(
                        domain, 
                        result.candidates, 
                        ['src', 'data-src', 'srcset'], 
                        []
                    )
                    
                    # Add to site recipes
                    if 'sites' not in self.site_recipes:
                        self.site_recipes['sites'] = {}
                    
                    self.site_recipes['sites'][domain] = recipe
                    
                    # Save updated recipes
                    save_site_recipes(self.site_recipes)
                    
                    logger.info(f"Successfully generated recipe for {domain}")
                    self.stats['selector_mining_successes'] += 1
                    return True
                else:
                    logger.warning(f"No selector candidates found for {domain}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error mining selectors for {domain}: {e}")
            return False
        finally:
            self.stats['selector_mining_attempts'] += 1
    
    async def crawl_site(self, url: str) -> Dict:
        """
        Crawl a single site.
        
        Args:
            url: URL to crawl
            
        Returns:
            Dictionary with crawling results
        """
        domain = self._extract_domain(url)
        logger.info(f"Starting crawl of {domain} ({url})")
        
        result = {
            'url': url,
            'domain': domain,
            'success': False,
            'error': None,
            'images_saved': 0,
            'thumbnails_saved': 0,
            'pages_crawled': 0
        }
        
        try:
            # Create crawler with list-specific settings
            crawler = ImageCrawler(
                max_pages=LIST_CRAWL_MAX_PAGES_PER_SITE,
                max_total_images=LIST_CRAWL_MAX_IMAGES_PER_SITE,
                require_face=False,  # Set to True if you want face detection
                crop_faces=True
            )
            
            # Run the crawl using async context manager
            async with crawler:
                crawl_result = await crawler.crawl_site(url)
            
            result.update({
                'success': True,
                'images_saved': crawl_result.raw_images_saved,
                'thumbnails_saved': crawl_result.thumbnails_saved,
                'pages_crawled': crawl_result.pages_crawled
            })
            
            # Update global stats
            self.stats['total_images_saved'] += crawl_result.raw_images_saved
            self.stats['total_thumbnails_saved'] += crawl_result.thumbnails_saved
            
            logger.info(f"Completed crawl of {domain}: {crawl_result.raw_images_saved} images, {crawl_result.thumbnails_saved} thumbnails")
            
        except Exception as e:
            logger.error(f"Error crawling {domain}: {e}")
            result['error'] = str(e)
            self.stats['sites_failed'] += 1
        
        return result
    
    async def process_site(self, url: str) -> Dict:
        """
        Process a single site (mine selectors if needed, then crawl).
        
        Args:
            url: URL to process
            
        Returns:
            Dictionary with processing results
        """
        domain = self._extract_domain(url)
        
        # Check if we need to mine selectors
        needs_mining = (
            LIST_CRAWL_AUTO_SELECTOR_MINING and 
            (not LIST_CRAWL_SKIP_EXISTING_RECIPES or not self._site_has_recipe(domain))
        )
        
        mining_success = None
        if needs_mining:
            logger.info(f"Mining selectors for {domain}")
            mining_success = await self.mine_site_selectors(url)
            if not mining_success:
                logger.warning(f"Selector mining failed for {domain}, proceeding with crawl anyway")
        else:
            logger.info(f"Using existing recipe for {domain}")
        
        # Crawl the site
        crawl_result = await self.crawl_site(url)
        
        return {
            'domain': domain,
            'url': url,
            'mining_attempted': needs_mining,
            'mining_success': mining_success,
            **crawl_result
        }
    
    async def crawl_list(self, max_concurrent: int = 2) -> Dict:
        """
        Crawl all sites in the list file.
        
        Args:
            max_concurrent: Maximum number of sites to crawl concurrently
            
        Returns:
            Dictionary with overall results
        """
        sites = self.parse_sites_file()
        
        if not sites:
            logger.error("No sites to crawl")
            return {'error': 'No sites to crawl'}
        
        logger.info(f"Starting list crawl of {len(sites)} sites")
        
        # Create semaphore to limit concurrent crawls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(url):
            async with semaphore:
                return await self.process_site(url)
        
        # Process all sites
        results = []
        for url in sites:
            try:
                result = await process_with_semaphore(url)
                results.append(result)
                
                if result['success']:
                    self.stats['sites_processed'] += 1
                else:
                    self.stats['sites_failed'] += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {url}: {e}")
                results.append({
                    'domain': self._extract_domain(url),
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
                self.stats['sites_failed'] += 1
        
        # Save results
        self._save_results(results)
        
        logger.info(f"List crawl completed: {self.stats['sites_processed']} successful, {self.stats['sites_failed']} failed")
        
        return {
            'results': results,
            'stats': self.stats,
            'total_sites': len(sites)
        }
    
    def _save_results(self, results: List[Dict]):
        """Save crawling results to files."""
        # Save detailed results as JSON
        import json
        results_file = self.output_dir / "crawl_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate actual statistics from results
        actual_stats = {
            'sites_processed': sum(1 for r in results if r['success']),
            'sites_failed': sum(1 for r in results if not r['success']),
            'total_images_saved': sum(r.get('images_saved', 0) for r in results),
            'total_thumbnails_saved': sum(r.get('thumbnails_saved', 0) for r in results),
            'selector_mining_attempts': sum(1 for r in results if r.get('mining_attempted', False)),
            'selector_mining_successes': sum(1 for r in results if r.get('mining_success', False))
        }
        
        # Save summary report
        summary_file = self.output_dir / "crawl_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("LIST CRAWL SUMMARY\n")
            f.write("==================\n\n")
            f.write(f"Total sites processed: {len(results)}\n")
            f.write(f"Successful crawls: {actual_stats['sites_processed']}\n")
            f.write(f"Failed crawls: {actual_stats['sites_failed']}\n")
            f.write(f"Total images saved: {actual_stats['total_images_saved']}\n")
            f.write(f"Total thumbnails saved: {actual_stats['total_thumbnails_saved']}\n")
            f.write(f"Selector mining attempts: {actual_stats['selector_mining_attempts']}\n")
            f.write(f"Selector mining successes: {actual_stats['selector_mining_successes']}\n\n")
            
            f.write("PER-SITE DETAILED RESULTS:\n")
            f.write("==========================\n\n")
            
            for result in results:
                f.write(f"Domain: {result['domain']}\n")
                f.write(f"URL: {result['url']}\n")
                f.write(f"Crawl Success: {result['success']}\n")
                
                # Mining information
                if result.get('mining_attempted', False):
                    f.write(f"Selector Mining: Attempted - {'SUCCESS' if result.get('mining_success', False) else 'FAILED'}\n")
                else:
                    f.write(f"Selector Mining: Skipped (using existing recipe)\n")
                
                if result['success']:
                    f.write(f"Images saved: {result.get('images_saved', 0)}\n")
                    f.write(f"Thumbnails saved: {result.get('thumbnails_saved', 0)}\n")
                    f.write(f"Pages crawled: {result.get('pages_crawled', 0)}\n")
                else:
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                
                f.write("-" * 50 + "\n\n")
        
        logger.info(f"Results saved to {self.output_dir}")


async def main():
    """Main function for testing the list crawler."""
    crawler = ListCrawler()
    results = await crawler.crawl_list(max_concurrent=2)
    print(f"Crawl completed: {results}")


if __name__ == "__main__":
    asyncio.run(main())
