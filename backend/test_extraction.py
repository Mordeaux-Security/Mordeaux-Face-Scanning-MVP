#!/usr/bin/env python3
"""
Extraction Test Harness

Tests individual extraction strategies on saved HTML samples.
Validates that found elements are actually posts (not nav/errors).
Generates strategy effectiveness reports.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from new_crawler.selector_miner import SelectorMiner
from new_crawler.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_html_sample(file_path: Path) -> Optional[str]:
    """Load HTML from a saved sample file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load HTML from {file_path}: {e}")
        return None


def validate_post_content(element, min_length: int = 50) -> tuple[bool, str]:
    """
    Validate that an element is actually a post, not an error/nav/menu.
    Returns (is_valid, reason).
    """
    if not element:
        return False, "Element is None"
    
    text = element.get_text(strip=True)
    
    # Check minimum length
    if len(text) < min_length:
        return False, f"Content too short: {len(text)} chars"
    
    # Check for error messages
    error_keywords = ['error', 'not found', '404', '500', 'forbidden', 'unauthorized', 
                     'page not found', 'access denied', 'try again', 'something went wrong']
    text_lower = text.lower()
    for keyword in error_keywords:
        if keyword in text_lower:
            return False, f"Contains error keyword: {keyword}"
    
    # Check for navigation/menu (too many links relative to text)
    links = element.find_all('a')
    link_count = len(links)
    text_length = len(text)
    if text_length > 0 and link_count > text_length / 50:  # More than 1 link per 50 chars
        return False, f"Too many links ({link_count}) relative to text ({text_length} chars)"
    
    # Check for post-like structure (has title, content, or author indicators)
    has_title = bool(element.find(['h1', 'h2', 'h3', 'h4', '[class*="title"]', '[class*="heading"]']))
    has_author = bool(element.find(['[class*="author"]', '[class*="user"]', '[class*="poster"]']))
    has_date = bool(element.find(['time', '[class*="date"]', '[class*="time"]']))
    
    if not (has_title or has_author or has_date):
        # Still might be valid if it has substantial content
        if len(text) < 200:
            return False, "No post structure indicators and content too short"
    
    return True, "Valid post content"


def test_extraction_strategies(html: str, url: str, site_id: str = "test") -> Dict[str, Any]:
    """Test all extraction strategies on a given HTML sample."""
    soup = BeautifulSoup(html, 'html.parser')
    miner = SelectorMiner()
    
    results = {
        'url': url,
        'site_id': site_id,
        'strategies_tested': [],
        'successful_strategies': [],
        'failed_strategies': [],
        'best_result': None
    }
    
    # Test Strategy 0: Reddit
    try:
        shreddit_posts = soup.select('shreddit-post')
        if shreddit_posts:
            main_post = shreddit_posts[0]
            content = main_post.get_text(strip=True)
            is_valid, reason = validate_post_content(main_post)
            results['strategies_tested'].append({
                'strategy': 'strategy_0_reddit',
                'elements_found': len(shreddit_posts),
                'content_length': len(content),
                'is_valid': is_valid,
                'validation_reason': reason
            })
            if is_valid:
                results['successful_strategies'].append('strategy_0_reddit')
                if not results['best_result'] or len(content) > len(results['best_result'].get('content', '')):
                    results['best_result'] = {
                        'strategy': 'strategy_0_reddit',
                        'content_length': len(content),
                        'title': main_post.select_one('h1, [slot="title"]'),
                        'has_author': bool(main_post.select_one('[slot="author"]')),
                        'has_date': bool(main_post.select_one('time[datetime]'))
                    }
            else:
                results['failed_strategies'].append({
                    'strategy': 'strategy_0_reddit',
                    'reason': reason
                })
    except Exception as e:
        results['failed_strategies'].append({
            'strategy': 'strategy_0_reddit',
            'reason': f"Exception: {str(e)}"
        })
    
    # Test Strategy 1: vBulletin
    try:
        thread_posts = soup.select('div.threadpost, div[id*="edit"][id*="post"]')
        if thread_posts:
            main_post = thread_posts[0]
            content = main_post.get_text(strip=True)
            is_valid, reason = validate_post_content(main_post)
            results['strategies_tested'].append({
                'strategy': 'strategy_1_vbulletin',
                'elements_found': len(thread_posts),
                'content_length': len(content),
                'is_valid': is_valid,
                'validation_reason': reason
            })
            if is_valid:
                results['successful_strategies'].append('strategy_1_vbulletin')
                if not results['best_result'] or len(content) > len(results['best_result'].get('content_length', 0)):
                    results['best_result'] = {
                        'strategy': 'strategy_1_vbulletin',
                        'content_length': len(content)
                    }
            else:
                results['failed_strategies'].append({
                    'strategy': 'strategy_1_vbulletin',
                    'reason': reason
                })
    except Exception as e:
        results['failed_strategies'].append({
            'strategy': 'strategy_1_vbulletin',
            'reason': f"Exception: {str(e)}"
        })
    
    # Test Strategy 2: Post message divs
    try:
        post_messages = soup.select('div[id*="post_message"]')
        if post_messages:
            main_post = post_messages[0]
            content = main_post.get_text(strip=True)
            is_valid, reason = validate_post_content(main_post)
            results['strategies_tested'].append({
                'strategy': 'strategy_2_post_message',
                'elements_found': len(post_messages),
                'content_length': len(content),
                'is_valid': is_valid,
                'validation_reason': reason
            })
            if is_valid:
                results['successful_strategies'].append('strategy_2_post_message')
                if not results['best_result'] or len(content) > results['best_result'].get('content_length', 0):
                    results['best_result'] = {
                        'strategy': 'strategy_2_post_message',
                        'content_length': len(content)
                    }
            else:
                results['failed_strategies'].append({
                    'strategy': 'strategy_2_post_message',
                    'reason': reason
                })
    except Exception as e:
        results['failed_strategies'].append({
            'strategy': 'strategy_2_post_message',
            'reason': f"Exception: {str(e)}"
        })
    
    # Test Strategy 3: Standard post patterns (simplified)
    try:
        post_selectors = [
            '[class*="post-content"]', '[class*="message-content"]',
            '[class*="post-body"]', '[class*="message-body"]',
            '[class*="post"]', '[class*="message"]'
        ]
        for selector in post_selectors:
            posts = soup.select(selector)
            if posts:
                main_post = posts[0]
                content = main_post.get_text(strip=True)
                if len(content) >= 20:  # Minimum content length
                    is_valid, reason = validate_post_content(main_post, min_length=20)
                    results['strategies_tested'].append({
                        'strategy': f'strategy_3_standard_{selector}',
                        'elements_found': len(posts),
                        'content_length': len(content),
                        'is_valid': is_valid,
                        'validation_reason': reason
                    })
                    if is_valid:
                        results['successful_strategies'].append(f'strategy_3_standard_{selector}')
                        if not results['best_result'] or len(content) > results['best_result'].get('content_length', 0):
                            results['best_result'] = {
                                'strategy': f'strategy_3_standard_{selector}',
                                'content_length': len(content)
                            }
                        break  # Found a valid one, stop trying other selectors
    except Exception as e:
        results['failed_strategies'].append({
            'strategy': 'strategy_3_standard',
            'reason': f"Exception: {str(e)}"
        })
    
    return results


def generate_strategy_report(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a report on strategy effectiveness."""
    report = {
        'total_tests': len(test_results),
        'strategy_stats': {},
        'overall_success_rate': 0.0,
        'best_strategies': []
    }
    
    strategy_counts = {}
    strategy_successes = {}
    
    for result in test_results:
        for strategy_info in result.get('strategies_tested', []):
            strategy = strategy_info['strategy']
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
                strategy_successes[strategy] = 0
            strategy_counts[strategy] += 1
            if strategy_info.get('is_valid', False):
                strategy_successes[strategy] += 1
    
    for strategy, count in strategy_counts.items():
        success_count = strategy_successes.get(strategy, 0)
        report['strategy_stats'][strategy] = {
            'attempts': count,
            'successes': success_count,
            'success_rate': success_count / count if count > 0 else 0.0
        }
    
    # Calculate overall success rate
    total_successful = sum(1 for r in test_results if r.get('successful_strategies'))
    report['overall_success_rate'] = total_successful / len(test_results) if test_results else 0.0
    
    # Find best strategies
    sorted_strategies = sorted(
        report['strategy_stats'].items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )
    report['best_strategies'] = [s[0] for s in sorted_strategies[:5]]
    
    return report


def main():
    """Main entry point for testing extraction strategies."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test extraction strategies on HTML samples")
    parser.add_argument("html_file", type=Path, help="Path to HTML file to test")
    parser.add_argument("--url", type=str, default="http://example.com", help="URL of the HTML sample")
    parser.add_argument("--site-id", type=str, default="test", help="Site ID for testing")
    parser.add_argument("--output", type=Path, help="Path to save test results JSON")
    
    args = parser.parse_args()
    
    # Load HTML
    html = load_html_sample(args.html_file)
    if not html:
        logger.error("Failed to load HTML sample")
        return 1
    
    # Test extraction
    logger.info(f"Testing extraction strategies on {args.html_file}")
    result = test_extraction_strategies(html, args.url, args.site_id)
    
    # Print results
    print("\n" + "="*80)
    print("EXTRACTION TEST RESULTS")
    print("="*80)
    print(f"URL: {result['url']}")
    print(f"Strategies tested: {len(result['strategies_tested'])}")
    print(f"Successful strategies: {len(result['successful_strategies'])}")
    print(f"Failed strategies: {len(result['failed_strategies'])}")
    
    if result['best_result']:
        print(f"\nBest result:")
        print(f"  Strategy: {result['best_result']['strategy']}")
        print(f"  Content length: {result['best_result'].get('content_length', 0)}")
    
    if result['failed_strategies']:
        print(f"\nFailed strategies:")
        for failed in result['failed_strategies']:
            print(f"  {failed['strategy']}: {failed.get('reason', 'Unknown')}")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    
    return 0 if result['successful_strategies'] else 1


if __name__ == "__main__":
    sys.exit(main())

