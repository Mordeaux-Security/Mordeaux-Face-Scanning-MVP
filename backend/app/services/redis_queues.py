"""
Redis Queue Helper Functions

Provides helper functions for Redis-based queue operations used in multiprocessing crawler.
"""

import json
import logging
import redis
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_redis_client(redis_url: Optional[str] = None) -> redis.Redis:
    """
    Get Redis client instance.
    
    Args:
        redis_url: Redis connection URL. If None, uses environment variable.
        
    Returns:
        Redis client instance
    """
    if redis_url is None:
        import os
        # Check if running in Docker
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
    
    # Use decode_responses=True to match existing cache.py pattern
    client = redis.from_url(redis_url, decode_responses=True, db=0)
    
    # Test connection
    try:
        # Force connection to DB 0
        client.execute_command('SELECT', 0)
        client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    return client


def setup_redis_queues(redis_client: redis.Redis) -> None:
    """
    Initialize Redis queues for multiprocessing.
    Clears any existing queues.
    
    Args:
        redis_client: Redis client instance
    """
    try:
        redis_client.delete(
            'sites_to_crawl',
            'crawled_pages',
            'cpu_fallback_queue',
            'cpu_results',
            'extraction_results'
        )
        logger.info("Redis queues initialized")
    except Exception as e:
        logger.error(f"Error setting up Redis queues: {e}")
        raise


def push_sites_to_queue(redis_client: redis.Redis, sites: List[str]) -> None:
    """
    Push sites to crawling queue.
    
    Args:
        redis_client: Redis client instance
        sites: List of URLs to crawl
    """
    logger.info(f"Pushing {len(sites)} sites to crawl queue")
    
    for i, site in enumerate(sites, 1):
        result = redis_client.rpush('sites_to_crawl', site)
        logger.info(f"Pushed site {i}/{len(sites)}: {site}, result={result}")
    
    length = redis_client.llen('sites_to_crawl')
    logger.info(f"Queue length after pushing: {length}")


def get_site_from_queue(redis_client: redis.Redis, timeout: int = 5) -> Optional[str]:
    """
    Blocking pop from sites queue.
    
    Args:
        redis_client: Redis client instance
        timeout: Timeout in seconds
        
    Returns:
        Site URL or None if timeout
    """
    result = redis_client.blpop('sites_to_crawl', timeout=timeout)
    if result:
        return result[1]
    return None


def push_crawled_page(redis_client: redis.Redis, page_data: Dict[str, Any]) -> None:
    """
    Push crawled page data to queue for extraction workers.
    
    Args:
        redis_client: Redis client instance
        page_data: Dict containing site, html, images, etc.
    """
    try:
        json_data = json.dumps(page_data)
        redis_client.rpush('crawled_pages', json_data)
    except Exception as e:
        logger.error(f"Error pushing crawled page: {e}")


def get_crawled_page(redis_client: redis.Redis, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Blocking pop from crawled pages queue.
    
    Args:
        redis_client: Redis client instance
        timeout: Timeout in seconds
        
    Returns:
        Page data dict or None if timeout
    """
    result = redis_client.blpop('crawled_pages', timeout=timeout)
    if result:
        try:
            return json.loads(result[1])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding crawled page JSON: {e}")
            return None
    return None


def push_cpu_fallback_job(redis_client: redis.Redis, image_data: bytes, image_info: Dict[str, Any]) -> None:
    """
    Push image to CPU fallback queue.
    
    Args:
        redis_client: Redis client instance
        image_data: Raw image bytes
        image_info: Image metadata
    """
    try:
        job_data = {
            'image_data': image_data.hex(),  # Convert bytes to hex string for JSON
            'image_info': image_info
        }
        json_data = json.dumps(job_data)
        redis_client.rpush('cpu_fallback_queue', json_data)
    except Exception as e:
        logger.error(f"Error pushing CPU fallback job: {e}")


def get_cpu_fallback_job(redis_client: redis.Redis, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Blocking pop from CPU fallback queue.
    
    Args:
        redis_client: Redis client instance
        timeout: Timeout in seconds
        
    Returns:
        Job data dict or None if timeout
    """
    result = redis_client.blpop('cpu_fallback_queue', timeout=timeout)
    if result:
        try:
            job_data = json.loads(result[1])
            # Convert hex string back to bytes
            job_data['image_data'] = bytes.fromhex(job_data['image_data'])
            return job_data
        except Exception as e:
            logger.error(f"Error decoding CPU fallback job: {e}")
            return None
    return None


def push_extraction_result(redis_client: redis.Redis, result: Dict[str, Any]) -> None:
    """
    Push extraction result to results queue.
    
    Args:
        redis_client: Redis client instance
        result: Result data
    """
    try:
        json_data = json.dumps(result)
        redis_client.rpush('extraction_results', json_data)
    except Exception as e:
        logger.error(f"Error pushing extraction result: {e}")


def get_extraction_result(redis_client: redis.Redis, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Blocking pop from extraction results queue.
    
    Args:
        redis_client: Redis client instance
        timeout: Timeout in seconds
        
    Returns:
        Result dict or None if timeout
    """
    result = redis_client.blpop('extraction_results', timeout=timeout)
    if result:
        try:
            return json.loads(result[1])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding extraction result: {e}")
            return None
    return None 