#!/usr/bin/env python3
"""
Prestart script to initialize MinIO buckets before the API starts.
This ensures buckets exist even if the startup event handler fails.
"""

import logging
import sys
import time
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_buckets():
    """Initialize MinIO buckets."""
    try:
        from app.core.config import get_settings
        from app.services.storage import _minio
        
        settings = get_settings()
        
        # Check if using MinIO
        if not settings.using_minio:
            logger.info("Not using MinIO, skipping bucket initialization")
            return
        
        logger.info("Initializing MinIO buckets...")
        logger.info(f"MinIO endpoint: {settings.s3_endpoint}")
        
        # Wait for MinIO to be ready (retry up to 30 times with 2 second intervals)
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                client = _minio()
                # Try to list buckets to verify connection
                client.list_buckets()
                logger.info("✓ MinIO connection successful")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"✗ MinIO connection failed after {max_retries} retries: {e}")
                    logger.error("Continuing anyway - buckets will be created on startup")
                    return
                logger.warning(f"MinIO not ready yet (attempt {retry_count}/{max_retries}), retrying in 2 seconds...")
                time.sleep(2)
        
        # Required buckets
        REQUIRED_BUCKETS = [
            'raw-images',
            'thumbnails',
            'audit-logs'
        ]
        
        # Initialize each bucket
        for bucket_name in REQUIRED_BUCKETS:
            try:
                exists = client.bucket_exists(bucket_name)
                if exists:
                    logger.info(f"✓ Bucket '{bucket_name}' already exists")
                else:
                    client.make_bucket(bucket_name)
                    logger.info(f"✓ Created bucket '{bucket_name}'")
            except Exception as e:
                logger.error(f"✗ Failed to initialize bucket '{bucket_name}': {e}")
        
        logger.info("✓ MinIO bucket initialization completed")
        
    except Exception as e:
        logger.error(f"✗ Error during bucket initialization: {e}")
        logger.error("Continuing anyway - buckets may be created on startup")

if __name__ == "__main__":
    init_buckets()

