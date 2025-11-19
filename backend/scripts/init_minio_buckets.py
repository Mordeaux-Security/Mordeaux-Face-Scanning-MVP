#!/usr/bin/env python3
"""
MinIO Bucket Initialization Script

Ensures all required buckets exist for the face scanning application:
- raw-images: Stores original crawled images
- thumbnails: Stores processed thumbnail images
- audit-logs: Stores audit log files
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.storage import _minio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REQUIRED_BUCKETS = [
    'raw-images',
    'thumbnails', 
    'audit-logs'
]

async def init_minio_buckets():
    """Initialize all required MinIO buckets."""
    settings = get_settings()
    
    logger.info("Initializing MinIO buckets...")
    logger.info(f"MinIO endpoint: {settings.s3_endpoint}")
    logger.info(f"MinIO access key: {settings.s3_access_key[:8]}...")
    
    try:
        # Get MinIO client
        client = _minio()
        logger.info("✓ MinIO client created successfully")
        
        # Initialize each bucket
        for bucket_name in REQUIRED_BUCKETS:
            try:
                # Check if bucket exists
                exists = client.bucket_exists(bucket_name)
                
                if exists:
                    logger.info(f"✓ Bucket '{bucket_name}' already exists")
                else:
                    # Create bucket
                    client.make_bucket(bucket_name)
                    logger.info(f"✓ Created bucket '{bucket_name}'")
                    
            except Exception as e:
                logger.error(f"✗ Failed to initialize bucket '{bucket_name}': {e}")
                return False
        
        logger.info("✓ All MinIO buckets initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ MinIO connection failed: {e}")
        logger.error("Check MinIO service is running and accessible")
        return False

async def verify_buckets():
    """Verify all buckets are accessible."""
    client = _minio()
    
    logger.info("Verifying bucket accessibility...")
    
    for bucket_name in REQUIRED_BUCKETS:
        try:
            exists = client.bucket_exists(bucket_name)
            if exists:
                logger.info(f"✓ Bucket '{bucket_name}' is accessible")
            else:
                logger.error(f"✗ Bucket '{bucket_name}' is not accessible")
                return False
        except Exception as e:
            logger.error(f"✗ Error checking bucket '{bucket_name}': {e}")
            return False
    
    logger.info("✓ All buckets are accessible")
    return True

async def main():
    """Main function to initialize and verify MinIO buckets."""
    logger.info("Starting MinIO bucket initialization...")
    
    # Initialize buckets
    success = await init_minio_buckets()
    if not success:
        logger.error("Bucket initialization failed")
        sys.exit(1)
    
    # Verify buckets
    success = await verify_buckets()
    if not success:
        logger.error("Bucket verification failed")
        sys.exit(1)
    
    logger.info("MinIO bucket initialization completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
