#!/usr/bin/env python3
"""
Download all thumbnail images from MinIO to a local 'thumbs' folder.

Usage:
    python backend/scripts/download_all_thumbs.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from minio import Minio
from minio.error import S3Error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_minio_client() -> Minio:
    """Create and return a MinIO client."""
    # Get configuration from environment variables
    endpoint = os.getenv("S3_ENDPOINT", "localhost:9000").replace("https://", "").replace("http://", "")
    access_key = os.getenv("S3_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("S3_SECRET_KEY", "minioadmin")
    use_ssl = os.getenv("S3_USE_SSL", "false").lower() == "true"
    
    logger.info(f"Connecting to MinIO at {endpoint}")
    
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=use_ssl
    )
    
    return client


def download_all_thumbs(output_dir: str = "thumbs") -> None:
    """
    Download all thumbnail images from MinIO to a local folder.
    
    Args:
        output_dir: Directory to save downloaded images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Get MinIO client
    client = get_minio_client()
    
    # Bucket name
    bucket_name = os.getenv("S3_BUCKET_THUMBS", "thumbnails")
    
    try:
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            logger.error(f"Bucket '{bucket_name}' does not exist")
            return
        
        logger.info(f"Listing objects in bucket '{bucket_name}'...")
        
        # List all objects in the bucket
        objects = client.list_objects(bucket_name, recursive=True)
        
        downloaded_count = 0
        error_count = 0
        
        for obj in objects:
            try:
                # Get object name (key)
                object_name = obj.object_name
                
                # Skip if it's not an image file
                if not any(object_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    continue
                
                # Create local file path (flatten the directory structure)
                # Use just the filename to avoid nested directories
                filename = Path(object_name).name
                local_file_path = output_path / filename
                
                # Skip if file already exists
                if local_file_path.exists():
                    logger.debug(f"Skipping {object_name} (already exists)")
                    continue
                
                # Download the object
                logger.info(f"Downloading {object_name}...")
                client.fget_object(bucket_name, object_name, str(local_file_path))
                downloaded_count += 1
                
                if downloaded_count % 100 == 0:
                    logger.info(f"Downloaded {downloaded_count} images...")
                    
            except Exception as e:
                logger.error(f"Error downloading {object_name}: {e}")
                error_count += 1
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Download complete!")
        logger.info(f"Successfully downloaded: {downloaded_count} images")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Output directory: {output_path.absolute()}")
        logger.info(f"{'='*60}")
        
    except S3Error as e:
        logger.error(f"MinIO error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    download_all_thumbs()



