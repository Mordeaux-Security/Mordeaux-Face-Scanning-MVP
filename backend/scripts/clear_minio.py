#!/usr/bin/env python3
"""
Clear MinIO buckets script.
"""

import sys
import os
from minio import Minio

# Add the backend directory to Python path
sys.path.insert(0, '/app')

from app.core.config import get_settings

def clear_minio_buckets():
    """Clear all objects from MinIO buckets."""
    try:
        # Get settings
        settings = get_settings()
        
        # Create MinIO client
        endpoint = os.getenv('S3_ENDPOINT', 'http://minio:9000').replace('http://', '').replace('https://', '')
        client = Minio(
            endpoint,
            access_key=os.getenv('S3_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('S3_SECRET_KEY', 'minioadmin'),
            secure=False
        )
        
        # Clear the main bucket (crawled-images) which contains both images/ and thumbnails/ folders
        bucket_name = settings.s3_bucket_raw  # This should be "crawled-images"
        
        try:
            # Check if bucket exists
            if client.bucket_exists(bucket_name):
                # List and remove all objects from both images/ and thumbnails/ folders
                objects = list(client.list_objects(bucket_name, recursive=True))
                for obj in objects:
                    client.remove_object(bucket_name, obj.object_name)
                print(f'Cleared {bucket_name} bucket ({len(objects)} objects from images/ and thumbnails/ folders)')
            else:
                print(f'Bucket {bucket_name} does not exist - nothing to clear')
        except Exception as e:
            print(f'Error clearing {bucket_name}: {e}')
        
        print('MinIO buckets cleared.')
        
    except Exception as e:
        print(f'Error connecting to MinIO: {e}')
        sys.exit(1)

if __name__ == '__main__':
    clear_minio_buckets()
