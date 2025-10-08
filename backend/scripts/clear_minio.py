#!/usr/bin/env python3
"""
Clear MinIO buckets script.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, '/app')

from app.services.storage import _minio
from app.core.config import get_settings

def clear_minio_buckets():
    """Clear all objects from MinIO buckets."""
    try:
        client = _minio()
        settings = get_settings()
        
        # Clear raw-images bucket
        try:
            objects = list(client.list_objects(settings.s3_bucket_raw, recursive=True))
            for obj in objects:
                client.remove_object(settings.s3_bucket_raw, obj.object_name)
            print(f'Cleared {settings.s3_bucket_raw} bucket ({len(objects)} objects)')
        except Exception as e:
            print(f'Error clearing {settings.s3_bucket_raw}: {e}')
        
        # Clear thumbnails bucket
        try:
            objects = list(client.list_objects(settings.s3_bucket_thumbs, recursive=True))
            for obj in objects:
                client.remove_object(settings.s3_bucket_thumbs, obj.object_name)
            print(f'Cleared {settings.s3_bucket_thumbs} bucket ({len(objects)} objects)')
        except Exception as e:
            print(f'Error clearing {settings.s3_bucket_thumbs}: {e}')
        
        print('MinIO buckets cleared.')
        
    except Exception as e:
        print(f'Error connecting to MinIO: {e}')
        sys.exit(1)

if __name__ == '__main__':
    clear_minio_buckets()
