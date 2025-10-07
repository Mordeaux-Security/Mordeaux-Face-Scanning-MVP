#!/usr/bin/env python3
"""
Clear MinIO buckets script.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, '/app')

from app.services.storage import get_minio_client, BUCKET_RAW, BUCKET_THUMBS

def clear_minio_buckets():
    """Clear all objects from MinIO buckets."""
    try:
        client = get_minio_client()
        
        # Clear raw-images bucket
        try:
            objects = list(client.list_objects(BUCKET_RAW, recursive=True))
            for obj in objects:
                client.remove_object(BUCKET_RAW, obj.object_name)
            print(f'Cleared {BUCKET_RAW} bucket ({len(objects)} objects)')
        except Exception as e:
            print(f'Error clearing {BUCKET_RAW}: {e}')
        
        # Clear thumbnails bucket
        try:
            objects = list(client.list_objects(BUCKET_THUMBS, recursive=True))
            for obj in objects:
                client.remove_object(BUCKET_THUMBS, obj.object_name)
            print(f'Cleared {BUCKET_THUMBS} bucket ({len(objects)} objects)')
        except Exception as e:
            print(f'Error clearing {BUCKET_THUMBS}: {e}')
        
        print('MinIO buckets cleared.')
        
    except Exception as e:
        print(f'Error connecting to MinIO: {e}')
        sys.exit(1)

if __name__ == '__main__':
    clear_minio_buckets()
