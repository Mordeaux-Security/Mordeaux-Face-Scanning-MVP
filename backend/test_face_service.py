#!/usr/bin/env python3
"""Test face service GPU worker integration."""

import asyncio
import sys
import os
import logging

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Set up logging to see diagnostic messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_face_service():
    """Test face service GPU worker integration."""
    try:
        from app.services.face import detect_faces_batch_async
        from app.core.settings import get_settings
        
        settings = get_settings()
        print(f"GPU_WORKER_ENABLED: {settings.gpu_worker_enabled}")
        print(f"GPU_WORKER_URL: {settings.gpu_worker_url}")
        
        # Create a dummy image
        from PIL import Image
        import io
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='JPEG')
        image_bytes = byte_arr.getvalue()
        
        print(f"Testing face detection with GPU worker...")
        results = await detect_faces_batch_async([image_bytes])
        print(f"Results: {len(results)} images processed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_face_service())
