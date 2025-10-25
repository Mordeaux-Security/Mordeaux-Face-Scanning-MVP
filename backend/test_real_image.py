#!/usr/bin/env python3
"""Test GPU worker with a real image."""

import asyncio
import base64
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_real_image():
    """Test GPU worker with a real image."""
    try:
        from app.services.gpu_client import get_gpu_client
        
        print("=== Real Image Test ===")
        
        # Create a simple 100x100 PNG image
        import io
        from PIL import Image
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        print(f"Created test image: {len(img_data)} bytes")
        
        # Test GPU client
        client = await get_gpu_client()
        result = await client.detect_faces_batch_async([img_data], 0.5, True, True, 0.2)
        print(f"Result: {result}")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_image())
