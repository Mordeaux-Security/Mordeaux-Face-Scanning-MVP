#!/usr/bin/env python3
"""Test individual face detection with GPU worker."""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_individual_gpu():
    """Test individual face detection with GPU worker."""
    try:
        from app.services.face import detect_and_embed_async
        
        print("=== Individual GPU Test ===")
        
        # Create a simple test image
        import io
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        print(f"Created test image: {len(img_data)} bytes")
        
        # Test individual face detection
        print("Testing individual face detection with GPU worker...")
        result = await detect_and_embed_async(img_data)
        print(f"Individual face detection result: {result}")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_individual_gpu())
