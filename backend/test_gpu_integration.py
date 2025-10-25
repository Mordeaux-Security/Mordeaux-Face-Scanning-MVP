#!/usr/bin/env python3
"""Test GPU worker integration."""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_gpu_integration():
    """Test GPU worker integration."""
    try:
        from app.services.gpu_client import get_gpu_client
        from app.services.face import detect_and_embed_batch_async
        
        print("=== GPU Integration Test ===")
        
        # Test GPU client directly
        print("Testing GPU client directly...")
        client = await get_gpu_client()
        
        # Test with a simple image (1x1 pixel PNG)
        test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        result = await client.detect_faces_batch_async([test_image], 0.5, True, True, 0.2)
        print(f"GPU client result: {result}")
        
        # Test face service integration
        print("\nTesting face service integration...")
        result2 = await detect_and_embed_batch_async([test_image], 1)
        print(f"Face service result: {result2}")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"✗ Error testing GPU integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gpu_integration())
