#!/usr/bin/env python3
"""Debug GPU worker integration."""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def debug_gpu_worker():
    """Debug GPU worker integration."""
    try:
        from app.services.gpu_client import get_gpu_client
        from app.core.settings import get_settings
        
        print("=== GPU Worker Debug ===")
        
        # Check settings
        settings = get_settings()
        print(f"GPU Worker Enabled: {settings.gpu_worker_enabled}")
        print(f"GPU Worker URL: {settings.gpu_worker_url}")
        
        # Test GPU client
        client = await get_gpu_client()
        print("GPU Client created")
        
        # Test health check
        print("Testing health check...")
        health = await client._check_health()
        print(f"Health check result: {health}")
        
        # Test with a simple image
        import io
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        print(f"Created test image: {len(img_data)} bytes")
        
        # Test GPU worker request
        print("Testing GPU worker request...")
        result = await client.detect_faces_batch_async([img_data], 0.5, True, True, 0.2)
        print(f"GPU worker result: {result}")
        
        print("\n=== Debug Complete ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_gpu_worker())
