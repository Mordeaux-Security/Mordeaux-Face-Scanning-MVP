#!/usr/bin/env python3
"""Test GPU client initialization."""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_gpu_client():
    """Test GPU client initialization."""
    try:
        from app.services.gpu_client import get_gpu_client
        from app.core.settings import get_settings
        
        settings = get_settings()
        print(f"GPU_WORKER_ENABLED: {settings.gpu_worker_enabled}")
        print(f"GPU_WORKER_URL: {settings.gpu_worker_url}")
        
        if settings.gpu_worker_enabled:
            client = await get_gpu_client()
            print(f"GPU Client: {client}")
            
            # Test health check
            health = await client._check_health()
            print(f"Health check: {health}")
        else:
            print("GPU worker is disabled")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gpu_client())
