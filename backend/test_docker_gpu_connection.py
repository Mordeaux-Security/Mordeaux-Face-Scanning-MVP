#!/usr/bin/env python3
"""Test GPU worker connection from Docker container."""

import httpx
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gpu_connection():
    """Test GPU worker connection from Docker container."""
    urls_to_test = [
        "http://host.docker.internal:8765/health",
        "http://192.168.68.51:8765/health", 
        "http://localhost:8765/health"
    ]
    
    for url in urls_to_test:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                logger.info(f"✅ {url}: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"   Status: {data.get('status')}")
                    logger.info(f"   GPU Available: {data.get('gpu_available')}")
                    return url
        except Exception as e:
            logger.warning(f"❌ {url}: {e}")
    
    return None

if __name__ == "__main__":
    result = asyncio.run(test_gpu_connection())
    if result:
        print(f"\n🎉 Working URL: {result}")
    else:
        print("\n❌ No working URL found")
