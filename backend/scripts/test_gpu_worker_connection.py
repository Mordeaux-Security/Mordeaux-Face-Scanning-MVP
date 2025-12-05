#!/usr/bin/env python3
"""
GPU Worker Connectivity Test

Tests if the Windows GPU worker is reachable from the Linux container.
This script should be run from within the backend-gpu container to verify
network connectivity to the Windows GPU worker.
"""

import asyncio
import sys
import os
import logging
from urllib.parse import urlparse

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import httpx
from app.core.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_gpu_worker_connection():
    """Test GPU worker connectivity and health."""
    try:
        settings = get_settings()
        
        print("=== GPU Worker Connectivity Test ===")
        print(f"GPU Worker Enabled: {settings.gpu_worker_enabled}")
        print(f"GPU Worker URL: {settings.gpu_worker_url}")
        print(f"GPU Worker Timeout: {settings.gpu_worker_timeout}s")
        print()
        
        if not settings.gpu_worker_enabled:
            print("❌ GPU Worker is disabled in settings")
            return False
        
        # Parse URL to get host and port
        parsed_url = urlparse(settings.gpu_worker_url)
        host = parsed_url.hostname
        port = parsed_url.port or 8765
        
        print(f"Testing connection to {host}:{port}...")
        
        # Test basic connectivity
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                # Test health endpoint
                health_url = f"{settings.gpu_worker_url}/health"
                print(f"Testing health endpoint: {health_url}")
                
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    health_data = response.json()
                    print("✅ GPU Worker is reachable and healthy!")
                    print(f"   Status: {health_data.get('status', 'unknown')}")
                    print(f"   GPU Available: {health_data.get('gpu_available', False)}")
                    print(f"   DirectML Available: {health_data.get('directml_available', False)}")
                    print(f"   Model Loaded: {health_data.get('model_loaded', False)}")
                    print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
                    
                    if health_data.get('directml_available', False):
                        print("✅ DirectML GPU acceleration is available")
                    else:
                        print("⚠️  DirectML not available, worker will use CPU")
                    
                    return True
                else:
                    print(f"❌ GPU Worker returned status {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
            except httpx.ConnectError as e:
                print(f"❌ Connection failed: {e}")
                print("   Possible causes:")
                print("   1. GPU Worker is not running on Windows")
                print("   2. Windows Firewall is blocking port 8765")
                print("   3. host.docker.internal is not resolving correctly")
                print("   4. Docker network configuration issue")
                return False
                
            except httpx.TimeoutException:
                print("❌ Connection timed out")
                print("   GPU Worker may be running but not responding")
                return False
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Test setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gpu_worker_processing():
    """Test actual GPU worker processing with a simple image."""
    try:
        from app.services.gpu_client import get_gpu_client
        
        print("\n=== GPU Worker Processing Test ===")
        
        # Create a simple test image (1x1 pixel PNG)
        test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        print("Testing GPU worker face detection...")
        
        gpu_client = await get_gpu_client()
        results = await gpu_client.detect_faces_batch_async(
            [test_image],
            min_face_quality=0.5,
            require_face=True,
            crop_faces=True,
            face_margin=0.2
        )
        
        print(f"✅ GPU worker processing test completed")
        print(f"   Results: {len(results)} image results")
        print(f"   First image faces: {len(results[0]) if results else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU worker processing test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("GPU Worker Connectivity Test for Docker Container")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    connectivity_ok = await test_gpu_worker_connection()
    
    if connectivity_ok:
        # Test 2: Processing test
        processing_ok = await test_gpu_worker_processing()
        
        if processing_ok:
            print("\n🎉 All tests passed! GPU worker is ready for use.")
            return 0
        else:
            print("\n⚠️  Connectivity OK but processing failed.")
            return 1
    else:
        print("\n❌ Connectivity test failed. GPU worker is not accessible.")
        print("\nTroubleshooting steps:")
        print("1. Start the GPU worker on Windows: cd backend/gpu_worker && launch.bat")
        print("2. Verify worker is accessible: curl http://localhost:8765/health")
        print("3. Check Windows Firewall allows port 8765")
        print("4. Verify host.docker.internal resolves in container")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
