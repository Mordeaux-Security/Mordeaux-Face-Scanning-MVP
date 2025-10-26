"""
GPU Worker Health Check Utility

Provides health check functionality to verify GPU worker connectivity
and availability before starting multiprocess crawlers.
"""

import asyncio
import logging
import httpx
from typing import Optional, Dict, Any
from ..core.settings import get_settings

logger = logging.getLogger(__name__)


async def check_gpu_worker_health(timeout: float = 10.0) -> Dict[str, Any]:
    """
    Check GPU worker health and return status information.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Dict with health status information
    """
    settings = get_settings()
    
    if not settings.gpu_worker_enabled:
        return {
            'status': 'disabled',
            'gpu_worker_enabled': False,
            'message': 'GPU worker is disabled in settings'
        }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            health_url = f"{settings.gpu_worker_url}/health"
            logger.info(f"Checking GPU worker health at: {health_url}")
            
            response = await client.get(health_url)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info("GPU worker health check successful")
                return {
                    'status': 'healthy',
                    'gpu_worker_enabled': True,
                    'gpu_worker_url': settings.gpu_worker_url,
                    'health_data': health_data,
                    'message': 'GPU worker is healthy and accessible'
                }
            else:
                logger.error(f"GPU worker returned status {response.status_code}")
                return {
                    'status': 'unhealthy',
                    'gpu_worker_enabled': True,
                    'gpu_worker_url': settings.gpu_worker_url,
                    'status_code': response.status_code,
                    'message': f'GPU worker returned HTTP {response.status_code}'
                }
                
    except httpx.ConnectError as e:
        logger.error(f"Failed to connect to GPU worker: {e}")
        return {
            'status': 'unreachable',
            'gpu_worker_enabled': True,
            'gpu_worker_url': settings.gpu_worker_url,
            'error': str(e),
            'message': f'Cannot connect to GPU worker at {settings.gpu_worker_url}'
        }
    except httpx.TimeoutException as e:
        logger.error(f"GPU worker health check timed out: {e}")
        return {
            'status': 'timeout',
            'gpu_worker_enabled': True,
            'gpu_worker_url': settings.gpu_worker_url,
            'error': str(e),
            'message': f'GPU worker health check timed out after {timeout}s'
        }
    except Exception as e:
        logger.error(f"Unexpected error during GPU worker health check: {e}")
        return {
            'status': 'error',
            'gpu_worker_enabled': True,
            'gpu_worker_url': settings.gpu_worker_url,
            'error': str(e),
            'message': f'Unexpected error: {e}'
        }


def check_gpu_worker_health_sync(timeout: float = 10.0) -> Dict[str, Any]:
    """
    Synchronous wrapper for GPU worker health check.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Dict with health status information
    """
    return asyncio.run(check_gpu_worker_health(timeout))


def validate_gpu_worker_startup() -> bool:
    """
    Validate GPU worker is ready for multiprocess crawler startup.
    
    Returns:
        True if GPU worker is ready, False otherwise
    """
    logger.info("Validating GPU worker startup...")
    
    health_result = check_gpu_worker_health_sync()
    
    if health_result['status'] == 'healthy':
        health_data = health_result.get('health_data', {})
        
        # Check critical health indicators
        gpu_available = health_data.get('gpu_available', False)
        model_loaded = health_data.get('model_loaded', False)
        
        if gpu_available and model_loaded:
            logger.info("✅ GPU worker validation successful - ready for processing")
            return True
        else:
            logger.warning(f"⚠️ GPU worker validation failed - GPU available: {gpu_available}, Model loaded: {model_loaded}")
            return False
    else:
        logger.error(f"❌ GPU worker validation failed: {health_result['message']}")
        return False


def print_gpu_worker_troubleshooting():
    """Print troubleshooting steps for GPU worker issues."""
    print("\n🔧 GPU Worker Troubleshooting:")
    print("1. Start the GPU worker: cd backend/gpu_worker && python launch.py")
    print("2. Verify worker is accessible: curl http://localhost:8765/health")
    print("3. Check Windows Firewall allows port 8765")
    print("4. For Docker: ensure GPU_WORKER_URL=http://host.docker.internal:8765")
    print("5. For native Windows: ensure GPU_WORKER_URL=http://localhost:8765")
    print("6. Check GPU worker logs for errors")
    print()


if __name__ == "__main__":
    # Command line health check
    import sys
    
    health_result = check_gpu_worker_health_sync()
    
    print("GPU Worker Health Check")
    print("=" * 30)
    print(f"Status: {health_result['status']}")
    print(f"Message: {health_result['message']}")
    
    if health_result['status'] != 'healthy':
        print_gpu_worker_troubleshooting()
        sys.exit(1)
    else:
        print("✅ GPU worker is healthy and ready!")
        sys.exit(0)
