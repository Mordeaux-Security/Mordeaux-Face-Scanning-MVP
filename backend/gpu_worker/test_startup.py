"""
Simple GPU Worker Startup Test

Tests DirectML availability, port binding, and model loading.
This is a temporary diagnostic script.
"""

import os
import sys
import socket
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_directml_availability():
    """Test if DirectML is available."""
    logger.info("=== Testing DirectML Availability ===")
    
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        directml_available = 'DmlExecutionProvider' in available_providers
        if directml_available:
            logger.info("✓ DirectML is available")
        else:
            logger.warning("✗ DirectML is not available - will use CPU")
        
        return directml_available
    except ImportError as e:
        logger.error(f"✗ ONNX Runtime not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error checking DirectML: {e}")
        return False

def test_port_binding():
    """Test if port 8765 is available."""
    logger.info("=== Testing Port Binding ===")
    
    port = 8765
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            logger.info(f"✓ Port {port} is available")
            return True
    except OSError as e:
        logger.error(f"✗ Port {port} is not available: {e}")
        return False

def test_model_loading():
    """Test if InsightFace model can be loaded."""
    logger.info("=== Testing Model Loading ===")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        logger.info("Loading InsightFace model...")
        home = os.path.expanduser("~/.insightface")
        os.makedirs(home, exist_ok=True)
        
        app = FaceAnalysis(name='buffalo_l', root=home)
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        logger.info("✓ InsightFace model loaded successfully")
        
        # Check which providers are being used
        if hasattr(app, 'models'):
            for model_name, model in app.models.items():
                if hasattr(model, 'session'):
                    providers = model.session.get_providers()
                    logger.info(f"Model '{model_name}' using providers: {providers}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False

def test_gpu_worker_endpoints():
    """Test if GPU worker endpoints are accessible."""
    logger.info("=== Testing GPU Worker Endpoints ===")
    
    try:
        import requests
        
        base_url = "http://localhost:8765"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Health endpoint accessible")
                health_data = response.json()
                logger.info(f"Health data: {health_data}")
            else:
                logger.warning(f"✗ Health endpoint returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Health endpoint not accessible: {e}")
            return False
        
        # Test GPU info endpoint
        try:
            response = requests.get(f"{base_url}/gpu_info", timeout=5)
            if response.status_code == 200:
                logger.info("✓ GPU info endpoint accessible")
                gpu_data = response.json()
                logger.info(f"GPU info: {gpu_data}")
            else:
                logger.warning(f"✗ GPU info endpoint returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ GPU info endpoint not accessible: {e}")
            return False
        
        return True
    except ImportError:
        logger.error("✗ Requests library not available")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing endpoints: {e}")
        return False

def main():
    """Run all startup tests."""
    logger.info("=== GPU Worker Startup Diagnostics ===")
    
    tests = [
        ("DirectML Availability", test_directml_availability),
        ("Port Binding", test_port_binding),
        ("Model Loading", test_model_loading),
        ("GPU Worker Endpoints", test_gpu_worker_endpoints)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! GPU worker should be ready.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
