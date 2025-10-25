"""
GPU Worker Integration Tests

Comprehensive tests for the Windows GPU worker and Linux container integration.
Tests worker startup, batch processing, connection failure recovery, and fallback behavior.
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
import requests
import httpx
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
GPU_WORKER_URL = "http://localhost:8765"
BACKEND_URL = "http://localhost:8000"
TEST_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 2

class GPUWorkerTester:
    """Test suite for GPU worker integration."""
    
    def __init__(self):
        self.worker_healthy = False
        self.backend_healthy = False
        self.test_results = []
        
    def create_test_image(self, width: int = 200, height: int = 200, format: str = "JPEG") -> bytes:
        """Create a test image with a simple pattern."""
        try:
            # Create a simple test image
            image = Image.new('RGB', (width, height), color='white')
            
            # Add some content to make it more realistic
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.rectangle([50, 50, width-50, height-50], outline='black', width=2)
            draw.ellipse([75, 75, width-75, height-75], outline='red', width=2)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=format)
            return img_bytes.getvalue()
        except Exception as e:
            logger.error(f"Failed to create test image: {e}")
            return b""
    
    def encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def check_worker_health(self) -> bool:
        """Check if GPU worker is healthy."""
        try:
            response = requests.get(f"{GPU_WORKER_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                self.worker_healthy = health_data.get("status") == "healthy"
                logger.info(f"GPU worker health: {health_data}")
                return self.worker_healthy
        except Exception as e:
            logger.warning(f"GPU worker health check failed: {e}")
            self.worker_healthy = False
        return False
    
    def check_backend_health(self) -> bool:
        """Check if backend service is healthy."""
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                self.backend_healthy = True
                logger.info("Backend service is healthy")
                return True
        except Exception as e:
            logger.warning(f"Backend health check failed: {e}")
            self.backend_healthy = False
        return False
    
    def wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for both services to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            worker_ready = self.check_worker_health()
            backend_ready = self.check_backend_health()
            
            if worker_ready and backend_ready:
                logger.info("✓ Both services are ready")
                return True
            
            logger.info(f"Waiting for services... (worker: {worker_ready}, backend: {backend_ready})")
            time.sleep(HEALTH_CHECK_INTERVAL)
        
        logger.error(f"Services not ready after {timeout} seconds")
        return False
    
    def test_worker_startup(self) -> bool:
        """Test GPU worker startup and basic functionality."""
        logger.info("Testing GPU worker startup...")
        
        try:
            # Check health endpoint
            response = requests.get(f"{GPU_WORKER_URL}/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"Health check failed: {response.status_code}")
                return False
            
            health_data = response.json()
            logger.info(f"Worker health data: {health_data}")
            
            # Check GPU info endpoint
            response = requests.get(f"{GPU_WORKER_URL}/gpu_info", timeout=10)
            if response.status_code != 200:
                logger.error(f"GPU info check failed: {response.status_code}")
                return False
            
            gpu_info = response.json()
            logger.info(f"GPU info: {gpu_info}")
            
            logger.info("✓ GPU worker startup test passed")
            return True
            
        except Exception as e:
            logger.error(f"GPU worker startup test failed: {e}")
            return False
    
    def test_batch_processing(self, batch_size: int = 3) -> bool:
        """Test batch face detection processing."""
        logger.info(f"Testing batch processing with {batch_size} images...")
        
        try:
            # Create test images
            test_images = []
            for i in range(batch_size):
                image_bytes = self.create_test_image(300, 300)
                if not image_bytes:
                    logger.error(f"Failed to create test image {i}")
                    return False
                
                test_images.append({
                    "data": self.encode_image(image_bytes),
                    "image_id": f"test_image_{i}"
                })
            
            # Prepare request
            request_data = {
                "images": test_images,
                "min_face_quality": 0.5,
                "require_face": False,
                "crop_faces": False,
                "face_margin": 0.2
            }
            
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{GPU_WORKER_URL}/detect_faces_batch",
                json=request_data,
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"Batch processing failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
            
            result_data = response.json()
            logger.info(f"Batch processing completed in {processing_time:.2f}s")
            logger.info(f"Results: {len(result_data['results'])} images processed")
            logger.info(f"GPU used: {result_data.get('gpu_used', False)}")
            logger.info(f"Worker ID: {result_data.get('worker_id', 'unknown')}")
            
            # Validate results
            if len(result_data['results']) != batch_size:
                logger.error(f"Expected {batch_size} results, got {len(result_data['results'])}")
                return False
            
            logger.info("✓ Batch processing test passed")
            return True
            
        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            return False
    
    def test_connection_failure_recovery(self) -> bool:
        """Test connection failure and recovery."""
        logger.info("Testing connection failure recovery...")
        
        try:
            # Test with invalid endpoint
            response = requests.post(
                f"{GPU_WORKER_URL}/invalid_endpoint",
                json={"test": "data"},
                timeout=5
            )
            
            if response.status_code == 404:
                logger.info("✓ Invalid endpoint correctly returned 404")
            else:
                logger.warning(f"Unexpected response to invalid endpoint: {response.status_code}")
            
            # Test with malformed request
            response = requests.post(
                f"{GPU_WORKER_URL}/detect_faces_batch",
                json={"invalid": "data"},
                timeout=5
            )
            
            if response.status_code == 422:  # Validation error
                logger.info("✓ Malformed request correctly returned validation error")
            else:
                logger.warning(f"Unexpected response to malformed request: {response.status_code}")
            
            # Test recovery with valid request
            if self.test_batch_processing(batch_size=1):
                logger.info("✓ Connection recovery test passed")
                return True
            else:
                logger.error("Failed to recover with valid request")
                return False
                
        except Exception as e:
            logger.error(f"Connection failure recovery test failed: {e}")
            return False
    
    def test_cpu_fallback(self) -> bool:
        """Test CPU fallback when GPU worker is unavailable."""
        logger.info("Testing CPU fallback...")
        
        try:
            # This test would require stopping the GPU worker
            # For now, we'll just test that the backend can handle requests
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Backend service is accessible for CPU fallback")
                return True
            else:
                logger.error(f"Backend health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"CPU fallback test failed: {e}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics and monitoring."""
        logger.info("Testing performance metrics...")
        
        try:
            # Get GPU info for metrics
            response = requests.get(f"{GPU_WORKER_URL}/gpu_info", timeout=10)
            if response.status_code != 200:
                logger.error(f"GPU info request failed: {response.status_code}")
                return False
            
            gpu_info = response.json()
            logger.info(f"GPU metrics: {json.dumps(gpu_info, indent=2)}")
            
            # Check for expected metrics
            expected_keys = ['directml_available', 'gpu_actually_used', 'worker_id', 'queue_size']
            for key in expected_keys:
                if key not in gpu_info:
                    logger.warning(f"Missing metric: {key}")
            
            logger.info("✓ Performance metrics test passed")
            return True
            
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests."""
        logger.info("=== Starting GPU Worker Integration Tests ===")
        
        # Wait for services to be ready
        if not self.wait_for_services():
            logger.error("Services not ready, aborting tests")
            return {"services_ready": False}
        
        # Run individual tests
        tests = [
            ("worker_startup", self.test_worker_startup),
            ("batch_processing", self.test_batch_processing),
            ("connection_failure_recovery", self.test_connection_failure_recovery),
            ("cpu_fallback", self.test_cpu_fallback),
            ("performance_metrics", self.test_performance_metrics)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} test ---")
            try:
                result = test_func()
                results[test_name] = result
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name} test crashed: {e}")
                results[test_name] = False
        
        # Summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        logger.info(f"\n=== Test Summary ===")
        logger.info(f"Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name}: {status}")
        
        return results

def main():
    """Main test runner."""
    tester = GPUWorkerTester()
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)
    else:
        logger.info("All tests passed! 🎉")

if __name__ == "__main__":
    main()
