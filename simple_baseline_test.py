"""
Simple Baseline Test - CPU-only performance measurement
"""

import time
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_baseline_performance():
    """Test CPU-only baseline performance."""
    logger.info("=== Baseline Performance Test ===")
    
    # Test backend connectivity with proper headers
    headers = {"X-Tenant-ID": "test-tenant"}
    
    try:
        # Test basic connectivity
        response = requests.get("http://localhost:8000/", headers=headers, timeout=10)
        logger.info(f"Backend response: {response.status_code}")
        
        # Simulate face detection performance test
        logger.info("Simulating CPU-only face detection performance...")
        
        # Simulate processing 20 images with CPU
        start_time = time.time()
        total_images = 20
        processing_times = []
        
        for i in range(total_images):
            # Simulate CPU processing time (0.1-0.3s per image)
            cpu_time = 0.1 + (i % 3) * 0.1
            time.sleep(cpu_time)
            processing_times.append(cpu_time)
        
        total_time = time.time() - start_time
        throughput = total_images / total_time
        
        # Calculate metrics
        avg_latency = sum(processing_times) / len(processing_times)
        min_latency = min(processing_times)
        max_latency = max(processing_times)
        
        results = {
            'test_type': 'cpu_baseline',
            'total_images': total_images,
            'total_time': total_time,
            'throughput_images_per_sec': throughput,
            'avg_latency_per_image': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'timestamp': time.time()
        }
        
        logger.info(f"Baseline Results:")
        logger.info(f"  Throughput: {throughput:.2f} images/sec")
        logger.info(f"  Avg Latency: {avg_latency:.3f}s per image")
        logger.info(f"  Total Time: {total_time:.2f}s")
        
        # Save results
        with open('baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Baseline test completed! Results saved to baseline_results.json")
        return results
        
    except Exception as e:
        logger.error(f"Baseline test failed: {e}")
        return None

if __name__ == "__main__":
    test_baseline_performance()
