"""
GPU Performance Test

Comprehensive performance testing for GPU worker vs CPU fallback.
Measures throughput, latency, GPU utilization, and resource usage.
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
from typing import List, Dict, Any, Tuple
import requests
import httpx
from PIL import Image, ImageDraw
import numpy as np
import psutil
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
GPU_WORKER_URL = "http://localhost:8765"
BACKEND_URL = "http://localhost:8000"
TEST_IMAGES_COUNT = 50
BATCH_SIZES = [1, 4, 8, 16, 32]
CONCURRENT_REQUESTS = [1, 2, 4, 8]

class PerformanceTester:
    """Performance testing suite for GPU worker."""
    
    def __init__(self):
        self.test_images = []
        self.results = {}
        
    def create_test_images(self, count: int, sizes: List[Tuple[int, int]] = None) -> List[bytes]:
        """Create test images of various sizes."""
        if sizes is None:
            sizes = [(200, 200), (400, 300), (600, 400), (800, 600)]
        
        images = []
        for i in range(count):
            # Cycle through different sizes
            width, height = sizes[i % len(sizes)]
            image_bytes = self.create_test_image(width, height)
            images.append(image_bytes)
        
        return images
    
    def create_test_image(self, width: int, height: int) -> bytes:
        """Create a test image with realistic content."""
        try:
            # Create a more realistic test image
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Add some geometric shapes to simulate faces
            center_x, center_y = width // 2, height // 2
            
            # Face outline (circle)
            face_radius = min(width, height) // 4
            draw.ellipse([
                center_x - face_radius, center_y - face_radius,
                center_x + face_radius, center_y + face_radius
            ], outline='black', width=2)
            
            # Eyes
            eye_y = center_y - face_radius // 3
            left_eye_x = center_x - face_radius // 3
            right_eye_x = center_x + face_radius // 3
            draw.ellipse([left_eye_x-5, eye_y-5, left_eye_x+5, eye_y+5], fill='black')
            draw.ellipse([right_eye_x-5, eye_y-5, right_eye_x+5, eye_y+5], fill='black')
            
            # Nose
            nose_y = center_y
            draw.line([center_x, nose_y-5, center_x, nose_y+5], fill='black', width=2)
            
            # Mouth
            mouth_y = center_y + face_radius // 3
            mouth_width = face_radius // 2
            draw.arc([
                center_x - mouth_width, mouth_y - 5,
                center_x + mouth_width, mouth_y + 5
            ], 0, 180, fill='black', width=2)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG', quality=85)
            return img_bytes.getvalue()
        except Exception as e:
            logger.error(f"Failed to create test image: {e}")
            return b""
    
    def encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def test_gpu_worker_throughput(self, batch_size: int, num_batches: int) -> Dict[str, Any]:
        """Test GPU worker throughput with different batch sizes."""
        logger.info(f"Testing GPU worker throughput: batch_size={batch_size}, num_batches={num_batches}")
        
        results = {
            'batch_size': batch_size,
            'num_batches': num_batches,
            'total_images': batch_size * num_batches,
            'latencies': [],
            'success_count': 0,
            'failure_count': 0,
            'gpu_used_count': 0,
            'total_processing_time': 0
        }
        
        # Create test images
        test_images = self.create_test_images(batch_size * num_batches)
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch_images = test_images[batch_start:batch_end]
            
            # Prepare request
            request_data = {
                "images": [
                    {"data": self.encode_image(img), "image_id": f"batch_{batch_idx}_img_{i}"}
                    for i, img in enumerate(batch_images)
                ],
                "min_face_quality": 0.5,
                "require_face": False,
                "crop_faces": False,
                "face_margin": 0.2
            }
            
            # Send request
            batch_start_time = time.time()
            try:
                response = requests.post(
                    f"{GPU_WORKER_URL}/detect_faces_batch",
                    json=request_data,
                    timeout=60
                )
                batch_latency = time.time() - batch_start_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    results['success_count'] += 1
                    results['latencies'].append(batch_latency)
                    
                    if result_data.get('gpu_used', False):
                        results['gpu_used_count'] += 1
                    
                    logger.debug(f"Batch {batch_idx}: {batch_latency:.2f}s, GPU: {result_data.get('gpu_used', False)}")
                else:
                    results['failure_count'] += 1
                    logger.warning(f"Batch {batch_idx} failed: {response.status_code}")
                    
            except Exception as e:
                results['failure_count'] += 1
                logger.error(f"Batch {batch_idx} error: {e}")
        
        results['total_processing_time'] = time.time() - start_time
        
        # Calculate metrics
        if results['latencies']:
            results['avg_latency'] = statistics.mean(results['latencies'])
            results['p50_latency'] = statistics.median(results['latencies'])
            results['p95_latency'] = statistics.quantiles(results['latencies'], n=20)[18] if len(results['latencies']) >= 20 else results['p50_latency']
            results['min_latency'] = min(results['latencies'])
            results['max_latency'] = max(results['latencies'])
        else:
            results['avg_latency'] = 0
            results['p50_latency'] = 0
            results['p95_latency'] = 0
            results['min_latency'] = 0
            results['max_latency'] = 0
        
        results['success_rate'] = results['success_count'] / num_batches
        results['gpu_usage_rate'] = results['gpu_used_count'] / results['success_count'] if results['success_count'] > 0 else 0
        results['throughput'] = results['total_images'] / results['total_processing_time'] if results['total_processing_time'] > 0 else 0
        
        logger.info(f"Throughput test completed: {results['throughput']:.2f} images/sec, "
                   f"success_rate: {results['success_rate']:.2f}, gpu_usage: {results['gpu_usage_rate']:.2f}")
        
        return results
    
    def test_concurrent_requests(self, concurrent_count: int, requests_per_client: int) -> Dict[str, Any]:
        """Test concurrent request handling."""
        logger.info(f"Testing concurrent requests: {concurrent_count} clients, {requests_per_client} requests each")
        
        results = {
            'concurrent_count': concurrent_count,
            'requests_per_client': requests_per_client,
            'total_requests': concurrent_count * requests_per_client,
            'success_count': 0,
            'failure_count': 0,
            'latencies': [],
            'start_time': 0,
            'end_time': 0
        }
        
        def make_request(client_id: int, request_id: int) -> Dict[str, Any]:
            """Make a single request."""
            try:
                # Create a small test image
                image_bytes = self.create_test_image(200, 200)
                
                request_data = {
                    "images": [{
                        "data": self.encode_image(image_bytes),
                        "image_id": f"client_{client_id}_req_{request_id}"
                    }],
                    "min_face_quality": 0.5,
                    "require_face": False,
                    "crop_faces": False,
                    "face_margin": 0.2
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{GPU_WORKER_URL}/detect_faces_batch",
                    json=request_data,
                    timeout=30
                )
                latency = time.time() - start_time
                
                return {
                    'success': response.status_code == 200,
                    'latency': latency,
                    'gpu_used': response.json().get('gpu_used', False) if response.status_code == 200 else False
                }
                
            except Exception as e:
                logger.error(f"Request failed (client {client_id}, req {request_id}): {e}")
                return {'success': False, 'latency': 0, 'gpu_used': False}
        
        # Create threads for concurrent requests
        threads = []
        results_lock = threading.Lock()
        
        def client_worker(client_id: int):
            """Worker function for each client."""
            client_results = []
            for req_id in range(requests_per_client):
                result = make_request(client_id, req_id)
                client_results.append(result)
            
            # Update shared results
            with results_lock:
                for result in client_results:
                    if result['success']:
                        results['success_count'] += 1
                        results['latencies'].append(result['latency'])
                    else:
                        results['failure_count'] += 1
        
        # Start all threads
        start_time = time.time()
        for client_id in range(concurrent_count):
            thread = threading.Thread(target=client_worker, args=(client_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        results['start_time'] = start_time
        results['end_time'] = end_time
        results['total_time'] = end_time - start_time
        
        # Calculate metrics
        if results['latencies']:
            results['avg_latency'] = statistics.mean(results['latencies'])
            results['p50_latency'] = statistics.median(results['latencies'])
            results['p95_latency'] = statistics.quantiles(results['latencies'], n=20)[18] if len(results['latencies']) >= 20 else results['p50_latency']
        else:
            results['avg_latency'] = 0
            results['p50_latency'] = 0
            results['p95_latency'] = 0
        
        results['success_rate'] = results['success_count'] / results['total_requests']
        results['throughput'] = results['total_requests'] / results['total_time']
        
        logger.info(f"Concurrent test completed: {results['throughput']:.2f} requests/sec, "
                   f"success_rate: {results['success_rate']:.2f}")
        
        return results
    
    def test_resource_usage(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Monitor resource usage during processing."""
        logger.info(f"Monitoring resource usage for {duration_seconds} seconds...")
        
        # Get initial resource usage
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().percent
        
        # Start monitoring thread
        monitoring_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'start_time': time.time()
        }
        
        def monitor_resources():
            """Monitor resource usage in background."""
            while time.time() - monitoring_data['start_time'] < duration_seconds:
                monitoring_data['cpu_usage'].append(psutil.cpu_percent())
                monitoring_data['memory_usage'].append(psutil.virtual_memory().percent)
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Run processing during monitoring
        test_images = self.create_test_images(10)
        processing_results = []
        
        for i in range(duration_seconds // 3):  # Process every 3 seconds
            batch_start = time.time()
            
            request_data = {
                "images": [
                    {"data": self.encode_image(img), "image_id": f"monitor_{i}_{j}"}
                    for j, img in enumerate(test_images[:5])  # Process 5 images at a time
                ],
                "min_face_quality": 0.5,
                "require_face": False,
                "crop_faces": False,
                "face_margin": 0.2
            }
            
            try:
                response = requests.post(
                    f"{GPU_WORKER_URL}/detect_faces_batch",
                    json=request_data,
                    timeout=10
                )
                processing_time = time.time() - batch_start
                processing_results.append({
                    'success': response.status_code == 200,
                    'processing_time': processing_time,
                    'gpu_used': response.json().get('gpu_used', False) if response.status_code == 200 else False
                })
            except Exception as e:
                logger.warning(f"Processing batch {i} failed: {e}")
                processing_results.append({'success': False, 'processing_time': 0, 'gpu_used': False})
            
            time.sleep(2)  # Wait 2 seconds between batches
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Calculate resource usage statistics
        results = {
            'duration_seconds': duration_seconds,
            'initial_cpu': initial_cpu,
            'initial_memory': initial_memory,
            'avg_cpu': statistics.mean(monitoring_data['cpu_usage']) if monitoring_data['cpu_usage'] else 0,
            'max_cpu': max(monitoring_data['cpu_usage']) if monitoring_data['cpu_usage'] else 0,
            'avg_memory': statistics.mean(monitoring_data['memory_usage']) if monitoring_data['memory_usage'] else 0,
            'max_memory': max(monitoring_data['memory_usage']) if monitoring_data['memory_usage'] else 0,
            'processing_success_rate': sum(1 for r in processing_results if r['success']) / len(processing_results) if processing_results else 0,
            'gpu_usage_rate': sum(1 for r in processing_results if r.get('gpu_used', False)) / len(processing_results) if processing_results else 0
        }
        
        logger.info(f"Resource monitoring completed: avg_cpu={results['avg_cpu']:.1f}%, "
                   f"avg_memory={results['avg_memory']:.1f}%, gpu_usage={results['gpu_usage_rate']:.2f}")
        
        return results
    
    def run_performance_suite(self) -> Dict[str, Any]:
        """Run complete performance test suite."""
        logger.info("=== Starting GPU Performance Test Suite ===")
        
        suite_results = {
            'timestamp': time.time(),
            'throughput_tests': {},
            'concurrent_tests': {},
            'resource_tests': {}
        }
        
        # Test 1: Throughput with different batch sizes
        logger.info("\n--- Throughput Tests ---")
        for batch_size in BATCH_SIZES:
            num_batches = max(1, TEST_IMAGES_COUNT // batch_size)
            result = self.test_gpu_worker_throughput(batch_size, num_batches)
            suite_results['throughput_tests'][f'batch_{batch_size}'] = result
        
        # Test 2: Concurrent request handling
        logger.info("\n--- Concurrent Request Tests ---")
        for concurrent_count in CONCURRENT_REQUESTS:
            requests_per_client = max(1, 10 // concurrent_count)
            result = self.test_concurrent_requests(concurrent_count, requests_per_client)
            suite_results['concurrent_tests'][f'concurrent_{concurrent_count}'] = result
        
        # Test 3: Resource usage monitoring
        logger.info("\n--- Resource Usage Tests ---")
        resource_result = self.test_resource_usage(duration_seconds=30)
        suite_results['resource_tests']['monitoring'] = resource_result
        
        # Generate summary
        self.generate_performance_summary(suite_results)
        
        return suite_results
    
    def generate_performance_summary(self, results: Dict[str, Any]):
        """Generate performance summary report."""
        logger.info("\n=== Performance Test Summary ===")
        
        # Throughput summary
        throughput_results = results['throughput_tests']
        if throughput_results:
            best_throughput = max(
                (test['throughput'] for test in throughput_results.values()),
                default=0
            )
            best_batch_size = max(
                throughput_results.keys(),
                key=lambda k: throughput_results[k]['throughput']
            )
            logger.info(f"Best throughput: {best_throughput:.2f} images/sec (batch_size: {best_batch_size})")
        
        # Concurrent summary
        concurrent_results = results['concurrent_tests']
        if concurrent_results:
            best_concurrent = max(
                (test['throughput'] for test in concurrent_results.values()),
                default=0
            )
            logger.info(f"Best concurrent throughput: {best_concurrent:.2f} requests/sec")
        
        # Resource summary
        resource_results = results['resource_tests']['monitoring']
        if resource_results:
            logger.info(f"Resource usage: CPU avg={resource_results['avg_cpu']:.1f}%, "
                       f"Memory avg={resource_results['avg_memory']:.1f}%")
            logger.info(f"GPU usage rate: {resource_results['gpu_usage_rate']:.2f}")
        
        # Save detailed results
        results_file = "gpu_performance_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {results_file}")

def main():
    """Main performance test runner."""
    tester = PerformanceTester()
    results = tester.run_performance_suite()
    
    logger.info("Performance testing completed! 🚀")

if __name__ == "__main__":
    main()
