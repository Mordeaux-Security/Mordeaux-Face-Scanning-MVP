"""
GPU Performance Testing

Comprehensive benchmarks for GPU vs CPU performance across different operations.
Tests face detection, image processing, and quality checks with various configurations.
"""

import asyncio
import logging
import os
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image
import io
import json
import psutil
import gc

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.gpu_manager import get_gpu_manager, GPUBackend
from app.services.face import get_face_service
from app.core.settings import get_settings

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.operation_times: List[float] = []
        self.throughput: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.gpu_memory_usage: List[float] = []
        self.error_count: int = 0
        self.total_operations: int = 0
        self.successful_operations: int = 0
    
    def add_measurement(self, operation_time: float, throughput: float, 
                       memory_usage: float, cpu_usage: float, gpu_memory: float = 0.0):
        """Add a performance measurement."""
        self.operation_times.append(operation_time)
        self.throughput.append(throughput)
        self.memory_usage.append(memory_usage)
        self.cpu_usage.append(cpu_usage)
        self.gpu_memory_usage.append(gpu_memory)
        self.total_operations += 1
        self.successful_operations += 1
    
    def add_error(self):
        """Record an error."""
        self.error_count += 1
        self.total_operations += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary."""
        if not self.operation_times:
            return {}
        
        return {
            'mean_time_ms': statistics.mean(self.operation_times),
            'median_time_ms': statistics.median(self.operation_times),
            'std_time_ms': statistics.stdev(self.operation_times) if len(self.operation_times) > 1 else 0,
            'min_time_ms': min(self.operation_times),
            'max_time_ms': max(self.operation_times),
            'mean_throughput': statistics.mean(self.throughput),
            'mean_memory_mb': statistics.mean(self.memory_usage),
            'mean_cpu_percent': statistics.mean(self.cpu_usage),
            'mean_gpu_memory_mb': statistics.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'error_rate': self.error_count / self.total_operations if self.total_operations > 0 else 0,
            'success_rate': self.successful_operations / self.total_operations if self.total_operations > 0 else 0
        }


class GPUPerformanceTester:
    """Comprehensive GPU performance testing framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_manager = get_gpu_manager()
        self.face_service = get_face_service()
        self.settings = get_settings()
        
        # Test images of different sizes
        self.test_images = self._create_test_images()
        
        # Performance results
        self.results: Dict[str, Dict[str, PerformanceMetrics]] = {}
    
    def _create_test_images(self) -> Dict[str, bytes]:
        """Create test images of various sizes."""
        images = {}
        
        # Small image (< 500px)
        small_img = Image.new('RGB', (300, 300), color='red')
        small_bytes = io.BytesIO()
        small_img.save(small_bytes, format='JPEG', quality=95)
        images['small'] = small_bytes.getvalue()
        
        # Medium image (500-1000px)
        medium_img = Image.new('RGB', (800, 600), color='green')
        medium_bytes = io.BytesIO()
        medium_img.save(medium_bytes, format='JPEG', quality=95)
        images['medium'] = medium_bytes.getvalue()
        
        # Large image (> 1000px)
        large_img = Image.new('RGB', (1920, 1080), color='blue')
        large_bytes = io.BytesIO()
        large_img.save(large_bytes, format='JPEG', quality=95)
        images['large'] = large_bytes.getvalue()
        
        return images
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent()
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        try:
            memory_info = self.gpu_manager.get_memory_info()
            return memory_info.get('used', 0)
        except Exception:
            return 0.0
    
    async def benchmark_face_detection(self, image_size: str, use_gpu: bool, 
                                     batch_size: int = 1, iterations: int = 10) -> PerformanceMetrics:
        """Benchmark face detection performance."""
        self.logger.info(f"Benchmarking face detection: {image_size}, GPU={use_gpu}, batch={batch_size}")
        
        # Set GPU configuration
        os.environ['FACE_DETECTION_GPU'] = str(use_gpu).lower()
        
        metrics = PerformanceMetrics()
        image_bytes = self.test_images[image_size]
        
        for i in range(iterations):
            try:
                # Force garbage collection before each test
                gc.collect()
                
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()
                start_gpu_memory = self._get_gpu_memory_usage()
                start_time = time.perf_counter()
                
                if batch_size == 1:
                    # Single image detection
                    faces = await self.face_service.detect_and_embed_async(image_bytes)
                else:
                    # Batch detection (simulate by running multiple single detections)
                    faces_list = []
                    for _ in range(batch_size):
                        faces = await self.face_service.detect_and_embed_async(image_bytes)
                        faces_list.append(faces)
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                end_cpu = self._get_cpu_usage()
                end_gpu_memory = self._get_gpu_memory_usage()
                
                operation_time = (end_time - start_time) * 1000  # Convert to ms
                throughput = batch_size / (operation_time / 1000)  # Operations per second
                memory_usage = end_memory - start_memory
                cpu_usage = end_cpu
                gpu_memory_usage = end_gpu_memory - start_gpu_memory
                
                metrics.add_measurement(operation_time, throughput, memory_usage, 
                                      cpu_usage, gpu_memory_usage)
                
                self.logger.debug(f"Iteration {i+1}: {operation_time:.2f}ms, {throughput:.2f} ops/s")
                
            except Exception as e:
                self.logger.error(f"Error in face detection benchmark: {e}")
                metrics.add_error()
        
        return metrics
    
    async def benchmark_image_enhancement(self, image_size: str, use_gpu: bool,
                                        iterations: int = 10) -> PerformanceMetrics:
        """Benchmark image enhancement performance."""
        self.logger.info(f"Benchmarking image enhancement: {image_size}, GPU={use_gpu}")
        
        metrics = PerformanceMetrics()
        image_bytes = self.test_images[image_size]
        
        for i in range(iterations):
            try:
                gc.collect()
                
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()
                start_gpu_memory = self._get_gpu_memory_usage()
                start_time = time.perf_counter()
                
                # Image enhancement
                enhanced_bytes, scale = self.face_service.enhance_image_for_face_detection(image_bytes)
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                end_cpu = self._get_cpu_usage()
                end_gpu_memory = self._get_gpu_memory_usage()
                
                operation_time = (end_time - start_time) * 1000
                throughput = 1 / (operation_time / 1000)
                memory_usage = end_memory - start_memory
                cpu_usage = end_cpu
                gpu_memory_usage = end_gpu_memory - start_gpu_memory
                
                metrics.add_measurement(operation_time, throughput, memory_usage,
                                      cpu_usage, gpu_memory_usage)
                
                self.logger.debug(f"Iteration {i+1}: {operation_time:.2f}ms, {throughput:.2f} ops/s")
                
            except Exception as e:
                self.logger.error(f"Error in image enhancement benchmark: {e}")
                metrics.add_error()
        
        return metrics
    
    async def benchmark_thumbnail_creation(self, image_size: str, use_gpu: bool,
                                         iterations: int = 10) -> PerformanceMetrics:
        """Benchmark thumbnail creation performance."""
        self.logger.info(f"Benchmarking thumbnail creation: {image_size}, GPU={use_gpu}")
        
        metrics = PerformanceMetrics()
        image_bytes = self.test_images[image_size]
        
        for i in range(iterations):
            try:
                gc.collect()
                
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()
                start_gpu_memory = self._get_gpu_memory_usage()
                start_time = time.perf_counter()
                
                # Create thumbnail
                thumbnail_bytes = self.face_service.create_thumbnail(image_bytes)
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                end_cpu = self._get_cpu_usage()
                end_gpu_memory = self._get_gpu_memory_usage()
                
                operation_time = (end_time - start_time) * 1000
                throughput = 1 / (operation_time / 1000)
                memory_usage = end_memory - start_memory
                cpu_usage = end_cpu
                gpu_memory_usage = end_gpu_memory - start_gpu_memory
                
                metrics.add_measurement(operation_time, throughput, memory_usage,
                                      cpu_usage, gpu_memory_usage)
                
                self.logger.debug(f"Iteration {i+1}: {operation_time:.2f}ms, {throughput:.2f} ops/s")
                
            except Exception as e:
                self.logger.error(f"Error in thumbnail creation benchmark: {e}")
                metrics.add_error()
        
        return metrics
    
    async def benchmark_quality_checks(self, image_size: str, use_gpu: bool,
                                     iterations: int = 10) -> PerformanceMetrics:
        """Benchmark quality checks performance."""
        self.logger.info(f"Benchmarking quality checks: {image_size}, GPU={use_gpu}")
        
        metrics = PerformanceMetrics()
        image_bytes = self.test_images[image_size]
        
        for i in range(iterations):
            try:
                gc.collect()
                
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()
                start_gpu_memory = self._get_gpu_memory_usage()
                start_time = time.perf_counter()
                
                # Simulate quality checks (blur, brightness, contrast)
                # This would be implemented in the quality module
                import cv2
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # Blur detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Brightness
                brightness = np.mean(gray)
                
                # Contrast
                contrast = np.std(gray)
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                end_cpu = self._get_cpu_usage()
                end_gpu_memory = self._get_gpu_memory_usage()
                
                operation_time = (end_time - start_time) * 1000
                throughput = 1 / (operation_time / 1000)
                memory_usage = end_memory - start_memory
                cpu_usage = end_cpu
                gpu_memory_usage = end_gpu_memory - start_gpu_memory
                
                metrics.add_measurement(operation_time, throughput, memory_usage,
                                      cpu_usage, gpu_memory_usage)
                
                self.logger.debug(f"Iteration {i+1}: {operation_time:.2f}ms, {throughput:.2f} ops/s")
                
            except Exception as e:
                self.logger.error(f"Error in quality checks benchmark: {e}")
                metrics.add_error()
        
        return metrics
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all operations and configurations."""
        self.logger.info("Starting comprehensive GPU performance benchmark...")
        
        # Test configurations
        image_sizes = ['small', 'medium', 'large']
        gpu_configs = [False, True]  # CPU, GPU
        batch_sizes = [1, 10, 50]  # Single, small batch, large batch
        
        results = {}
        
        for operation in ['face_detection', 'image_enhancement', 'thumbnail_creation', 'quality_checks']:
            results[operation] = {}
            
            for image_size in image_sizes:
                results[operation][image_size] = {}
                
                for use_gpu in gpu_configs:
                    config_key = f"gpu_{use_gpu}"
                    results[operation][image_size][config_key] = {}
                    
                    if operation == 'face_detection':
                        for batch_size in batch_sizes:
                            self.logger.info(f"Testing {operation} - {image_size} - GPU={use_gpu} - batch={batch_size}")
                            metrics = await self.benchmark_face_detection(
                                image_size, use_gpu, batch_size, iterations=5
                            )
                            results[operation][image_size][config_key][f"batch_{batch_size}"] = metrics.get_stats()
                    else:
                        self.logger.info(f"Testing {operation} - {image_size} - GPU={use_gpu}")
                        if operation == 'image_enhancement':
                            metrics = await self.benchmark_image_enhancement(image_size, use_gpu, iterations=5)
                        elif operation == 'thumbnail_creation':
                            metrics = await self.benchmark_thumbnail_creation(image_size, use_gpu, iterations=5)
                        elif operation == 'quality_checks':
                            metrics = await self.benchmark_quality_checks(image_size, use_gpu, iterations=5)
                        
                        results[operation][image_size][config_key] = metrics.get_stats()
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results and provide recommendations."""
        analysis = {
            'recommendations': {},
            'performance_gains': {},
            'optimal_configurations': {}
        }
        
        for operation, operation_results in results.items():
            analysis['recommendations'][operation] = {}
            analysis['performance_gains'][operation] = {}
            analysis['optimal_configurations'][operation] = {}
            
            for image_size, size_results in operation_results.items():
                if 'gpu_true' in size_results and 'gpu_false' in size_results:
                    cpu_stats = size_results['gpu_false']
                    gpu_stats = size_results['gpu_true']
                    
                    # Calculate performance gain
                    if cpu_stats and gpu_stats:
                        time_improvement = (cpu_stats['mean_time_ms'] - gpu_stats['mean_time_ms']) / cpu_stats['mean_time_ms']
                        throughput_improvement = (gpu_stats['mean_throughput'] - cpu_stats['mean_throughput']) / cpu_stats['mean_throughput']
                        
                        analysis['performance_gains'][operation][image_size] = {
                            'time_improvement_percent': time_improvement * 100,
                            'throughput_improvement_percent': throughput_improvement * 100
                        }
                        
                        # Recommendation based on performance gain
                        if time_improvement > 0.1:  # 10% improvement
                            analysis['recommendations'][operation][image_size] = 'enable_gpu'
                            analysis['optimal_configurations'][operation][image_size] = True
                        else:
                            analysis['recommendations'][operation][image_size] = 'use_cpu'
                            analysis['optimal_configurations'][operation][image_size] = False
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], filename: str = "gpu_benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Benchmark results saved to {output_path}")


async def main():
    """Main benchmark execution."""
    logging.basicConfig(level=logging.INFO)
    
    tester = GPUPerformanceTester()
    
    # Run comprehensive benchmark
    results = await tester.run_comprehensive_benchmark()
    
    # Analyze results
    analysis = tester.analyze_results(results)
    
    # Save results
    tester.save_results({
        'benchmark_results': results,
        'analysis': analysis,
        'timestamp': time.time(),
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    })
    
    # Print summary
    print("\n" + "="*80)
    print("GPU PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    
    for operation, operation_analysis in analysis['performance_gains'].items():
        print(f"\n{operation.upper()}:")
        for image_size, gains in operation_analysis.items():
            print(f"  {image_size}: {gains['time_improvement_percent']:.1f}% time improvement, "
                  f"{gains['throughput_improvement_percent']:.1f}% throughput improvement")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
