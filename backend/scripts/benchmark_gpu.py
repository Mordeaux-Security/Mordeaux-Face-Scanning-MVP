#!/usr/bin/env python3
"""
Standalone GPU Benchmark Script

Run comprehensive GPU performance benchmarks to determine optimal configurations.
This script can be run independently to test GPU acceleration performance.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_gpu_performance import GPUPerformanceTester


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def run_benchmark(operation: str = None, image_size: str = None, 
                       use_gpu: bool = None, batch_size: int = None,
                       iterations: int = 10, output_file: str = None):
    """Run GPU performance benchmark."""
    logger = logging.getLogger(__name__)
    
    tester = GPUPerformanceTester()
    
    if operation and image_size is not None and use_gpu is not None:
        # Run specific benchmark
        logger.info(f"Running {operation} benchmark: {image_size}, GPU={use_gpu}")
        
        if operation == 'face_detection':
            metrics = await tester.benchmark_face_detection(
                image_size, use_gpu, batch_size or 1, iterations
            )
        elif operation == 'image_enhancement':
            metrics = await tester.benchmark_image_enhancement(
                image_size, use_gpu, iterations
            )
        elif operation == 'thumbnail_creation':
            metrics = await tester.benchmark_thumbnail_creation(
                image_size, use_gpu, iterations
            )
        elif operation == 'quality_checks':
            metrics = await tester.benchmark_quality_checks(
                image_size, use_gpu, iterations
            )
        else:
            logger.error(f"Unknown operation: {operation}")
            return
        
        # Print results
        stats = metrics.get_stats()
        print(f"\n{operation.upper()} BENCHMARK RESULTS")
        print("="*50)
        print(f"Image size: {image_size}")
        print(f"GPU enabled: {use_gpu}")
        print(f"Batch size: {batch_size or 1}")
        print(f"Iterations: {iterations}")
        print(f"Mean time: {stats.get('mean_time_ms', 0):.2f} ms")
        print(f"Mean throughput: {stats.get('mean_throughput', 0):.2f} ops/s")
        print(f"Success rate: {stats.get('success_rate', 0)*100:.1f}%")
        print(f"Error rate: {stats.get('error_rate', 0)*100:.1f}%")
        
        if output_file:
            tester.save_results({operation: stats}, output_file)
    
    else:
        # Run comprehensive benchmark
        logger.info("Running comprehensive GPU benchmark...")
        results = await tester.run_comprehensive_benchmark()
        analysis = tester.analyze_results(results)
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE GPU BENCHMARK RESULTS")
        print("="*80)
        
        for op, op_results in analysis['performance_gains'].items():
            print(f"\n{op.upper()}:")
            for size, gains in op_results.items():
                print(f"  {size}: {gains['time_improvement_percent']:.1f}% time improvement, "
                      f"{gains['throughput_improvement_percent']:.1f}% throughput improvement")
        
        # Save results
        output_file = output_file or "gpu_benchmark_results.json"
        tester.save_results({
            'benchmark_results': results,
            'analysis': analysis
        }, output_file)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description='GPU Performance Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive benchmark
  python scripts/benchmark_gpu.py

  # Run specific face detection benchmark
  python scripts/benchmark_gpu.py --operation face_detection --image-size medium --gpu --batch-size 10

  # Run image enhancement benchmark with CPU
  python scripts/benchmark_gpu.py --operation image_enhancement --image-size large --no-gpu

  # Run with custom iterations and output
  python scripts/benchmark_gpu.py --operation quality_checks --image-size small --gpu --iterations 20 --output results.json
        """
    )
    
    # Operation selection
    parser.add_argument('--operation', 
                       choices=['face_detection', 'image_enhancement', 'thumbnail_creation', 'quality_checks'],
                       help='Specific operation to benchmark (default: run all)')
    
    # Image configuration
    parser.add_argument('--image-size', 
                       choices=['small', 'medium', 'large'],
                       help='Image size to test (required for specific operations)')
    
    # GPU configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu', action='store_true', 
                          help='Enable GPU acceleration')
    gpu_group.add_argument('--no-gpu', action='store_true', 
                          help='Disable GPU acceleration (use CPU)')
    
    # Batch configuration
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for face detection (default: 1)')
    
    # Test configuration
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per test (default: 10)')
    
    # Output configuration
    parser.add_argument('--output', '-o',
                       help='Output file for results (default: gpu_benchmark_results.json)')
    
    # Logging configuration
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if args.operation and not args.image_size:
        print("Error: --image-size is required when specifying --operation")
        sys.exit(1)
    
    if args.operation and args.gpu is None and args.no_gpu is None:
        print("Error: --gpu or --no-gpu is required when specifying --operation")
        sys.exit(1)
    
    # Determine GPU setting
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    # Run benchmark
    try:
        asyncio.run(run_benchmark(
            operation=args.operation,
            image_size=args.image_size,
            use_gpu=use_gpu,
            batch_size=args.batch_size,
            iterations=args.iterations,
            output_file=args.output
        ))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
