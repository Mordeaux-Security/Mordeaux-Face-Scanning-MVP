"""
Benchmark script to compare single-image vs batched face detection performance.

Usage:
    python bench_detector.py --image-dir /path/to/images --num-images 1000
    python bench_detector.py --image-list /path/to/image_list.txt --num-images 1000
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu_worker.worker import _detect_faces_batch, _load_face_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_images_from_dir(image_dir: str, num_images: int) -> List[np.ndarray]:
    """Load images from directory."""
    images = []
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ][:num_images]
    
    if len(image_files) < num_images:
        logger.warning(f"Only found {len(image_files)} images, requested {num_images}")
    
    for img_file in image_files:
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {img_file}: {e}")
    
    return images


def load_images_from_list(image_list: str, num_images: int) -> List[np.ndarray]:
    """Load images from list file (one path per line)."""
    images = []
    
    with open(image_list, 'r') as f:
        lines = f.readlines()[:num_images]
    
    for line in lines:
        img_path = line.strip()
        if not img_path:
            continue
        
        try:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    return images


def benchmark_single_image(images: List[np.ndarray]) -> dict:
    """Benchmark single-image processing (original logic)."""
    logger.info(f"Benchmarking single-image mode with {len(images)} images")
    
    # Temporarily disable batched detector
    import gpu_worker.worker as worker_module
    original_enabled = worker_module._batched_detector_enabled
    worker_module._batched_detector_enabled = False
    
    try:
        face_app = _load_face_model()
        
        total_time = 0.0
        det_forward_times = []
        total_faces = 0
        
        for i, image in enumerate(images):
            start_time = time.time()
            
            # Single image detection
            faces = face_app.get(image)
            
            det_time = (time.time() - start_time) * 1000  # ms
            total_time += det_time
            det_forward_times.append(det_time)
            total_faces += len(faces)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} images")
        
        avg_time_per_image = total_time / len(images)
        imgs_per_sec = 1000.0 / avg_time_per_image if avg_time_per_image > 0 else 0
        
        det_forward_times_sorted = sorted(det_forward_times)
        p50 = det_forward_times_sorted[len(det_forward_times_sorted) // 2]
        p95 = det_forward_times_sorted[int(len(det_forward_times_sorted) * 0.95)]
        
        return {
            'mode': 'single-image',
            'num_images': len(images),
            'total_time_ms': total_time,
            'avg_time_per_image_ms': avg_time_per_image,
            'imgs_per_sec': imgs_per_sec,
            'p50_det_forward_ms': p50,
            'p95_det_forward_ms': p95,
            'total_faces': total_faces,
            'avg_faces_per_image': total_faces / len(images) if images else 0
        }
    finally:
        # Restore original setting
        worker_module._batched_detector_enabled = original_enabled


def benchmark_batched(images: List[np.ndarray]) -> dict:
    """Benchmark batched processing."""
    logger.info(f"Benchmarking batched mode with {len(images)} images")
    
    # Enable batched detector
    import gpu_worker.worker as worker_module
    original_enabled = worker_module._batched_detector_enabled
    worker_module._batched_detector_enabled = True
    
    # Load batched detector if not already loaded
    if worker_module._batched_detector is None:
        worker_module._load_batched_detector()
    
    try:
        # Process in batches
        batch_size = 16
        total_time = 0.0
        det_forward_times = []
        total_faces = 0
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            start_time = time.time()
            
            # Batched detection
            results = _detect_faces_batch(batch, min_quality=0.5)
            
            batch_time = (time.time() - start_time) * 1000  # ms
            total_time += batch_time
            
            # Calculate per-image time
            time_per_image = batch_time / len(batch)
            det_forward_times.extend([time_per_image] * len(batch))
            
            for face_detections in results:
                total_faces += len(face_detections)
            
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, len(images))}/{len(images)} images")
        
        avg_time_per_image = total_time / len(images)
        imgs_per_sec = 1000.0 / avg_time_per_image if avg_time_per_image > 0 else 0
        
        det_forward_times_sorted = sorted(det_forward_times)
        p50 = det_forward_times_sorted[len(det_forward_times_sorted) // 2]
        p95 = det_forward_times_sorted[int(len(det_forward_times_sorted) * 0.95)]
        
        return {
            'mode': 'batched',
            'num_images': len(images),
            'total_time_ms': total_time,
            'avg_time_per_image_ms': avg_time_per_image,
            'imgs_per_sec': imgs_per_sec,
            'p50_det_forward_ms': p50,
            'p95_det_forward_ms': p95,
            'total_faces': total_faces,
            'avg_faces_per_image': total_faces / len(images) if images else 0
        }
    finally:
        # Restore original setting
        worker_module._batched_detector_enabled = original_enabled


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-image vs batched face detection")
    parser.add_argument('--image-dir', type=str, help='Directory containing images')
    parser.add_argument('--image-list', type=str, help='Text file with image paths (one per line)')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images to process')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup images')
    
    args = parser.parse_args()
    
    if not args.image_dir and not args.image_list:
        parser.error("Either --image-dir or --image-list must be provided")
    
    # Load images
    logger.info(f"Loading {args.num_images} images...")
    if args.image_dir:
        images = load_images_from_dir(args.image_dir, args.num_images)
    else:
        images = load_images_from_list(args.image_list, args.num_images)
    
    if len(images) < args.warmup:
        logger.error(f"Not enough images: {len(images)} < {args.warmup}")
        return
    
    logger.info(f"Loaded {len(images)} images")
    
    # Warmup
    logger.info(f"Warming up with {args.warmup} images...")
    warmup_images = images[:args.warmup]
    _detect_faces_batch(warmup_images, min_quality=0.5)
    
    # Benchmark single-image
    logger.info("\n" + "="*60)
    single_results = benchmark_single_image(images)
    
    # Benchmark batched
    logger.info("\n" + "="*60)
    batched_results = benchmark_batched(images)
    
    # Print comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    
    print(f"\n{'Metric':<30} {'Single-Image':<20} {'Batched':<20} {'Improvement':<15}")
    print("-" * 85)
    
    metrics = [
        ('Avg time per image (ms)', 'avg_time_per_image_ms'),
        ('Images per second', 'imgs_per_sec'),
        ('P50 det_forward (ms)', 'p50_det_forward_ms'),
        ('P95 det_forward (ms)', 'p95_det_forward_ms'),
        ('Total faces detected', 'total_faces'),
        ('Avg faces per image', 'avg_faces_per_image'),
    ]
    
    for metric_name, metric_key in metrics:
        single_val = single_results[metric_key]
        batched_val = batched_results[metric_key]
        
        if 'time' in metric_key or 'per_sec' in metric_key:
            if single_val > 0:
                improvement = ((single_val - batched_val) / single_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            # For face counts, show difference
            diff = batched_val - single_val
            improvement_str = f"{diff:+.1f}"
        
        print(f"{metric_name:<30} {single_val:<20.2f} {batched_val:<20.2f} {improvement_str:<15}")
    
    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()

