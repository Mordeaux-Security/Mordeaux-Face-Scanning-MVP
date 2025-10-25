#!/usr/bin/env python3
"""
Simple test script for GPU worker functionality.
Tests single image processing and batch processing.
"""

import base64
import json
import requests
import time
from pathlib import Path

# GPU Worker URL
GPU_WORKER_URL = "http://localhost:8765"

def create_test_image():
    """Create a simple test image."""
    from PIL import Image
    import numpy as np
    
    # Create a simple test image with a face-like pattern
    img = Image.new('RGB', (200, 200), color='white')
    
    # Add some basic shapes that could represent a face
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    draw.ellipse([50, 50, 150, 150], fill='lightblue', outline='black', width=2)
    draw.ellipse([70, 80, 85, 95], fill='black')  # Left eye
    draw.ellipse([115, 80, 130, 95], fill='black')  # Right eye
    draw.arc([80, 100, 120, 120], 0, 180, fill='black', width=2)  # Smile
    
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def test_health():
    """Test GPU worker health endpoint."""
    print("Testing GPU worker health...")
    try:
        response = requests.get(f"{GPU_WORKER_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"[OK] Health check passed: {health_data}")
            return True
        else:
            print(f"[FAIL] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Health check error: {e}")
        return False

def test_single_image():
    """Test single image processing."""
    print("\nTesting single image processing...")
    
    try:
        # Create test image
        test_image = create_test_image()
        image_data = image_to_base64(test_image)
        
        # Prepare request
        request_data = {
            "images": [
                {
                    "data": image_data,
                    "image_id": "test_image_1"
                }
            ],
            "min_face_quality": 0.3,
            "require_face": False,
            "crop_faces": True,
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
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Single image processing successful")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  GPU used: {result.get('gpu_used', False)}")
            print(f"  Faces detected: {len(result['results'][0])}")
            
            if result['results'][0]:
                face = result['results'][0][0]
                print(f"  Face quality: {face['quality']:.3f}")
                print(f"  Face bbox: {face['bbox']}")
            
            return True
        else:
            print(f"[FAIL] Single image processing failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Single image processing error: {e}")
        return False

def test_batch_processing():
    """Test batch image processing."""
    print("\nTesting batch image processing...")
    
    try:
        # Create multiple test images
        images = []
        for i in range(3):
            test_image = create_test_image()
            image_data = image_to_base64(test_image)
            images.append({
                "data": image_data,
                "image_id": f"test_image_{i+1}"
            })
        
        # Prepare request
        request_data = {
            "images": images,
            "min_face_quality": 0.3,
            "require_face": False,
            "crop_faces": True,
            "face_margin": 0.2
        }
        
        # Send request
        start_time = time.time()
        response = requests.post(
            f"{GPU_WORKER_URL}/detect_faces_batch",
            json=request_data,
            timeout=60
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Batch processing successful")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  GPU used: {result.get('gpu_used', False)}")
            print(f"  Images processed: {len(result['results'])}")
            
            total_faces = sum(len(faces) for faces in result['results'])
            print(f"  Total faces detected: {total_faces}")
            
            return True
        else:
            print(f"[FAIL] Batch processing failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Batch processing error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== GPU Worker Test Suite ===")
    
    # Test health
    if not test_health():
        print("Health check failed, stopping tests")
        return
    
    # Test single image
    test_single_image()
    
    # Test batch processing
    test_batch_processing()
    
    print("\n=== Test Suite Complete ===")

if __name__ == "__main__":
    main()
