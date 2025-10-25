#!/usr/bin/env python3
"""
Direct GPU Worker Test

Test the GPU worker directly with a proper image to verify it's working.
"""

import asyncio
import base64
import requests
import json

def test_gpu_worker_direct():
    """Test GPU worker directly with a proper image."""
    try:
        print("=== Direct GPU Worker Test ===")
        
        # Create a proper test image (100x100 pixel PNG)
        from PIL import Image
        import io
        
        # Create a 100x100 pixel image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        test_image = img_bytes.getvalue()
        
        print(f"Test image size: {len(test_image)} bytes")
        print(f"Test image header: {test_image[:10]}")
        
        # Encode to base64
        encoded_image = base64.b64encode(test_image).decode('utf-8')
        print(f"Base64 length: {len(encoded_image)}")
        print(f"Base64 preview: {encoded_image[:50]}...")
        
        # Test health first
        print("\nTesting health endpoint...")
        health_response = requests.get("http://localhost:8765/health", timeout=10)
        print(f"Health status: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Health data: {health_data}")
        else:
            print(f"Health error: {health_response.text}")
            return False
        
        # Test face detection
        print("\nTesting face detection...")
        request_data = {
            "images": [{
                "data": encoded_image,
                "image_id": "test_image"
            }],
            "min_face_quality": 0.5,
            "require_face": True,
            "crop_faces": True,
            "face_margin": 0.2
        }
        
        response = requests.post(
            "http://localhost:8765/detect_faces_batch",
            json=request_data,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS! Result: {result}")
            print(f"Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            print(f"GPU used: {result.get('gpu_used', False)}")
            return True
        else:
            print(f"ERROR: {response.text}")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_worker_direct()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)
