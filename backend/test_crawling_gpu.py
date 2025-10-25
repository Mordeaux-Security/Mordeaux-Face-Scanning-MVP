#!/usr/bin/env python3
"""
Test GPU acceleration during crawling simulation.
"""

import asyncio
import sys
import os
from PIL import Image
import io

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_crawling_gpu():
    """Test GPU acceleration with realistic crawling scenario."""
    try:
        from app.services.face import detect_and_embed_batch_async
        
        print("=== GPU Crawling Test ===")
        
        # Create multiple test images (simulating crawled images)
        test_images = []
        for i in range(3):
            # Create different colored images to simulate real crawled content
            colors = ['white', 'lightblue', 'lightgray']
            img = Image.new('RGB', (200, 200), color=colors[i])
            
            # Add some pattern to make it more realistic
            pixels = img.load()
            for x in range(200):
                for y in range(200):
                    if (x + y) % 30 < 15:
                        pixels[x, y] = (150, 150, 150)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            test_images.append(img_bytes.getvalue())
        
        print(f"Created {len(test_images)} test images")
        print(f"Image sizes: {[len(img) for img in test_images]}")
        
        # Test batch face detection (this is what crawling would do)
        print("\nTesting batch face detection with GPU worker...")
        results = await detect_and_embed_batch_async(test_images, batch_size=2)
        
        print(f"Results: {len(results)} image results")
        for i, result in enumerate(results):
            print(f"  Image {i+1}: {len(result)} faces detected")
            for j, face in enumerate(result):
                print(f"    Face {j+1}: bbox={face.get('bbox', 'N/A')}, quality={face.get('quality', 'N/A'):.2f}")
        
        print("\n✅ GPU crawling test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GPU crawling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_crawling_gpu())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
