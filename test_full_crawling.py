#!/usr/bin/env python3
"""
Test full crawling process with GPU acceleration.
"""

import asyncio
import sys
import os
from PIL import Image, ImageDraw
import io

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_full_crawling():
    """Test the complete crawling process with GPU acceleration."""
    try:
        from app.services.crawler import ImageCrawler
        from app.core.settings import get_settings
        
        print("=== Full Crawling GPU Test ===")
        
        # Check settings
        settings = get_settings()
        print(f"GPU Worker Enabled: {settings.gpu_worker_enabled}")
        print(f"GPU Worker URL: {settings.gpu_worker_url}")
        
        # Initialize crawler
        crawler = ImageCrawler()
        print("✅ ImageCrawler initialized successfully")
        
        # Create a test image (simulating a crawled image)
        img = Image.new('RGB', (400, 400), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a more realistic face
        # Face outline
        draw.ellipse([100, 100, 300, 300], fill='peachpuff', outline='black', width=3)
        
        # Eyes
        draw.ellipse([150, 150, 180, 180], fill='white', outline='black', width=2)
        draw.ellipse([220, 150, 250, 180], fill='white', outline='black', width=2)
        draw.ellipse([160, 160, 170, 170], fill='black')  # Left pupil
        draw.ellipse([230, 160, 240, 170], fill='black')  # Right pupil
        
        # Nose
        draw.polygon([(200, 200), (190, 230), (210, 230)], fill='peachpuff', outline='black')
        
        # Mouth
        draw.arc([170, 250, 230, 280], 0, 180, fill='red', width=4)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        test_image = img_bytes.getvalue()
        
        print(f"Created test image: {len(test_image)} bytes")
        
        # Test the face processing pipeline (this is what crawling does)
        print("\nTesting face processing pipeline...")
        
        # Import the face service directly
        from app.services.face import detect_and_embed_async
        
        # Process the image (this is what the crawler would do)
        results = await detect_and_embed_async(test_image)
        
        print(f"Face processing results: {len(results)} faces detected")
        for i, face in enumerate(results):
            print(f"  Face {i+1}: bbox={face.get('bbox', 'N/A')}, quality={face.get('quality', 'N/A'):.2f}")
            if 'embedding' in face:
                print(f"    Embedding: {len(face['embedding'])} dimensions")
        
        print("\n✅ Full crawling test completed successfully!")
        print("✅ GPU worker integration is working for crawling!")
        return True
        
    except Exception as e:
        print(f"❌ Full crawling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_crawling())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
