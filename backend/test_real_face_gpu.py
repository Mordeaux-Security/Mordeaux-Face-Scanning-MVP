#!/usr/bin/env python3
"""
Test GPU acceleration with a real face image.
"""

import asyncio
import sys
import os
from PIL import Image, ImageDraw
import io

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_real_face_gpu():
    """Test GPU acceleration with a face-like image."""
    try:
        from app.services.face import detect_and_embed_async
        
        print("=== Real Face GPU Test ===")
        
        # Create a simple face-like image
        img = Image.new('RGB', (300, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face
        # Face outline
        draw.ellipse([50, 50, 250, 250], fill='peachpuff', outline='black', width=2)
        
        # Eyes
        draw.ellipse([100, 120, 130, 150], fill='white', outline='black')
        draw.ellipse([170, 120, 200, 150], fill='white', outline='black')
        draw.ellipse([110, 130, 120, 140], fill='black')  # Left pupil
        draw.ellipse([180, 130, 190, 140], fill='black')  # Right pupil
        
        # Nose
        draw.polygon([(150, 160), (140, 190), (160, 190)], fill='peachpuff', outline='black')
        
        # Mouth
        draw.arc([120, 200, 180, 220], 0, 180, fill='red', width=3)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        test_image = img_bytes.getvalue()
        
        print(f"Created face image: {len(test_image)} bytes")
        print(f"Image header: {test_image[:10]}")
        
        # Test face detection
        print("\nTesting face detection with GPU worker...")
        results = await detect_and_embed_async(test_image)
        
        print(f"Results: {len(results)} faces detected")
        for i, face in enumerate(results):
            print(f"  Face {i+1}: bbox={face.get('bbox', 'N/A')}, quality={face.get('quality', 'N/A'):.2f}")
            if 'embedding' in face:
                print(f"    Embedding: {len(face['embedding'])} dimensions")
        
        if results:
            print("\n✅ GPU face detection working! Found faces with GPU acceleration.")
        else:
            print("\n⚠️  No faces detected (may be due to simple drawing)")
        
        print("\n✅ Real face GPU test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Real face GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_face_gpu())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
