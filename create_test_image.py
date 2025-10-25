#!/usr/bin/env python3
"""
Create a proper test image for GPU worker testing.
"""

from PIL import Image
import io
import base64

def create_test_image():
    """Create a proper test image."""
    # Create a 100x100 pixel image with a simple pattern
    img = Image.new('RGB', (100, 100), color='white')
    
    # Add a simple pattern to make it more realistic
    pixels = img.load()
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pixels[x, y] = (200, 200, 200)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    print(f"Created test image: {len(img_bytes)} bytes")
    print(f"Image header: {img_bytes[:10]}")
    
    # Encode to base64
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    print(f"Base64 length: {len(encoded)}")
    print(f"Base64 preview: {encoded[:50]}...")
    
    return img_bytes, encoded

if __name__ == "__main__":
    img_bytes, encoded = create_test_image()
    
    # Test decoding
    decoded = base64.b64decode(encoded)
    print(f"Decoded length: {len(decoded)}")
    print(f"Decoded matches original: {decoded == img_bytes}")
    
    # Test with PIL
    from PIL import Image
    test_img = Image.open(io.BytesIO(decoded))
    print(f"PIL can open decoded image: {test_img.size}")
