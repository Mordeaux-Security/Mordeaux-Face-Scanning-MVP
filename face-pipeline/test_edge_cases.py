#!/usr/bin/env python3
"""
Test edge cases and error handling for the embedding pipeline.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.embedder import embed
from pipeline.detector import detect_faces, align_and_crop
from pipeline.indexer import make_point

print("=" * 60)
print("EDGE CASE TESTING FOR EMBEDDING PIPELINE")
print("=" * 60)

# Test 1: None input
print("\n[TEST 1] None input to embed()")
try:
    embed(None)
    print("✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"✓ PASS: Caught ValueError: {e}")
except Exception as e:
    print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")

# Test 2: Wrong shape
print("\n[TEST 2] Wrong shape (100x100 instead of 112x112)")
try:
    embed(np.zeros((100, 100, 3), dtype=np.uint8))
    print("✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"✓ PASS: Caught ValueError: {e}")
except Exception as e:
    print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")

# Test 3: Float dtype (should auto-convert)
print("\n[TEST 3] Float dtype (should auto-convert to uint8)")
try:
    random_img = np.random.rand(112, 112, 3) * 255
    result = embed(random_img)
    print(f"✓ PASS: Result shape: {result.shape}, dtype: {result.dtype}")
    if result.shape != (512,):
        print(f"✗ FAIL: Expected shape (512,), got {result.shape}")
    if result.dtype != np.float32:
        print(f"✗ FAIL: Expected dtype float32, got {result.dtype}")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 4: All black image
print("\n[TEST 4] All black image (edge case)")
try:
    black_img = np.zeros((112, 112, 3), dtype=np.uint8)
    result = embed(black_img)
    print(f"✓ PASS: Generated embedding for black image, shape: {result.shape}")
    print(f"  Embedding norm: {np.linalg.norm(result):.4f}")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 5: All white image
print("\n[TEST 5] All white image (edge case)")
try:
    white_img = np.ones((112, 112, 3), dtype=np.uint8) * 255
    result = embed(white_img)
    print(f"✓ PASS: Generated embedding for white image, shape: {result.shape}")
    print(f"  Embedding norm: {np.linalg.norm(result):.4f}")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 6: Random noise
print("\n[TEST 6] Random noise image")
try:
    noise_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    result = embed(noise_img)
    print(f"✓ PASS: Generated embedding for noise, shape: {result.shape}")
    print(f"  Embedding norm: {np.linalg.norm(result):.4f}")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 7: detect_faces with no faces
print("\n[TEST 7] detect_faces on image with no faces")
try:
    noise_img = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    faces = detect_faces(noise_img)
    print(f"✓ PASS: Detected {len(faces)} faces (expected 0)")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 8: align_and_crop with insufficient landmarks
print("\n[TEST 8] align_and_crop with insufficient landmarks")
try:
    img = np.zeros((640, 480, 3), dtype=np.uint8)
    landmarks = [[100, 100], [200, 100]]  # Only 2 landmarks instead of 5
    result = align_and_crop(img, landmarks)
    print(f"✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"✓ PASS: Caught ValueError: {e}")
except Exception as e:
    print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")

# Test 9: make_point with various face_id formats
print("\n[TEST 9] make_point with various face_id formats")
test_ids = [
    "simple",
    "with:colon",
    "abc123_def456:face_0",
    "very_long_id_" + "x" * 200,
    "unicode_测试",
]
for test_id in test_ids:
    try:
        vec = [0.1] * 512
        payload = {"test": "data"}
        point = make_point(test_id, vec, payload)
        print(f"✓ PASS: '{test_id[:50]}...' -> UUID: {point.id[:36]}")
    except Exception as e:
        print(f"✗ FAIL: '{test_id}' -> {type(e).__name__}: {e}")

# Test 10: Embedding consistency
print("\n[TEST 10] Embedding consistency (same input -> same output)")
try:
    test_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    result1 = embed(test_img)
    result2 = embed(test_img)
    diff = np.linalg.norm(result1 - result2)
    if diff < 1e-6:
        print(f"✓ PASS: Embeddings are consistent (diff: {diff:.10f})")
    else:
        print(f"✗ FAIL: Embeddings differ (diff: {diff:.10f})")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 11: Real face image if available
print("\n[TEST 11] Real face image (image0-6.jpeg)")
try:
    img_path = Path(__file__).parent / "samples" / "image0-6.jpeg"
    if img_path.exists():
        arr = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        faces = detect_faces(img)
        if faces:
            crop = align_and_crop(img, faces[0]["landmarks"])
            result = embed(crop)
            print(f"✓ PASS: Real face -> embedding shape: {result.shape}, norm: {np.linalg.norm(result):.4f}")
        else:
            print(f"⚠ WARNING: No faces detected in image0-6.jpeg")
    else:
        print(f"⚠ SKIP: image0-6.jpeg not found")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 12: Memory/performance test
print("\n[TEST 12] Memory efficiency (100 embeddings)")
try:
    import time
    start = time.time()
    for i in range(100):
        test_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = embed(test_img)
    elapsed = time.time() - start
    print(f"✓ PASS: Generated 100 embeddings in {elapsed:.2f}s ({elapsed/100*1000:.1f}ms per embedding)")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("EDGE CASE TESTING COMPLETE")
print("=" * 60)

