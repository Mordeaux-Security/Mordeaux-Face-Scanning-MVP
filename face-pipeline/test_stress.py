#!/usr/bin/env python3
"""
Stress tests and concurrent access tests for the embedding pipeline.
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.embedder import embed
from pipeline.detector import detect_faces, align_and_crop
from pipeline.indexer import make_point, get_client
from pipeline.storage import get_client as get_minio_client

print("=" * 60)
print("STRESS TESTING FOR EMBEDDING PIPELINE")
print("=" * 60)

# Test 1: Concurrent embedding generation
print("\n[TEST 1] Concurrent embedding generation (10 threads)")
try:
    def generate_embedding(idx):
        img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = embed(img)
        return idx, result.shape, np.linalg.norm(result)
    
    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_embedding, i) for i in range(50)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.time() - start
    
    print(f"✓ PASS: Generated {len(results)} embeddings concurrently in {elapsed:.2f}s")
    print(f"  Throughput: {len(results)/elapsed:.1f} embeddings/sec")
    
    # Check all results are valid
    all_valid = all(shape == (512,) and 0.99 < norm < 1.01 for _, shape, norm in results)
    if all_valid:
        print(f"✓ PASS: All embeddings are valid (shape=(512,), normalized)")
    else:
        print(f"✗ FAIL: Some embeddings are invalid")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 2: Large batch processing
print("\n[TEST 2] Large batch processing (500 images)")
try:
    start = time.time()
    embeddings = []
    for i in range(500):
        img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        result = embed(img)
        embeddings.append(result)
    elapsed = time.time() - start
    
    print(f"✓ PASS: Processed 500 images in {elapsed:.2f}s ({elapsed/500*1000:.1f}ms per image)")
    print(f"  Memory: {sum(e.nbytes for e in embeddings) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 3: Extreme values
print("\n[TEST 3] Extreme pixel values")
try:
    # All zeros
    img_zeros = np.zeros((112, 112, 3), dtype=np.uint8)
    emb_zeros = embed(img_zeros)
    
    # All max
    img_max = np.full((112, 112, 3), 255, dtype=np.uint8)
    emb_max = embed(img_max)
    
    # Check they're different
    diff = np.linalg.norm(emb_zeros - emb_max)
    if diff > 0.1:
        print(f"✓ PASS: Embeddings differ for extreme values (diff: {diff:.4f})")
    else:
        print(f"⚠ WARNING: Embeddings are too similar for extreme values (diff: {diff:.4f})")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 4: Model singleton behavior
print("\n[TEST 4] Model singleton behavior (load_model called multiple times)")
try:
    from pipeline.embedder import load_model
    
    start = time.time()
    model1 = load_model()
    time1 = time.time() - start
    
    start = time.time()
    model2 = load_model()
    time2 = time.time() - start
    
    start = time.time()
    model3 = load_model()
    time3 = time.time() - start
    
    if model1 is model2 is model3:
        print(f"✓ PASS: Singleton working correctly (same instance returned)")
        print(f"  First call: {time1*1000:.1f}ms, Second: {time2*1000:.1f}ms, Third: {time3*1000:.1f}ms")
    else:
        print(f"✗ FAIL: Different instances returned")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 5: UUID collision test
print("\n[TEST 5] UUID collision test (10000 unique face IDs)")
try:
    from pipeline.indexer import make_point
    
    uuids = set()
    collisions = 0
    
    for i in range(10000):
        face_id = f"test_face_{i}"
        point = make_point(face_id, [0.0] * 512, {})
        if point.id in uuids:
            collisions += 1
        uuids.add(point.id)
    
    if collisions == 0:
        print(f"✓ PASS: No UUID collisions in 10000 face IDs")
    else:
        print(f"✗ FAIL: {collisions} UUID collisions detected")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 6: Qdrant client singleton
print("\n[TEST 6] Qdrant client singleton behavior")
try:
    client1 = get_client()
    client2 = get_client()
    
    if client1 is client2:
        print(f"✓ PASS: Qdrant client singleton working")
    else:
        print(f"✗ FAIL: Different Qdrant client instances")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 7: MinIO client singleton
print("\n[TEST 7] MinIO client singleton behavior")
try:
    minio1 = get_minio_client()
    minio2 = get_minio_client()
    
    if minio1 is minio2:
        print(f"✓ PASS: MinIO client singleton working")
    else:
        print(f"✗ FAIL: Different MinIO client instances")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 8: Embedding normalization
print("\n[TEST 8] Embedding L2 normalization (should be ~1.0)")
try:
    norms = []
    for i in range(100):
        img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = embed(img)
        norms.append(np.linalg.norm(emb))
    
    min_norm = min(norms)
    max_norm = max(norms)
    avg_norm = sum(norms) / len(norms)
    
    if 0.99 < min_norm and max_norm < 1.01:
        print(f"✓ PASS: All embeddings normalized (min: {min_norm:.6f}, max: {max_norm:.6f}, avg: {avg_norm:.6f})")
    else:
        print(f"✗ FAIL: Normalization out of range (min: {min_norm:.6f}, max: {max_norm:.6f})")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 9: Face detection on various image sizes
print("\n[TEST 9] Face detection on various image sizes")
try:
    sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080), (4000, 3000)]
    
    for width, height in sizes:
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        start = time.time()
        faces = detect_faces(img)
        elapsed = time.time() - start
        print(f"  {width}x{height}: {len(faces)} faces, {elapsed*1000:.0f}ms")
    
    print(f"✓ PASS: Face detection works on various sizes")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

# Test 10: Embedding determinism with same random seed
print("\n[TEST 10] Embedding determinism (same input, different runs)")
try:
    np.random.seed(42)
    img1 = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    
    np.random.seed(42)
    img2 = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    
    emb1 = embed(img1)
    emb2 = embed(img2)
    
    diff = np.linalg.norm(emb1 - emb2)
    
    if diff < 1e-6:
        print(f"✓ PASS: Embeddings are deterministic (diff: {diff:.10f})")
    else:
        print(f"✗ FAIL: Embeddings differ for same input (diff: {diff:.10f})")
except Exception as e:
    print(f"✗ FAIL: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("STRESS TESTING COMPLETE")
print("=" * 60)

