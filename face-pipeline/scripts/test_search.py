"""
Test search script to debug search results.

Usage:
    cd face-pipeline
    python -m scripts.test_search <image_path>
    
Or test with a sample from Qdrant:
    python -m scripts.test_search --sample
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.indexer import get_client
from pipeline.embedder import embed
from pipeline.detector import detect_faces, align_and_crop
from config.settings import settings
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


FACES_COLLECTION = settings.QDRANT_COLLECTION
TENANT_ID = "demo-tenant"


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")


def search_with_threshold(query_vector: np.ndarray, threshold: float, top_k: int = 50):
    """Search with a specific threshold and show results."""
    qc = get_client()
    
    # Normalize vector
    query_vec = query_vector / (np.linalg.norm(query_vector) + 1e-9)
    
    print(f"\n=== Search with threshold={threshold}, top_k={top_k} ===")
    
    hits = qc.search(
        collection_name=FACES_COLLECTION,
        query_vector=query_vec.tolist(),
        limit=top_k,
        score_threshold=threshold,
        query_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=TENANT_ID))]
        ),
        with_payload=True,
        with_vectors=False,
    )
    
    print(f"Found {len(hits)} results above threshold {threshold}")
    
    if hits:
        print("\nTop 10 results:")
        for i, hit in enumerate(hits[:10], 1):
            payload = hit.payload or {}
            url = payload.get("source_url") or payload.get("url", "N/A")
            site = payload.get("site_id", "N/A")
            print(f"  {i}. Score: {hit.score:.4f} | Site: {site} | URL: {url[:60]}")
    
    return hits


def test_with_sample():
    """Test search using a sample point from Qdrant."""
    print("=== Testing search with a sample face from Qdrant ===\n")
    
    qc = get_client()
    
    # Get a sample point
    points, _ = qc.scroll(
        collection_name=FACES_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=TENANT_ID))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    
    if not points:
        print("❌ No points found in Qdrant with tenant_id='demo-tenant'")
        return
    
    sample = points[0]
    sample_vector = np.array(sample.vector, dtype=np.float32)
    payload = sample.payload or {}
    
    print(f"Sample point ID: {sample.id}")
    print(f"Sample URL: {payload.get('source_url', 'N/A')}")
    print()
    
    # Test with different thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.85]
    
    for threshold in thresholds:
        hits = search_with_threshold(sample_vector, threshold, top_k=20)
        if hits:
            print(f"✅ Threshold {threshold}: Found {len(hits)} matches")
        else:
            print(f"❌ Threshold {threshold}: No matches found")
        print()


def test_with_image(image_path: Path):
    """Test search with a user-provided image."""
    print(f"=== Testing search with image: {image_path} ===\n")
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load and process image
    from pipeline.image_utils import load_image
    
    try:
        img_bgr = load_image(str(image_path))
        print(f"✅ Image loaded: {img_bgr.shape}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Detect faces
    faces = detect_faces(img_bgr)
    print(f"✅ Detected {len(faces)} face(s)")
    
    if not faces:
        print("❌ No faces detected in image")
        return
    
    # Use first face
    face_data = faces[0]
    face_crop = align_and_crop(img_bgr, face_data["landmarks"])
    
    # Generate embedding
    try:
        embedding = embed(face_crop)
        print(f"✅ Generated embedding: {embedding.shape}")
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return
    
    # Test with different thresholds
    print("\n=== Testing with multiple thresholds ===\n")
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.85]
    
    for threshold in thresholds:
        hits = search_with_threshold(embedding, threshold, top_k=20)
        if hits:
            print(f"✅ Threshold {threshold}: Found {len(hits)} matches")
            # Show best match
            best = hits[0]
            best_payload = best.payload or {}
            print(f"   Best match: score={best.score:.4f}, URL={best_payload.get('source_url', 'N/A')[:60]}")
        else:
            print(f"❌ Threshold {threshold}: No matches found")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test face search with different thresholds")
    parser.add_argument("image_path", nargs="?", type=Path, help="Path to image file to search")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Test with a sample face from Qdrant instead of user image"
    )
    args = parser.parse_args()
    
    if args.sample:
        test_with_sample()
    elif args.image_path:
        test_with_image(args.image_path)
    else:
        print("Please provide an image path or use --sample flag")
        print("\nUsage:")
        print("  python -m scripts.test_search <image_path>")
        print("  python -m scripts.test_search --sample")


if __name__ == "__main__":
    main()

