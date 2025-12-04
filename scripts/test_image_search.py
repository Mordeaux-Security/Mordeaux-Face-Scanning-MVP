#!/usr/bin/env python3
"""
Test search with an actual image to see why it's not matching.
"""
import sys
import base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "face-pipeline"))

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from config.settings import settings
from pipeline.face_helpers import embed_one_b64_strict
from face_quality import SEARCH_QUALITY

FACE_ID = "0b7598f7-498a-48de-963c-d87b4a3fe20b"
TENANT_ID = "demo-tenant"
IMAGE_URL = "https://cdn5-thumbs.motherlessmedia.com/thumbs/3488B64-small.jpg"

def image_to_base64(url_or_path):
    """Convert image URL or path to base64."""
    import urllib.request
    from io import BytesIO
    
    if url_or_path.startswith('http'):
        with urllib.request.urlopen(url_or_path) as response:
            img_data = response.read()
    else:
        with open(url_or_path, 'rb') as f:
            img_data = f.read()
    
    # Convert to base64 data URL
    import base64
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"

def main():
    # Connect to Qdrant
    client = QdrantClient(
        url=settings.QDRANT_URL, 
        api_key=settings.QDRANT_API_KEY or None,
        prefer_grpc=False
    )
    
    # Get stored face
    print(f"🔍 Retrieving stored face {FACE_ID}...")
    points = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[FACE_ID],
        with_vectors=True,
        with_payload=True
    )
    
    if not points:
        print(f"❌ Face not found!")
        return
    
    stored_point = points[0]
    stored_vector = np.array(stored_point.vector, dtype=np.float32)
    stored_payload = stored_point.payload or {}
    
    print(f"✅ Stored face:")
    print(f"   URL: {stored_payload.get('source_url', 'N/A')}")
    print(f"   Vector norm: {np.linalg.norm(stored_vector):.6f}")
    
    # Try to get the image and generate embedding
    print(f"\n🖼️  Downloading image from: {IMAGE_URL}")
    try:
        image_b64 = image_to_base64(IMAGE_URL)
        print(f"✅ Image downloaded, generating embedding...")
        
        # Generate embedding using the same method as search endpoint
        query_vec, fwq = embed_one_b64_strict(
            image_b64,
            require_single_face=False,
            quality_cfg=SEARCH_QUALITY,
        )
        
        query_vec = query_vec.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        print(f"✅ Query embedding generated:")
        print(f"   Vector norm: {query_norm:.6f}")
        print(f"   Face quality: usable={fwq.quality.is_usable}, score={fwq.quality.score:.4f}")
        
        # Compare embeddings
        similarity = float(np.dot(stored_vector, query_vec))
        print(f"\n🔍 Comparison:")
        print(f"   Cosine similarity: {similarity:.6f}")
        print(f"   L2 distance: {np.linalg.norm(stored_vector - query_vec):.6f}")
        
        if similarity < 0.10:
            print(f"   ❌ Similarity is VERY LOW! This explains why it's not matching.")
            print(f"   Possible causes:")
            print(f"   - Different face detection/cropping")
            print(f"   - Different alignment")
            print(f"   - Image preprocessing differences")
        elif similarity < 0.50:
            print(f"   ⚠️  Similarity is low but above threshold")
        else:
            print(f"   ✅ Similarity is good!")
        
        # Now search with this query vector
        print(f"\n🔍 Searching with query embedding (tenant_id='{TENANT_ID}', threshold=0.10)...")
        hits = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vec.tolist(),
            limit=10,
            score_threshold=0.10,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(
                    key="tenant_id",
                    match=qm.MatchValue(value=TENANT_ID)
                )]
            ),
            with_payload=True,
            with_vectors=False
        ).points
        
        print(f"   Found {len(hits)} results:")
        found_self = False
        for i, hit in enumerate(hits[:10], 1):
            is_match = str(hit.id) == FACE_ID
            if is_match:
                found_self = True
            marker = "⭐ TARGET" if is_match else "  "
            score = getattr(hit, 'score', 0.0)
            print(f"   {marker} {i}. ID: {str(hit.id)[:8]}..., score: {score:.4f}")
        
        if found_self:
            print(f"\n✅ SUCCESS: Face found in search results!")
        else:
            print(f"\n❌ FAILED: Face NOT found in search results")
            print(f"   The query embedding similarity was {similarity:.4f}, which is {'above' if similarity >= 0.10 else 'below'} the 0.10 threshold")
            if similarity < 0.10:
                print(f"   This is why it's not showing up - the embeddings don't match!")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

