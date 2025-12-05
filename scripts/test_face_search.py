#!/usr/bin/env python3
"""
Test why a specific face isn't showing up in search results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "face-pipeline"))

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from config.settings import settings

FACE_ID = "0b7598f7-498a-48de-963c-d87b4a3fe20b"
TENANT_ID = "demo-tenant"

def main():
    # Connect to Qdrant
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    
    # Get the point from Qdrant
    print(f"🔍 Retrieving face {FACE_ID} from Qdrant...")
    points = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[FACE_ID],
        with_vectors=True,
        with_payload=True
    )
    
    if not points:
        print(f"❌ Face {FACE_ID} not found in Qdrant!")
        return
    
    point = points[0]
    payload = point.payload or {}
    vector = np.array(point.vector, dtype=np.float32)
    vec_norm = np.linalg.norm(vector)
    
    print(f"✅ Found face {FACE_ID}")
    print(f"   Tenant ID: {payload.get('tenant_id', 'MISSING')}")
    print(f"   Source URL: {payload.get('source_url', payload.get('url', 'MISSING'))}")
    print(f"   Vector norm: {vec_norm:.6f} (should be ~1.0)")
    print(f"   Thumb key: {payload.get('thumb_key', 'N/A')}")
    
    stored_tenant = payload.get('tenant_id', 'MISSING')
    if stored_tenant != TENANT_ID:
        print(f"\n⚠️  WARNING: Stored tenant_id='{stored_tenant}' != search tenant_id='{TENANT_ID}'")
        print(f"   This could be why it's not showing up in search!")
    
    # Test 1: Search with no filters (should find itself)
    print(f"\n🔍 Test 1: Searching with face's own vector (NO filters)...")
    hits = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector.tolist(),
        limit=10,
        score_threshold=0.0,
        with_payload=True,
        with_vectors=False
    ).points
    
    print(f"   Found {len(hits)} results:")
    found_self = False
    for i, hit in enumerate(hits[:10], 1):
        is_match = str(hit.id) == FACE_ID
        if is_match:
            found_self = True
        marker = "⭐ SELF" if is_match else "  "
        hit_tenant = hit.payload.get('tenant_id', 'N/A') if hit.payload else 'N/A'
        score = getattr(hit, 'score', 0.0)
        print(f"   {marker} {i}. ID: {str(hit.id)[:8]}..., score: {score:.4f}, tenant: {hit_tenant}")
    
    if not found_self:
        print(f"   ❌ Face did NOT find itself! This is very strange.")
    
    # Test 2: Search with tenant filter
    print(f"\n🔍 Test 2: Searching with tenant_id filter: '{TENANT_ID}'...")
    hits = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector.tolist(),
        limit=10,
        score_threshold=0.0,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(
                key="tenant_id",
                match=qm.MatchValue(value=TENANT_ID)
            )]
        ),
        with_payload=True,
        with_vectors=False
    ).points
    
    print(f"   Found {len(hits)} results with tenant filter:")
    found_self = False
    for i, hit in enumerate(hits[:10], 1):
        is_match = str(hit.id) == FACE_ID
        if is_match:
            found_self = True
        marker = "⭐ SELF" if is_match else "  "
        score = getattr(hit, 'score', 0.0)
        print(f"   {marker} {i}. ID: {str(hit.id)[:8]}..., score: {score:.4f}")
    
    if not found_self and stored_tenant == TENANT_ID:
        print(f"   ❌ Face not found even with correct tenant filter!")
    elif not found_self:
        print(f"   ⚠️  Face not found because tenant_id mismatch (stored: '{stored_tenant}', filter: '{TENANT_ID}')")
    
    # Test 3: Search with stored tenant_id
    if stored_tenant != TENANT_ID and stored_tenant != 'MISSING':
        print(f"\n🔍 Test 3: Searching with stored tenant_id filter: '{stored_tenant}'...")
        hits = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=vector.tolist(),
            limit=10,
            score_threshold=0.0,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(
                    key="tenant_id",
                    match=qm.MatchValue(value=stored_tenant)
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
            marker = "⭐ SELF" if is_match else "  "
            score = getattr(hit, 'score', 0.0)
            print(f"   {marker} {i}. ID: {str(hit.id)[:8]}..., score: {score:.4f}")
        
        if found_self:
            print(f"   ✅ Face found with correct tenant_id!")
    
    # Test 4: Check threshold
    print(f"\n🔍 Test 4: Testing different thresholds...")
    for threshold in [0.0, 0.10, 0.50, 0.75, 0.90]:
        filter_obj = qm.Filter(
            must=[qm.FieldCondition(
                key="tenant_id",
                match=qm.MatchValue(value=TENANT_ID)
            )]
        ) if stored_tenant == TENANT_ID else None
        hits = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=vector.tolist(),
            limit=10,
            score_threshold=threshold,
            query_filter=filter_obj,
            with_payload=True,
            with_vectors=False
        ).points
        found = any(str(h.id) == FACE_ID for h in hits)
        print(f"   Threshold {threshold:.2f}: {len(hits)} results, self found: {'✅' if found else '❌'}")

if __name__ == "__main__":
    main()

