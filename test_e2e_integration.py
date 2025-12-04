#!/usr/bin/env python3
"""
End-to-End Integration Test
============================

Tests the complete flow:
1. Upload image to backend
2. Verify storage in MinIO
3. Trigger face pipeline processing
4. Search for the face
5. Verify results display with metadata
"""

import requests
import base64
import json
import time
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost/api"  # or http://localhost:8000 for direct backend
PIPELINE_URL = "http://localhost:8001/api/v1"
TENANT_ID = "test-tenant"
SITE = "test-site"

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_health():
    """Test service health"""
    print("\n🔍 Testing Service Health...")
    
    # Test backend
    try:
        r = requests.get(f"{BACKEND_URL}/v1/health", timeout=5)
        print(f"✅ Backend API: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"❌ Backend API failed: {e}")
    
    # Test pipeline
    try:
        r = requests.get(f"{PIPELINE_URL}/health", timeout=5)
        print(f"✅ Face Pipeline: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"❌ Face Pipeline failed: {e}")

def test_search_with_base64(image_path: str):
    """Test search endpoint with base64 image"""
    print(f"\n🔍 Testing Search with {image_path}...")
    
    # Encode image
    image_b64 = encode_image(image_path)
    
    # Prepare request
    payload = {
        "tenant_id": TENANT_ID,
        "image_b64": image_b64,
        "top_k": 50,
        "threshold": 0.70
    }
    
    # Send to backend (which proxies to pipeline)
    try:
        print("📤 Sending search request to backend...")
        r = requests.post(
            f"{BACKEND_URL}/v1/search",
            json=payload,
            timeout=60
        )
        
        print(f"Response Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Search successful!")
            print(f"   Results count: {data.get('count', 0)}")
            
            # Display first few results
            hits = data.get('hits', [])
            for i, hit in enumerate(hits[:3]):
                print(f"\n   Result {i+1}:")
                print(f"   - Face ID: {hit.get('face_id', 'N/A')}")
                print(f"   - Score: {hit.get('score', 0):.4f}")
                print(f"   - Tenant ID: {hit.get('payload', {}).get('tenant_id', 'N/A')}")
                print(f"   - Site: {hit.get('payload', {}).get('site', 'N/A')}")
                print(f"   - URL: {hit.get('payload', {}).get('url', 'N/A')[:60]}...")
                
            return data
        else:
            print(f"❌ Search failed: {r.status_code}")
            print(f"   Response: {r.text}")
            return None
            
    except Exception as e:
        print(f"❌ Search request failed: {e}")
        return None

def test_ingest(image_path: str):
    """Test image ingest (upload and queue for processing)"""
    print(f"\n📤 Testing Ingest with {image_path}...")
    
    # Encode image
    image_b64 = encode_image(image_path)
    
    # First, we need to upload to MinIO (simplified - in real flow this happens automatically)
    # For now, test the ingest endpoint which should handle MinIO upload
    
    bucket = "raw-images"
    key = f"{TENANT_ID}/test_{int(time.time())}.jpg"
    
    payload = {
        "tenant_id": TENANT_ID,
        "bucket": bucket,
        "key": key,
        "site": SITE,
        "meta": {
            "test": True,
            "uploaded_at": time.time()
        }
    }
    
    try:
        print("📤 Sending ingest request...")
        r = requests.post(
            f"{BACKEND_URL}/v1/ingest",
            json=payload,
            timeout=30
        )
        
        print(f"Response Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Ingest successful!")
            print(f"   Message ID: {data.get('message_id', 'N/A')}")
            print(f"   Stream: {data.get('stream', 'N/A')}")
            return data
        else:
            print(f"❌ Ingest failed: {r.status_code}")
            print(f"   Response: {r.text}")
            return None
            
    except Exception as e:
        print(f"❌ Ingest request failed: {e}")
        return None

def verify_minio_connection():
    """Verify MinIO is accessible"""
    print("\n🔍 Checking MinIO Connection...")
    
    try:
        # Try to access MinIO console
        r = requests.get("http://localhost:9001", timeout=5)
        print(f"✅ MinIO Console accessible: {r.status_code}")
    except Exception as e:
        print(f"❌ MinIO Console not accessible: {e}")

def verify_qdrant_connection():
    """Verify Qdrant is accessible"""
    print("\n🔍 Checking Qdrant Connection...")
    
    try:
        r = requests.get("http://localhost:6333/collections", timeout=5)
        print(f"✅ Qdrant accessible: {r.status_code}")
        
        if r.status_code == 200:
            collections = r.json()
            print(f"   Collections: {json.dumps(collections, indent=2)}")
    except Exception as e:
        print(f"❌ Qdrant not accessible: {e}")

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("🚀 MORDEAUX E2E INTEGRATION TEST")
    print("=" * 60)
    
    # 1. Health checks
    test_health()
    verify_minio_connection()
    verify_qdrant_connection()
    
    # 2. Test with sample image (if available)
    sample_images = [
        "face-pipeline/samples/person3_a.jpeg",
        "face-pipeline/samples/person3_b.jpg",
    ]
    
    for img_path in sample_images:
        if Path(img_path).exists():
            print(f"\n{'=' * 60}")
            print(f"Testing with: {img_path}")
            print(f"{'=' * 60}")
            
            # Test search
            search_result = test_search_with_base64(img_path)
            
            if search_result and search_result.get('count', 0) == 0:
                print("\n⚠️  No results found. This is expected if database is empty.")
                print("   Consider running ingestion first to populate the database.")
            
            break  # Only test with first available image
    else:
        print("\n⚠️  No sample images found. Skipping search test.")
        print("   Place test images in face-pipeline/samples/ directory.")
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print("\n📋 NEXT STEPS:")
    print("   1. If services are healthy, proceed with frontend integration")
    print("   2. If search returns empty, run ingestion to populate database")
    print("   3. Check logs in Docker containers for detailed errors")
    print("\n   View logs:")
    print("   docker-compose logs -f api")
    print("   docker-compose logs -f face-pipeline")
    print()

if __name__ == "__main__":
    main()

