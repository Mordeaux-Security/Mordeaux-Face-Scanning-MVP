#!/usr/bin/env python3
"""
Test script to search for faces using an image file.
Usage: python test_search_image.py <image_path>
"""

import sys
import base64
import requests
import json
from pathlib import Path

# API configuration
API_BASE = "http://localhost/pipeline/api/v1"
TENANT_ID = "demo-tenant"

def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        # Detect MIME type from extension
        ext = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png"
        }
        mime = mime_map.get(ext, "image/jpeg")
        return f"{b64}"  # Return just base64, not data URL

def search_with_image(image_path: str):
    """Search for faces using an image."""
    print(f"🔍 Testing search with image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    # Convert image to base64
    print("📸 Converting image to base64...")
    try:
        image_b64 = image_to_base64(Path(image_path))
        print(f"✅ Image encoded ({len(image_b64)} chars)")
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return False
    
    # Prepare search request
    search_url = f"{API_BASE}/search"
    payload = {
        "tenant_id": TENANT_ID,
        "image_b64": image_b64,
        "top_k": 10,
        "threshold": 0.10
    }
    
    print(f"\n🚀 Sending search request to: {search_url}")
    print(f"   Tenant: {TENANT_ID}")
    print(f"   Top K: 10, Threshold: 0.10")
    
    try:
        response = requests.post(
            search_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            count = data.get("count", 0)
            hits = data.get("hits", [])
            
            print(f"✅ Search successful!")
            print(f"   Found {count} matches")
            
            if count > 0:
                print(f"\n📋 Top matches:")
                for i, hit in enumerate(hits[:5], 1):
                    score = hit.get("score", 0)
                    similarity_pct = hit.get("similarity_pct", score * 100)
                    payload = hit.get("payload", {})
                    site = payload.get("site", "unknown")
                    print(f"   {i}. Similarity: {similarity_pct:.1f}% | Site: {site}")
                    if hit.get("image_url"):
                        print(f"      Image: {hit['image_url'][:80]}...")
                
                return True
            else:
                print("⚠️  No matches found (even with low threshold)")
                print("   This could mean:")
                print("   - The face is not in the database")
                print("   - There might be an embedding mismatch issue")
                return False
        else:
            error_data = response.json()
            print(f"❌ Search failed with status {response.status_code}")
            print(f"   Error details:")
            print(json.dumps(error_data, indent=2))
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def search_with_file_upload(image_path: str):
    """Test the file upload endpoint."""
    print(f"\n" + "="*60)
    print(f"📤 Testing file upload endpoint")
    print(f"="*60)
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    search_url = f"{API_BASE}/search/file"
    
    print(f"🚀 Uploading file to: {search_url}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            params = {
                'tenant_id': TENANT_ID,
                'top_k': 10,
                'threshold': 0.10
            }
            
            response = requests.post(
                search_url,
                files=files,
                params=params,
                timeout=30
            )
            
            print(f"\n📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("count", 0)
                hits = data.get("hits", [])
                
                print(f"✅ File upload search successful!")
                print(f"   Found {count} matches")
                
                if count > 0:
                    print(f"\n📋 Top matches:")
                    for i, hit in enumerate(hits[:5], 1):
                        score = hit.get("score", 0)
                        similarity_pct = hit.get("similarity_pct", score * 100)
                        payload = hit.get("payload", {})
                        site = payload.get("site", "unknown")
                        print(f"   {i}. Similarity: {similarity_pct:.1f}% | Site: {site}")
                
                return True
            else:
                error_data = response.json()
                print(f"❌ File upload search failed with status {response.status_code}")
                print(f"   Error details:")
                print(json.dumps(error_data, indent=2))
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_search_image.py <image_path>")
        print("\nExample:")
        print("  python test_search_image.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("="*60)
    print("🧪 Face Search Test")
    print("="*60)
    
    # Test 1: Base64 search
    print("\n" + "="*60)
    print("Test 1: Base64 Image Search (JSON endpoint)")
    print("="*60)
    result1 = search_with_image(image_path)
    
    # Test 2: File upload search
    result2 = search_with_file_upload(image_path)
    
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"Base64 search: {'✅ Success' if result1 else '❌ Failed'}")
    print(f"File upload search: {'✅ Success' if result2 else '❌ Failed'}")

