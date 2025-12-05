#!/usr/bin/env python3
"""Test search connectivity and functionality"""

import requests
import json
import base64
from pathlib import Path

# Create a tiny 1x1 red pixel PNG (valid image)
tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

print("=" * 60)
print("TESTING SEARCH CONNECTIVITY")
print("=" * 60)

# Test 1: Frontend -> Nginx -> Backend
print("\n1. Testing Frontend -> Nginx -> Backend")
try:
    response = requests.get("http://localhost/api/v1/health", timeout=5)
    print(f"   ✅ Nginx routing works: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Nginx routing failed: {e}")

# Test 2: Backend -> Face Pipeline
print("\n2. Testing Backend -> Face Pipeline")
try:
    response = requests.get("http://localhost:8001/api/v1/health", timeout=5)
    print(f"   ✅ Face Pipeline reachable: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Face Pipeline unreachable: {e}")

# Test 3: Search endpoint through full chain
print("\n3. Testing Search Endpoint (Frontend -> Backend -> Pipeline)")
search_payload = {
    "tenant_id": "demo-tenant",
    "image_b64": tiny_png_b64,
    "top_k": 10,
    "threshold": 0.0  # Very low threshold - should match everything
}

try:
    response = requests.post(
        "http://localhost/api/v1/search",
        json=search_payload,
        timeout=30,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Search endpoint works!")
        print(f"   Count: {data.get('count', 'N/A')}")
        print(f"   Hits: {len(data.get('hits', []))}")
        if data.get('count', 0) == 0:
            print(f"   ⚠️  No matches found even with threshold=0.0")
            print(f"   Response: {json.dumps(data, indent=2)[:500]}")
    else:
        print(f"   ❌ Search failed: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
except Exception as e:
    print(f"   ❌ Search request failed: {e}")

# Test 4: Direct Face Pipeline search
print("\n4. Testing Direct Face Pipeline Search")
try:
    response = requests.post(
        "http://localhost:8001/api/v1/search",
        json=search_payload,
        timeout=30,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Direct pipeline search works!")
        print(f"   Count: {data.get('count', 'N/A')}")
        print(f"   Hits: {len(data.get('hits', []))}")
    else:
        print(f"   ❌ Direct search failed: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
except Exception as e:
    print(f"   ❌ Direct search request failed: {e}")

print("\n" + "=" * 60)
print("CONNECTIVITY TEST COMPLETE")
print("=" * 60)

