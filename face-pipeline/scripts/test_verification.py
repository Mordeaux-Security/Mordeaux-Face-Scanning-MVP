#!/usr/bin/env python3
"""
Verification-First Flow: Testing Helper Script

Quick helper to test enrollment and verification with real photos.

Usage:
    python scripts/test_verification.py enroll person_a photo1.jpg photo2.jpg photo3.jpg
    python scripts/test_verification.py verify person_a person_a probe.jpg
    python scripts/test_verification.py verify person_a person_b probe.jpg  # Should fail
"""

import sys
import base64
import requests
import json
from pathlib import Path

API_BASE = "http://localhost/pipeline/api/v1"
TENANT_ID = "test-tenant"

def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        # Detect MIME type from extension
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png"
        }
        mime = mime_map.get(ext, "image/jpeg")
        return f"data:{mime};base64,{b64}"

def enroll(identity_id: str, image_paths: list[str]):
    """Enroll an identity with multiple photos."""
    print(f"📸 Enrolling identity: {identity_id}")
    print(f"   Using {len(image_paths)} photos: {', '.join(Path(p).name for p in image_paths)}")
    
    # Convert images to base64
    images_b64 = []
    for path in image_paths:
        if not Path(path).exists():
            print(f"❌ Error: {path} not found")
            return False
        images_b64.append(image_to_base64(path))
    
    # Enroll
    try:
        response = requests.post(
            f"{API_BASE}/enroll_identity",
            json={
                "tenant_id": TENANT_ID,
                "identity_id": identity_id,
                "images_b64": images_b64
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                print(f"✅ Enrollment successful!")
                print(f"   Identity: {result['identity']}")
                return True
            else:
                print(f"❌ Enrollment failed: {result}")
                return False
        else:
            print(f"❌ Enrollment failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def verify(identity_id: str, probe_image_path: str, threshold: float = 0.78):
    """Verify a probe photo against an identity."""
    print(f"🔍 Verifying identity: {identity_id}")
    print(f"   Probe photo: {Path(probe_image_path).name}")
    print(f"   Threshold: {threshold}")
    
    if not Path(probe_image_path).exists():
        print(f"❌ Error: {probe_image_path} not found")
        return None
    
    # Convert image to base64
    probe_b64 = image_to_base64(probe_image_path)
    
    # Verify
    try:
        response = requests.post(
            f"{API_BASE}/verify",
            json={
                "tenant_id": TENANT_ID,
                "identity_id": identity_id,
                "image_b64": probe_b64,
                "hi_threshold": threshold,
                "top_k": 50
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            verified = result.get("verified", False)
            similarity = result.get("similarity", 0.0)
            count = result.get("count", 0)
            
            if verified:
                print(f"✅ Verification PASSED!")
                print(f"   Similarity: {similarity:.3f} (>= {threshold})")
                print(f"   Found {count} faces")
                
                # Show top results
                results = result.get("results", [])
                if results:
                    print(f"   Top faces:")
                    for i, face in enumerate(results[:5], 1):
                        score = face.get("score", 0.0)
                        face_id = face.get("id", "unknown")
                        print(f"     {i}. {face_id} (score: {score:.3f})")
            else:
                print(f"❌ Verification FAILED")
                print(f"   Similarity: {similarity:.3f} (< {threshold})")
                print(f"   Results: {count} faces (should be 0)")
                
                # Check if results array is empty (critical!)
                results = result.get("results", [])
                if len(results) > 0:
                    print(f"   ⚠️  WARNING: Results array is not empty! ({len(results)} faces)")
                    print(f"      This should not happen when verified=false")
                else:
                    print(f"   ✅ Results array is empty (correct behavior)")
            
            return result
        elif response.status_code == 404:
            print(f"❌ Identity not enrolled: {identity_id}")
            print(f"   Response: {response.json().get('detail', 'Not found')}")
            return None
        else:
            print(f"❌ Verification failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def test_matrix(person_ids: list[str], photo_map: dict[str, str], threshold: float = 0.78):
    """Test verification matrix: verify each person with each person's photo."""
    print(f"\n🧪 Running Verification Matrix Test")
    print(f"   People: {', '.join(person_ids)}")
    print(f"   Threshold: {threshold}\n")
    
    results = {}
    
    for identity_id in person_ids:
        for probe_id, probe_photo in photo_map.items():
            test_name = f"{identity_id} -> {probe_id}"
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"{'='*60}")
            
            result = verify(identity_id, probe_photo, threshold)
            results[test_name] = result
            
            # Expected behavior
            if identity_id == probe_id:
                expected = "PASS (same person)"
            else:
                expected = "FAIL (different person)"
            
            print(f"Expected: {expected}")
            
            if result:
                if identity_id == probe_id:
                    if result.get("verified"):
                        print(f"✅ Correct: Same person passed")
                    else:
                        print(f"⚠️  Unexpected: Same person failed (may be threshold too high or quality issue)")
                else:
                    if not result.get("verified"):
                        print(f"✅ Correct: Different person correctly rejected")
                    else:
                        print(f"❌ ERROR: Different person incorrectly passed (false accept!)")
                        print(f"   This is a critical issue - Person {probe_id} should not verify as Person {identity_id}")
    
    return results

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "enroll":
        if len(sys.argv) < 4:
            print("Usage: python test_verification.py enroll <identity_id> <photo1> <photo2> [photo3...]")
            sys.exit(1)
        
        identity_id = sys.argv[2]
        image_paths = sys.argv[3:]
        
        success = enroll(identity_id, image_paths)
        sys.exit(0 if success else 1)
    
    elif command == "verify":
        if len(sys.argv) < 4:
            print("Usage: python test_verification.py verify <identity_id> <probe_photo> [threshold]")
            sys.exit(1)
        
        identity_id = sys.argv[2]
        probe_photo = sys.argv[3]
        threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.78
        
        result = verify(identity_id, probe_photo, threshold)
        sys.exit(0 if (result and result.get("verified")) else 1)
    
    elif command == "matrix":
        if len(sys.argv) < 3:
            print("Usage: python test_verification.py matrix <person1>:<photo1> <person2>:<photo2> [person3]:[photo3]...")
            print("Example: python test_verification.py matrix person_a:photo_a.jpg person_b:photo_b.jpg person_c:photo_c.jpg")
            sys.exit(1)
        
        # Parse person:photo pairs
        person_ids = []
        photo_map = {}
        
        for arg in sys.argv[2:]:
            if ":" in arg:
                person_id, photo = arg.split(":", 1)
                person_ids.append(person_id)
                photo_map[person_id] = photo
            else:
                print(f"Error: Invalid format: {arg} (expected person_id:photo_path)")
                sys.exit(1)
        
        threshold = float(sys.argv[-1]) if len(sys.argv) > 2 and sys.argv[-1].replace(".", "").isdigit() else 0.78
        
        results = test_matrix(person_ids, photo_map, threshold)
        
        # Summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        
        correct = 0
        incorrect = 0
        for test_name, result in results.items():
            identity_id, probe_id = test_name.split(" -> ")
            if result:
                if identity_id == probe_id:
                    if result.get("verified"):
                        correct += 1
                    else:
                        incorrect += 1
                else:
                    if not result.get("verified"):
                        correct += 1
                    else:
                        incorrect += 1
        
        total = len(results)
        print(f"Total tests: {total}")
        print(f"Correct: {correct} ({correct/total*100:.1f}%)")
        print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
        
        sys.exit(0 if incorrect == 0 else 1)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()

