#!/usr/bin/env python3
"""
Test script for enrolling and verifying persons 1, 2, 3, and 6
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test helper
from face_pipeline.scripts.test_verification import enroll, verify  # This won't work, let me inline it

import base64
import requests
import json

API_BASE = "http://localhost/pipeline/api/v1"
TENANT_ID = "test-tenant"
SAMPLES_DIR = Path(__file__).parent.parent / "samples"

def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        ext = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png"
        }
        mime = mime_map.get(ext, "image/jpeg")
        return f"data:{mime};base64,{b64}"

def enroll_person(person_id: str, photo_names: list[str]):
    """Enroll a person with their photos."""
    print(f"\n📸 Enrolling {person_id} with {len(photo_names)} photos...")
    
    # Convert photos to base64
    images_b64 = []
    for photo_name in photo_names:
        photo_path = SAMPLES_DIR / photo_name
        if not photo_path.exists():
            print(f"❌ Error: {photo_path} not found")
            return False
        images_b64.append(image_to_base64(photo_path))
        print(f"   ✓ Loaded: {photo_name}")
    
    # Enroll
    try:
        response = requests.post(
            f"{API_BASE}/enroll_identity",
            json={
                "tenant_id": TENANT_ID,
                "identity_id": person_id,
                "images_b64": images_b64
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                print(f"✅ {person_id} enrolled successfully!")
                print(f"   Identity: {result['identity']}")
                return True
            else:
                print(f"❌ Enrollment failed: {result}")
                return False
        else:
            print(f"❌ Enrollment failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def verify_person(identity_id: str, probe_photo: str, threshold: float = 0.78):
    """Verify a probe photo against an identity."""
    print(f"\n🔍 Verifying {identity_id} with {probe_photo}...")
    
    probe_path = SAMPLES_DIR / probe_photo
    if not probe_path.exists():
        print(f"❌ Error: {probe_path} not found")
        return None
    
    probe_b64 = image_to_base64(probe_path)
    
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
            else:
                print(f"❌ Verification FAILED")
                print(f"   Similarity: {similarity:.3f} (< {threshold})")
                print(f"   Results: {count} faces (should be 0)")
                
                results = result.get("results", [])
                if len(results) > 0:
                    print(f"   ⚠️  WARNING: Results array is not empty! ({len(results)} faces)")
                else:
                    print(f"   ✅ Results array is empty (correct behavior)")
            
            return result
        elif response.status_code == 404:
            print(f"❌ Identity not enrolled: {identity_id}")
            return None
        else:
            print(f"❌ Verification failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    print("=" * 70)
    print("Verification-First Flow: Testing Persons 1, 2, 3, and 6")
    print("=" * 70)
    
    # Phase 1: Enrollment
    print("\n" + "=" * 70)
    print("Phase 1: Enrollment")
    print("=" * 70)
    
    enrollments = {
        "person_1": ["person1_A.jpg", "person1_B.jpeg", "person1_C.jpg"],
        "person_2": ["person2_A.jpg", "person2_B.jpg"],
        "person_3": ["person3_a.jpeg", "person3_b.jpg", "person3_C.jpg"],
        "person_6": ["person6_a.jpeg", "person6_b.jpeg", "person6_C.jpg", "person6_D.jpg"],
    }
    
    enrollment_results = {}
    for person_id, photos in enrollments.items():
        success = enroll_person(person_id, photos)
        enrollment_results[person_id] = success
    
    # Wait a moment for enrollment to complete
    import time
    print("\n⏳ Waiting 2 seconds for enrollment to complete...")
    time.sleep(2)
    
    # Phase 2: Verification Tests
    print("\n" + "=" * 70)
    print("Phase 2: Verification Tests")
    print("=" * 70)
    
    tests = [
        ("person_1", "person1_A.jpg", True, "Person 1 → Person 1 (should pass)"),
        ("person_1", "person2_A.jpg", False, "Person 1 → Person 2 (should fail)"),
        ("person_2", "person2_A.jpg", True, "Person 2 → Person 2 (should pass)"),
        ("person_2", "person3_a.jpeg", False, "Person 2 → Person 3 (should fail)"),
        ("person_3", "person3_a.jpeg", True, "Person 3 → Person 3 (should pass)"),
        ("person_3", "person6_a.jpeg", False, "Person 3 → Person 6 (should fail)"),
        ("person_6", "person6_a.jpeg", True, "Person 6 → Person 6 (should pass)"),
        ("person_6", "person1_A.jpg", False, "Person 6 → Person 1 (should fail)"),
    ]
    
    test_results = {}
    for identity_id, probe_photo, should_pass, description in tests:
        print(f"\n{'=' * 70}")
        print(f"Test: {description}")
        print(f"{'=' * 70}")
        
        result = verify_person(identity_id, probe_photo, 0.78)
        test_results[description] = result
        
        if result:
            verified = result.get("verified", False)
            if should_pass:
                if verified:
                    print(f"✅ Correct: Same person passed")
                else:
                    print(f"⚠️  Unexpected: Same person failed (may be threshold too high or quality issue)")
            else:
                if not verified:
                    print(f"✅ Correct: Different person correctly rejected")
                else:
                    print(f"❌ ERROR: Different person incorrectly passed (false accept!)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    print("\nEnrollment Results:")
    for person_id, success in enrollment_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {person_id}: {status}")
    
    print("\nVerification Results:")
    correct = 0
    incorrect = 0
    for description, result in test_results.items():
        identity_id, probe_photo, should_pass, _ = next(
            (id, probe, pass_exp, desc) 
            for id, probe, pass_exp, desc in tests 
            if desc == description
        )
        if result:
            verified = result.get("verified", False)
            if (should_pass and verified) or (not should_pass and not verified):
                correct += 1
                print(f"  ✅ {description}")
            else:
                incorrect += 1
                print(f"  ❌ {description}")
        else:
            incorrect += 1
            print(f"  ❌ {description} (error)")
    
    total = len(test_results)
    print(f"\nTotal tests: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
    
    print("\n" + "=" * 70)
    if incorrect == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {incorrect} test(s) failed")
    print("=" * 70)

if __name__ == "__main__":
    main()

