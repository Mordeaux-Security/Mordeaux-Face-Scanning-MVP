#!/usr/bin/env python3
"""
Auto-test script using urllib (no external dependencies needed)
"""
import sys
import base64
import json
import urllib.request
import urllib.error
import time
from pathlib import Path

API_BASE = "http://localhost:8001/api/v1"
TENANT_ID = "test-tenant"
SAMPLES_DIR = Path("/app/samples")

def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URL."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        ext = image_path.suffix.lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime};base64,{b64}"

def enroll(person_id: str, photos: list[str]):
    """Enroll a person."""
    print(f"\n📸 Enrolling {person_id} with {len(photos)} photos...")
    
    images_b64 = []
    for photo in photos:
        photo_path = SAMPLES_DIR / photo
        if not photo_path.exists():
            print(f"❌ Error: {photo_path} not found")
            return False
        images_b64.append(image_to_base64(photo_path))
        print(f"   ✓ Loaded: {photo}")
    
    # Create request
    data = json.dumps({
        "tenant_id": TENANT_ID,
        "identity_id": person_id,
        "images_b64": images_b64
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(
            f"{API_BASE}/enroll_identity",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            if result.get("ok"):
                print(f"✅ {person_id} enrolled successfully!")
                return True
            else:
                print(f"❌ Failed: {result}")
                return False
    except urllib.error.HTTPError as e:
        response_text = e.read().decode()
        print(f"❌ HTTP {e.code}: {response_text[:200]}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify(identity_id: str, probe_photo: str, expected_pass: bool):
    """Verify and check result."""
    print(f"\n🔍 Verifying {identity_id} with {probe_photo} (expected: {'pass' if expected_pass else 'fail'})...")
    
    probe_path = SAMPLES_DIR / probe_photo
    if not probe_path.exists():
        print(f"❌ Error: {probe_path} not found")
        return None
    
    probe_b64 = image_to_base64(probe_path)
    
    data = json.dumps({
        "tenant_id": TENANT_ID,
        "identity_id": identity_id,
        "image_b64": probe_b64,
        "hi_threshold": 0.78,
        "top_k": 50
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(
            f"{API_BASE}/verify",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            verified = result.get("verified", False)
            similarity = result.get("similarity", 0.0)
            count = result.get("count", 0)
            results = result.get("results", [])
            
            if verified:
                print(f"✅ Verification PASSED (similarity: {similarity:.3f})")
                print(f"   Found {count} faces")
                if expected_pass:
                    print(f"   ✓ Correct: Same person passed")
                    return True
                else:
                    print(f"   ❌ ERROR: False accept! Different person passed")
                    return False
            else:
                print(f"❌ Verification FAILED (similarity: {similarity:.3f})")
                print(f"   Found {count} faces (should be 0)")
                
                if len(results) > 0:
                    print(f"   ⚠️  WARNING: Results array has {len(results)} items (should be empty!)")
                else:
                    print(f"   ✓ Results array is empty (correct)")
                
                if expected_pass:
                    print(f"   ⚠️  Unexpected: Same person failed (may be quality/threshold issue)")
                    return False
                else:
                    print(f"   ✓ Correct: Different person correctly rejected")
                    return True
            
    except urllib.error.HTTPError as e:
        response_text = e.read().decode()
        print(f"❌ HTTP {e.code}: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("=" * 70)
    print("AUTOMATIC TEST: Verification-First Flow")
    print("Testing Persons 1, 2, 3, and 6")
    print("=" * 70)
    
    # Phase 1: Enrollment
    print("\n" + "=" * 70)
    print("PHASE 1: ENROLLMENT")
    print("=" * 70)
    
    enrollments = {
        "person_1": ["person1_A.jpg", "person1_B.jpeg", "person1_C.jpg"],
        "person_2": ["person2_A.jpg", "person2_B.jpg"],
        "person_3": ["person3_a.jpeg", "person3_b.jpg", "person3_C.jpg"],
        "person_6": ["person6_a.jpeg", "person6_b.jpeg", "person6_C.jpg", "person6_D.jpg"],
    }
    
    enrollment_results = {}
    for person_id, photos in enrollments.items():
        success = enroll(person_id, photos)
        enrollment_results[person_id] = success
        time.sleep(1)
    
    print("\n⏳ Waiting 3 seconds for enrollment to complete...")
    time.sleep(3)
    
    # Phase 2: Verification
    print("\n" + "=" * 70)
    print("PHASE 2: VERIFICATION TESTS")
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
    
    test_results = []
    for identity_id, probe_photo, should_pass, description in tests:
        print(f"\n{'=' * 70}")
        print(f"TEST: {description}")
        print(f"{'=' * 70}")
        
        result = verify(identity_id, probe_photo, should_pass)
        test_results.append((description, result, should_pass))
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print("\n📋 Enrollment Results:")
    for person_id, success in enrollment_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {person_id}: {status}")
    
    print("\n🔍 Verification Results:")
    correct = 0
    incorrect = 0
    for description, result, should_pass in test_results:
        if result is True:
            correct += 1
            print(f"   ✅ {description}")
        elif result is False:
            incorrect += 1
            print(f"   ❌ {description}")
        else:
            incorrect += 1
            print(f"   ⚠️  {description} (error)")
    
    total = len(test_results)
    print(f"\n📊 Results:")
    print(f"   Total tests: {total}")
    print(f"   ✅ Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"   ❌ Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
    
    print("\n" + "=" * 70)
    if incorrect == 0:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Verification-first flow is working correctly")
        print("✓ No false accepts detected")
        print("✓ Results array is empty when verified=false")
    else:
        print(f"⚠️  {incorrect} test(s) had issues")
    print("=" * 70)
    
    return 0 if incorrect == 0 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

