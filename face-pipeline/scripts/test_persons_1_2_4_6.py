#!/usr/bin/env python3
"""
Test persons 1, 2, 4, and 6 to prevent cross-facial recognition
Reports which faces are matched with which
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
        
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode())
            if result.get("ok"):
                print(f"✅ {person_id} enrolled successfully!")
                return True
            else:
                print(f"❌ Failed: {result}")
                return False
    except urllib.error.HTTPError as e:
        response_text = e.read().decode()
        print(f"❌ HTTP {e.code}: {response_text[:300]}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify(identity_id: str, probe_photo: str, expected_pass: bool):
    """Verify and check result - returns similarity and whether verification passed."""
    print(f"\n🔍 Verifying {identity_id} with {probe_photo} (expected: {'PASS' if expected_pass else 'FAIL'})...")
    
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
                print(f"✅ Verification PASSED")
                print(f"   Similarity: {similarity:.4f} (threshold: 0.78)")
                print(f"   Found {count} matching faces")
                if expected_pass:
                    print(f"   ✓ CORRECT: Same person passed verification")
                    return {"verified": True, "similarity": similarity, "correct": True, "count": count}
                else:
                    print(f"   ❌ FALSE ACCEPT! Different person incorrectly passed!")
                    return {"verified": True, "similarity": similarity, "correct": False, "count": count}
            else:
                print(f"❌ Verification FAILED")
                print(f"   Similarity: {similarity:.4f} (threshold: 0.78)")
                print(f"   Found {count} matching faces (should be 0)")
                
                if len(results) > 0:
                    print(f"   ⚠️  WARNING: Results array has {len(results)} items (should be empty!)")
                else:
                    print(f"   ✓ Results array is empty (correct)")
                
                if expected_pass:
                    print(f"   ⚠️  UNEXPECTED: Same person failed (threshold may be too high)")
                    return {"verified": False, "similarity": similarity, "correct": False, "count": count}
                else:
                    print(f"   ✓ CORRECT: Different person correctly rejected")
                    return {"verified": False, "similarity": similarity, "correct": True, "count": count}
            
    except urllib.error.HTTPError as e:
        response_text = e.read().decode()
        print(f"❌ HTTP {e.code}: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("=" * 80)
    print("CROSS-FACIAL RECOGNITION PREVENTION TEST")
    print("Testing Persons 1, 2, 4, and 6")
    print("=" * 80)
    
    # Phase 1: Enrollment
    print("\n" + "=" * 80)
    print("PHASE 1: ENROLLMENT")
    print("=" * 80)
    
    enrollments = {
        "person_1": ["person1_A.jpg", "person1_B.jpeg", "person1_C.jpg"],
        "person_2": ["person2_A.jpg", "person2_B.jpg"],
        "person_4": ["person4_a.jpg", "person4_b.jpg", "person4_c.jpg"],
        "person_6": ["person6_a.jpeg", "person6_b.jpeg", "person6_C.jpg", "person6_D.jpg"],
    }
    
    enrollment_results = {}
    for person_id, photos in enrollments.items():
        success = enroll(person_id, photos)
        enrollment_results[person_id] = success
        time.sleep(1)
    
    print("\n⏳ Waiting 3 seconds for enrollment to complete...")
    time.sleep(3)
    
    # Phase 2: Verification Tests - Create a comprehensive matrix
    print("\n" + "=" * 80)
    print("PHASE 2: CROSS-FACIAL RECOGNITION TESTS")
    print("Testing all person combinations to detect false accepts")
    print("=" * 80)
    
    # Define all test cases
    tests = [
        # Same person tests (should PASS)
        ("person_1", "person1_A.jpg", True, "Person 1 → Person 1 photo A"),
        ("person_2", "person2_A.jpg", True, "Person 2 → Person 2 photo A"),
        ("person_4", "person4_a.jpg", True, "Person 4 → Person 4 photo a"),
        ("person_6", "person6_a.jpeg", True, "Person 6 → Person 6 photo a"),
        
        # Cross-person tests (should FAIL - prevent cross recognition)
        ("person_1", "person2_A.jpg", False, "Person 1 → Person 2 (cross-match)"),
        ("person_1", "person4_a.jpg", False, "Person 1 → Person 4 (cross-match)"),
        ("person_1", "person6_a.jpeg", False, "Person 1 → Person 6 (cross-match)"),
        
        ("person_2", "person1_A.jpg", False, "Person 2 → Person 1 (cross-match)"),
        ("person_2", "person4_a.jpg", False, "Person 2 → Person 4 (cross-match)"),
        ("person_2", "person6_a.jpeg", False, "Person 2 → Person 6 (cross-match)"),
        
        ("person_4", "person1_A.jpg", False, "Person 4 → Person 1 (cross-match)"),
        ("person_4", "person2_A.jpg", False, "Person 4 → Person 2 (cross-match)"),
        ("person_4", "person6_a.jpeg", False, "Person 4 → Person 6 (cross-match)"),
        
        ("person_6", "person1_A.jpg", False, "Person 6 → Person 1 (cross-match)"),
        ("person_6", "person2_A.jpg", False, "Person 6 → Person 2 (cross-match)"),
        ("person_6", "person4_a.jpg", False, "Person 6 → Person 4 (cross-match)"),
    ]
    
    test_results = []
    for identity_id, probe_photo, should_pass, description in tests:
        print(f"\n{'=' * 80}")
        print(f"TEST: {description}")
        print(f"{'=' * 80}")
        
        result = verify(identity_id, probe_photo, should_pass)
        if result:
            test_results.append((description, result, should_pass))
        else:
            test_results.append((description, None, should_pass))
        time.sleep(0.3)
    
    # Summary
    print("\n" + "=" * 80)
    print("DETAILED TEST SUMMARY")
    print("=" * 80)
    
    print("\n📋 Enrollment Results:")
    for person_id, success in enrollment_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {person_id}: {status}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION MATRIX")
    print("=" * 80)
    
    print("\n🟢 Same-Person Tests (should all PASS):")
    same_person_correct = 0
    same_person_total = 0
    for description, result, should_pass in test_results:
        if should_pass:
            same_person_total += 1
            if result and result['correct']:
                same_person_correct += 1
                sim = result['similarity']
                print(f"   ✅ {description} - Similarity: {sim:.4f}")
            elif result:
                sim = result['similarity']
                print(f"   ⚠️  {description} - Similarity: {sim:.4f} (FAILED - threshold too high?)")
            else:
                print(f"   ❌ {description} (ERROR)")
    
    print(f"\n   Same-person accuracy: {same_person_correct}/{same_person_total} " +
          f"({same_person_correct/same_person_total*100:.1f}%)")
    
    print("\n🔴 Cross-Person Tests (should all FAIL to prevent cross-recognition):")
    cross_person_correct = 0
    cross_person_total = 0
    false_accepts = []
    
    for description, result, should_pass in test_results:
        if not should_pass:
            cross_person_total += 1
            if result and result['correct']:
                cross_person_correct += 1
                sim = result['similarity']
                print(f"   ✅ {description} - Similarity: {sim:.4f} (Correctly rejected)")
            elif result and not result['correct']:
                sim = result['similarity']
                print(f"   ❌ {description} - Similarity: {sim:.4f} (FALSE ACCEPT!)")
                false_accepts.append((description, sim))
            else:
                print(f"   ⚠️  {description} (ERROR)")
    
    print(f"\n   Cross-person rejection accuracy: {cross_person_correct}/{cross_person_total} " +
          f"({cross_person_correct/cross_person_total*100:.1f}%)")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    total_correct = same_person_correct + cross_person_correct
    total_tests = same_person_total + cross_person_total
    
    print(f"\n📊 Overall Accuracy: {total_correct}/{total_tests} ({total_correct/total_tests*100:.1f}%)")
    print(f"   ✅ Same-person matches: {same_person_correct}/{same_person_total}")
    print(f"   ✅ Cross-person rejections: {cross_person_correct}/{cross_person_total}")
    
    if false_accepts:
        print(f"\n❌ CRITICAL: {len(false_accepts)} FALSE ACCEPT(S) DETECTED!")
        print("   These are cases where different people were incorrectly matched:")
        for desc, sim in false_accepts:
            print(f"   - {desc} (similarity: {sim:.4f})")
        print("\n   ⚠️  Cross-facial recognition is NOT prevented with current threshold!")
        print("   💡 Consider raising the threshold above 0.78 or retraining with better quality images.")
    else:
        print("\n✅ SUCCESS: No false accepts detected!")
        print("   Cross-facial recognition is successfully prevented.")
        print("   The verification-first flow is working correctly.")
    
    print("\n" + "=" * 80)
    
    return 0 if not false_accepts and total_correct == total_tests else 1

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

