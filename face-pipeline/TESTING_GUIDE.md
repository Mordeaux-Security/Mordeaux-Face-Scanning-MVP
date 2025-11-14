# Verification-First Flow: Testing Guide

**Ready for full testing with real photos!**

---

## 🧪 Testing Checklist

### Pre-Testing Setup

- [ ] Service is running and healthy (`/api/v1/health`)
- [ ] `identities_v1` collection exists in Qdrant
- [ ] Test tenant ID ready (e.g., "test-tenant" or "demo")
- [ ] Have photos ready for each person:
  - Person A: 3-5 photos (frontal + angles)
  - Person B: 3-5 photos (frontal + angles)
  - Person C: 3-5 photos (frontal + angles)
  - Probe photos for verification

---

## 📋 Test Scenarios

### 1. Enroll Person A

**Steps:**
1. Enroll Person A with 3-5 photos
2. Verify enrollment succeeded
3. Check `identities_v1` collection

**Expected Result:**
```json
{
  "ok": true,
  "identity": {
    "tenant_id": "test-tenant",
    "identity_id": "person_a"
  },
  "vector_dim": 512
}
```

**Verify in Qdrant:**
```bash
curl -s http://localhost:6333/collections/identities_v1/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "identity_id", "match": {"value": "person_a"}}]}}'
```

### 2. Enroll Person B

**Steps:**
1. Enroll Person B with 3-5 photos
2. Verify enrollment succeeded
3. Verify Person A still exists (no overwrite)

**Expected:** Both Person A and Person B exist in `identities_v1`

### 3. Enroll Person C

**Steps:**
1. Enroll Person C with 3-5 photos
2. Verify all three identities exist

**Expected:** Person A, B, and C all exist independently

---

## ✅ Verification Tests

### Test 4: Verify Person A with Person A's Photo (Should Pass)

```bash
curl -s -X POST http://localhost/pipeline/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "test-tenant",
    "identity_id": "person_a",
    "image_b64": "DATA_FOR_PERSON_A_PROBE",
    "hi_threshold": 0.78,
    "top_k": 50
  }'
```

**Expected Result:**
- `verified: true`
- `similarity: >= 0.78` (should be 0.80-0.95 for same person)
- `results: [...]` (faces belonging to Person A, if any exist)

### Test 5: Verify Person A with Person B's Photo (Should Fail)

```bash
curl -s -X POST http://localhost/pipeline/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "test-tenant",
    "identity_id": "person_a",
    "image_b64": "DATA_FOR_PERSON_B_PROBE",
    "hi_threshold": 0.78,
    "top_k": 50
  }'
```

**Expected Result:**
- `verified: false`
- `similarity: < 0.78` (should be 0.50-0.75 for different person)
- `results: []` (empty array - **critical!**)

**✅ Key Success Criteria:** `results` is empty when verified=false

### Test 6: Verify Person B with Person A's Photo (Should Fail)

Same as Test 5, but reversed.

**Expected:** `verified: false`, `results: []`

### Test 7: Verify Person C with Person C's Photo (Should Pass)

**Expected:** `verified: true`, `similarity >= 0.78`

---

## 🔍 Edge Cases to Test

### Test 8: Non-Existent Identity

```bash
curl -s -X POST http://localhost/pipeline/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "test-tenant",
    "identity_id": "non_existent",
    "image_b64": "DATA",
    "hi_threshold": 0.78
  }'
```

**Expected:** HTTP 404 with `{"detail": "identity_not_enrolled"}`

### Test 9: Threshold Edge Cases

**Test with threshold = 0.76 (lower):**
- Same person should still pass (similarity should be higher)
- May catch some borderline cases

**Test with threshold = 0.80 (higher):**
- Same person should still pass (if good quality)
- More strict verification

### Test 10: Poor Quality Photo

**Test with blurry/off-angle photo:**
- May fail verification even for same person
- Expected behavior: `verified: false` with low similarity

---

## 📊 What to Monitor During Testing

### Success Metrics

1. **Enrollment Success Rate**
   - Target: 100% (all enrollments should succeed)
   - Monitor: Check for any 422/500 errors

2. **Verification Pass Rate (Same Person)**
   - Target: 85-95% (some variation expected)
   - Good photos: Should pass consistently
   - Poor photos: May fail (acceptable)

3. **Verification Reject Rate (Different Person)**
   - Target: 100% (all different-person verifications should fail)
   - **Critical:** No false accepts (Person B should never verify as Person A)

4. **Similarity Score Distribution**
   - Same person: Usually 0.80-0.95
   - Different person: Usually 0.50-0.75
   - Very different: < 0.60

### Key Verification Points

✅ **Person A does NOT show up for Person B**
- When verifying Person B with Person A's photo
- When verifying Person A with Person B's photo
- `results` should be empty in both cases

✅ **Only correct identity's faces are returned**
- When `verified: true`, only that person's faces
- No cross-contamination between identities

---

## 🐛 Troubleshooting

### Issue: Enrollment Returns 422 "no_face_detected"

**Possible Causes:**
- Image doesn't contain a detectable face
- Image is too blurry/low quality
- Face too small in image

**Solutions:**
- Use clearer photos
- Ensure face is clearly visible
- Try different photos

### Issue: Verification Always Returns `verified: false`

**Possible Causes:**
- Threshold too high (0.78 might be strict)
- Photo quality differences between enrollment and verification
- Different lighting/angles

**Solutions:**
- Try lowering threshold to 0.76
- Ensure verification photo matches enrollment quality
- Use similar lighting conditions

### Issue: Verification Returns `verified: true` for Different Person

**Possible Causes:**
- Threshold too low (< 0.76)
- Very similar-looking people
- Photo quality issues

**Solutions:**
- Increase threshold to 0.80
- Improve enrollment quality (more diverse angles)
- Review similarity scores (should be > 0.80 for same person)

### Issue: `results` Array Contains Wrong Person's Faces

**Critical Issue!** This should not happen.

**Possible Causes:**
- Faces not tagged with `identity_id` during ingestion
- Filtering not working correctly

**Debug:**
```bash
# Check if faces are tagged with identity_id
curl -s http://localhost:6333/collections/faces_v1/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "tenant_id", "match": {"value": "test-tenant"}}]}}' \
  | jq '.result.points[] | {id, payload: {tenant_id, identity_id}}'
```

---

## 📝 Testing Log Template

Use this template to log your tests:

```
Date: ___________
Tester: ___________

=== Enrollment Tests ===
[ ] Person A enrolled: ☐ Success ☐ Failed
[ ] Person B enrolled: ☐ Success ☐ Failed
[ ] Person C enrolled: ☐ Success ☐ Failed

=== Verification Tests ===
[ ] Person A → Person A: ☐ Passed (verified=true) ☐ Failed
  Similarity: _____
  Results count: _____

[ ] Person A → Person B: ☐ Rejected (verified=false) ☐ Passed (ERROR!)
  Similarity: _____
  Results count: _____ (should be 0)

[ ] Person B → Person A: ☐ Rejected (verified=false) ☐ Passed (ERROR!)
  Similarity: _____
  Results count: _____ (should be 0)

[ ] Person B → Person B: ☐ Passed (verified=true) ☐ Failed
  Similarity: _____
  Results count: _____

[ ] Person C → Person C: ☐ Passed (verified=true) ☐ Failed
  Similarity: _____
  Results count: _____

=== Key Findings ===
1. False accept rate: _____ (should be 0%)
2. False reject rate: _____ (acceptable: 5-15%)
3. Average similarity (same person): _____
4. Average similarity (different person): _____

=== Issues Encountered ===
1. _____________________________________
2. _____________________________________
3. _____________________________________

=== Recommendations ===
1. _____________________________________
2. _____________________________________
```

---

## 🎯 Quick Test Commands

### Enroll Multiple People (Batch)

```bash
# Enroll Person A
curl -s -X POST http://localhost/pipeline/api/v1/enroll_identity \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"test","identity_id":"person_a","images_b64":[...]}' | jq

# Enroll Person B  
curl -s -X POST http://localhost/pipeline/api/v1/enroll_identity \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"test","identity_id":"person_b","images_b64":[...]}' | jq

# Enroll Person C
curl -s -X POST http://localhost/pipeline/api/v1/enroll_identity \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"test","identity_id":"person_c","images_b64":[...]}' | jq
```

### Verify All Combinations

```bash
# Test matrix
for person in a b c; do
  for probe in a b c; do
    echo "Testing Person $person with Person $probe photo:"
    curl -s -X POST http://localhost/pipeline/api/v1/verify \
      -H 'Content-Type: application/json' \
      -d "{\"tenant_id\":\"test\",\"identity_id\":\"person_$person\",\"image_b64\":\"PROBE_$probe\",\"hi_threshold\":0.78}" \
      | jq '{verified, similarity, count}'
    echo ""
  done
done
```

---

## ✅ Success Criteria

Your testing is successful if:

1. ✅ All enrollments succeed (100%)
2. ✅ Same-person verifications pass (85-95%)
3. ✅ Different-person verifications fail (100%)
4. ✅ No false accepts (Person B never verifies as Person A)
5. ✅ Results array is empty when `verified=false`
6. ✅ Results array contains only correct identity's faces when `verified=true`

---

## 🚀 Ready to Test!

When you're ready with your photos, let me know if you encounter any issues. I can help with:

- Debugging enrollment problems
- Analyzing verification results
- Adjusting thresholds if needed
- Checking Qdrant collections
- Verifying face tagging
- Interpreting similarity scores

**Good luck with testing!** 🎉

