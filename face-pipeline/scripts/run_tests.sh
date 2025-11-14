#!/bin/bash
# Auto-test script for persons 1, 2, 3, and 6
# Runs inside the container to access the API directly

set -e

cd /app
API_BASE="http://localhost:8001/api/v1"
TENANT_ID="test-tenant"
SAMPLES_DIR="samples"

echo "========================================"
echo "AUTOMATIC TEST: Verification-First Flow"
echo "Testing Persons 1, 2, 3, and 6"
echo "========================================"

# Helper function
encode_image() {
    local img=$1
    local mime="image/jpeg"
    [[ $img == *.png ]] && mime="image/png"
    echo "data:${mime};base64,$(base64 -w 0 < "$SAMPLES_DIR/$img" 2>/dev/null || base64 < "$SAMPLES_DIR/$img" | tr -d '\n')"
}

# Phase 1: Enrollment
echo ""
echo "========================================"
echo "PHASE 1: ENROLLMENT"
echo "========================================"

enroll() {
    local person=$1
    shift
    local photos=("$@")
    
    echo ""
    echo "📸 Enrolling $person with ${#photos[@]} photos..."
    
    local images_json="["
    for i in "${!photos[@]}"; do
        [ $i -gt 0 ] && images_json+=","
        local encoded=$(encode_image "${photos[$i]}")
        images_json+="\"${encoded}\""
    done
    images_json+="]"
    
    local response=$(curl -s -X POST "${API_BASE}/enroll_identity" \
        -H 'Content-Type: application/json' \
        -d "{
            \"tenant_id\": \"${TENANT_ID}\",
            \"identity_id\": \"${person}\",
            \"images_b64\": ${images_json}
        }")
    
    if echo "$response" | grep -q '"ok":true'; then
        echo "✅ $person enrolled successfully"
        return 0
    else
        echo "❌ $person enrollment failed"
        echo "   Response: $response"
        return 1
    fi
}

# Enroll all persons
enroll "person_1" "person1_A.jpg" "person1_B.jpeg" "person1_C.jpg"
enroll "person_2" "person2_A.jpg" "person2_B.jpg"
enroll "person_3" "person3_a.jpeg" "person3_b.jpg" "person3_C.jpg"
enroll "person_6" "person6_a.jpeg" "person6_b.jpeg" "person6_C.jpg" "person6_D.jpg"

echo ""
echo "⏳ Waiting 3 seconds..."
sleep 3

# Phase 2: Verification
echo ""
echo "========================================"
echo "PHASE 2: VERIFICATION TESTS"
echo "========================================"

verify() {
    local identity=$1
    local probe=$2
    local expected=$3  # true or false
    
    echo ""
    echo "🔍 Verifying $identity with $probe (expected: $expected)..."
    
    local probe_b64=$(encode_image "$probe")
    
    local response=$(curl -s -X POST "${API_BASE}/verify" \
        -H 'Content-Type: application/json' \
        -d "{
            \"tenant_id\": \"${TENANT_ID}\",
            \"identity_id\": \"${identity}\",
            \"image_b64\": \"${probe_b64}\",
            \"hi_threshold\": 0.78,
            \"top_k\": 50
        }")
    
    # Extract values
    local verified=$(echo "$response" | grep -o '"verified":[^,}]*' | cut -d: -f2 | tr -d ' "')
    local similarity=$(echo "$response" | grep -o '"similarity":[^,}]*' | cut -d: -f2 | tr -d ' "')
    local count=$(echo "$response" | grep -o '"count":[^,}]*' | cut -d: -f2 | tr -d ' "')
    
    if [ "$verified" = "true" ]; then
        echo "✅ Verification PASSED (similarity: $similarity)"
        echo "   Found $count faces"
        if [ "$expected" = "true" ]; then
            echo "   ✓ Correct: Same person passed"
            return 0
        else
            echo "   ❌ ERROR: False accept! Different person passed"
            return 1
        fi
    elif [ "$verified" = "false" ]; then
        echo "❌ Verification FAILED (similarity: $similarity)"
        echo "   Found $count faces (should be 0)"
        if [ "$count" = "0" ]; then
            echo "   ✓ Results array is empty (correct)"
        else
            echo "   ⚠️  WARNING: Results array has $count items!"
        fi
        if [ "$expected" = "false" ]; then
            echo "   ✓ Correct: Different person correctly rejected"
            return 0
        else
            echo "   ⚠️  Unexpected: Same person failed"
            return 1
        fi
    else
        echo "❌ Error parsing response: $response"
        return 1
    fi
}

# Run tests
CORRECT=0
INCORRECT=0

verify "person_1" "person1_A.jpg" "true" && ((CORRECT++)) || ((INCORRECT++))
verify "person_1" "person2_A.jpg" "false" && ((CORRECT++)) || ((INCORRECT++))
verify "person_2" "person2_A.jpg" "true" && ((CORRECT++)) || ((INCORRECT++))
verify "person_2" "person3_a.jpeg" "false" && ((CORRECT++)) || ((INCORRECT++))
verify "person_3" "person3_a.jpeg" "true" && ((CORRECT++)) || ((INCORRECT++))
verify "person_3" "person6_a.jpeg" "false" && ((CORRECT++)) || ((INCORRECT++))
verify "person_6" "person6_a.jpeg" "true" && ((CORRECT++)) || ((INCORRECT++))
verify "person_6" "person1_A.jpg" "false" && ((CORRECT++)) || ((INCORRECT++))

# Summary
echo ""
echo "========================================"
echo "TEST SUMMARY"
echo "========================================"
TOTAL=$((CORRECT + INCORRECT))
echo "Total tests: $TOTAL"
echo "✅ Correct: $CORRECT ($(awk "BEGIN {printf \"%.1f\", ($CORRECT/$TOTAL)*100}")%)"
echo "❌ Incorrect: $INCORRECT ($(awk "BEGIN {printf \"%.1f\", ($INCORRECT/$TOTAL)*100}")%)"

if [ $INCORRECT -eq 0 ]; then
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    echo "✓ Verification-first flow is working correctly"
    echo "✓ No false accepts detected"
    exit 0
else
    echo ""
    echo "⚠️  $INCORRECT test(s) had issues"
    exit 1
fi

