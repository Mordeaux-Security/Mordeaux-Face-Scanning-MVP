#!/bin/bash
#
# Test script for enrolling and verifying persons 1, 2, 3, and 6
# Uses curl and jq for testing
#

API_BASE="http://localhost/pipeline/api/v1"
TENANT_ID="test-tenant"
SAMPLES_DIR="samples"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Verification-First Flow: Testing Persons 1, 2, 3, and 6 ===${NC}"
echo ""

# Helper function to encode image to base64
encode_image() {
    local img=$1
    local mime="image/jpeg"
    if [[ $img == *.png ]]; then
        mime="image/png"
    fi
    echo "data:${mime};base64,$(base64 -i "$SAMPLES_DIR/$img" | tr -d '\n')"
}

# Function to enroll a person
enroll() {
    local person=$1
    shift
    local photos=("$@")
    
    echo -e "${BLUE}📸 Enrolling ${person} with ${#photos[@]} photos...${NC}"
    
    # Build JSON array of base64 images
    local images_json="["
    for i in "${!photos[@]}"; do
        if [ $i -gt 0 ]; then
            images_json+=","
        fi
        local encoded=$(encode_image "${photos[$i]}")
        images_json+="\"${encoded}\""
    done
    images_json+="]"
    
    # Enroll via API
    local response=$(curl -s -X POST "${API_BASE}/enroll_identity" \
        -H 'Content-Type: application/json' \
        -d "{
            \"tenant_id\": \"${TENANT_ID}\",
            \"identity_id\": \"${person}\",
            \"images_b64\": ${images_json}
        }")
    
    if echo "$response" | grep -q '"ok":true'; then
        echo -e "${GREEN}✅ ${person} enrolled successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ ${person} enrollment failed${NC}"
        echo "   Response: $response"
        return 1
    fi
}

# Function to verify
verify() {
    local identity=$1
    local probe=$2
    local threshold=${3:-0.78}
    
    echo -e "${BLUE}🔍 Verifying ${identity} with ${probe}...${NC}"
    
    local probe_b64=$(encode_image "$probe")
    
    local response=$(curl -s -X POST "${API_BASE}/verify" \
        -H 'Content-Type: application/json' \
        -d "{
            \"tenant_id\": \"${TENANT_ID}\",
            \"identity_id\": \"${identity}\",
            \"image_b64\": \"${probe_b64}\",
            \"hi_threshold\": ${threshold},
            \"top_k\": 50
        }")
    
    # Check if jq is available
    if command -v jq &> /dev/null; then
        local verified=$(echo "$response" | jq -r '.verified // "null"')
        local similarity=$(echo "$response" | jq -r '.similarity // 0')
        local count=$(echo "$response" | jq -r '.count // 0')
        
        if [ "$verified" = "true" ]; then
            echo -e "${GREEN}✅ Verification PASSED${NC}"
            echo "   Similarity: $similarity (>= $threshold)"
            echo "   Found $count faces"
        elif [ "$verified" = "false" ]; then
            echo -e "${YELLOW}❌ Verification FAILED${NC}"
            echo "   Similarity: $similarity (< $threshold)"
            echo "   Found $count faces (should be 0)"
        else
            echo -e "${RED}❌ Verification error${NC}"
            echo "   Response: $response"
        fi
    else
        # Fallback if jq not available
        if echo "$response" | grep -q '"verified":true'; then
            echo -e "${GREEN}✅ Verification PASSED${NC}"
        elif echo "$response" | grep -q '"verified":false'; then
            echo -e "${YELLOW}❌ Verification FAILED${NC}"
        else
            echo -e "${RED}❌ Verification error${NC}"
            echo "   Response: $response"
        fi
    fi
    
    return 0
}

# Change to face-pipeline directory
cd "$(dirname "$0")/.." || exit 1

# Phase 1: Enrollment
echo -e "${BLUE}=== Phase 1: Enrollment ===${NC}"
echo ""

enroll "person_1" "person1_A.jpg" "person1_B.jpeg" "person1_C.jpg"
echo ""

enroll "person_2" "person2_A.jpg" "person2_B.jpg"
echo ""

enroll "person_3" "person3_a.jpeg" "person3_b.jpg" "person3_C.jpg"
echo ""

enroll "person_6" "person6_a.jpeg" "person6_b.jpeg" "person6_C.jpg" "person6_D.jpg"
echo ""

# Wait a moment
sleep 2

# Phase 2: Verification Tests
echo -e "${BLUE}=== Phase 2: Verification Tests ===${NC}"
echo ""

echo -e "${BLUE}Test 1: Person 1 → Person 1 (should pass)${NC}"
verify "person_1" "person1_A.jpg" 0.78
echo ""

echo -e "${BLUE}Test 2: Person 1 → Person 2 (should fail)${NC}"
verify "person_1" "person2_A.jpg" 0.78
echo ""

echo -e "${BLUE}Test 3: Person 2 → Person 2 (should pass)${NC}"
verify "person_2" "person2_A.jpg" 0.78
echo ""

echo -e "${BLUE}Test 4: Person 2 → Person 3 (should fail)${NC}"
verify "person_2" "person3_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 5: Person 3 → Person 3 (should pass)${NC}"
verify "person_3" "person3_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 6: Person 3 → Person 6 (should fail)${NC}"
verify "person_3" "person6_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 7: Person 6 → Person 6 (should pass)${NC}"
verify "person_6" "person6_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 8: Person 6 → Person 1 (should fail)${NC}"
verify "person_6" "person1_A.jpg" 0.78
echo ""

echo -e "${GREEN}=== Testing Complete ===${NC}"
echo ""
echo "Summary:"
echo "- Same person verifications should PASS (verified=true)"
echo "- Different person verifications should FAIL (verified=false, results=[])"

