#!/bin/bash
#
# Test script for enrolling and verifying persons 1, 2, 3, and 6
#

set -e

API_BASE="http://localhost/pipeline/api/v1"
TENANT_ID="test-tenant"
SAMPLES_DIR="face-pipeline/samples"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Verification-First Flow: Testing Persons 1, 2, 3, and 6 ===${NC}"
echo ""

# Function to convert image to base64
img_to_b64() {
    local img_path=$1
    if [ ! -f "$img_path" ]; then
        echo "" > /dev/stderr
        echo -e "${RED}Error: File not found: $img_path${NC}" > /dev/stderr
        return 1
    fi
    
    # Detect MIME type from extension
    local ext="${img_path##*.}"
    case "$ext" in
        jpg|jpeg) echo "data:image/jpeg;base64,$(base64 -w 0 < "$img_path")" ;;
        png) echo "data:image/png;base64,$(base64 -w 0 < "$img_path")" ;;
        *) echo "data:image/jpeg;base64,$(base64 -w 0 < "$img_path")" ;;
    esac
}

# Function to enroll a person
enroll_person() {
    local person_id=$1
    shift
    local photos=("$@")
    
    echo -e "${BLUE}📸 Enrolling ${person_id} with ${#photos[@]} photos...${NC}"
    
    # Convert photos to base64
    local images_b64="["
    local first=true
    for photo in "${photos[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            images_b64+=","
        fi
        local b64=$(img_to_b64 "$SAMPLES_DIR/$photo")
        images_b64+="\"$b64\""
    done
    images_b64+="]"
    
    # Enroll
    local response=$(curl -s -X POST "$API_BASE/enroll_identity" \
        -H 'Content-Type: application/json' \
        -d "{
            \"tenant_id\": \"$TENANT_ID\",
            \"identity_id\": \"$person_id\",
            \"images_b64\": $images_b64
        }")
    
    if echo "$response" | grep -q '"ok":true'; then
        echo -e "${GREEN}✅ ${person_id} enrolled successfully${NC}"
        echo "   $response" | jq '.' 2>/dev/null || echo "   $response"
        return 0
    else
        echo -e "${RED}❌ ${person_id} enrollment failed${NC}"
        echo "   $response" | jq '.' 2>/dev/null || echo "   $response"
        return 1
    fi
}

# Function to verify a person
verify_person() {
    local identity_id=$1
    local probe_photo=$2
    local threshold=${3:-0.78}
    
    echo -e "${BLUE}🔍 Verifying ${identity_id} with ${probe_photo}...${NC}"
    
    local probe_b64=$(img_to_b64 "$SAMPLES_DIR/$probe_photo")
    
    local response=$(curl -s -X POST "$API_BASE/verify" \
        -H 'Content-Type: application/json' \
        -d "{
            \"tenant_id\": \"$TENANT_ID\",
            \"identity_id\": \"$identity_id\",
            \"image_b64\": \"$probe_b64\",
            \"hi_threshold\": $threshold,
            \"top_k\": 50
        }")
    
    local verified=$(echo "$response" | jq -r '.verified' 2>/dev/null || echo "null")
    local similarity=$(echo "$response" | jq -r '.similarity' 2>/dev/null || echo "0")
    local count=$(echo "$response" | jq -r '.count' 2>/dev/null || echo "0")
    
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
        echo "   $response" | jq '.' 2>/dev/null || echo "   $response"
    fi
    
    return 0
}

# Enroll all persons
echo -e "${BLUE}=== Phase 1: Enrollment ===${NC}"
echo ""

enroll_person "person_1" "person1_A.jpg" "person1_B.jpeg" "person1_C.jpg"
echo ""

enroll_person "person_2" "person2_A.jpg" "person2_B.jpg"
echo ""

enroll_person "person_3" "person3_a.jpeg" "person3_b.jpg" "person3_C.jpg"
echo ""

enroll_person "person_6" "person6_a.jpeg" "person6_b.jpeg" "person6_C.jpg" "person6_D.jpg"
echo ""

# Wait a moment for enrollment to complete
sleep 2

# Test verification matrix
echo -e "${BLUE}=== Phase 2: Verification Tests ===${NC}"
echo ""

echo -e "${BLUE}Test 1: Person 1 → Person 1 (should pass)${NC}"
verify_person "person_1" "person1_A.jpg" 0.78
echo ""

echo -e "${BLUE}Test 2: Person 1 → Person 2 (should fail)${NC}"
verify_person "person_1" "person2_A.jpg" 0.78
echo ""

echo -e "${BLUE}Test 3: Person 2 → Person 2 (should pass)${NC}"
verify_person "person_2" "person2_A.jpg" 0.78
echo ""

echo -e "${BLUE}Test 4: Person 2 → Person 3 (should fail)${NC}"
verify_person "person_2" "person3_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 5: Person 3 → Person 3 (should pass)${NC}"
verify_person "person_3" "person3_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 6: Person 3 → Person 6 (should fail)${NC}"
verify_person "person_3" "person6_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 7: Person 6 → Person 6 (should pass)${NC}"
verify_person "person_6" "person6_a.jpeg" 0.78
echo ""

echo -e "${BLUE}Test 8: Person 6 → Person 1 (should fail)${NC}"
verify_person "person_6" "person1_A.jpg" 0.78
echo ""

echo -e "${GREEN}=== Testing Complete ===${NC}"
echo ""
echo "Summary:"
echo "- Same person verifications should PASS (verified=true)"
echo "- Different person verifications should FAIL (verified=false, results=[])"

