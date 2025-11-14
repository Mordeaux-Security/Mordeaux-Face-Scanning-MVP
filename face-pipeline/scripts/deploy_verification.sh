#!/bin/bash
#
# Deployment Script for Verification-First Flow
# Usage: ./deploy_verification.sh [dev|staging|prod]
#
# This script automates the deployment of the verification-first flow.
# It performs pre-deployment checks, deploys code, verifies collections,
# and runs smoke tests.
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENV=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Environment-specific configs
case "$ENV" in
  dev)
    QDRANT_URL=${QDRANT_URL:-"http://localhost:6333"}
    API_URL=${API_URL:-"http://localhost:8001"}
    ;;
  staging)
    QDRANT_URL=${QDRANT_URL:-"http://qdrant-staging:6333"}
    API_URL=${API_URL:-"http://api-staging.example.com"}
    ;;
  prod)
    QDRANT_URL=${QDRANT_URL:-"http://qdrant-prod:6333"}
    API_URL=${API_URL:-"http://api-prod.example.com"}
    echo -e "${RED}WARNING: Deploying to PRODUCTION${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
      echo "Deployment cancelled."
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [dev|staging|prod]"
    exit 1
    ;;
esac

echo -e "${GREEN}=== Verification-First Flow Deployment ===${NC}"
echo "Environment: $ENV"
echo "Qdrant URL: $QDRANT_URL"
echo "API URL: $API_URL"
echo ""

# Functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

check_prereqs() {
  log_info "Checking prerequisites..."
  
  # Check if Python is available
  if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found"
    exit 1
  fi
  
  # Check if curl is available
  if ! command -v curl &> /dev/null; then
    log_error "curl not found"
    exit 1
  fi
  
  # Check if jq is available (optional, for JSON parsing)
  if ! command -v jq &> /dev/null; then
    log_warn "jq not found (optional, for JSON parsing)"
  fi
  
  log_info "Prerequisites check passed"
}

check_qdrant() {
  log_info "Checking Qdrant connectivity..."
  
  if ! curl -s -f "${QDRANT_URL}/health" > /dev/null; then
    log_error "Qdrant not accessible at ${QDRANT_URL}"
    exit 1
  fi
  
  log_info "Qdrant is accessible"
}

check_api() {
  log_info "Checking API connectivity..."
  
  if ! curl -s -f "${API_URL}/api/v1/health" > /dev/null; then
    log_warn "API not accessible at ${API_URL} (may not be deployed yet)"
    return 1
  fi
  
  log_info "API is accessible"
  return 0
}

check_collections() {
  log_info "Checking Qdrant collections..."
  
  # Get list of collections
  COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[]?.name // empty' 2>/dev/null || echo "")
  
  if [ -z "$COLLECTIONS" ]; then
    log_warn "Could not fetch collections (may need API key or jq not installed)"
    return 0
  fi
  
  # Check for faces_v1
  if echo "$COLLECTIONS" | grep -q "faces_v1"; then
    log_info "✓ faces_v1 collection exists"
  else
    log_warn "faces_v1 collection not found (will be created on startup)"
  fi
  
  # Check for identities_v1
  if echo "$COLLECTIONS" | grep -q "identities_v1"; then
    log_info "✓ identities_v1 collection exists"
  else
    log_warn "identities_v1 collection not found (will be created on startup)"
  fi
}

verify_collections_post_deploy() {
  log_info "Verifying collections after deployment..."
  
  # Wait for collections to be created (if needed)
  sleep 5
  
  # Get list of collections
  COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[]?.name // empty' 2>/dev/null || echo "")
  
  if [ -z "$COLLECTIONS" ]; then
    log_warn "Could not verify collections (may need API key)"
    return 0
  fi
  
  # Verify faces_v1 exists
  if echo "$COLLECTIONS" | grep -q "faces_v1"; then
    log_info "✓ faces_v1 collection verified"
  else
    log_error "faces_v1 collection not found"
    return 1
  fi
  
  # Verify identities_v1 exists
  if echo "$COLLECTIONS" | grep -q "identities_v1"; then
    log_info "✓ identities_v1 collection verified"
  else
    log_error "identities_v1 collection not found"
    return 1
  fi
  
  return 0
}

verify_indexes() {
  log_info "Verifying indexes..."
  
  # Check identities_v1 indexes
  IDENTITY_INDEXES=$(curl -s "${QDRANT_URL}/collections/identities_v1" | jq -r '.result.config.payload // {}' 2>/dev/null || echo "{}")
  
  if [ "$IDENTITY_INDEXES" != "{}" ]; then
    log_info "✓ identities_v1 indexes verified"
  else
    log_warn "Could not verify identities_v1 indexes (may be created lazily)"
  fi
}

test_enrollment() {
  log_info "Testing enrollment endpoint..."
  
  # Create a test request (minimal - will fail if images are invalid, but that's OK)
  TEST_REQUEST='{
    "tenant_id": "deployment-test",
    "identity_id": "deploy-test-'$(date +%s)'",
    "images_b64": ["data:image/jpeg;base64,/9j/4AAQSkZJRg=="]
  }'
  
  RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/api/v1/enroll_identity" \
    -H "Content-Type: application/json" \
    -d "$TEST_REQUEST")
  
  HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
  
  if [ "$HTTP_CODE" == "200" ] || [ "$HTTP_CODE" == "422" ]; then
    # 200 = success, 422 = validation error (expected with dummy data)
    log_info "✓ Enrollment endpoint responds (HTTP $HTTP_CODE)"
    return 0
  else
    log_error "Enrollment endpoint returned HTTP $HTTP_CODE"
    return 1
  fi
}

test_verification() {
  log_info "Testing verification endpoint..."
  
  # Create a test request (will fail with 404 if identity not enrolled, but that's OK)
  TEST_REQUEST='{
    "tenant_id": "deployment-test",
    "identity_id": "non-existent-'$(date +%s)'",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
  }'
  
  RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/api/v1/verify" \
    -H "Content-Type: application/json" \
    -d "$TEST_REQUEST")
  
  HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
  
  if [ "$HTTP_CODE" == "200" ] || [ "$HTTP_CODE" == "404" ] || [ "$HTTP_CODE" == "422" ]; then
    # 200 = success, 404 = not enrolled (expected), 422 = validation error (expected)
    log_info "✓ Verification endpoint responds (HTTP $HTTP_CODE)"
    return 0
  else
    log_error "Verification endpoint returned HTTP $HTTP_CODE"
    return 1
  fi
}

test_search() {
  log_info "Testing search endpoint (legacy)..."
  
  # Create a test request
  TEST_REQUEST='{
    "tenant_id": "deployment-test",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
    "top_k": 10
  }'
  
  RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/api/v1/search" \
    -H "Content-Type: application/json" \
    -d "$TEST_REQUEST")
  
  HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
  
  if [ "$HTTP_CODE" == "200" ] || [ "$HTTP_CODE" == "422" ]; then
    # 200 = success, 422 = validation error (expected with dummy data)
    log_info "✓ Search endpoint responds (HTTP $HTTP_CODE)"
    return 0
  else
    log_error "Search endpoint returned HTTP $HTTP_CODE"
    return 1
  fi
}

run_smoke_tests() {
  log_info "Running smoke tests..."
  
  TESTS_PASSED=0
  TESTS_FAILED=0
  
  # Test enrollment
  if test_enrollment; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
  
  # Test verification
  if test_verification; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
  
  # Test search
  if test_search; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
  
  log_info "Smoke tests: $TESTS_PASSED passed, $TESTS_FAILED failed"
  
  if [ $TESTS_FAILED -gt 0 ]; then
    log_error "Some smoke tests failed"
    return 1
  fi
  
  return 0
}

# Main deployment flow
main() {
  echo -e "${GREEN}Starting deployment to $ENV...${NC}"
  echo ""
  
  # Pre-deployment checks
  log_info "Phase 1: Pre-deployment checks"
  check_prereqs
  check_qdrant
  check_collections
  echo ""
  
  # Code deployment (placeholder - customize for your deployment method)
  log_info "Phase 2: Code deployment"
  log_warn "Code deployment step must be customized for your environment"
  log_warn "Options:"
  log_warn "  - Docker: docker-compose up -d"
  log_warn "  - Kubernetes: kubectl apply -f k8s/"
  log_warn "  - Direct: systemctl restart face-pipeline"
  echo ""
  
  # Wait for service to be ready
  log_info "Waiting for service to be ready..."
  sleep 10
  
  # Post-deployment verification
  log_info "Phase 3: Post-deployment verification"
  
  # Check API is up
  if ! check_api; then
    log_error "API not accessible after deployment"
    exit 1
  fi
  
  # Verify collections
  if ! verify_collections_post_deploy; then
    log_error "Collections verification failed"
    exit 1
  fi
  
  # Verify indexes
  verify_indexes
  echo ""
  
  # Smoke tests
  log_info "Phase 4: Smoke tests"
  if ! run_smoke_tests; then
    log_error "Smoke tests failed"
    exit 1
  fi
  echo ""
  
  # Success
  log_info "${GREEN}Deployment completed successfully!${NC}"
  echo ""
  log_info "Next steps:"
  log_info "  1. Monitor logs for any errors"
  log_info "  2. Check monitoring dashboards"
  log_info "  3. Run integration tests"
  log_info "  4. Verify with real user data (if safe)"
  echo ""
}

# Run main function
main

