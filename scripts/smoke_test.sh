#!/bin/bash
# Mordeaux Face Scanning MVP - Proxy Smoke Tests
# Tests Nginx reverse proxy routing and API endpoints

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NGINX_HOST="localhost"
NGINX_PORT="80"
BACKEND_PORT="8000"
FRONTEND_PORT="3000"
PIPELINE_PORT="8001"

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_status="$3"
    local max_latency="$4"
    
    ((TOTAL_TESTS++))
    log_info "Running: $test_name"
    
    # Run the test and capture output
    local start_time=$(date +%s%3N)
    local response=$(eval "$test_command" 2>&1)
    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))
    
    # Extract HTTP status code
    local status_code=$(echo "$response" | grep -o 'HTTP/[0-9.]* [0-9]*' | tail -1 | awk '{print $2}')
    
    # Check if we got a response
    if [ -z "$status_code" ]; then
        log_error "$test_name - No HTTP response received"
        return 1
    fi
    
    # Check status code
    if [ "$status_code" = "$expected_status" ]; then
        # Check latency if specified
        if [ -n "$max_latency" ] && [ "$latency" -gt "$max_latency" ]; then
            log_warning "$test_name - Status OK ($status_code) but latency too high (${latency}ms > ${max_latency}ms)"
        else
            log_success "$test_name - Status: $status_code, Latency: ${latency}ms"
        fi
    else
        log_error "$test_name - Expected status $expected_status, got $status_code"
        echo "Response: $response"
        return 1
    fi
}

# Check if services are running
check_services() {
    log_info "Checking if services are running..."
    
    # Check if Docker containers are running
    if ! docker-compose ps | grep -q "Up"; then
        log_error "Docker services are not running. Please run 'make start' first."
        exit 1
    fi
    
    # Check if ports are accessible
    local services=("nginx:80" "backend:8000" "frontend:3000" "pipeline:8001")
    
    for service in "${services[@]}"; do
        local name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "$name service is accessible on port $port"
        else
            log_error "$name service is not accessible on port $port"
        fi
    done
}

# Test Nginx routing
test_nginx_routing() {
    log_info "Testing Nginx reverse proxy routing..."
    
    # Test frontend routing
    run_test "Frontend routing" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/" \
        "200"
    
    # Test backend API routing
    run_test "Backend API routing" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/health" \
        "200"
    
    # Test face-pipeline routing
    run_test "Face Pipeline routing" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/face-pipeline/health" \
        "200"
}

# Test CORS headers
test_cors_headers() {
    log_info "Testing CORS headers..."
    
    # Test CORS preflight request
    run_test "CORS preflight request" \
        "curl -s -w '%{http_code}' -o /dev/null -X OPTIONS -H 'Origin: http://localhost:3000' -H 'Access-Control-Request-Method: GET' http://$NGINX_HOST:$NGINX_PORT/api/health" \
        "200"
    
    # Test CORS headers in response
    local cors_response=$(curl -s -I -H 'Origin: http://localhost:3000' http://$NGINX_HOST:$NGINX_PORT/api/health)
    
    if echo "$cors_response" | grep -q "Access-Control-Allow-Origin"; then
        log_success "CORS headers present in API response"
    else
        log_error "CORS headers missing in API response"
    fi
}

# Test port mapping
test_port_mapping() {
    log_info "Testing port mapping..."
    
    # Test direct backend access
    run_test "Direct backend access" \
        "curl -s -w '%{http_code}' -o /dev/null http://localhost:$BACKEND_PORT/health" \
        "200"
    
    # Test direct frontend access
    run_test "Direct frontend access" \
        "curl -s -w '%{http_code}' -o /dev/null http://localhost:$FRONTEND_PORT/" \
        "200"
    
    # Test direct pipeline access
    run_test "Direct pipeline access" \
        "curl -s -w '%{http_code}' -o /dev/null http://localhost:$PIPELINE_PORT/health" \
        "200"
}

# Test API endpoints through proxy
test_api_endpoints() {
    log_info "Testing API endpoints through proxy..."
    
    # Test health endpoints
    run_test "Backend health through proxy" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/health" \
        "200" \
        "200"
    
    run_test "Pipeline health through proxy" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/face-pipeline/health" \
        "200" \
        "200"
    
    # Test ready endpoints
    run_test "Backend ready through proxy" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/ready" \
        "200" \
        "200"
    
    run_test "Pipeline ready through proxy" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/face-pipeline/ready" \
        "503" \
        "200"
    
    # Test search endpoint (should return 405 Method Not Allowed for GET)
    run_test "Search endpoint through proxy" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/v1/search" \
        "405"
}

# Test performance
test_performance() {
    log_info "Testing performance requirements..."
    
    # Test health endpoint latency
    local health_latency=$(curl -s -w '%{time_total}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/health)
    local health_latency_ms=$(echo "$health_latency * 1000" | bc)
    
    if (( $(echo "$health_latency_ms < 200" | bc -l) )); then
        log_success "Health endpoint latency: ${health_latency_ms}ms (under 200ms requirement)"
    else
        log_warning "Health endpoint latency: ${health_latency_ms}ms (exceeds 200ms requirement)"
    fi
    
    # Test multiple concurrent requests
    log_info "Testing concurrent request handling..."
    local concurrent_requests=5
    local success_count=0
    
    for i in $(seq 1 $concurrent_requests); do
        if curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/health | grep -q "200"; then
            ((success_count++))
        fi
    done
    
    if [ "$success_count" -eq "$concurrent_requests" ]; then
        log_success "Concurrent request handling: $success_count/$concurrent_requests successful"
    else
        log_warning "Concurrent request handling: $success_count/$concurrent_requests successful"
    fi
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."
    
    # Test 404 handling
    run_test "404 error handling" \
        "curl -s -w '%{http_code}' -o /dev/null http://$NGINX_HOST:$NGINX_PORT/api/nonexistent" \
        "404"
    
    # Test invalid method
    run_test "Invalid method handling" \
        "curl -s -w '%{http_code}' -o /dev/null -X DELETE http://$NGINX_HOST:$NGINX_PORT/api/health" \
        "405"
}

# Test Nginx configuration
test_nginx_config() {
    log_info "Testing Nginx configuration..."
    
    # Test if Nginx is serving the correct content
    local frontend_response=$(curl -s http://$NGINX_HOST:$NGINX_PORT/)
    if echo "$frontend_response" | grep -q "html\|<!DOCTYPE"; then
        log_success "Nginx is serving frontend content"
    else
        log_error "Nginx is not serving frontend content properly"
    fi
    
    # Test if API routes are properly proxied
    local api_response=$(curl -s http://$NGINX_HOST:$NGINX_PORT/api/health)
    if echo "$api_response" | grep -q "status\|healthy"; then
        log_success "Nginx is properly proxying API requests"
    else
        log_error "Nginx is not properly proxying API requests"
    fi
}

# Main test execution
main() {
    echo "🧪 Mordeaux Face Scanning MVP - Proxy Smoke Tests"
    echo "=================================================="
    echo ""
    
    # Check prerequisites
    check_services
    
    echo ""
    log_info "Starting smoke tests..."
    echo ""
    
    # Run all tests
    test_nginx_routing
    test_cors_headers
    test_port_mapping
    test_api_endpoints
    test_performance
    test_error_handling
    test_nginx_config
    
    # Print summary
    echo ""
    echo "📊 Test Summary"
    echo "==============="
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo ""
        log_success "All smoke tests passed! 🎉"
        exit 0
    else
        echo ""
        log_error "Some smoke tests failed. Please check the configuration."
        exit 1
    fi
}

# Run main function
main "$@"
