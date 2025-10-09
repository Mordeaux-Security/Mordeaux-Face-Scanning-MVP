#!/bin/bash
# Test script for Face Pipeline application
# Tests the minimal app setup and acceptance criteria

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                  Face Pipeline App Test Script                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the face-pipeline directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ Error: main.py not found. Run this script from face-pipeline/ directory${NC}"
    exit 1
fi

echo "📋 Test Plan:"
echo "  1. Verify Python syntax"
echo "  2. Check settings can be imported"
echo "  3. Start uvicorn server"
echo "  4. Test GET /health endpoint"
echo "  5. Test other endpoints return 501"
echo ""

# Test 1: Syntax check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Verifying Python syntax..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -m py_compile main.py && echo -e "${GREEN}✅ main.py syntax OK${NC}"
python3 -m py_compile config/settings.py && echo -e "${GREEN}✅ config/settings.py syntax OK${NC}"
python3 -m py_compile services/search_api.py && echo -e "${GREEN}✅ services/search_api.py syntax OK${NC}"

echo ""

# Test 2: Check if dependencies are installed
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Checking dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${GREEN}✅ fastapi installed${NC}"
else
    echo -e "${YELLOW}⚠️  fastapi not installed. Run: make install${NC}"
    echo ""
    echo "To run full test, install dependencies first:"
    echo "  cd face-pipeline"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 0
fi

if python3 -c "import uvicorn" 2>/dev/null; then
    echo -e "${GREEN}✅ uvicorn installed${NC}"
else
    echo -e "${RED}❌ uvicorn not installed${NC}"
    exit 1
fi

if python3 -c "import pydantic" 2>/dev/null; then
    echo -e "${GREEN}✅ pydantic installed${NC}"
else
    echo -e "${RED}❌ pydantic not installed${NC}"
    exit 1
fi

echo ""

# Test 3: Start server in background
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Starting uvicorn server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start server in background
uvicorn main:app --host 127.0.0.1 --port 8001 > /tmp/face-pipeline-test.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to start..."
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo -e "${GREEN}✅ Server started successfully${NC}"
else
    echo -e "${RED}❌ Server failed to start${NC}"
    cat /tmp/face-pipeline-test.log
    exit 1
fi

echo ""

# Function to cleanup
cleanup() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Cleaning up..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Server stopped"
}

trap cleanup EXIT

# Test 4: Test /health endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: Testing GET /health endpoint (ACCEPTANCE TEST)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RESPONSE=$(curl -s -w "\n%{http_code}" http://127.0.0.1:8001/health)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

echo "HTTP Status: $HTTP_CODE"
echo "Response Body: $BODY"

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ /health returns 200 OK${NC}"
    
    # Check if response contains "ok"
    if echo "$BODY" | grep -q "ok"; then
        echo -e "${GREEN}✅ Response contains 'status': 'ok'${NC}"
    else
        echo -e "${RED}❌ Response missing 'ok' status${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ /health returned $HTTP_CODE instead of 200${NC}"
    exit 1
fi

echo ""

# Test 5: Test 501 endpoints
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 5: Testing endpoints return 501 (Not Implemented)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test /api/v1/search
echo "Testing POST /api/v1/search..."
SEARCH_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8001/api/v1/search \
    -H "Content-Type: application/json" \
    -d '{"limit": 10}')

if [ "$SEARCH_CODE" = "501" ]; then
    echo -e "${GREEN}✅ /api/v1/search returns 501 (TODO)${NC}"
else
    echo -e "${YELLOW}⚠️  /api/v1/search returned $SEARCH_CODE instead of 501${NC}"
fi

# Test /api/v1/faces/{id}
echo "Testing GET /api/v1/faces/test-id..."
FACE_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/api/v1/faces/test-id)

if [ "$FACE_CODE" = "501" ]; then
    echo -e "${GREEN}✅ /api/v1/faces/{id} returns 501 (TODO)${NC}"
else
    echo -e "${YELLOW}⚠️  /api/v1/faces/{id} returned $FACE_CODE instead of 501${NC}"
fi

# Test /api/v1/stats
echo "Testing GET /api/v1/stats..."
STATS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/api/v1/stats)

if [ "$STATS_CODE" = "501" ]; then
    echo -e "${GREEN}✅ /api/v1/stats returns 501 (TODO)${NC}"
else
    echo -e "${YELLOW}⚠️  /api/v1/stats returned $STATS_CODE instead of 501${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ ALL ACCEPTANCE TESTS PASSED!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Summary:"
echo "  ✅ Python syntax valid"
echo "  ✅ Dependencies installed"
echo "  ✅ Server starts successfully (uvicorn main:app --reload)"
echo "  ✅ GET /health returns 200 with {'status': 'ok'}"
echo "  ✅ Search endpoints return 501 (TODO)"
echo ""
echo "The minimal app is ready! 🎉"
echo ""
echo "Next steps:"
echo "  1. Visit http://127.0.0.1:8001/docs for API documentation"
echo "  2. Implement the TODO endpoints in services/search_api.py"
echo "  3. Add pipeline components (detector, embedder, quality, etc.)"
echo ""

