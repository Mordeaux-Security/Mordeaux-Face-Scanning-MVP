#!/bin/bash

# Start Mock Server - Phase 3
# ============================

echo "=========================================="
echo "🎭 Mock Server - Phase 3"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Print fixture summary
echo ""
echo "Generating fixtures..."
python fixtures.py

# Start server
echo ""
echo "Starting server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo "Mock config: http://localhost:8000/mock/fixtures"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

python app.py

