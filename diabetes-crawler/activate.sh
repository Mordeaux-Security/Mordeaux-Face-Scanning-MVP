#!/bin/bash
# Quick activation script for diabetes-crawler
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=src
echo "✓ Virtual environment activated"
echo "✓ PYTHONPATH set to: src"
echo ""
echo "Run the crawler with:"
echo "  python -m diabetes_crawler.main --help"
