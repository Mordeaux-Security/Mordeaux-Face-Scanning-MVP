#!/bin/bash
# Simple crawler runner with log capture
# Usage: ./run-crawler.sh [--sites-file sites.txt] [--sites url1 url2 ...]

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Run: python3 -m venv .venv"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=src

# Default log file
LOG_FILE="${LOG_FILE:-debugamd.txt}"

# Build command
CMD="python -m diabetes_crawler.main"

# Parse arguments
SITES_FILE=""
SITES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --sites-file)
            SITES_FILE="$2"
            shift 2
            ;;
        --sites)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SITES+=("$1")
                shift
            done
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Add sites to command
if [ -n "$SITES_FILE" ] && [ -f "$SITES_FILE" ]; then
    CMD="$CMD --sites-file $SITES_FILE"
elif [ ${#SITES[@]} -gt 0 ]; then
    CMD="$CMD --sites ${SITES[*]}"
else
    # Default to sites.txt if it exists
    if [ -f "sites.txt" ]; then
        CMD="$CMD --sites-file sites.txt"
    else
        echo "Error: No sites provided. Use --sites-file or --sites"
        exit 1
    fi
fi

echo "Running: $CMD"
echo "Logging to: $LOG_FILE"
echo ""

# Run command and capture all output (stdout + stderr) to both console and file
eval "$CMD" 2>&1 | tee "$LOG_FILE"


