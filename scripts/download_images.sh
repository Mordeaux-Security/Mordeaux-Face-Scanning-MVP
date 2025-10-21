#!/bin/bash

# Download and flatten MinIO images script
# Usage: ./scripts/download_images.sh [thumbnails|raw-images|both]

set -e

BUCKET_TYPE="${1:-both}"
MC_ALIAS="${MC_ALIAS:-myminio}"
TEMP_DIR="${TEMP_DIR:-dl}"
OUTPUT_DIR="${OUTPUT_DIR:-flat}"
ZIP_DIR="${ZIP_DIR:-zips}"

echo "=== MinIO Image Downloader ==="
echo "Bucket type: $BUCKET_TYPE"
echo "MinIO alias: $MC_ALIAS"
echo "Temp directory: $TEMP_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Zip directory: $ZIP_DIR"
echo ""

# Check if mc command exists
if ! command -v mc &> /dev/null; then
    echo "Error: MinIO client (mc) not found. Please install it:"
    echo "  brew install minio/stable/mc"
    echo "  or download from: https://min.io/download"
    exit 1
fi

# Check if mc alias is configured
if ! mc ls "$MC_ALIAS" &> /dev/null; then
    echo "Error: MinIO alias '$MC_ALIAS' not configured or not accessible."
    echo "Configure it with: mc alias set $MC_ALIAS <endpoint> <access-key> <secret-key>"
    exit 1
fi

# Create directories
mkdir -p "$TEMP_DIR"/{thumbs,raw}
mkdir -p "$OUTPUT_DIR"/{thumbs,raw}
mkdir -p "$ZIP_DIR"

# Download based on bucket type
echo "Downloading images..."
if [[ "$BUCKET_TYPE" == "thumbnails" || "$BUCKET_TYPE" == "both" ]]; then
    echo "  - Downloading thumbnails..."
    mc mirror --overwrite "$MC_ALIAS/crawled-images/thumbnails" "$TEMP_DIR/thumbs"
    # Remove JSON files from thumbnails
    find "$TEMP_DIR/thumbs" -name "*.json" -type f -delete
    echo "    - Removed JSON files from thumbnails"
fi

if [[ "$BUCKET_TYPE" == "raw-images" || "$BUCKET_TYPE" == "both" ]]; then
    echo "  - Downloading raw images..."
    mc mirror --overwrite "$MC_ALIAS/crawled-images/images" "$TEMP_DIR/raw"
    # Remove JSON files from raw images
    find "$TEMP_DIR/raw" -name "*.json" -type f -delete
    echo "    - Removed JSON files from raw images"
fi

# Flatten the directory structure
echo "Flattening directory structure..."
if [[ "$BUCKET_TYPE" == "thumbnails" || "$BUCKET_TYPE" == "both" ]]; then
    echo "  - Flattening thumbnails..."
    find "$TEMP_DIR/thumbs" -type f -exec cp {} "$OUTPUT_DIR/thumbs/" \;
fi

if [[ "$BUCKET_TYPE" == "raw-images" || "$BUCKET_TYPE" == "both" ]]; then
    echo "  - Flattening raw images..."
    find "$TEMP_DIR/raw" -type f -exec cp {} "$OUTPUT_DIR/raw/" \;
fi

# Create zip files
echo "Creating zip files..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [[ "$BUCKET_TYPE" == "thumbnails" || "$BUCKET_TYPE" == "both" ]]; then
    THUMB_COUNT=$(find "$OUTPUT_DIR/thumbs" -type f | wc -l)
    if [ "$THUMB_COUNT" -gt 0 ]; then
        echo "  - Creating thumbnails zip ($THUMB_COUNT files)..."
        cd "$OUTPUT_DIR/thumbs"
        zip -r "../../$ZIP_DIR/thumbnails_$TIMESTAMP.zip" . -q
        cd - > /dev/null
        echo "    ✓ Created: $ZIP_DIR/thumbnails_$TIMESTAMP.zip"
    else
        echo "  - No thumbnails found to zip"
    fi
fi

if [[ "$BUCKET_TYPE" == "raw-images" || "$BUCKET_TYPE" == "both" ]]; then
    RAW_COUNT=$(find "$OUTPUT_DIR/raw" -type f | wc -l)
    if [ "$RAW_COUNT" -gt 0 ]; then
        echo "  - Creating raw images zip ($RAW_COUNT files)..."
        cd "$OUTPUT_DIR/raw"
        zip -r "../../$ZIP_DIR/raw_images_$TIMESTAMP.zip" . -q
        cd - > /dev/null
        echo "    ✓ Created: $ZIP_DIR/raw_images_$TIMESTAMP.zip"
    else
        echo "  - No raw images found to zip"
    fi
fi

if [[ "$BUCKET_TYPE" == "both" ]]; then
    TOTAL_COUNT=$((THUMB_COUNT + RAW_COUNT))
    if [ "$TOTAL_COUNT" -gt 0 ]; then
        echo "  - Creating combined zip ($TOTAL_COUNT files)..."
        cd "$OUTPUT_DIR"
        zip -r "../$ZIP_DIR/all_images_$TIMESTAMP.zip" . -q
        cd - > /dev/null
        echo "    ✓ Created: $ZIP_DIR/all_images_$TIMESTAMP.zip"
    fi
fi

# Cleanup temp directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Download Complete ==="
echo "Files downloaded to: $OUTPUT_DIR/"
echo "Zip files created in: $ZIP_DIR/"
if [[ "$BUCKET_TYPE" == "both" ]]; then
    echo "Total images: $TOTAL_COUNT"
else
    echo "Images processed: $(find "$OUTPUT_DIR" -type f | wc -l)"
fi
