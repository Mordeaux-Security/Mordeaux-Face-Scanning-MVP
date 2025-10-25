# Simple script to download both thumbnails and raw images from MinIO
# Usage: .\scripts\download_both_simple.ps1

$MC_ALIAS = "myminio"
$TEMP_DIR = "dl"
$OUTPUT_DIR = "flat"
$ZIP_DIR = "zips"

Write-Host "=== MinIO Image Downloader ===" -ForegroundColor Green
Write-Host "Downloading both thumbnails and raw images"
Write-Host ""

# Check if mc command exists
try {
    $null = Get-Command mc -ErrorAction Stop
} catch {
    Write-Host "Error: MinIO client (mc) not found. Please install it:" -ForegroundColor Red
    Write-Host "  Download from: https://min.io/download" -ForegroundColor Yellow
    exit 1
}

# Create directories
$directories = @("$TEMP_DIR\thumbs", "$TEMP_DIR\raw", "$OUTPUT_DIR\thumbs", "$OUTPUT_DIR\raw", $ZIP_DIR)
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Download images
Write-Host "Downloading images..." -ForegroundColor Cyan

Write-Host "  - Downloading thumbnails..." -ForegroundColor Yellow
mc mirror --overwrite "$MC_ALIAS/thumbnails" "$TEMP_DIR/thumbs"

Write-Host "  - Downloading raw images..." -ForegroundColor Yellow
mc mirror --overwrite "$MC_ALIAS/raw-images" "$TEMP_DIR/raw"

# Flatten the directory structure
Write-Host "Flattening directory structure..." -ForegroundColor Cyan

Write-Host "  - Flattening thumbnails..." -ForegroundColor Yellow
Get-ChildItem -Path "$TEMP_DIR/thumbs" -Recurse -File | ForEach-Object {
    Copy-Item $_.FullName -Destination "$OUTPUT_DIR/thumbs/" -Force
}

Write-Host "  - Flattening raw images..." -ForegroundColor Yellow
Get-ChildItem -Path "$TEMP_DIR/raw" -Recurse -File | ForEach-Object {
    Copy-Item $_.FullName -Destination "$OUTPUT_DIR/raw/" -Force
}

# Create zip files
Write-Host "Creating zip files..." -ForegroundColor Cyan
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# Create thumbnails zip
$thumbFiles = Get-ChildItem -Path "$OUTPUT_DIR/thumbs" -File
if ($thumbFiles.Count -gt 0) {
    Write-Host "  - Creating thumbnails zip..." -ForegroundColor Yellow
    $thumbZipPath = "$ZIP_DIR/thumbnails_$TIMESTAMP.zip"
    Compress-Archive -Path "$OUTPUT_DIR/thumbs/*" -DestinationPath $thumbZipPath -Force
    Write-Host "    ✓ Created: $thumbZipPath" -ForegroundColor Green
}

# Create raw images zip
$rawFiles = Get-ChildItem -Path "$OUTPUT_DIR/raw" -File
if ($rawFiles.Count -gt 0) {
    Write-Host "  - Creating raw images zip..." -ForegroundColor Yellow
    $rawZipPath = "$ZIP_DIR/raw_images_$TIMESTAMP.zip"
    Compress-Archive -Path "$OUTPUT_DIR/raw/*" -DestinationPath $rawZipPath -Force
    Write-Host "    ✓ Created: $rawZipPath" -ForegroundColor Green
}

# Create combined zip
$totalCount = $thumbFiles.Count + $rawFiles.Count
if ($totalCount -gt 0) {
    Write-Host "  - Creating combined zip..." -ForegroundColor Yellow
    $combinedZipPath = "$ZIP_DIR/all_images_$TIMESTAMP.zip"
    Compress-Archive -Path "$OUTPUT_DIR/*" -DestinationPath $combinedZipPath -Force
    Write-Host "    ✓ Created: $combinedZipPath" -ForegroundColor Green
}

# Cleanup temp directory
Write-Host "Cleaning up temporary files..." -ForegroundColor Cyan
Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Download Complete ===" -ForegroundColor Green
Write-Host "Files downloaded to: $OUTPUT_DIR/"
Write-Host "Zip files created in: $ZIP_DIR/"
Write-Host "Total images: $totalCount"
