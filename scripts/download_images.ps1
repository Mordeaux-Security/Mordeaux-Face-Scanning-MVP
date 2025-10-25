# Download and flatten MinIO images script (PowerShell version)
# Usage: .\scripts\download_images.ps1 [thumbnails|raw-images|both]

param(
    [string]$BucketType = "both"
)

$MC_ALIAS = if ($env:MC_ALIAS) { $env:MC_ALIAS } else { "myminio" }
$TEMP_DIR = if ($env:TEMP_DIR) { $env:TEMP_DIR } else { "dl" }
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "flat" }
$ZIP_DIR = if ($env:ZIP_DIR) { $env:ZIP_DIR } else { "zips" }

Write-Host "=== MinIO Image Downloader ===" -ForegroundColor Green
Write-Host "Bucket type: $BucketType"
Write-Host "MinIO alias: $MC_ALIAS"
Write-Host "Temp directory: $TEMP_DIR"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host "Zip directory: $ZIP_DIR"
Write-Host ""

# Check if mc command exists
try {
    $null = Get-Command mc -ErrorAction Stop
} catch {
    Write-Host "Error: MinIO client (mc) not found. Please install it:" -ForegroundColor Red
    Write-Host "  Download from: https://min.io/download" -ForegroundColor Yellow
    Write-Host "  Or use: winget install MinIO.MinIO" -ForegroundColor Yellow
    exit 1
}

# Check if mc alias is configured
try {
    $null = mc ls $MC_ALIAS 2>$null
} catch {
    Write-Host "Error: MinIO alias '$MC_ALIAS' not configured or not accessible." -ForegroundColor Red
    Write-Host "Configure it with: mc alias set $MC_ALIAS <endpoint> <access-key> <secret-key>" -ForegroundColor Yellow
    exit 1
}

# Create directories
$directories = @(
    "$TEMP_DIR\thumbs",
    "$TEMP_DIR\raw",
    "$OUTPUT_DIR\thumbs",
    "$OUTPUT_DIR\raw",
    $ZIP_DIR
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Download based on bucket type
Write-Host "Downloading images..." -ForegroundColor Cyan

if ($BucketType -eq "thumbnails" -or $BucketType -eq "both") {
    Write-Host "  - Downloading thumbnails..." -ForegroundColor Yellow
    mc mirror --overwrite "$MC_ALIAS/thumbnails" "$TEMP_DIR/thumbs"
}

if ($BucketType -eq "raw-images" -or $BucketType -eq "both") {
    Write-Host "  - Downloading raw images..." -ForegroundColor Yellow
    mc mirror --overwrite "$MC_ALIAS/raw-images" "$TEMP_DIR/raw"
}

# Flatten the directory structure
Write-Host "Flattening directory structure..." -ForegroundColor Cyan

if ($BucketType -eq "thumbnails" -or $BucketType -eq "both") {
    Write-Host "  - Flattening thumbnails..." -ForegroundColor Yellow
    Get-ChildItem -Path "$TEMP_DIR/thumbs" -Recurse -File | ForEach-Object {
        Copy-Item $_.FullName -Destination "$OUTPUT_DIR/thumbs/" -Force
    }
}

if ($BucketType -eq "raw-images" -or $BucketType -eq "both") {
    Write-Host "  - Flattening raw images..." -ForegroundColor Yellow
    Get-ChildItem -Path "$TEMP_DIR/raw" -Recurse -File | ForEach-Object {
        Copy-Item $_.FullName -Destination "$OUTPUT_DIR/raw/" -Force
    }
}

# Create zip files
Write-Host "Creating zip files..." -ForegroundColor Cyan
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# Initialize variables
$thumbFiles = @()
$rawFiles = @()

if ($BucketType -eq "thumbnails" -or $BucketType -eq "both") {
    $thumbFiles = Get-ChildItem -Path "$OUTPUT_DIR/thumbs" -File
    if ($thumbFiles.Count -gt 0) {
        $thumbCount = $thumbFiles.Count
        Write-Host "  - Creating thumbnails zip ($thumbCount files)..." -ForegroundColor Yellow
        $thumbZipPath = "$ZIP_DIR/thumbnails_$TIMESTAMP.zip"
        Compress-Archive -Path "$OUTPUT_DIR/thumbs/*" -DestinationPath $thumbZipPath -Force
        Write-Host "    ✓ Created: $thumbZipPath" -ForegroundColor Green
    } else {
        Write-Host "  - No thumbnails found to zip" -ForegroundColor Yellow
    }
}

if ($BucketType -eq "raw-images" -or $BucketType -eq "both") {
    $rawFiles = Get-ChildItem -Path "$OUTPUT_DIR/raw" -File
    if ($rawFiles.Count -gt 0) {
        $rawCount = $rawFiles.Count
        Write-Host "  - Creating raw images zip ($rawCount files)..." -ForegroundColor Yellow
        $rawZipPath = "$ZIP_DIR/raw_images_$TIMESTAMP.zip"
        Compress-Archive -Path "$OUTPUT_DIR/raw/*" -DestinationPath $rawZipPath -Force
        Write-Host "    ✓ Created: $rawZipPath" -ForegroundColor Green
    } else {
        Write-Host "  - No raw images found to zip" -ForegroundColor Yellow
    }
}

if ($BucketType -eq "both") {
    $totalCount = $thumbFiles.Count + $rawFiles.Count
    if ($totalCount -gt 0) {
        Write-Host "  - Creating combined zip ($totalCount files)..." -ForegroundColor Yellow
        $combinedZipPath = "$ZIP_DIR/all_images_$TIMESTAMP.zip"
        Compress-Archive -Path "$OUTPUT_DIR/*" -DestinationPath $combinedZipPath -Force
        Write-Host "    ✓ Created: $combinedZipPath" -ForegroundColor Green
    }
}

# Cleanup temp directory
Write-Host "Cleaning up temporary files..." -ForegroundColor Cyan
Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Download Complete ===" -ForegroundColor Green
Write-Host "Files downloaded to: $OUTPUT_DIR/"
Write-Host "Zip files created in: $ZIP_DIR/"
if ($BucketType -eq "both") {
    Write-Host "Total images: $totalCount"
} else {
    $allFiles = Get-ChildItem -Path $OUTPUT_DIR -Recurse -File
    Write-Host "Images processed: $($allFiles.Count)"
}
