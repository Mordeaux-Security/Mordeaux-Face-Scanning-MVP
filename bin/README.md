# CLI Tools for Selector Mining

This directory contains command-line tools for mining and reviewing image selectors as part of Phase 2.2 of the Mordeaux Face Scanning MVP.

## Tools Overview

### `mine-selectors`
Mines and validates image selectors for a domain by analyzing HTML content from provided URLs.

### `review-selectors`
Downloads sample thumbnails and provides a console interface for reviewing and approving selectors.

## Installation & Setup

Make sure the CLI scripts are executable:
```bash
chmod +x bin/mine-selectors
chmod +x bin/review-selectors
```

## Usage

### mine-selectors

Mine and validate image selectors for a domain.

```bash
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml [--append] [--verbose]
```

**Arguments:**
- `--domain` (required): Domain name to analyze
- `--urls` (required): File containing URLs (one per line)
- `--out` (required): Output YAML file path
- `--append`: Append to existing YAML file (default: True)
- `--js`: Enable JavaScript fallback for dynamic content (requires Playwright)
- `--min-candidates N`: Minimum candidates before JS fallback (default: 3)
- `--verbose, -v`: Enable verbose logging

**Examples:**
```bash
# Basic usage
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml

# With verbose logging
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --verbose

# With JavaScript fallback for dynamic content
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --js

# JavaScript fallback with custom minimum candidates
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --js --min-candidates 2

# Create new file (don't append)
python bin/mine-selectors --domain example.com --urls urls.txt --out new_recipes.yaml
```

**URLs File Format:**
```
# Comments start with #
https://example.com/page1
https://example.com/page2
https://example.com/page3

# Empty lines are ignored
```

**Process:**
1. Loads URLs from the specified file
2. Fetches HTML content from the first few URLs (max 3)
3. Analyzes HTML to mine candidate selectors
4. Validates selectors by testing sample image URLs
5. Merges successful recipes into the output YAML file

### review-selectors

Review and approve image selectors for a domain.

```bash
python bin/review-selectors --domain example.com [--yaml site_recipes.yaml] [--download-dir /tmp/review] [--keep-downloads] [--verbose]
```

**Arguments:**
- `--domain` (required): Domain name to review
- `--yaml`: YAML file path (default: site_recipes.yaml)
- `--download-dir`: Directory to download sample images (default: temporary directory)
- `--keep-downloads`: Keep downloaded images after review
- `--verbose, -v`: Enable verbose logging

**Examples:**
```bash
# Basic usage
python bin/review-selectors --domain example.com

# Specify custom YAML file
python bin/review-selectors --domain example.com --yaml custom_recipes.yaml

# Keep downloaded images for inspection
python bin/review-selectors --domain example.com --keep-downloads --download-dir /tmp/review
```

**Process:**
1. Loads the recipe for the specified domain from YAML file
2. Downloads 6-8 sample images for each selector
3. Displays a console interface showing selectors and sample images
4. Prompts user to approve (y) or reject (n) each selector
5. Updates the YAML file with only approved selectors

**Review Interface:**
```
================================================================================
SELECTOR REVIEW INTERFACE
================================================================================
Review each selector and approve (y) or reject (n)
Press Enter for default (approve)
================================================================================

[1/3] SELECTOR REVIEW
------------------------------------------------------------
Selector: body > div.video-gallery > img.thumbnail
Description: Thumbnail images in video gallery
Downloaded Images: 6
Sample Images:
  1. /tmp/review/selector_1/sample_1.jpg
  2. /tmp/review/selector_1/sample_2.jpg
  3. /tmp/review/selector_1/sample_3.jpg
  4. /tmp/review/selector_1/sample_4.jpg

Approve this selector? [Y/n]: y
✅ APPROVED
```

## JavaScript Fallback (Phase 2.3)

For pages that inject thumbnails dynamically with JavaScript, the `--js` flag enables Playwright-based rendering fallback.

### When to Use JavaScript Fallback

- **Static mining finds < N candidates**: When static HTML analysis finds too few image selectors
- **Dynamic content**: Pages that load images via JavaScript after initial page load
- **SPA (Single Page Applications)**: React, Vue, Angular apps that render content client-side
- **Lazy loading**: Sites that load images on scroll or interaction

### How It Works

1. **Static Analysis First**: Always tries static HTML parsing first
2. **Threshold Check**: If fewer than `--min-candidates` selectors found, triggers JS fallback
3. **Playwright Rendering**: Launches headless Chrome to render the page with JavaScript
4. **Content Extraction**: Extracts fully-rendered HTML after 2-3 second timeout
5. **Re-analysis**: Runs the same selector mining on JavaScript-rendered content
6. **Best Result**: Returns whichever method found more/better candidates

### Playwright Requirements

```bash
# Install Playwright (if not already installed)
pip install playwright

# Install browser binaries
playwright install chromium
```

### JavaScript Fallback Examples

```bash
# Enable JS fallback with default threshold (3 candidates)
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --js

# Lower threshold to trigger JS fallback more aggressively
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --js --min-candidates 1

# JS fallback with verbose logging to see the process
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --js --verbose
```

### Performance Notes

- **Headless Mode**: Runs in headless Chrome for speed
- **Image Blocking**: Blocks images during rendering for faster load times
- **Timeout**: 2-3 second wait for dynamic content (configurable)
- **Fallback Graceful**: If Playwright not available, falls back to static results
- **Resource Cleanup**: Automatically closes browser instances

### Troubleshooting JavaScript Fallback

**Playwright not installed:**
```
WARNING: Playwright not available - skipping JavaScript rendering
```
Solution: Install Playwright and browser binaries

**JavaScript rendering fails:**
```
WARNING: JavaScript rendering failed for https://example.com: timeout
```
Solution: Check if URL is accessible, increase timeout, or use static mining

**No improvement with JS:**
```
INFO: JavaScript fallback found 2 candidates (same or fewer than static)
```
Normal behavior - static results returned when JS doesn't improve

## Workflow Example

Here's a complete workflow for analyzing a new domain:

### 1. Prepare URLs File
Create a file with URLs from the target domain:
```bash
cat > urls.txt << EOF
https://example.com/page1
https://example.com/page2
https://example.com/page3
EOF
```

### 2. Mine Selectors
```bash
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --verbose
```

### 3. Review Selectors
```bash
python bin/review-selectors --domain example.com --yaml site_recipes.yaml --verbose
```

### 4. Verify Results
Check the updated YAML file:
```bash
cat site_recipes.yaml
```

## Integration with Existing System

These CLI tools integrate seamlessly with the existing Mordeaux Face Scanning MVP:

- **Compatible YAML Format**: Generated recipes match the existing `site_recipes.yaml` structure
- **Validation**: All selectors are validated with anti-malware guards before acceptance
- **Merge Support**: Safely merges new recipes into existing configuration files
- **Error Handling**: Robust error handling with informative logging

## Troubleshooting

### Common Issues

**No selectors found:**
- Ensure URLs return valid HTML content
- Check that pages contain image elements
- Use `--verbose` flag for detailed logging

**Validation failures:**
- Image URLs may be inaccessible or return non-image content
- Check network connectivity and URL accessibility
- Some domains may block automated requests

**YAML merge errors:**
- Ensure output file is writable
- Check YAML file format and permissions
- Use `--verbose` flag to see detailed error messages

### Debug Mode

Use the `--verbose` flag to enable detailed logging:
```bash
python bin/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml --verbose
```

This will show:
- HTTP request details
- Selector mining progress
- Validation results
- YAML merge operations

## Security Features

- **Anti-malware Guards**: Blocks malicious URL schemes (javascript:, data:, file:, ftp:)
- **Content Type Validation**: Ensures URLs return valid image MIME types
- **Size Limits**: Prevents processing of overly large files (>50MB)
- **Timeout Protection**: HTTP requests timeout after 30 seconds

## Performance Notes

- **Parallel Processing**: HTTP requests are made concurrently for better performance
- **Selective Download**: Only downloads first 6-8 sample images per selector
- **Temporary Storage**: Uses temporary directories for downloads (cleaned up automatically)
- **Efficient Validation**: Uses HEAD requests first, then GET for first 1KB

## Dependencies

- Python 3.7+
- httpx (for HTTP requests)
- PyYAML (for YAML processing)
- beautifulsoup4 (for HTML parsing)
- pytest-asyncio (for testing)

All dependencies are included in the project's virtual environment.
