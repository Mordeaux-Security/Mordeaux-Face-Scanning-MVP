# Selector Miner Tools

This directory contains tools for the Mordeaux Face Scanning MVP, specifically the Phase 2 Selector-Miner Core.

## Selector Miner (`selector_miner.py`)

A deterministic selector-miner that analyzes HTML content to generate candidate CSS selectors for image extraction with evidence-based scoring.

### Features

- **HTML Parsing**: Uses BeautifulSoup to parse HTML and extract image elements
- **Multi-source Detection**: Finds images in:
  - `<img>` tags with various `src` attributes
  - `<source>` tags
  - `<video>` tags with `poster` attribute
  - `<meta>` tags with `og:image` property
  - `<link>` tags with `image_src` rel
  - Elements with `background-image` CSS

- **Smart Selector Generation**: Creates minimal, stable selectors that:
  - Favor classes over IDs for stability
  - Avoid random tokens and `nth-child` selectors
  - Limit depth to 4 levels maximum
  - Skip document root elements

- **Evidence Gathering**: Analyzes context for scoring:
  - Repetition count (how many images match the selector)
  - Duration text nearby (regex `\b\d{1,2}:\d{2}\b`)
  - Video URL patterns in ancestor links (`/video/`, `/watch/`, `/embed/`)
  - Class name hints (positive: `thumb`, `preview`, `image`; negative: `logo`, `icon`, `avatar`)
  - Srcset richness (responsive image indicators)
  - URL quality assessment

- **Intelligent Scoring**: Combines evidence with weighted scoring:
  - Repetition score (30%)
  - Duration score (20%)
  - Video URL score (15%)
  - Class hint score (15%)
  - Srcset score (10%)
  - URL quality score (10%)

### Usage

```python
from tools.selector_miner import mine_selectors_for_url, SelectorMiner

# Simple usage
html_content = "<html><body><img src='/thumb.jpg' class='thumbnail'></body></html>"
candidates = mine_selectors_for_url(html_content, "https://example.com")

for candidate in candidates:
    print(f"Selector: {candidate.selector}")
    print(f"Score: {candidate.score}")
    print(f"Description: {candidate.description}")
    print(f"Sample URLs: {candidate.sample_urls}")

# Advanced usage
miner = SelectorMiner("https://example.com")
candidates = miner.mine_selectors(html_content)
```

### Command Line

```bash
python tools/selector_miner.py
```

Runs with a sample HTML fixture and displays the top candidate selectors.

## Tests (`test_miner_core.py`)

Comprehensive unit tests covering:

- Basic image extraction
- Video thumbnail detection with duration text
- Class-based selector generation
- Background image detection
- Meta image detection
- Evidence gathering (duration patterns, video URLs, class hints)
- Scoring system validation
- Edge cases (empty HTML, invalid URLs, random tokens)
- Integration tests with realistic HTML fixtures

### Running Tests

```bash
python -m pytest tools/test_miner_core.py -v
```

## Phase 2.1 Features - Validation Loop + YAML Emission

### URL Validation
- **HEAD/GET Validation**: Tests top 2-3 candidates with HEAD requests followed by GET for first 1KB
- **Anti-malware Guards**: Blocks malicious schemes (javascript:, data:, file:, ftp:)
- **Content Type Validation**: Ensures URLs return valid image MIME types
- **Size Limits**: Prevents processing of overly large files (>50MB)
- **Success Criteria**: Candidate accepted if ≥2 out of 3 sample URLs succeed

### Recipe Proposal
```python
# Propose a complete recipe for a domain
recipe = await propose_recipe_for_domain("example.com", html_content, "https://example.com")

if recipe:
    print(f"Confidence: {recipe.confidence}")
    print(f"Selectors: {len(recipe.selectors)}")
    print(f"Attributes: {recipe.attributes_priority}")
```

### YAML Emission
- **Compatible Structure**: Generates YAML blocks compatible with existing `site_recipes.yaml`
- **Attribute Inference**: Automatically infers `attributes_priority` from observed image attributes
- **Extra Sources Detection**: Identifies additional image source attributes (`data-src`, `data-lazy-src`, etc.)
- **Merge Functionality**: Safely merges new recipes into existing YAML files

### Usage Example
```python
import asyncio
from tools.selector_miner import propose_recipe_for_domain

async def analyze_site():
    html_content = """<html><body><img src="/thumb.jpg" class="thumbnail" alt="Video"></body></html>"""
    
    # Propose recipe with validation
    recipe = await propose_recipe_for_domain("example.com", html_content)
    
    if recipe and recipe.confidence > 0.5:
        # Generate YAML
        miner = SelectorMiner()
        yaml_output = miner.generate_yaml_recipe(recipe)
        print(yaml_output)
        
        # Merge into existing file
        success = miner.merge_recipe_to_yaml_file(recipe, "site_recipes.yaml")
        print(f"Merged successfully: {success}")

asyncio.run(analyze_site())
```

## Integration with Existing System

The selector miner is designed to work alongside the existing site recipes system in the Mordeaux Face Scanning MVP. It can:

1. **Analyze HTML** from crawled pages
2. **Generate candidate selectors** with evidence-based scoring
3. **Validate URLs** with anti-malware protection
4. **Propose complete recipes** with inferred attributes and extra sources
5. **Emit YAML blocks** compatible with existing `site_recipes.yaml`
6. **Merge recipes** into existing configuration files

This tool enables automatic discovery of effective image selectors, reducing the need for manual configuration of site-specific crawling rules while maintaining security and compatibility with the existing system.
