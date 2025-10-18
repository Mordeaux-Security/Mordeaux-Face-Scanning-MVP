# Tools Directory

This directory contains various utility tools for the Mordeaux Face Scanning MVP.

## Available Tools

### Selector Mining Tools
The selector mining functionality has been moved to `backend/app/selector_miner/` for better organization. This includes:

- Core selector mining logic
- Site recipe management
- CLI tools for mining and reviewing selectors
- Test suites

See `backend/app/selector_miner/README.md` for detailed documentation.

### Remaining Tools

- `test_cli.py` - CLI testing utilities
- `redirect_utils.py` - HTTP redirect handling utilities (moved to selector_miner)

## Usage

For selector mining functionality, use the tools in `backend/app/selector_miner/`:

```bash
# Mine selectors for a domain
python backend/app/selector_miner/mine-selectors --domain example.com --urls urls.txt --out site_recipes.yaml

# Review selectors
python backend/app/selector_miner/review-selectors --domain example.com
```

## Migration Notes

The selector miner and related functionality has been reorganized into a dedicated module structure similar to the crawler module. This provides better separation of concerns and makes the codebase more maintainable.
