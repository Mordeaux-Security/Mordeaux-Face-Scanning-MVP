"""
Site Recipes Configuration Module

Provides functionality to load and manage site-specific crawling recipes from YAML files.
Site recipes allow per-domain customization of image extraction selectors, attributes,
and crawling behavior while maintaining fallback to default behavior.

Includes schema validation and auto-fix functionality for malformed entries.
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from urllib.parse import urlparse
import yaml

logger = logging.getLogger(__name__)

# Schema definitions for validation and normalization
VALID_URL_ATTRIBUTES = {
    # Standard image attributes
    'src', 'data-src', 'data-lazy-src', 'data-original', 'data-large', 
    'data-medium', 'data-thumb', 'data-mediumthumb', 'data-video-thumb',
    'data-sfwthumb', 'data-idcdn', 'data-videoid', 'data-thumbs',
    'data-custom-attr', 'poster', 'srcset'
}

VALID_DOM_SOURCES = {
    # Meta tags and structured data
    'og:image', 'link[rel=image_src]', 'video::attr(poster)', 
    'img/srcset', 'source/srcset', 
    "script[type='application/ld+json']::jsonpath($.thumbnailUrl,$.image)"
}

# Known non-URL attributes that should be filtered out
NON_URL_ATTRIBUTES = {
    'alt', 'title', 'data-title', 'data-alt', 'aria-label', 'id', 'class',
    'width', 'height', 'loading', 'decoding', 'crossorigin', 'referrerpolicy',
    'sizes', 'usemap', 'ismap', 'longdesc', 'name', 'data-id', 'data-class'
}

# Valid keys for site recipe entries
VALID_RECIPE_KEYS = {
    'selectors', 'attributes_priority', 'extra_sources', 'method'
}

# Default recipe configuration with normalized schema
DEFAULT_RECIPE = {
    "selectors": [
        {"selector": "img[data-mediumthumb]", "description": "data-mediumthumb attribute"},
        {"selector": "img.js-videoThumb", "description": "js-videoThumb class"},
        {"selector": ".phimage img", "description": "images in .phimage containers"},
        {"selector": "a.latestThumb img", "description": "images in .latestThumb links"},
        {"selector": "img[data-video-thumb]", "description": "data-video-thumb attribute"},
        {"selector": ".video-thumb img", "description": ".video-thumb container images"},
        {"selector": ".thumbnail img", "description": ".thumbnail container images"},
        {"selector": ".thumb img", "description": ".thumb container images"},
        {"selector": "img[width='320'][height='180']", "description": "320x180 dimensions"},
        {"selector": "img[width='640'][height='360']", "description": "640x360 dimensions"},
        {"selector": "img[width='1280'][height='720']", "description": "1280x720 dimensions"},
        {"selector": "img", "description": "all images"}
    ],
    "extra_sources": [
        "data-src", "data-lazy-src", "data-original", "data-large", 
        "data-medium", "data-thumb", "og:image", "link[rel=image_src]",
        "video::attr(poster)", "img/srcset", "source/srcset"
    ],
    "attributes_priority": [
        "src", "data-src", "data-lazy-src", "data-original", 
        "data-large", "data-medium", "data-thumb", "srcset"
    ],
    "method": "smart"
}

# Global cache for loaded recipes
_recipe_cache: Optional[Dict[str, Any]] = None
_recipe_cache_file: Optional[str] = None


def load_site_recipes(path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Load site recipes from YAML file with caching.
    
    Args:
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing all site recipes and defaults
        
    Raises:
        FileNotFoundError: If the recipe file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    global _recipe_cache, _recipe_cache_file
    
    # Normalize path to absolute path
    abs_path = os.path.abspath(path)
    
    # Check if we already have this file cached
    if _recipe_cache is not None and _recipe_cache_file == abs_path:
        logger.debug(f"Using cached recipes from {abs_path}")
        return _recipe_cache
    
    # Check if file exists
    if not os.path.exists(abs_path):
        logger.warning(f"Recipe file not found: {abs_path}, using default recipe only")
        _recipe_cache = {"defaults": DEFAULT_RECIPE, "sites": {}}
        _recipe_cache_file = abs_path
        return _recipe_cache
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as file:
            recipes = yaml.safe_load(file) or {}
        
        # Ensure we have defaults section
        if "defaults" not in recipes:
            logger.info("No defaults section found in recipe file, using built-in defaults")
            recipes["defaults"] = DEFAULT_RECIPE
        
        # Ensure we have sites section
        if "sites" not in recipes:
            recipes["sites"] = {}
        
        # Validate and auto-fix recipe structure
        _validate_and_fix_recipes(recipes)
        
        # Cache the loaded recipes
        _recipe_cache = recipes
        _recipe_cache_file = abs_path
        
        logger.info(f"Loaded {len(recipes.get('sites', {}))} site recipes from {abs_path}")
        return _recipe_cache
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {abs_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading recipe file {abs_path}: {e}")
        raise


def get_recipe_for_host(host: str, path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Get the recipe configuration for a specific host.
    
    Args:
        host: Hostname to get recipe for (e.g., "example.com")
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing the recipe for the host, merged with defaults
        
    Example:
        recipe = get_recipe_for_host("pornhub.com")
        selectors = recipe.get("selectors", [])
        method = recipe.get("method", "smart")
    """
    try:
        # Load all recipes
        all_recipes = load_site_recipes(path)
        
        # Get defaults
        defaults = all_recipes.get("defaults", DEFAULT_RECIPE).copy()
        
        # Get site-specific recipe if it exists
        sites = all_recipes.get("sites", {})
        site_recipe = sites.get(host, {})
        
        # Merge site recipe with defaults (site recipe takes precedence)
        merged_recipe = _merge_recipes(defaults, site_recipe)
        
        logger.debug(f"Recipe for host '{host}': {len(merged_recipe.get('selectors', []))} selectors, method='{merged_recipe.get('method', 'smart')}'")
        return merged_recipe
        
    except Exception as e:
        logger.warning(f"Error getting recipe for host '{host}': {e}, using defaults")
        return DEFAULT_RECIPE.copy()


def get_recipe_for_url(url: str, path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Get the recipe configuration for a specific URL by extracting the host.
    
    Args:
        url: Full URL to get recipe for
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing the recipe for the host, merged with defaults
    """
    try:
        parsed_url = urlparse(url)
        host = parsed_url.netloc.lower()
        
        # Remove port if present
        if ':' in host:
            host = host.split(':')[0]
        
        return get_recipe_for_host(host, path)
        
    except Exception as e:
        logger.warning(f"Error extracting host from URL '{url}': {e}, using defaults")
        return DEFAULT_RECIPE.copy()


def _validate_and_fix_recipes(recipes: Dict[str, Any]) -> None:
    """
    Validate and auto-fix recipe structure according to normalized schema.
    
    Args:
        recipes: Dictionary containing recipe data (modified in place)
    """
    # Validate and fix defaults
    if "defaults" in recipes:
        _validate_and_fix_recipe_entry(recipes["defaults"], "defaults")
    
    # Validate and fix sites
    if "sites" in recipes:
        sites = recipes["sites"]
        if not isinstance(sites, dict):
            logger.warning("Auto-fixing: sites section must be a dictionary, creating empty dict")
            recipes["sites"] = {}
            return
        
        for host, site_config in sites.items():
            if isinstance(site_config, dict):
                _validate_and_fix_recipe_entry(site_config, f"sites.{host}")
            else:
                logger.warning(f"Auto-fixing: sites.{host} must be a dictionary, skipping entry")
                continue


def _validate_and_fix_recipe_entry(recipe: Dict[str, Any], entry_path: str) -> None:
    """
    Validate and auto-fix a single recipe entry.
    
    Args:
        recipe: Recipe dictionary to validate and fix (modified in place)
        entry_path: Path to the entry for logging (e.g., "defaults", "sites.example.com")
    """
    if not isinstance(recipe, dict):
        logger.warning(f"Auto-fixing: {entry_path} must be a dictionary")
        return
    
    # Remove unknown keys
    unknown_keys = set(recipe.keys()) - VALID_RECIPE_KEYS
    if unknown_keys:
        for key in unknown_keys:
            del recipe[key]
        logger.warning(f"Auto-fixing {entry_path}: removed unknown keys {sorted(unknown_keys)}")
    
    # Validate and fix selectors
    if "selectors" in recipe:
        _validate_and_fix_selectors(recipe["selectors"], f"{entry_path}.selectors")
    
    # Validate and fix attributes_priority
    if "attributes_priority" in recipe:
        _validate_and_fix_attributes_priority(recipe["attributes_priority"], f"{entry_path}.attributes_priority")
    
    # Validate and fix extra_sources
    if "extra_sources" in recipe:
        _validate_and_fix_extra_sources(recipe["extra_sources"], f"{entry_path}.extra_sources")


def _validate_and_fix_selectors(selectors: Any, entry_path: str) -> None:
    """
    Validate and fix selectors list.
    
    Args:
        selectors: Selectors list to validate and fix (modified in place)
        entry_path: Path to the entry for logging
    """
    if not isinstance(selectors, list):
        logger.warning(f"Auto-fixing: {entry_path} must be a list, replacing with empty list")
        selectors = []
        return
    
    # Filter out invalid selectors
    valid_selectors = []
    for i, selector in enumerate(selectors):
        if isinstance(selector, dict) and "selector" in selector:
            valid_selectors.append(selector)
        else:
            logger.warning(f"Auto-fixing {entry_path}[{i}]: invalid selector entry, skipping")
    
    # Update the list in place
    selectors[:] = valid_selectors


def _validate_and_fix_attributes_priority(attrs: Any, entry_path: str) -> None:
    """
    Validate and fix attributes_priority list - only URL-bearing attributes allowed.
    
    Args:
        attrs: Attributes list to validate and fix (modified in place)
        entry_path: Path to the entry for logging
    """
    if not isinstance(attrs, list):
        logger.warning(f"Auto-fixing: {entry_path} must be a list, replacing with empty list")
        attrs = []
        return
    
    # Filter to only URL-bearing attributes
    original_attrs = set(attrs)
    url_attrs = [attr for attr in attrs if attr in VALID_URL_ATTRIBUTES]
    
    # Remove non-URL attributes
    removed_attrs = original_attrs - set(url_attrs)
    if removed_attrs:
        logger.warning(f"Auto-fixing {entry_path}: removed non-URL attributes {sorted(removed_attrs)}")
    
    # Update the list in place
    attrs[:] = url_attrs


def _validate_and_fix_extra_sources(sources: Any, entry_path: str) -> None:
    """
    Validate and fix extra_sources list - only URL-bearing attributes and DOM sources allowed.
    
    Args:
        sources: Sources list to validate and fix (modified in place)
        entry_path: Path to the entry for logging
    """
    if not isinstance(sources, list):
        logger.warning(f"Auto-fixing: {entry_path} must be a list, replacing with empty list")
        sources = []
        return
    
    # Filter to only valid URL attributes and DOM sources
    original_sources = set(sources)
    valid_sources = [source for source in sources if source in VALID_URL_ATTRIBUTES or source in VALID_DOM_SOURCES]
    
    # Remove invalid sources
    removed_sources = original_sources - set(valid_sources)
    if removed_sources:
        logger.warning(f"Auto-fixing {entry_path}: removed invalid sources {sorted(removed_sources)}")
    
    # Update the list in place
    sources[:] = valid_sources


def _merge_recipes(defaults: Dict[str, Any], site_recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge site-specific recipe with defaults.
    
    Args:
        defaults: Default recipe configuration
        site_recipe: Site-specific recipe configuration
        
    Returns:
        Merged recipe with site settings taking precedence
    """
    merged = defaults.copy()
    
    # Merge simple fields (site recipe overwrites defaults)
    for key in ["method", "extra_sources", "attributes_priority"]:
        if key in site_recipe:
            merged[key] = site_recipe[key]
    
    # Merge selectors (site recipe replaces defaults entirely if present)
    if "selectors" in site_recipe:
        merged["selectors"] = site_recipe["selectors"]
    
    return merged


def clear_recipe_cache() -> None:
    """Clear the recipe cache to force reload on next access."""
    global _recipe_cache, _recipe_cache_file
    _recipe_cache = None
    _recipe_cache_file = None
    logger.debug("Recipe cache cleared")


def get_recipe_info(path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Get information about loaded recipes without loading them.
    
    Args:
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing recipe file information
    """
    abs_path = os.path.abspath(path)
    
    info = {
        "file_path": abs_path,
        "exists": os.path.exists(abs_path),
        "cached": _recipe_cache is not None and _recipe_cache_file == abs_path,
        "site_count": 0,
        "has_defaults": False
    }
    
    if info["exists"]:
        try:
            with open(abs_path, 'r', encoding='utf-8') as file:
                recipes = yaml.safe_load(file) or {}
            
            info["site_count"] = len(recipes.get("sites", {}))
            info["has_defaults"] = "defaults" in recipes
            
        except Exception as e:
            info["error"] = str(e)
    
    return info
