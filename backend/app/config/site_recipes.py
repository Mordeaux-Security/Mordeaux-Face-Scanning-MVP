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

# Schema v1 definitions for validation and normalization
SCHEMA_V1_URL_ATTRIBUTES = ["data-src", "data-srcset", "srcset", "src"]

# Valid extra_sources patterns for schema v1
VALID_EXTRA_SOURCE_PATTERNS = [
    r'^.+::attr\([^)]+\)$',  # <css>::attr(<name>)
    r'^script\[type=[\'"]application/ld\+json[\'"]\]::jsonpath\([^)]+\)$',  # JSON-LD jsonpath
    r'^::style\(background-image\)$'  # Background image style
]

# Valid keys for site recipe entries (schema v1)
VALID_SCHEMA_V1_KEYS = {
    'schema_version', 'defaults', 'sites'
}

# Valid keys for recipe entries (within defaults/sites)
VALID_RECIPE_ENTRY_KEYS = {
    'selectors', 'attributes_priority', 'extra_sources', 'method'
}

# Schema v1 default recipe configuration
DEFAULT_RECIPE_V1 = {
    "selectors": [
        {"selector": ".video-thumb img", "description": ".video-thumb container images"},
        {"selector": ".thumbnail img", "description": ".thumbnail container images"},
        {"selector": ".thumb img", "description": ".thumb container images"},
        {"selector": "picture img", "description": "picture element images"},
        {"selector": "img.lazy", "description": "lazy-loaded images"},
        {"selector": "img", "description": "all images"}
    ],
    "extra_sources": [
        "meta[property='og:image']::attr(content)",
        "link[rel='image_src']::attr(href)",
        "video::attr(poster)",
        "img::attr(srcset)",
        "source::attr(srcset)",
        "source::attr(data-srcset)",
        "script[type='application/ld+json']::jsonpath($.thumbnailUrl, $.image, $..thumbnailUrl, $..image, $..url)",
        "::style(background-image)"
    ],
    "attributes_priority": SCHEMA_V1_URL_ATTRIBUTES.copy(),
    "method": "smart"
}

# Global cache for loaded recipes
_recipe_cache: Optional[Dict[str, Any]] = None
_recipe_cache_file: Optional[str] = None


def load_site_recipes(path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Load site recipes from YAML file with schema v1 validation and auto-fixing.
    
    Args:
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing all site recipes and defaults (schema v1 compliant)
        
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
        logger.warning(f"Recipe file not found: {abs_path}, using schema v1 defaults")
        _recipe_cache = {
            "schema_version": 1,
            "defaults": DEFAULT_RECIPE_V1,
            "sites": {}
        }
        _recipe_cache_file = abs_path
        return _recipe_cache
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as file:
            recipes = yaml.safe_load(file) or {}
        
        # Enforce schema v1 structure
        recipes = _enforce_schema_v1(recipes)
        
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
    Get the recipe configuration for a specific host (schema v1).
    
    Args:
        host: Hostname to get recipe for (e.g., "example.com")
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing the recipe for the host, merged with defaults (schema v1 compliant)
        
    Example:
        recipe = get_recipe_for_host("pornhub.com")
        selectors = recipe.get("selectors", [])
        method = recipe.get("method", "smart")
    """
    try:
        # Load all recipes
        all_recipes = load_site_recipes(path)
        
        # Get defaults (schema v1)
        defaults = all_recipes.get("defaults", DEFAULT_RECIPE_V1).copy()
        
        # Get site-specific recipe if it exists
        sites = all_recipes.get("sites", {})
        site_recipe = sites.get(host, {})
        
        # Merge site recipe with defaults (site recipe takes precedence)
        merged_recipe = _merge_recipes_v1(defaults, site_recipe)
        
        logger.debug(f"Recipe for host '{host}': {len(merged_recipe.get('selectors', []))} selectors, method='{merged_recipe.get('method', 'smart')}'")
        return merged_recipe
        
    except Exception as e:
        logger.warning(f"Error getting recipe for host '{host}': {e}, using defaults")
        return DEFAULT_RECIPE_V1.copy()


def get_recipe_for_url(url: str, path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Get the recipe configuration for a specific URL by extracting the host (schema v1).
    
    Args:
        url: Full URL to get recipe for
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing the recipe for the host, merged with defaults (schema v1 compliant)
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
        return DEFAULT_RECIPE_V1.copy()


def _enforce_schema_v1(recipes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce schema v1 structure with auto-fixing and validation.
    
    Args:
        recipes: Dictionary containing recipe data
        
    Returns:
        Schema v1 compliant recipes dictionary
    """
    # Start with schema v1 structure
    schema_v1_recipes = {
        "schema_version": 1,
        "defaults": DEFAULT_RECIPE_V1.copy(),
        "sites": {}
    }
    
    # Remove invalid top-level keys
    invalid_keys = set(recipes.keys()) - VALID_SCHEMA_V1_KEYS
    if invalid_keys:
        logger.warning(f"Auto-fixed: removed invalid top-level keys {sorted(invalid_keys)}")
    
    # Process defaults section
    if "defaults" in recipes and isinstance(recipes["defaults"], dict):
        schema_v1_recipes["defaults"] = _normalize_recipe_entry_v1(recipes["defaults"], "defaults")
    else:
        logger.warning("Auto-fixed: invalid defaults section, using schema v1 defaults")
    
    # Process sites section
    if "sites" in recipes and isinstance(recipes["sites"], dict):
        for host, site_config in recipes["sites"].items():
            if isinstance(site_config, dict):
                schema_v1_recipes["sites"][host] = _normalize_recipe_entry_v1(site_config, f"sites.{host}")
            else:
                logger.warning(f"Auto-fixed: sites.{host} must be a dictionary, skipping entry")
    else:
        logger.warning("Auto-fixed: invalid sites section, creating empty dict")
    
    return schema_v1_recipes


def _normalize_recipe_entry_v1(recipe: Dict[str, Any], entry_path: str) -> Dict[str, Any]:
    """
    Normalize a recipe entry to schema v1 compliance.
    
    Args:
        recipe: Recipe dictionary to normalize
        entry_path: Path to the entry for logging (e.g., "defaults", "sites.example.com")
        
    Returns:
        Schema v1 compliant recipe dictionary
    """
    if not isinstance(recipe, dict):
        logger.warning(f"Auto-fixed: {entry_path} must be a dictionary, using defaults")
        return DEFAULT_RECIPE_V1.copy()
    
    normalized = {}
    
    # Remove unknown keys
    unknown_keys = set(recipe.keys()) - VALID_RECIPE_ENTRY_KEYS
    if unknown_keys:
        logger.warning(f"Auto-fixed {entry_path}: removed unknown keys {sorted(unknown_keys)}")
    
    # Normalize selectors
    if "selectors" in recipe:
        normalized["selectors"] = _normalize_selectors_v1(recipe["selectors"], f"{entry_path}.selectors")
    else:
        normalized["selectors"] = DEFAULT_RECIPE_V1["selectors"].copy()
    
    # Normalize attributes_priority (enforce schema v1 URL attributes only)
    if "attributes_priority" in recipe:
        normalized["attributes_priority"] = _normalize_attributes_priority_v1(recipe["attributes_priority"], f"{entry_path}.attributes_priority")
    else:
        normalized["attributes_priority"] = SCHEMA_V1_URL_ATTRIBUTES.copy()
    
    # Normalize extra_sources (enforce schema v1 patterns only)
    if "extra_sources" in recipe:
        normalized["extra_sources"] = _normalize_extra_sources_v1(recipe["extra_sources"], f"{entry_path}.extra_sources")
    else:
        normalized["extra_sources"] = DEFAULT_RECIPE_V1["extra_sources"].copy()
    
    # Normalize method
    normalized["method"] = recipe.get("method", "smart")
    
    return normalized


def _normalize_selectors_v1(selectors: Any, entry_path: str) -> List[Dict[str, str]]:
    """
    Normalize selectors list to schema v1 compliance.
    
    Args:
        selectors: Selectors list to normalize
        entry_path: Path to the entry for logging
        
    Returns:
        List of normalized selector dictionaries
    """
    if not isinstance(selectors, list):
        logger.warning(f"Auto-fixed: {entry_path} must be a list, using default selectors")
        return DEFAULT_RECIPE_V1["selectors"].copy()
    
    # Filter out invalid selectors and ensure proper format
    valid_selectors = []
    for i, selector in enumerate(selectors):
        if isinstance(selector, dict) and "selector" in selector:
            # Ensure description exists
            normalized_selector = {
                "selector": str(selector["selector"]),
                "description": selector.get("description", selector["selector"])
            }
            valid_selectors.append(normalized_selector)
        else:
            logger.warning(f"Auto-fixed {entry_path}[{i}]: invalid selector entry, skipping")
    
    return valid_selectors if valid_selectors else DEFAULT_RECIPE_V1["selectors"].copy()


def _normalize_attributes_priority_v1(attrs: Any, entry_path: str) -> List[str]:
    """
    Normalize attributes_priority list to schema v1 compliance (only URL attributes).
    
    Args:
        attrs: Attributes list to normalize
        entry_path: Path to the entry for logging
        
    Returns:
        List of schema v1 compliant URL attributes
    """
    if not isinstance(attrs, list):
        logger.warning(f"Auto-fixed: {entry_path} must be a list, using schema v1 defaults")
        return SCHEMA_V1_URL_ATTRIBUTES.copy()
    
    # Filter to only schema v1 URL attributes, preserving order
    original_attrs = set(attrs)
    url_attrs = []
    for attr in attrs:
        if attr in SCHEMA_V1_URL_ATTRIBUTES and attr not in url_attrs:
            url_attrs.append(attr)
    
    # Remove non-URL attributes
    removed_attrs = original_attrs - set(url_attrs)
    if removed_attrs:
        logger.warning(f"Auto-fixed {entry_path}: removed non-URL attributes {sorted(removed_attrs)}")
    
    return url_attrs if url_attrs else SCHEMA_V1_URL_ATTRIBUTES.copy()


def _normalize_extra_sources_v1(sources: Any, entry_path: str) -> List[str]:
    """
    Normalize extra_sources list to schema v1 compliance.
    
    Args:
        sources: Sources list to normalize
        entry_path: Path to the entry for logging
        
    Returns:
        List of schema v1 compliant extra sources
    """
    if not isinstance(sources, list):
        logger.warning(f"Auto-fixed: {entry_path} must be a list, using schema v1 defaults")
        return DEFAULT_RECIPE_V1["extra_sources"].copy()
    
    # Filter to only valid schema v1 patterns
    valid_sources = []
    for source in sources:
        if _is_valid_extra_source_v1(source):
            valid_sources.append(source)
    
    # Remove invalid sources
    original_sources = set(sources)
    removed_sources = original_sources - set(valid_sources)
    if removed_sources:
        logger.warning(f"Auto-fixed {entry_path}: removed invalid sources {sorted(removed_sources)}")
    
    return valid_sources if valid_sources else DEFAULT_RECIPE_V1["extra_sources"].copy()


def _is_valid_extra_source_v1(source: str) -> bool:
    """
    Check if an extra source matches schema v1 patterns.
    
    Args:
        source: Source string to validate
        
    Returns:
        True if source matches schema v1 patterns
    """
    for pattern in VALID_EXTRA_SOURCE_PATTERNS:
        if re.match(pattern, source):
            return True
    return False


def _merge_recipes_v1(defaults: Dict[str, Any], site_recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge site-specific recipe with defaults (schema v1).
    
    Args:
        defaults: Default recipe configuration (schema v1)
        site_recipe: Site-specific recipe configuration (schema v1)
        
    Returns:
        Merged recipe with site settings taking precedence (schema v1 compliant)
    """
    merged = defaults.copy()
    
    # Merge simple fields (site recipe overwrites defaults)
    for key in ["method"]:
        if key in site_recipe:
            merged[key] = site_recipe[key]
    
    # Merge attributes_priority (site recipe overwrites defaults)
    if "attributes_priority" in site_recipe:
        merged["attributes_priority"] = site_recipe["attributes_priority"]
    
    # Merge extra_sources (site recipe overwrites defaults)
    if "extra_sources" in site_recipe:
        merged["extra_sources"] = site_recipe["extra_sources"]
    
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
    Get information about loaded recipes without loading them (schema v1).
    
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
        "has_defaults": False,
        "schema_version": None
    }
    
    if info["exists"]:
        try:
            with open(abs_path, 'r', encoding='utf-8') as file:
                recipes = yaml.safe_load(file) or {}
            
            info["site_count"] = len(recipes.get("sites", {}))
            info["has_defaults"] = "defaults" in recipes
            info["schema_version"] = recipes.get("schema_version")
            
        except Exception as e:
            info["error"] = str(e)
    
    return info
