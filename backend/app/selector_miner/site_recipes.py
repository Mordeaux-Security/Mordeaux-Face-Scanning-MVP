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
from typing import Dict, Any, Optional, List, Set, Tuple, Literal, Union
from urllib.parse import urlparse
import yaml

logger = logging.getLogger(__name__)

# Schema definitions for validation and normalization
SCHEMA_URL_ATTRIBUTES = ["data-src", "data-srcset", "srcset", "src"]
SCHEMA_SELECTOR_KINDS = ["video_grid", "album_grid", "gallery_images", "wordpress_content", "direct_image"]

# Valid extra_sources patterns
VALID_EXTRA_SOURCE_PATTERNS = [
    r'^.+::attr\([^)]+\)$',  # <css>::attr(<name>)
    r'^script\[type=[\'"]application/ld\+json[\'"]\]::jsonpath\([^)]+\)$',  # JSON-LD jsonpath
    r'^::style\(background-image\)$'  # Background image style
]

# Valid keys for site recipe entries
VALID_SCHEMA_KEYS = {
    'schema_version', 'defaults', 'sites'
}

# Valid keys for recipe entries (within defaults/sites)
VALID_RECIPE_ENTRY_KEYS = {
    'selectors', 'attributes_priority', 'extra_sources', 'method'
}

# Default recipe configuration
DEFAULT_RECIPE = {
    "selectors": [
        {"kind": "video_grid", "css": ".video-thumb img"},
        {"kind": "album_grid", "css": ".album-thumb img"},
        {"kind": "album_grid", "css": "a[href*='/album'] img"},
        {"kind": "gallery_images", "css": ".gallery img"},
        {"kind": "gallery_images", "css": "figure img"}
    ],
    "extra_sources": [
        "meta[property='og:image']::attr(content)",
        "img::attr(srcset)",
        "source::attr(data-srcset)",
        "source::attr(srcset)",
        "script[type='application/ld+json']::jsonpath($.image, $.associatedMedia[*].contentUrl, $..image, $..contentUrl)"
    ],
    "attributes_priority": SCHEMA_URL_ATTRIBUTES.copy(),
    "method": "smart"
}

# Global cache for loaded recipes
_recipe_cache: Optional[Dict[str, Any]] = None
_recipe_cache_file: Optional[str] = None


def load_site_recipes(path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Load site recipes from YAML file with schema validation and auto-fixing.
    
    Args:
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing all site recipes and defaults (schema compliant)
        
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
        logger.warning(f"Recipe file not found: {abs_path}, using defaults")
        _recipe_cache = {
            "schema_version": 2,
            "defaults": DEFAULT_RECIPE,
            "sites": {}
        }
        _recipe_cache_file = abs_path
        return _recipe_cache
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as file:
            recipes = yaml.safe_load(file) or {}
        
        # Enforce schema structure
        recipes = _enforce_schema(recipes)
        
        # Cache the loaded recipes
        _recipe_cache = recipes
        _recipe_cache_file = abs_path
        
        logger.debug(f"Loaded recipes from {abs_path}: {len(recipes.get('sites', {}))} sites")
        return recipes
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {abs_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading recipe file {abs_path}: {e}")
        raise


def save_site_recipes(recipes: Dict[str, Any], path: str = "site_recipes.yaml") -> bool:
    """
    Save site recipes to YAML file.
    
    Args:
        recipes: Dictionary containing site recipes to save
        path: Path to save the YAML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import yaml
        
        # Normalize path to absolute path
        abs_path = os.path.abspath(path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        # Write the recipes to file
        with open(abs_path, 'w', encoding='utf-8') as f:
            yaml.dump(recipes, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Update cache
        global _recipe_cache, _recipe_cache_file
        _recipe_cache = recipes
        _recipe_cache_file = abs_path
        
        logger.info(f"Saved site recipes to {abs_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving site recipes to {path}: {e}")
        return False


def get_recipe_for_host(host: str, path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Get the recipe configuration for a specific host.
    
    Args:
        host: Hostname to get recipe for (e.g., "example.com")
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing the recipe for the host, merged with defaults (schema compliant)
        
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
        Dictionary containing the recipe for the host, merged with defaults (schema compliant)
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


def _enforce_schema(recipes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce schema structure with auto-fixing and validation.
    
    Args:
        recipes: Dictionary containing recipe data
        
    Returns:
        Schema compliant recipes dictionary
    """
    # Start with schema structure
    schema_recipes = {
        "schema_version": 2,
        "defaults": DEFAULT_RECIPE.copy(),
        "sites": {}
    }
    
    # Remove invalid top-level keys
    invalid_keys = set(recipes.keys()) - VALID_SCHEMA_KEYS
    if invalid_keys:
        logger.warning(f"Auto-fixed: removed invalid top-level keys {sorted(invalid_keys)}")
    
    # Process defaults section
    if "defaults" in recipes and isinstance(recipes["defaults"], dict):
        schema_recipes["defaults"] = _normalize_recipe_entry(recipes["defaults"], "defaults")
    else:
        logger.warning("Auto-fixed: invalid defaults section, using defaults")
    
    # Process sites section
    if "sites" in recipes and isinstance(recipes["sites"], dict):
        for host, site_config in recipes["sites"].items():
            if isinstance(site_config, dict):
                schema_recipes["sites"][host] = _normalize_recipe_entry(site_config, f"sites.{host}")
            else:
                logger.warning(f"Auto-fixed: sites.{host} must be a dictionary, skipping entry")
    else:
        logger.warning("Auto-fixed: invalid sites section, creating empty dict")
    
    return schema_recipes


def _normalize_recipe_entry(recipe: Dict[str, Any], entry_path: str) -> Dict[str, Any]:
    """
    Normalize a recipe entry to schema compliance.
    
    Args:
        recipe: Recipe dictionary to normalize
        entry_path: Path to the entry for logging purposes
        
    Returns:
        Schema compliant recipe dictionary
    """
    if not isinstance(recipe, dict):
        logger.warning(f"Auto-fixed: {entry_path} must be a dictionary, using defaults")
        return DEFAULT_RECIPE.copy()
    
    normalized = {}
    
    # Remove invalid keys
    invalid_keys = set(recipe.keys()) - VALID_RECIPE_ENTRY_KEYS
    if invalid_keys:
        logger.warning(f"Auto-fixed {entry_path}: removed invalid keys {sorted(invalid_keys)}")
    
    # Normalize selectors
    if "selectors" in recipe:
        normalized["selectors"] = _normalize_selectors(recipe["selectors"], f"{entry_path}.selectors")
    else:
        normalized["selectors"] = DEFAULT_RECIPE["selectors"].copy()
    
    # Normalize attributes_priority (enforce URL attributes only)
    if "attributes_priority" in recipe:
        normalized["attributes_priority"] = _normalize_attributes_priority(recipe["attributes_priority"], f"{entry_path}.attributes_priority")
    else:
        normalized["attributes_priority"] = SCHEMA_URL_ATTRIBUTES.copy()
    
    # Normalize extra_sources (enforce valid patterns only)
    if "extra_sources" in recipe:
        normalized["extra_sources"] = _normalize_extra_sources(recipe["extra_sources"], f"{entry_path}.extra_sources")
    else:
        normalized["extra_sources"] = DEFAULT_RECIPE["extra_sources"].copy()
    
    # Normalize method
    if "method" in recipe and isinstance(recipe["method"], str):
        normalized["method"] = recipe["method"]
    else:
        normalized["method"] = DEFAULT_RECIPE["method"]
    
    return normalized


def _normalize_selectors(selectors: Any, entry_path: str) -> List[Dict[str, str]]:
    """
    Normalize selectors list to schema compliance with backward compatibility.
    
    Args:
        selectors: Selectors list to normalize
        entry_path: Path to the entry for logging purposes
        
    Returns:
        List of schema compliant selector dictionaries
    """
    if not isinstance(selectors, list):
        logger.warning(f"Auto-fixed: {entry_path} must be a list, using default selectors")
        return DEFAULT_RECIPE["selectors"].copy()
    
    # Filter out invalid selectors and ensure proper format
    valid_selectors = []
    for i, selector in enumerate(selectors):
        if isinstance(selector, dict):
            # Schema format: {"kind": "video_grid", "css": ".selector"}
            if "kind" in selector and "css" in selector:
                if selector["kind"] in SCHEMA_SELECTOR_KINDS:
                    normalized_selector = {
                        "kind": str(selector["kind"]),
                        "css": str(selector["css"])
                    }
                    valid_selectors.append(normalized_selector)
                else:
                    logger.warning(f"Auto-fixed {entry_path}[{i}]: invalid kind '{selector['kind']}', skipping")
            # Legacy format: {"selector": ".selector", "description": "..."} - convert to new format
            elif "selector" in selector:
                logger.info(f"Auto-converted {entry_path}[{i}]: legacy selector to new format")
                normalized_selector = {
                    "kind": "video_grid",  # Default kind for backward compatibility
                    "css": str(selector["selector"])
                }
                valid_selectors.append(normalized_selector)
            else:
                logger.warning(f"Auto-fixed {entry_path}[{i}]: invalid selector entry, skipping")
        # String format: ".selector" - convert to new format
        elif isinstance(selector, str):
            logger.info(f"Auto-converted {entry_path}[{i}]: string selector to new format")
            normalized_selector = {
                "kind": "video_grid",  # Default kind for backward compatibility
                "css": str(selector)
            }
            valid_selectors.append(normalized_selector)
        else:
            logger.warning(f"Auto-fixed {entry_path}[{i}]: invalid selector entry, skipping")
    
    return valid_selectors if valid_selectors else DEFAULT_RECIPE["selectors"].copy()


def _normalize_attributes_priority(attrs: Any, entry_path: str) -> List[str]:
    """
    Normalize attributes_priority list to schema compliance (only URL attributes).
    
    Args:
        attrs: Attributes list to normalize
        entry_path: Path to the entry for logging purposes
        
    Returns:
        List of schema compliant URL attributes
    """
    if not isinstance(attrs, list):
        logger.warning(f"Auto-fixed: {entry_path} must be a list, using defaults")
        return SCHEMA_URL_ATTRIBUTES.copy()
    
    # Filter to only URL attributes, preserving order
    original_attrs = set(attrs)
    url_attrs = []
    for attr in attrs:
        if attr in SCHEMA_URL_ATTRIBUTES and attr not in url_attrs:
            url_attrs.append(attr)
    
    # Log removed attributes
    removed_attrs = original_attrs - set(url_attrs)
    if removed_attrs:
        logger.warning(f"Auto-fixed {entry_path}: removed non-URL attributes {sorted(removed_attrs)}")
    
    return url_attrs if url_attrs else SCHEMA_URL_ATTRIBUTES.copy()


def _normalize_extra_sources(sources: Any, entry_path: str) -> List[str]:
    """
    Normalize extra_sources list to schema compliance.
    
    Args:
        sources: Sources list to normalize
        entry_path: Path to the entry for logging purposes
        
    Returns:
        List of schema compliant extra sources
    """
    if not isinstance(sources, list):
        logger.warning(f"Auto-fixed: {entry_path} must be a list, using defaults")
        return DEFAULT_RECIPE["extra_sources"].copy()
    
    # Filter to only valid patterns
    valid_sources = []
    for source in sources:
        if _is_valid_extra_source(source):
            valid_sources.append(source)
    
    # Log removed sources
    original_sources = set(sources)
    removed_sources = original_sources - set(valid_sources)
    if removed_sources:
        logger.warning(f"Auto-fixed {entry_path}: removed invalid sources {sorted(removed_sources)}")
    
    return valid_sources if valid_sources else DEFAULT_RECIPE["extra_sources"].copy()


def _is_valid_extra_source(source: str) -> bool:
    """
    Check if an extra source pattern is valid.
    
    Args:
        source: Source pattern to validate
        
    Returns:
        True if the source pattern is valid, False otherwise
    """
    if not isinstance(source, str):
        return False
    
    return any(re.match(pattern, source) for pattern in VALID_EXTRA_SOURCE_PATTERNS)


def _merge_recipes(defaults: Dict[str, Any], site_recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge site recipe with defaults, with site recipe taking precedence.
    
    Args:
        defaults: Default recipe configuration
        site_recipe: Site-specific recipe configuration
        
    Returns:
        Merged recipe configuration
    """
    merged = defaults.copy()
    
    # Merge each field, with site recipe taking precedence
    for key in VALID_RECIPE_ENTRY_KEYS:
        if key in site_recipe:
            merged[key] = site_recipe[key]
    
    return merged


def clear_recipe_cache():
    """Clear the global recipe cache."""
    global _recipe_cache, _recipe_cache_file
    _recipe_cache = None
    _recipe_cache_file = None
    logger.debug("Recipe cache cleared")


def get_recipe_info(path: str = "site_recipes.yaml") -> Dict[str, Any]:
    """
    Get information about the recipe file.
    
    Args:
        path: Path to the YAML file containing site recipes
        
    Returns:
        Dictionary containing recipe file information
    """
    info = {
        "file_path": os.path.abspath(path),
        "exists": False,
        "has_defaults": False,
        "schema_version": None,
        "site_count": 0,
        "error": None
    }
    
    try:
        if os.path.exists(info["file_path"]):
            info["exists"] = True
            recipes = load_site_recipes(path)
            info["site_count"] = len(recipes.get("sites", {}))
            info["has_defaults"] = "defaults" in recipes
            info["schema_version"] = recipes.get("schema_version")
            
    except Exception as e:
        info["error"] = str(e)
    
    return info
