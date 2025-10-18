"""
Selector Miner Module

A deterministic selector-miner that analyzes HTML content to generate
candidate CSS selectors for image extraction with evidence-based scoring.

This module provides functionality for:
- HTML parsing and image extraction
- CSS selector generation and validation
- Site recipe management and YAML emission
- CLI tools for mining and reviewing selectors
"""

from .selector_miner import (
    mine_page,
    emit_recipe_yaml_block,
    Limits,
    MinedResult,
    CandidateSelector,
    MinerNetworkError,
    MinerSchemaError,
    ATTR_PRIORITY,
    MAX_IMAGE_BYTES,
    HTTP_TIMEOUT
)

from .site_recipes import (
    load_site_recipes,
    get_recipe_for_host,
    get_recipe_for_url,
    clear_recipe_cache,
    get_recipe_info
)

from .redirect_utils import (
    create_safe_client,
    fetch_with_redirects,
    fetch_html_with_redirects,
    head_with_redirects,
    validate_url_security
)

__all__ = [
    # Core mining functions
    'mine_page',
    'emit_recipe_yaml_block',
    'Limits',
    'MinedResult',
    'CandidateSelector',
    'MinerNetworkError',
    'MinerSchemaError',
    'ATTR_PRIORITY',
    'MAX_IMAGE_BYTES',
    'HTTP_TIMEOUT',
    
    # Site recipes
    'load_site_recipes',
    'get_recipe_for_host',
    'get_recipe_for_url',
    'clear_recipe_cache',
    'get_recipe_info',
    
    # HTTP utilities
    'create_safe_client',
    'fetch_with_redirects',
    'fetch_html_with_redirects',
    'head_with_redirects',
    'validate_url_security'
]
