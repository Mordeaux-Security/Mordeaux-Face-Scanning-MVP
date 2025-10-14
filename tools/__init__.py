"""
Tools package for the Mordeaux Face Scanning MVP.

Contains utilities for selector mining and other crawling tools.
"""

from .selector_miner import (
    mine_page,
    emit_recipe_yaml_block,
    resolve_image_url,
    validate_image_request,
    extract_extra_sources,
    stable_selector,
    gather_evidence,
    score_candidate,
    render_js,
    discover_listing_links,
    Limits,
    MinedResult,
    CandidateSelector,
    MinerNetworkError,
    MinerSchemaError,
    ATTR_PRIORITY,
    HOST_DENY,
    MAX_REDIRECTS,
    MAX_IMAGE_BYTES,
    PLAYWRIGHT_AVAILABLE
)

__all__ = [
    'mine_page',
    'emit_recipe_yaml_block',
    'resolve_image_url',
    'validate_image_request',
    'extract_extra_sources',
    'stable_selector',
    'gather_evidence',
    'score_candidate',
    'render_js',
    'discover_listing_links',
    'Limits',
    'MinedResult',
    'CandidateSelector',
    'MinerNetworkError',
    'MinerSchemaError',
    'ATTR_PRIORITY',
    'HOST_DENY',
    'MAX_REDIRECTS',
    'MAX_IMAGE_BYTES',
    'PLAYWRIGHT_AVAILABLE'
]
