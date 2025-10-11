"""
Tools package for the Mordeaux Face Scanning MVP.

Contains utilities for selector mining and other crawling tools.
"""

from .selector_miner import (
    SelectorMiner, 
    CandidateSelector, 
    ImageNode, 
    ValidationResult,
    RecipeCandidate,
    mine_selectors_for_url,
    mine_selectors_with_js_fallback,
    propose_recipe_for_domain,
    PLAYWRIGHT_AVAILABLE
)

__all__ = [
    'SelectorMiner',
    'CandidateSelector', 
    'ImageNode',
    'ValidationResult',
    'RecipeCandidate',
    'mine_selectors_for_url',
    'mine_selectors_with_js_fallback',
    'propose_recipe_for_domain',
    'PLAYWRIGHT_AVAILABLE'
]
