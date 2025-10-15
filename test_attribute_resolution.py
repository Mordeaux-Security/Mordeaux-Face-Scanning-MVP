#!/usr/bin/env python3
"""
Test script to verify attribute resolution order follows schema v1 priority.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from tools.selector_miner import SelectorMiner
from bs4 import BeautifulSoup

def test_attribute_resolution_order():
    """Test that attribute resolution follows schema v1 priority order."""
    
    # Create test HTML with various attribute combinations
    test_html = """
    <html>
    <body>
        <!-- Test 1: data-src should be preferred over src -->
        <img src="http://example.com/fallback.jpg" data-src="http://example.com/preferred.jpg" alt="Test 1">
        
        <!-- Test 2: data-srcset should be preferred over srcset -->
        <img srcset="http://example.com/fallback1.jpg 1x, http://example.com/fallback2.jpg 2x" 
             data-srcset="http://example.com/preferred1.jpg 1x, http://example.com/preferred2.jpg 2x" 
             src="http://example.com/fallback.jpg" alt="Test 2">
        
        <!-- Test 3: srcset should be preferred over src -->
        <img srcset="http://example.com/responsive1.jpg 1x, http://example.com/responsive2.jpg 2x" 
             src="http://example.com/fallback.jpg" alt="Test 3">
        
        <!-- Test 4: data-src should be preferred over srcset -->
        <img data-src="http://example.com/lazy.jpg" 
             srcset="http://example.com/responsive1.jpg 1x, http://example.com/responsive2.jpg 2x" 
             src="http://example.com/fallback.jpg" alt="Test 4">
        
        <!-- Test 5: source tag should prefer data-srcset -->
        <picture>
            <source media="(min-width: 800px)" 
                    data-srcset="http://example.com/large.jpg 800w" 
                    srcset="http://example.com/fallback_large.jpg 800w">
            <source media="(min-width: 400px)" 
                    data-srcset="http://example.com/medium.jpg 400w" 
                    srcset="http://example.com/fallback_medium.jpg 400w">
            <img src="http://example.com/small.jpg" alt="Test 5">
        </picture>
    </body>
    </html>
    """
    
    # Create miner instance
    miner = SelectorMiner("http://example.com")
    
    # Parse HTML
    soup = BeautifulSoup(test_html, 'html.parser')
    img_tags = soup.find_all('img')
    source_tags = soup.find_all('source')
    
    print("Testing attribute resolution order...")
    print("=" * 50)
    
    # Test img tag resolution
    for i, img in enumerate(img_tags, 1):
        extracted_url = miner._extract_image_url(img)
        used_attr = miner._determine_used_attribute(img, extracted_url)
        print(f"Test {i} (img):")
        print(f"  Extracted URL: {extracted_url}")
        print(f"  Used attribute: {used_attr}")
        print(f"  All attributes: {dict(img.attrs)}")
        print()
    
    # Test source tag resolution
    for i, source in enumerate(source_tags, 1):
        extracted_url = miner._extract_image_url(source)
        used_attr = miner._determine_used_attribute(source, extracted_url)
        print(f"Test {i} (source):")
        print(f"  Extracted URL: {extracted_url}")
        print(f"  Used attribute: {used_attr}")
        print(f"  All attributes: {dict(source.attrs)}")
        print()
    
    # Test attributes priority inference
    print("Testing attributes_priority inference...")
    print("=" * 50)
    
    # Mine selectors to get image nodes
    miner.mine_selectors(test_html)
    
    # Get attributes priority
    attributes_priority = miner._infer_attributes_priority(miner.image_nodes)
    print(f"Inferred attributes_priority: {attributes_priority}")
    
    # Verify expected priority order
    expected_order = ["data-src", "data-srcset", "srcset", "src"]
    print(f"Expected order: {expected_order}")
    
    # Check that the inferred priority follows schema v1 order
    is_correct_order = True
    for i, attr in enumerate(attributes_priority):
        if i < len(expected_order) and attr in expected_order:
            expected_index = expected_order.index(attr)
            if expected_index < i:
                is_correct_order = False
                break
    
    print(f"Priority order is correct: {is_correct_order}")
    
    return is_correct_order

if __name__ == "__main__":
    success = test_attribute_resolution_order()
    if success:
        print("\n✅ All tests passed! Attribute resolution order is correct.")
    else:
        print("\n❌ Tests failed! Attribute resolution order needs fixing.")
        sys.exit(1)
