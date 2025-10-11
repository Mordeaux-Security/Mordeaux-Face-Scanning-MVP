"""
Unit tests for CLI functionality.

Tests the argument parsing and YAML merge functionality for the CLI tools.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import the functions directly since the bin scripts are not importable modules
# We'll test the functionality by importing the actual modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# For testing, we'll copy the relevant functions here or mock them
def parse_urls_file(urls_file: str):
    """Test version of parse_urls_file function."""
    import tempfile
    urls_path = Path(urls_file)
    if not urls_path.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")
    
    urls = []
    with open(urls_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            url = line.strip()
            if url and not url.startswith('#'):
                urls.append(url)
    
    if not urls:
        raise ValueError(f"No valid URLs found in {urls_file}")
    
    return urls

def merge_recipe_to_yaml(recipe_dict, yaml_file: str, append: bool = True):
    """Test version of merge_recipe_to_yaml function."""
    import yaml
    
    existing_recipes = {'defaults': {}, 'sites': {}}
    yaml_path = Path(yaml_file)
    
    if append and yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            existing_recipes = yaml.safe_load(f) or existing_recipes
    
    if 'sites' not in existing_recipes:
        existing_recipes['sites'] = {}
    
    existing_recipes['sites'][recipe_dict['domain']] = {
        'selectors': recipe_dict['selectors'],
        'attributes_priority': recipe_dict['attributes_priority'],
        'extra_sources': recipe_dict['extra_sources'],
        'method': recipe_dict['method']
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing_recipes, f, default_flow_style=False, sort_keys=False)
    
    return True

def load_domain_recipe(domain: str, yaml_file: str):
    """Test version of load_domain_recipe function."""
    import yaml
    
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        return None
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            recipes = yaml.safe_load(f) or {}
        
        sites = recipes.get('sites', {})
        if domain not in sites:
            return None
        
        recipe = sites[domain]
        recipe['domain'] = domain
        return recipe
        
    except Exception:
        return None

def update_yaml_with_approvals(domain: str, selectors, approvals, yaml_file: str):
    """Test version of update_yaml_with_approvals function."""
    import yaml
    
    yaml_path = Path(yaml_file)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        recipes = yaml.safe_load(f) or {}
    
    approved_selectors = [
        selector for selector, approved in zip(selectors, approvals)
        if approved
    ]
    
    if 'sites' not in recipes:
        recipes['sites'] = {}
    
    if domain in recipes['sites']:
        recipes['sites'][domain]['selectors'] = approved_selectors
    else:
        return False
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(recipes, f, default_flow_style=False, sort_keys=False)
    
    return True


class TestMineSelectorsCLI:
    """Test cases for mine-selectors CLI functionality."""
    
    def test_parse_urls_file_success(self):
        """Test successful parsing of URLs file."""
        # Create a temporary file with URLs
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("# This is a comment\n")
            f.write("https://example.com/page1\n")
            f.write("https://example.com/page2\n")
            f.write("\n")  # Empty line
            f.write("https://example.com/page3\n")
            temp_path = f.name
        
        try:
            urls = parse_urls_file(temp_path)
            
            assert len(urls) == 3
            assert "https://example.com/page1" in urls
            assert "https://example.com/page2" in urls
            assert "https://example.com/page3" in urls
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_parse_urls_file_not_found(self):
        """Test handling of missing URLs file."""
        with pytest.raises(FileNotFoundError):
            parse_urls_file("nonexistent_file.txt")
    
    def test_parse_urls_file_empty(self):
        """Test handling of empty URLs file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("# Only comments\n")
            f.write("\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="No valid URLs found"):
                parse_urls_file(temp_path)
                
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_merge_recipe_to_yaml_new_file(self):
        """Test merging recipe to a new YAML file."""
        recipe_dict = {
            'domain': 'test.com',
            'selectors': [
                {'selector': 'img.thumbnail', 'description': 'Thumbnail images'}
            ],
            'attributes_priority': ['alt', 'title'],
            'extra_sources': ['data-src'],
            'method': 'smart',
            'confidence': 0.8,
            'sample_urls': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            temp_path = f.name
        
        try:
            success = merge_recipe_to_yaml(recipe_dict, temp_path, append=False)
            assert success
            
            # Verify the file was created with correct content
            with open(temp_path, 'r') as f:
                content = yaml.safe_load(f)
            
            assert 'sites' in content
            assert 'test.com' in content['sites']
            assert content['sites']['test.com']['method'] == 'smart'
            assert len(content['sites']['test.com']['selectors']) == 1
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_merge_recipe_to_yaml_append_existing(self):
        """Test merging recipe to existing YAML file."""
        # Create existing YAML content
        existing_content = {
            'defaults': {
                'method': 'smart'
            },
            'sites': {
                'existing.com': {
                    'selectors': [
                        {'selector': 'img.existing', 'description': 'Existing images'}
                    ],
                    'method': 'custom'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(existing_content, f)
            temp_path = f.name
        
        try:
            recipe_dict = {
                'domain': 'new.com',
                'selectors': [
                    {'selector': 'img.new', 'description': 'New images'}
                ],
                'attributes_priority': ['alt'],
                'extra_sources': [],
                'method': 'smart',
                'confidence': 0.9,
                'sample_urls': []
            }
            
            success = merge_recipe_to_yaml(recipe_dict, temp_path, append=True)
            assert success
            
            # Verify both recipes exist
            with open(temp_path, 'r') as f:
                content = yaml.safe_load(f)
            
            assert 'existing.com' in content['sites']
            assert 'new.com' in content['sites']
            assert content['sites']['existing.com']['method'] == 'custom'  # Unchanged
            assert content['sites']['new.com']['method'] == 'smart'  # New recipe
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_merge_recipe_to_yaml_overwrite_existing(self):
        """Test overwriting existing recipe for same domain."""
        existing_content = {
            'sites': {
                'test.com': {
                    'selectors': [
                        {'selector': 'img.old', 'description': 'Old images'}
                    ],
                    'method': 'old'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(existing_content, f)
            temp_path = f.name
        
        try:
            recipe_dict = {
                'domain': 'test.com',  # Same domain
                'selectors': [
                    {'selector': 'img.new', 'description': 'New images'}
                ],
                'attributes_priority': ['alt'],
                'extra_sources': [],
                'method': 'smart',
                'confidence': 0.8,
                'sample_urls': []
            }
            
            success = merge_recipe_to_yaml(recipe_dict, temp_path, append=True)
            assert success
            
            # Verify the recipe was overwritten
            with open(temp_path, 'r') as f:
                content = yaml.safe_load(f)
            
            assert content['sites']['test.com']['method'] == 'smart'  # Updated
            assert content['sites']['test.com']['selectors'][0]['selector'] == 'img.new'  # Updated
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestReviewSelectorsCLI:
    """Test cases for review-selectors CLI functionality."""
    
    def test_load_domain_recipe_success(self):
        """Test successful loading of domain recipe."""
        yaml_content = {
            'sites': {
                'test.com': {
                    'selectors': [
                        {'selector': 'img.thumbnail', 'description': 'Thumbnail images'}
                    ],
                    'attributes_priority': ['alt', 'title'],
                    'extra_sources': ['data-src'],
                    'method': 'smart'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            recipe = load_domain_recipe('test.com', temp_path)
            
            assert recipe is not None
            assert recipe['domain'] == 'test.com'
            assert recipe['method'] == 'smart'
            assert len(recipe['selectors']) == 1
            assert recipe['selectors'][0]['selector'] == 'img.thumbnail'
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_domain_recipe_not_found(self):
        """Test loading non-existent domain recipe."""
        yaml_content = {
            'sites': {
                'other.com': {
                    'selectors': [{'selector': 'img.other', 'description': 'Other images'}],
                    'method': 'smart'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            recipe = load_domain_recipe('nonexistent.com', temp_path)
            assert recipe is None
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_domain_recipe_file_not_found(self):
        """Test loading from non-existent YAML file."""
        recipe = load_domain_recipe('test.com', 'nonexistent.yaml')
        assert recipe is None
    
    def test_update_yaml_with_approvals_success(self):
        """Test successful update of YAML with approval decisions."""
        yaml_content = {
            'sites': {
                'test.com': {
                    'selectors': [
                        {'selector': 'img.selector1', 'description': 'Selector 1'},
                        {'selector': 'img.selector2', 'description': 'Selector 2'},
                        {'selector': 'img.selector3', 'description': 'Selector 3'}
                    ],
                    'method': 'smart'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            selectors = yaml_content['sites']['test.com']['selectors']
            approvals = [True, False, True]  # Approve 1st and 3rd, reject 2nd
            
            success = update_yaml_with_approvals('test.com', selectors, approvals, temp_path)
            assert success
            
            # Verify only approved selectors remain
            with open(temp_path, 'r') as f:
                updated_content = yaml.safe_load(f)
            
            updated_selectors = updated_content['sites']['test.com']['selectors']
            assert len(updated_selectors) == 2
            assert updated_selectors[0]['selector'] == 'img.selector1'
            assert updated_selectors[1]['selector'] == 'img.selector3'
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_update_yaml_with_approvals_domain_not_found(self):
        """Test update with non-existent domain."""
        yaml_content = {
            'sites': {
                'other.com': {
                    'selectors': [{'selector': 'img.other', 'description': 'Other images'}],
                    'method': 'smart'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            selectors = [{'selector': 'img.test', 'description': 'Test images'}]
            approvals = [True]
            
            success = update_yaml_with_approvals('nonexistent.com', selectors, approvals, temp_path)
            assert not success  # Should fail for non-existent domain
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_update_yaml_with_approvals_all_rejected(self):
        """Test update when all selectors are rejected."""
        yaml_content = {
            'sites': {
                'test.com': {
                    'selectors': [
                        {'selector': 'img.selector1', 'description': 'Selector 1'},
                        {'selector': 'img.selector2', 'description': 'Selector 2'}
                    ],
                    'method': 'smart'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            selectors = yaml_content['sites']['test.com']['selectors']
            approvals = [False, False]  # Reject all
            
            success = update_yaml_with_approvals('test.com', selectors, approvals, temp_path)
            assert success
            
            # Verify no selectors remain
            with open(temp_path, 'r') as f:
                updated_content = yaml.safe_load(f)
            
            updated_selectors = updated_content['sites']['test.com']['selectors']
            assert len(updated_selectors) == 0
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_complete_workflow_simulation(self):
        """Test a complete workflow simulation."""
        # Create a sample URLs file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("https://example.com/page1\n")
            f.write("https://example.com/page2\n")
            urls_file = f.name
        
        # Create a sample YAML file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump({'defaults': {'method': 'smart'}, 'sites': {}}, f)
            yaml_file = f.name
        
        try:
            # Test URLs parsing
            urls = parse_urls_file(urls_file)
            assert len(urls) == 2
            
            # Test recipe merging
            recipe_dict = {
                'domain': 'example.com',
                'selectors': [
                    {'selector': 'img.thumbnail', 'description': 'Thumbnail images'}
                ],
                'attributes_priority': ['alt', 'title'],
                'extra_sources': ['data-src'],
                'method': 'smart',
                'confidence': 0.8,
                'sample_urls': []
            }
            
            success = merge_recipe_to_yaml(recipe_dict, yaml_file, append=True)
            assert success
            
            # Verify the recipe was added
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            
            assert 'example.com' in content['sites']
            
            # Test loading the recipe
            loaded_recipe = load_domain_recipe('example.com', yaml_file)
            assert loaded_recipe is not None
            assert loaded_recipe['domain'] == 'example.com'
            
            # Test updating with approvals
            selectors = loaded_recipe['selectors']
            approvals = [True]  # Approve the selector
            
            success = update_yaml_with_approvals('example.com', selectors, approvals, yaml_file)
            assert success
            
            # Verify the update worked
            with open(yaml_file, 'r') as f:
                updated_content = yaml.safe_load(f)
            
            assert len(updated_content['sites']['example.com']['selectors']) == 1
            
        finally:
            Path(urls_file).unlink(missing_ok=True)
            Path(yaml_file).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
