"""
Unit tests for the Selector Miner YAML emission functionality.

Tests the validation loop, recipe proposal, and YAML generation features
added in Phase 2.1.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from .selector_miner import (
    SelectorMiner,
    CandidateSelector,
    ImageNode,
    ValidationResult,
    RecipeCandidate,
    propose_recipe_for_domain
)


class TestValidationLoop:
    """Test cases for URL validation functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")

    def test_safe_url_validation(self):
        """Test URL safety validation."""
        # Safe URLs
        assert self.miner._is_safe_url("https://example.com/image.jpg")
        assert self.miner._is_safe_url("http://example.com/thumb.png")
        assert self.miner._is_safe_url("/relative/image.gif")

        # Unsafe URLs
        assert not self.miner._is_safe_url("javascript:alert('xss')")
        assert not self.miner._is_safe_url("data:text/html,<script>")
        assert not self.miner._is_safe_url("file:///etc/passwd")
        assert not self.miner._is_safe_url("vbscript:msgbox('xss')")

    def test_content_type_validation(self):
        """Test image content type validation."""
        # Valid image content types
        assert self.miner._is_valid_image_content_type("image/jpeg")
        assert self.miner._is_valid_image_content_type("image/png")
        assert self.miner._is_valid_image_content_type("image/gif")
        assert self.miner._is_valid_image_content_type("image/webp")
        assert self.miner._is_valid_image_content_type("image/svg+xml")

        # Invalid content types
        assert not self.miner._is_valid_image_content_type("text/html")
        assert not self.miner._is_valid_image_content_type("application/json")
        assert not self.miner._is_valid_image_content_type("")
        assert not self.miner._is_valid_image_content_type(None)

    @pytest.mark.asyncio
    async def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            url="https://example.com/test.jpg",
            is_valid=True,
            status_code=200,
            content_type="image/jpeg"
        )

        assert result.url == "https://example.com/test.jpg"
        assert result.is_valid is True
        assert result.status_code == 200
        assert result.content_type == "image/jpeg"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_unsafe_url_validation(self):
        """Test validation of unsafe URLs."""
        candidate = CandidateSelector(
            selector="img.unsafe",
            description="Test selector",
            evidence={},
            sample_urls=["javascript:alert('xss')"],
            repetition_count=1
        )

        results = await self.miner.validate_candidate_urls(candidate, max_samples=1)

        assert len(results) == 1
        assert not results[0].is_valid
        assert "Unsafe URL scheme" in results[0].error

    @pytest.mark.asyncio
    async def test_mock_http_validation(self):
        """Test HTTP validation with mocked responses."""
        # Create a mock candidate
        candidate = CandidateSelector(
            selector="img.test",
            description="Test selector",
            evidence={},
            sample_urls=["https://example.com/test.jpg"],
            repetition_count=1
        )

        # Mock successful HTTP response
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'image/jpeg', 'content-length': '1024'}

            mock_client.return_value.__aenter__.return_value.head.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            results = await self.miner.validate_candidate_urls(candidate, max_samples=1)

            assert len(results) == 1
            assert results[0].is_valid is True
            assert results[0].status_code == 200
            assert results[0].content_type == "image/jpeg"


class TestAttributesInference:
    """Test cases for attributes priority inference."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")

    def test_attributes_priority_inference(self):
        """Test inference of attributes priority from image nodes."""
        # Create mock image nodes with various attributes
        mock_nodes = [
            Mock(
                attributes={
                    'alt': 'Image 1 description',
                    'title': 'Image 1 title',
                    'src': '/image1.jpg',
                    'data-id': '12345'
                }
            ),
            Mock(
                attributes={
                    'alt': 'Image 2 description',
                    'title': 'Image 2 title',
                    'src': '/image2.jpg',
                    'data-title': 'Custom title'
                }
            ),
            Mock(
                attributes={
                    'alt': 'Image 3 description',
                    'src': '/image3.jpg',
                    'data-alt': 'Custom alt text'
                }
            )
        ]

        # Convert to ImageNode objects
        image_nodes = []
        for mock_node in mock_nodes:
            image_node = ImageNode(
                tag=Mock(),
                url="https://example.com/test.jpg",
                selector_path="img.test",
                attributes=mock_node.attributes,
                context={}
            )
            image_nodes.append(image_node)

        attributes_priority = self.miner._infer_attributes_priority(image_nodes)

        # Should prioritize semantic attributes
        assert 'alt' in attributes_priority
        assert 'title' in attributes_priority
        assert 'data-title' in attributes_priority
        assert 'data-alt' in attributes_priority

        # Alt should be first (highest semantic bonus)
        assert attributes_priority[0] == 'alt'

    def test_extra_sources_extraction(self):
        """Test extraction of extra sources from image nodes."""
        mock_nodes = [
            Mock(
                attributes={
                    'src': '/image1.jpg',
                    'data-src': '/lazy-image1.jpg',
                    'data-lazy-src': '/lazy-image1.jpg',
                    'data-original': '/original1.jpg'
                }
            ),
            Mock(
                attributes={
                    'src': '/image2.jpg',
                    'data-src': '/lazy-image2.jpg',
                    'data-large': '/large2.jpg'
                }
            )
        ]

        # Convert to ImageNode objects
        image_nodes = []
        for mock_node in mock_nodes:
            image_node = ImageNode(
                tag=Mock(),
                url="https://example.com/test.jpg",
                selector_path="img.test",
                attributes=mock_node.attributes,
                context={}
            )
            image_nodes.append(image_node)

        extra_sources = self.miner._extract_extra_sources(image_nodes)

        # Should find common data attributes
        assert 'data-src' in extra_sources
        assert 'data-lazy-src' in extra_sources
        assert 'data-original' in extra_sources
        assert 'data-large' in extra_sources


class TestRecipeGeneration:
    """Test cases for recipe generation and YAML emission."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")

    def test_recipe_candidate_creation(self):
        """Test RecipeCandidate creation."""
        recipe = RecipeCandidate(
            domain="example.com",
            selectors=[
                {"selector": "img.thumbnail", "description": "Thumbnail images"},
                {"selector": "img.preview", "description": "Preview images"}
            ],
            attributes_priority=["alt", "title"],
            extra_sources=["data-src", "data-lazy-src"],
            method="smart",
            confidence=0.85,
            sample_urls=["https://example.com/test1.jpg"],
            validation_results=[]
        )

        assert recipe.domain == "example.com"
        assert len(recipe.selectors) == 2
        assert recipe.attributes_priority == ["alt", "title"]
        assert recipe.extra_sources == ["data-src", "data-lazy-src"]
        assert recipe.method == "smart"
        assert recipe.confidence == 0.85

    def test_yaml_generation(self):
        """Test YAML recipe generation."""
        recipe = RecipeCandidate(
            domain="example.com",
            selectors=[
                {"selector": "img.thumbnail", "description": "Thumbnail images"},
                {"selector": "img.preview", "description": "Preview images"}
            ],
            attributes_priority=["alt", "title"],
            extra_sources=["data-src"],
            method="smart",
            confidence=0.85,
            sample_urls=[],
            validation_results=[]
        )

        yaml_output = self.miner.generate_yaml_recipe(recipe)

        # Parse the YAML to verify structure
        parsed = yaml.safe_load(yaml_output)

        assert "example.com" in parsed
        assert parsed["example.com"]["method"] == "smart"
        assert len(parsed["example.com"]["selectors"]) == 2
        assert parsed["example.com"]["attributes_priority"] == ["alt", "title"]
        assert parsed["example.com"]["extra_sources"] == ["data-src"]

    def test_yaml_file_merge(self):
        """Test merging recipe into existing YAML file."""
        recipe = RecipeCandidate(
            domain="test.com",
            selectors=[
                {"selector": "img.test", "description": "Test images"}
            ],
            attributes_priority=["alt"],
            extra_sources=[],
            method="smart",
            confidence=0.9,
            sample_urls=[],
            validation_results=[]
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Test merging into empty file
            success = self.miner.merge_recipe_to_yaml_file(recipe, temp_path)
            assert success

            # Verify the file was created and has correct content
            with open(temp_path, 'r') as f:
                content = yaml.safe_load(f)

            assert "sites" in content
            assert "test.com" in content["sites"]
            assert content["sites"]["test.com"]["method"] == "smart"

            # Test merging additional recipe
            recipe2 = RecipeCandidate(
                domain="test2.com",
                selectors=[
                    {"selector": "img.test2", "description": "Test 2 images"}
                ],
                attributes_priority=["title"],
                extra_sources=["data-src"],
                method="smart",
                confidence=0.8,
                sample_urls=[],
                validation_results=[]
            )

            success = self.miner.merge_recipe_to_yaml_file(recipe2, temp_path)
            assert success

            # Verify both recipes exist
            with open(temp_path, 'r') as f:
                content = yaml.safe_load(f)

            assert "test.com" in content["sites"]
            assert "test2.com" in content["sites"]

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    def test_yaml_file_merge_with_existing_content(self):
        """Test merging recipe into existing YAML file with content."""
        # Create existing YAML content
        existing_content = {
            'defaults': {
                'selectors': [
                    {"selector": "img", "description": "All images"}
                ],
                'method': 'smart'
            },
            'sites': {
                'existing.com': {
                    'selectors': [
                        {"selector": "img.existing", "description": "Existing images"}
                    ],
                    'method': 'custom'
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(existing_content, f)
            temp_path = f.name

        try:
            # Create new recipe
            recipe = RecipeCandidate(
                domain="new.com",
                selectors=[
                    {"selector": "img.new", "description": "New images"}
                ],
                attributes_priority=["alt"],
                extra_sources=[],
                method="smart",
                confidence=0.9,
                sample_urls=[],
                validation_results=[]
            )

            # Merge the recipe
            success = self.miner.merge_recipe_to_yaml_file(recipe, temp_path)
            assert success

            # Verify all content is preserved
            with open(temp_path, 'r') as f:
                content = yaml.safe_load(f)

            assert "defaults" in content
            assert "sites" in content
            assert "existing.com" in content["sites"]
            assert "new.com" in content["sites"]

            # Verify existing content is unchanged
            assert content["sites"]["existing.com"]["method"] == "custom"
            assert content["sites"]["new.com"]["method"] == "smart"

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


class TestProposeRecipe:
    """Test cases for the propose_recipe functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")

    @pytest.mark.asyncio
    async def test_propose_recipe_with_mock_validation(self):
        """Test recipe proposal with mocked validation."""
        html_content = """
        <html>
        <body>
            <img src="/thumb1.jpg" class="thumbnail" alt="Image 1" title="Title 1">
            <img src="/thumb2.jpg" class="thumbnail" alt="Image 2" title="Title 2">
            <img src="/thumb3.jpg" class="thumbnail" alt="Image 3" title="Title 3">
        </body>
        </html>
        """

        # Mock the validation to return successful results
        with patch.object(self.miner, 'validate_candidate_urls') as mock_validate:
            mock_validation_result = ValidationResult(
                url="https://example.com/test.jpg",
                is_valid=True,
                status_code=200,
                content_type="image/jpeg"
            )
            mock_validate.return_value = [mock_validation_result, mock_validation_result, mock_validation_result]

            recipe = await self.miner.propose_recipe("example.com", html_content, max_candidates=2)

            assert recipe is not None
            assert recipe.domain == "example.com"
            assert recipe.method == "smart"
            assert len(recipe.selectors) > 0
            assert "alt" in recipe.attributes_priority
            assert "title" in recipe.attributes_priority
            assert recipe.confidence > 0

    @pytest.mark.asyncio
    async def test_propose_recipe_validation_failure(self):
        """Test recipe proposal when validation fails."""
        html_content = """
        <html>
        <body>
            <img src="/thumb1.jpg" class="thumbnail">
        </body>
        </html>
        """

        # Mock validation to return failed results
        with patch.object(self.miner, 'validate_candidate_urls') as mock_validate:
            mock_validation_result = ValidationResult(
                url="https://example.com/test.jpg",
                is_valid=False,
                error="Invalid content type"
            )
            mock_validate.return_value = [mock_validation_result, mock_validation_result, mock_validation_result]

            recipe = await self.miner.propose_recipe("example.com", html_content)

            # Should return None when validation fails
            assert recipe is None

    @pytest.mark.asyncio
    async def test_propose_recipe_convenience_function(self):
        """Test the convenience function for recipe proposal."""
        html_content = """
        <html>
        <body>
            <img src="/thumb1.jpg" class="thumbnail" alt="Test">
            <img src="/thumb2.jpg" class="thumbnail" alt="Test">
            <img src="/thumb3.jpg" class="thumbnail" alt="Test">
        </body>
        </html>
        """

        # Mock validation
        with patch('tools.selector_miner.SelectorMiner.validate_candidate_urls') as mock_validate:
            mock_validation_result = ValidationResult(
                url="https://example.com/test.jpg",
                is_valid=True,
                status_code=200,
                content_type="image/jpeg"
            )
            mock_validate.return_value = [mock_validation_result, mock_validation_result, mock_validation_result]

            recipe = await propose_recipe_for_domain("example.com", html_content, "https://example.com")

            assert recipe is not None
            assert recipe.domain == "example.com"


class TestYAMLStructure:
    """Test cases for YAML structure compatibility."""

    def test_yaml_structure_compatibility(self):
        """Test that generated YAML is compatible with site_recipes.yaml structure."""
        recipe = RecipeCandidate(
            domain="test-site.com",
            selectors=[
                {"selector": "img.thumbnail", "description": "Thumbnail images"},
                {"selector": "img.preview", "description": "Preview images"},
                {"selector": "img", "description": "All images"}
            ],
            attributes_priority=["alt", "title", "data-title"],
            extra_sources=["data-src", "data-lazy-src"],
            method="smart",
            confidence=0.9,
            sample_urls=["https://test-site.com/img1.jpg"],
            validation_results=[]
        )

        miner = SelectorMiner()
        yaml_output = miner.generate_yaml_recipe(recipe)

        # Parse the YAML
        parsed = yaml.safe_load(yaml_output)

        # Check structure matches site_recipes.yaml format
        assert isinstance(parsed, dict)
        assert "test-site.com" in parsed

        site_config = parsed["test-site.com"]
        assert "selectors" in site_config
        assert "attributes_priority" in site_config
        assert "extra_sources" in site_config
        assert "method" in site_config

        # Check selectors format
        selectors = site_config["selectors"]
        assert isinstance(selectors, list)
        for selector in selectors:
            assert "selector" in selector
            assert "description" in selector

        # Check attributes_priority format
        assert isinstance(site_config["attributes_priority"], list)

        # Check extra_sources format
        assert isinstance(site_config["extra_sources"], list)

        # Check method format
        assert site_config["method"] == "smart"

    def test_yaml_merge_structure_preservation(self):
        """Test that YAML merging preserves the correct structure."""
        # Create a recipe that matches existing site_recipes.yaml structure
        recipe = RecipeCandidate(
            domain="new-domain.net",
            selectors=[
                {"selector": "img.custom", "description": "Custom images"}
            ],
            attributes_priority=["alt", "title"],
            extra_sources=["data-src"],
            method="smart",
            confidence=0.8,
            sample_urls=[],
            validation_results=[]
        )

        # Create a temporary file with existing structure
        existing_content = {
            'defaults': {
                'selectors': [
                    {"selector": "img", "description": "All images"}
                ],
                'attributes_priority': ["alt", "title"],
                'extra_sources': [],
                'method': 'smart'
            },
            'sites': {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(existing_content, f)
            temp_path = f.name

        try:
            miner = SelectorMiner()
            success = miner.merge_recipe_to_yaml_file(recipe, temp_path)
            assert success

            # Load and verify structure
            with open(temp_path, 'r') as f:
                merged_content = yaml.safe_load(f)

            # Verify structure is preserved
            assert "defaults" in merged_content
            assert "sites" in merged_content
            assert "new-domain.net" in merged_content["sites"]

            # Verify the new site has correct structure
            new_site = merged_content["sites"]["new-domain.net"]
            assert "selectors" in new_site
            assert "attributes_priority" in new_site
            assert "extra_sources" in new_site
            assert "method" in new_site

            # Verify defaults are unchanged
            assert merged_content["defaults"]["method"] == "smart"

        finally:
            Path(temp_path).unlink(missing_ok=True)


# Integration test fixtures
@pytest.fixture
def sample_html_with_attributes():
    """HTML fixture with various image attributes."""
    return """
    <html>
    <body>
        <div class="gallery">
            <img src="/img1.jpg" class="thumbnail" alt="Gallery Image 1" title="First Image" data-title="Custom Title 1">
            <img src="/img2.jpg" class="thumbnail" alt="Gallery Image 2" title="Second Image" data-alt="Custom Alt 2">
            <img src="/img3.jpg" class="thumbnail" alt="Gallery Image 3" title="Third Image" data-lazy-src="/lazy3.jpg">
        </div>
        <div class="sidebar">
            <img src="/logo.jpg" class="logo" alt="Site Logo">
        </div>
    </body>
    </html>
    """


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, sample_html_with_attributes):
        """Test the complete workflow from HTML to YAML."""
        # Mock validation to return successful results
        with patch('tools.selector_miner.SelectorMiner.validate_candidate_urls') as mock_validate:
            mock_validation_result = ValidationResult(
                url="https://example.com/test.jpg",
                is_valid=True,
                status_code=200,
                content_type="image/jpeg"
            )
            mock_validate.return_value = [mock_validation_result, mock_validation_result, mock_validation_result]

            # Propose recipe
            recipe = await propose_recipe_for_domain(
                "example.com",
                sample_html_with_attributes,
                "https://example.com"
            )

            assert recipe is not None
            assert recipe.domain == "example.com"

            # Generate YAML
            miner = SelectorMiner()
            yaml_output = miner.generate_yaml_recipe(recipe)

            # Verify YAML structure
            parsed = yaml.safe_load(yaml_output)
            assert "example.com" in parsed

            # Verify attributes priority includes semantic attributes
            attributes_priority = parsed["example.com"]["attributes_priority"]
            assert "alt" in attributes_priority
            assert "title" in attributes_priority

            # Verify selectors are generated
            selectors = parsed["example.com"]["selectors"]
            assert len(selectors) > 0

            # Verify extra sources are detected
            extra_sources = parsed["example.com"]["extra_sources"]
            assert "data-lazy-src" in extra_sources or len(extra_sources) > 0

    @pytest.mark.asyncio
    async def test_yaml_file_workflow(self, sample_html_with_attributes):
        """Test the complete workflow including YAML file merging."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Mock validation
            with patch('tools.selector_miner.SelectorMiner.validate_candidate_urls') as mock_validate:
                mock_validation_result = ValidationResult(
                    url="https://example.com/test.jpg",
                    is_valid=True,
                    status_code=200,
                    content_type="image/jpeg"
                )
                mock_validate.return_value = [mock_validation_result, mock_validation_result, mock_validation_result]

                # Propose recipe
                recipe = await propose_recipe_for_domain(
                    "integration-test.com",
                    sample_html_with_attributes,
                    "https://integration-test.com"
                )

                assert recipe is not None

                # Merge to YAML file
                miner = SelectorMiner()
                success = miner.merge_recipe_to_yaml_file(recipe, temp_path)
                assert success

                # Verify file was created with correct content
                with open(temp_path, 'r') as f:
                    content = yaml.safe_load(f)

                assert "sites" in content
                assert "integration-test.com" in content["sites"]

                site_config = content["sites"]["integration-test.com"]
                assert site_config["method"] == "smart"
                assert len(site_config["selectors"]) > 0
                assert len(site_config["attributes_priority"]) > 0

        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
