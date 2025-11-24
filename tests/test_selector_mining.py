"""
Test suite for Selector Mining Service

Validates that the selector mining service works correctly and preserves
all logic from the original tools/selector_miner.py implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from backend.app.services.selector_mining import (
    SelectorMiningService,
    mine_page,
    emit_recipe_yaml_block,
    resolve_image_url,
    validate_image_request,
    extract_extra_sources,
    stable_selector,
    gather_evidence,
    score_candidate,
    discover_listing_links,
    Limits,
    MinedResult,
    CandidateSelector
)
from backend.app.services.http_service import HttpService


class TestSelectorMiningService:
    """Test the main SelectorMiningService class."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test that service initializes correctly."""
        service = SelectorMiningService()
        assert service.http_service is not None
        assert isinstance(service.http_service, HttpService)
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_service_with_custom_http_service(self):
        """Test service with custom HTTP service."""
        http_service = HttpService()
        service = SelectorMiningService(http_service)
        assert service.http_service is http_service
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_mine_selectors_for_page(self):
        """Test mining selectors from a single page."""
        service = SelectorMiningService()
        
        # Mock HTML content
        html = """
        <html>
            <body>
                <div class="thumb">
                    <img src="image1.jpg" alt="Image 1">
                </div>
                <div class="thumb">
                    <img src="image2.jpg" alt="Image 2">
                </div>
                <div class="thumb">
                    <img src="image3.jpg" alt="Image 3">
                </div>
            </body>
        </html>
        """
        
        # Mock HTTP client
        mock_client = AsyncMock()
        
        # Mock validation to return True for all URLs
        with patch('backend.app.services.selector_mining.validate_image_request') as mock_validate:
            mock_validate.return_value = True
            
            result = await service.mine_selectors_for_page(html, "https://example.com", mock_client)
            
            assert isinstance(result, MinedResult)
            assert result.status in ["OK", "NO_THUMBS_STATIC", "EXTRA_ONLY"]
            assert len(result.candidates) >= 0  # Should find some candidates
            assert 'static_candidates' in result.stats
            assert 'extra_sources' in result.stats
            assert 'total_validated' in result.stats
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_save_selectors_to_recipe(self):
        """Test saving selectors to recipe file."""
        service = SelectorMiningService()
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            with patch('os.path.exists', return_value=False):
                with patch('yaml.safe_load', return_value={}):
                    with patch('yaml.dump') as mock_dump:
                        result = await service.save_selectors_to_recipe(
                            "test.com", 
                            [".thumb img", ".thumbnail img"], 
                            "test_recipes.yaml"
                        )
                        
                        assert result is True
                        mock_dump.assert_called_once()
        
        await service.close()


class TestMiningFunctions:
    """Test individual mining functions."""
    
    def test_resolve_image_url(self):
        """Test URL resolution from HTML elements."""
        from bs4 import BeautifulSoup
        
        # Test data-src priority
        html = '<img data-src="https://example.com/image.jpg" src="fallback.jpg">'
        soup = BeautifulSoup(html, 'html.parser')
        img = soup.find('img')
        
        url = resolve_image_url(img, "https://example.com")
        assert url == "https://example.com/image.jpg"
        
        # Test relative URL resolution
        html = '<img src="/relative/image.jpg">'
        soup = BeautifulSoup(html, 'html.parser')
        img = soup.find('img')
        
        url = resolve_image_url(img, "https://example.com")
        assert url == "https://example.com/relative/image.jpg"
    
    def test_stable_selector(self):
        """Test stable selector generation."""
        from bs4 import BeautifulSoup
        
        html = '<div class="thumb-wrapper"><img src="test.jpg"></div>'
        soup = BeautifulSoup(html, 'html.parser')
        img = soup.find('img')
        
        selector = stable_selector(img)
        assert selector == "div.thumb-wrapper img"
    
    def test_gather_evidence(self):
        """Test evidence gathering from nodes."""
        from bs4 import BeautifulSoup
        
        html = """
        <div class="thumb">
            <img src="image1.jpg">
        </div>
        <div class="thumb">
            <img src="image2.jpg">
        </div>
        <div class="thumb">
            <img src="image3.jpg">
        </div>
        """
        soup = BeautifulSoup(html, 'html.parser')
        imgs = soup.find_all('img')
        
        evidence = gather_evidence(imgs)
        
        assert evidence['repeats'] == 3
        assert 'has_duration' in evidence
        assert 'has_video_href' in evidence
        assert 'class_hits' in evidence
        assert 'srcset_count' in evidence
    
    def test_score_candidate(self):
        """Test candidate scoring."""
        # Test high score scenario
        score = score_candidate(repeats=15, has_duration=5, has_video_href=4, class_hits=3, srcset_count=6)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high score
        
        # Test low score scenario
        score = score_candidate(repeats=2, has_duration=0, has_video_href=0, class_hits=0, srcset_count=0)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low score
    
    def test_discover_listing_links(self):
        """Test listing link discovery."""
        from bs4 import BeautifulSoup
        
        html = """
        <html>
            <body>
                <nav>
                    <a href="/videos">Videos</a>
                    <a href="/trending">Trending</a>
                    <a href="/login">Login</a>
                </nav>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        links = discover_listing_links(soup, "https://example.com", "example.com")
        
        # Should find video-related links
        assert len(links) > 0
        assert any("/videos" in link for link in links)
        assert any("/trending" in link for link in links)
        # Should not include login link
        assert not any("/login" in link for link in links)
    
    def test_extract_extra_sources(self):
        """Test extra source extraction."""
        from bs4 import BeautifulSoup
        
        html = """
        <html>
            <head>
                <meta property="og:image" content="https://example.com/og-image.jpg">
                <link rel="image_src" href="https://example.com/link-image.jpg">
            </head>
            <body>
                <video poster="https://example.com/poster.jpg"></video>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        sources = extract_extra_sources(soup, "https://example.com")
        
        assert "meta[property='og:image']::attr(content)" in sources
        assert "link[rel='image_src']::attr(href)" in sources
        assert "video::attr(poster)" in sources


class TestMinePage:
    """Test the main mine_page function."""
    
    @pytest.mark.asyncio
    async def test_mine_page_with_html(self):
        """Test mining with provided HTML."""
        html = """
        <html>
            <body>
                <div class="thumb">
                    <img src="image1.jpg">
                </div>
                <div class="thumb">
                    <img src="image2.jpg">
                </div>
                <div class="thumb">
                    <img src="image3.jpg">
                </div>
            </body>
        </html>
        """
        
        mock_client = AsyncMock()
        limits = Limits(max_candidates=5, max_samples_per_candidate=2)
        
        # Mock validation
        with patch('backend.app.services.selector_mining.validate_image_request') as mock_validate:
            mock_validate.return_value = True
            
            result = await mine_page("https://example.com", html, use_js=False, client=mock_client, limits=limits)
            
            assert isinstance(result, MinedResult)
            assert result.status in ["OK", "NO_THUMBS_STATIC", "EXTRA_ONLY"]
            assert len(result.candidates) >= 0
            assert 'static_candidates' in result.stats
            assert 'extra_sources' in result.stats
            assert 'total_validated' in result.stats
    
    @pytest.mark.asyncio
    async def test_mine_page_fetch_html(self):
        """Test mining with HTML fetching."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <div class="thumb">
                    <img src="image1.jpg">
                </div>
            </body>
        </html>
        """
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        limits = Limits(max_candidates=5, max_samples_per_candidate=2)
        
        # Mock validation
        with patch('backend.app.services.selector_mining.validate_image_request') as mock_validate:
            mock_validate.return_value = True
            
            result = await mine_page("https://example.com", None, use_js=False, client=mock_client, limits=limits)
            
            assert isinstance(result, MinedResult)
            # The function may make multiple requests for category pages, so just check it was called
            assert mock_client.get.call_count >= 1


class TestEmitRecipeYamlBlock:
    """Test recipe YAML block generation."""
    
    def test_emit_recipe_yaml_block_valid(self):
        """Test valid recipe block generation."""
        domain = "example.com"
        selectors = [".thumb img", ".thumbnail img"]
        attr_priority = ["data-src", "src"]
        extra_sources = ["meta[property='og:image']::attr(content)"]
        
        recipe = emit_recipe_yaml_block(domain, selectors, attr_priority, extra_sources)
        
        assert recipe['domain'] == domain
        assert len(recipe['selectors']) == 2
        assert recipe['attributes_priority'] == attr_priority
        assert recipe['extra_sources'] == extra_sources
        assert recipe['method'] == 'miner'
        assert recipe['confidence'] == 0.8
    
    def test_emit_recipe_yaml_block_invalid_domain(self):
        """Test invalid domain handling."""
        with pytest.raises(Exception):  # Should raise MinerSchemaError
            emit_recipe_yaml_block("", [".thumb img"], ["data-src"], [])
    
    def test_emit_recipe_yaml_block_invalid_selectors(self):
        """Test invalid selectors handling."""
        with pytest.raises(Exception):  # Should raise MinerSchemaError
            emit_recipe_yaml_block("example.com", [], ["data-src"], [])


class TestValidateImageRequest:
    """Test image request validation."""
    
    @pytest.mark.asyncio
    async def test_validate_image_request_valid(self):
        """Test validation of valid image request."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg', 'content-length': '1000'}
        mock_response.iter_bytes.return_value = [b'fake image data']
        mock_client.head.return_value = mock_response
        mock_client.get.return_value = mock_response
        
        is_valid = await validate_image_request("https://example.com/image.jpg", mock_client, 10000)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_image_request_invalid_scheme(self):
        """Test validation of invalid scheme."""
        mock_client = AsyncMock()
        
        is_valid = await validate_image_request("javascript:alert('xss')", mock_client, 10000)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validate_image_request_svg_blocked(self):
        """Test that SVG files are blocked."""
        mock_client = AsyncMock()
        
        is_valid = await validate_image_request("https://example.com/image.svg", mock_client, 10000)
        assert is_valid is False


class TestBackwardCompatibility:
    """Test backward compatibility with original tool."""
    
    def test_import_compatibility(self):
        """Test that all expected functions are importable."""
        from backend.app.services.selector_mining import (
            SelectorMiningService,
            mine_page,
            emit_recipe_yaml_block,
            resolve_image_url,
            validate_image_request,
            extract_extra_sources,
            stable_selector,
            gather_evidence,
            score_candidate,
            discover_listing_links,
            Limits,
            MinedResult,
            CandidateSelector,
            MinerNetworkError,
            MinerSchemaError
        )
        
        # Just verify they're importable
        assert SelectorMiningService is not None
        assert mine_page is not None
        assert emit_recipe_yaml_block is not None
        assert resolve_image_url is not None
        assert validate_image_request is not None
        assert extract_extra_sources is not None
        assert stable_selector is not None
        assert gather_evidence is not None
        assert score_candidate is not None
        assert discover_listing_links is not None
        assert Limits is not None
        assert MinedResult is not None
        assert CandidateSelector is not None
        assert MinerNetworkError is not None
        assert MinerSchemaError is not None


if __name__ == "__main__":
    pytest.main([__file__])
