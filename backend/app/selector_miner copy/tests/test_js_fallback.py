"""
Unit tests for JavaScript fallback functionality.

Tests the JavaScript rendering and fallback logic for selector mining.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.app.selector_miner.selector_miner import SelectorMiner, PLAYWRIGHT_AVAILABLE, mine_selectors_with_js_fallback


class TestJavaScriptFallback:
    """Test cases for JavaScript fallback functionality."""
    
    def test_playwright_availability_check(self):
        """Test that Playwright availability is correctly detected."""
        # This test will pass regardless of whether Playwright is installed
        assert isinstance(PLAYWRIGHT_AVAILABLE, bool)
    
    @pytest.mark.asyncio
    async def test_render_with_javascript_no_playwright(self):
        """Test JavaScript rendering when Playwright is not available."""
        # Mock PLAYWRIGHT_AVAILABLE to False
        with patch('backend.app.selector_miner.selector_miner.PLAYWRIGHT_AVAILABLE', False):
            miner = SelectorMiner("https://example.com")
            result = await miner.render_with_javascript("https://example.com")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_render_with_javascript_success(self):
        """Test successful JavaScript rendering with mocked Playwright."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        # Mock Playwright components
        mock_page = Mock()
        mock_page.content = AsyncMock(return_value="<html><body><img src='test.jpg'></body></html>")
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.set_default_timeout = Mock()
        
        mock_context = Mock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        mock_browser = Mock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        
        mock_playwright = Mock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        
        with patch('backend.app.selector_miner.selector_miner.async_playwright') as mock_async_playwright:
            mock_async_playwright.return_value.__aenter__.return_value = mock_playwright
            
            miner = SelectorMiner("https://example.com")
            result = await miner.render_with_javascript("https://example.com", timeout=1000)
            
            assert result == "<html><body><img src='test.jpg'></body></html>"
            mock_page.goto.assert_called_once_with("https://example.com", wait_until='domcontentloaded')
            mock_page.wait_for_timeout.assert_called_once_with(1000)
    
    @pytest.mark.asyncio
    async def test_render_with_javascript_failure(self):
        """Test JavaScript rendering failure handling."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        # Mock Playwright to raise an exception
        with patch('backend.app.selector_miner.selector_miner.async_playwright') as mock_async_playwright:
            mock_async_playwright.side_effect = Exception("Browser launch failed")
            
            miner = SelectorMiner("https://example.com")
            result = await miner.render_with_javascript("https://example.com")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_sufficient_static(self):
        """Test JS fallback when static mining finds sufficient candidates."""
        miner = SelectorMiner("https://example.com")
        
        # Mock static mining to return enough candidates
        with patch.object(miner, 'mine_selectors') as mock_mine:
            mock_candidates = [Mock(), Mock(), Mock(), Mock()]  # 4 candidates
            mock_mine.return_value = mock_candidates
            
            result = await miner.mine_selectors_with_js_fallback(
                "<html><body><img src='test.jpg'></body></html>",
                "https://example.com",
                min_candidates=3
            )
            
            assert result == mock_candidates
            mock_mine.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_insufficient_static_no_js(self):
        """Test JS fallback when static mining finds insufficient candidates and no JS available."""
        # Mock PLAYWRIGHT_AVAILABLE to False
        with patch('backend.app.selector_miner.selector_miner.PLAYWRIGHT_AVAILABLE', False):
            miner = SelectorMiner("https://example.com")
            
            # Mock static mining to return insufficient candidates
            with patch.object(miner, 'mine_selectors') as mock_mine:
                mock_candidates = [Mock(), Mock()]  # 2 candidates
                mock_mine.return_value = mock_candidates
                
                result = await miner.mine_selectors_with_js_fallback(
                    "<html><body><img src='test.jpg'></body></html>",
                    "https://example.com",
                    min_candidates=3
                )
                
                assert result == mock_candidates
                mock_mine.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_insufficient_static_with_js(self):
        """Test JS fallback when static mining finds insufficient candidates and JS is available."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        miner = SelectorMiner("https://example.com")
        
        # Mock static mining to return insufficient candidates
        with patch.object(miner, 'mine_selectors') as mock_mine:
            mock_static_candidates = [Mock(), Mock()]  # 2 candidates
            mock_js_candidates = [Mock(), Mock(), Mock(), Mock()]  # 4 candidates
            mock_mine.side_effect = [mock_static_candidates, mock_js_candidates]
            
            # Mock JS rendering
            with patch.object(miner, 'render_with_javascript') as mock_render:
                mock_render.return_value = "<html><body><img src='test.jpg'></body></html>"
                
                result = await miner.mine_selectors_with_js_fallback(
                    "<html><body></body></html>",
                    "https://example.com",
                    min_candidates=3
                )
                
                assert result == mock_js_candidates
                assert mock_mine.call_count == 2  # Called twice: static + JS
                mock_render.assert_called_once_with("https://example.com", 3000)
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_js_fails(self):
        """Test JS fallback when JavaScript rendering fails."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        miner = SelectorMiner("https://example.com")
        
        # Mock static mining to return insufficient candidates
        with patch.object(miner, 'mine_selectors') as mock_mine:
            mock_static_candidates = [Mock(), Mock()]  # 2 candidates
            mock_mine.return_value = mock_static_candidates
            
            # Mock JS rendering to fail
            with patch.object(miner, 'render_with_javascript') as mock_render:
                mock_render.return_value = None
                
                result = await miner.mine_selectors_with_js_fallback(
                    "<html><body></body></html>",
                    "https://example.com",
                    min_candidates=3
                )
                
                assert result == mock_static_candidates
                mock_mine.assert_called_once()  # Only called once for static
                mock_render.assert_called_once_with("https://example.com", 3000)
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_js_no_improvement(self):
        """Test JS fallback when JavaScript rendering doesn't improve results."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        miner = SelectorMiner("https://example.com")
        
        # Mock static mining to return same or fewer candidates from JS
        with patch.object(miner, 'mine_selectors') as mock_mine:
            mock_static_candidates = [Mock(), Mock()]  # 2 candidates
            mock_js_candidates = [Mock()]  # 1 candidate (worse)
            mock_mine.side_effect = [mock_static_candidates, mock_js_candidates]
            
            # Mock JS rendering
            with patch.object(miner, 'render_with_javascript') as mock_render:
                mock_render.return_value = "<html><body><img src='test.jpg'></body></html>"
                
                result = await miner.mine_selectors_with_js_fallback(
                    "<html><body></body></html>",
                    "https://example.com",
                    min_candidates=3
                )
                
                assert result == mock_static_candidates  # Should return static results
                assert mock_mine.call_count == 2  # Called twice: static + JS
                mock_render.assert_called_once_with("https://example.com", 3000)
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_no_url(self):
        """Test JS fallback when no URL is provided."""
        miner = SelectorMiner("https://example.com")
        
        # Mock static mining to return insufficient candidates
        with patch.object(miner, 'mine_selectors') as mock_mine:
            mock_candidates = [Mock(), Mock()]  # 2 candidates
            mock_mine.return_value = mock_candidates
            
            result = await miner.mine_selectors_with_js_fallback(
                "<html><body></body></html>",
                url=None,  # No URL provided
                min_candidates=3
            )
            
            assert result == mock_candidates
            mock_mine.assert_called_once()


class TestConvenienceFunction:
    """Test cases for the convenience function."""
    
    @pytest.mark.asyncio
    async def test_mine_selectors_with_js_fallback_convenience(self):
        """Test the convenience function for JS fallback."""
        # Create mock candidates that will be returned
        mock_candidates = [Mock(), Mock()]
        
        # Mock the SelectorMiner class and its method
        with patch('backend.app.selector_miner.selector_miner.SelectorMiner') as mock_selector_miner_class:
            mock_miner = Mock()
            mock_miner.mine_selectors_with_js_fallback = AsyncMock(return_value=mock_candidates)
            mock_selector_miner_class.return_value = mock_miner
            
            result = await mine_selectors_with_js_fallback(
                "<html><body><img src='test.jpg'></body></html>",
                "https://example.com",
                "https://example.com",
                2
            )
            
            assert result == mock_candidates
            assert len(result) == 2
            mock_selector_miner_class.assert_called_once_with("https://example.com")
            mock_miner.mine_selectors_with_js_fallback.assert_called_once_with(
                "<html><body><img src='test.jpg'></body></html>",
                "https://example.com",
                2
            )


class TestIntegration:
    """Integration tests for JavaScript fallback."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_js_fallback_workflow(self):
        """Test complete end-to-end JS fallback workflow."""
        # Create a miner instance
        miner = SelectorMiner("https://example.com")
        
        # Test with static HTML that has few images
        static_html = """
        <html>
            <body>
                <img src="logo.png" alt="Logo">
            </body>
        </html>
        """
        
        # Mock PLAYWRIGHT_AVAILABLE to True to ensure JS fallback is attempted
        with patch('backend.app.selector_miner.selector_miner.PLAYWRIGHT_AVAILABLE', True):
            # Mock the methods to simulate the workflow
            with patch.object(miner, 'mine_selectors') as mock_mine:
                # Static mining returns 1 candidate (insufficient)
                mock_static_candidates = [Mock(selector="img", score=0.5)]
                
                # JS mining returns 3 candidates (sufficient)
                mock_js_candidates = [
                    Mock(selector="img.thumbnail", score=0.8),
                    Mock(selector="img.preview", score=0.7),
                    Mock(selector="img.gallery", score=0.6)
                ]
                mock_mine.side_effect = [mock_static_candidates, mock_js_candidates]
                
                # Mock JS rendering
                with patch.object(miner, 'render_with_javascript') as mock_render:
                    mock_render.return_value = """
                    <html>
                        <body>
                            <img class="thumbnail" src="thumb1.jpg">
                            <img class="preview" src="preview1.jpg">
                            <img class="gallery" src="gallery1.jpg">
                        </body>
                    </html>
                    """
                    
                    result = await miner.mine_selectors_with_js_fallback(
                        static_html,
                        "https://example.com",
                        min_candidates=2
                    )
                    
                    # Should return JS results since they're better
                    assert len(result) == 3
                    # Check that we got the JS candidates (they should have higher scores)
                    assert result[0].score == 0.8  # JS candidate with highest score
                    
                    # Verify the workflow was called correctly
                    assert mock_mine.call_count == 2
                    mock_render.assert_called_once_with("https://example.com")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
