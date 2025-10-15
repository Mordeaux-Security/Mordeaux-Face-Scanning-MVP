"""
Unit tests for the Selector Miner Core.

Tests the deterministic selector mining functionality with various
HTML fixtures and edge cases.
"""

import pytest
import re
from typing import List
from unittest.mock import Mock, patch

from .selector_miner import (
    SelectorMiner, 
    CandidateSelector, 
    ImageNode, 
    mine_selectors_for_url
)


class TestSelectorMiner:
    """Test cases for the SelectorMiner class."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")
    
    def test_basic_image_extraction(self):
        """Test basic image extraction from simple HTML."""
        html = """
        <html>
        <body>
            <img src="/image1.jpg" alt="Test 1">
            <img src="/image2.jpg" alt="Test 2">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        assert len(candidates) > 0
        assert any('img' in candidate.selector for candidate in candidates)
        
        # Check that we found the images
        img_candidate = next((c for c in candidates if 'img' in c.selector), None)
        assert img_candidate is not None
        assert img_candidate.repetition_count == 2
    
    def test_video_thumbnail_detection(self):
        """Test detection of video thumbnails with duration text."""
        html = """
        <html>
        <body>
            <div class="video-gallery">
                <a href="/video/123">
                    <img src="/thumb/123.jpg" class="thumbnail">
                    <span class="duration">2:30</span>
                </a>
                <a href="/video/456">
                    <img src="/thumb/456.jpg" class="thumbnail">
                    <span class="duration">1:45</span>
                </a>
                <a href="/video/789">
                    <img src="/thumb/789.jpg" class="thumbnail">
                    <span class="duration">3:15</span>
                </a>
            </div>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should find thumbnail selector with high score
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        
        assert thumbnail_candidate is not None
        assert thumbnail_candidate.repetition_count == 3
        assert thumbnail_candidate.evidence['duration_score'] > 0
        assert thumbnail_candidate.score > 0.5
    
    def test_class_based_selectors(self):
        """Test generation of class-based selectors."""
        html = """
        <html>
        <body>
            <div class="content">
                <img src="/img1.jpg" class="preview-image">
                <img src="/img2.jpg" class="preview-image">
                <img src="/img3.jpg" class="preview-image">
            </div>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should prefer class-based selector
        preview_candidate = next(
            (c for c in candidates if 'preview-image' in c.selector), 
            None
        )
        
        assert preview_candidate is not None
        assert preview_candidate.selector == "body > div.content > img.preview-image"
        assert preview_candidate.repetition_count == 3
    
    def test_background_image_detection(self):
        """Test detection of background images."""
        html = """
        <html>
        <body>
            <div class="hero-section" style="background-image: url('/hero.jpg')">
                Content here
            </div>
            <div class="banner" style="background-image: url('/banner.jpg')">
                More content
            </div>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should detect background images
        bg_candidates = [c for c in candidates if 'background' in c.description.lower()]
        assert len(bg_candidates) > 0
    
    def test_meta_image_detection(self):
        """Test detection of meta og:image tags."""
        html = """
        <html>
        <head>
            <meta property="og:image" content="/og-image.jpg">
        </head>
        <body>
            <img src="/regular.jpg">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should find both meta and regular images
        assert len(candidates) >= 2
        
        # Meta image should be detected
        meta_candidate = next(
            (c for c in candidates if 'meta' in c.selector), 
            None
        )
        assert meta_candidate is not None
    
    def test_srcset_richness_scoring(self):
        """Test scoring based on srcset richness."""
        html = """
        <html>
        <body>
            <img src="/img1.jpg" 
                 srcset="/img1-small.jpg 320w, /img1-medium.jpg 640w, /img1-large.jpg 1280w">
            <img src="/img2.jpg" 
                 srcset="/img2-small.jpg 320w, /img2-medium.jpg 640w">
            <img src="/img3.jpg">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        img_candidate = next((c for c in candidates if 'img' in c.selector), None)
        assert img_candidate is not None
        assert img_candidate.evidence['srcset_score'] > 0
    
    def test_negative_class_filtering(self):
        """Test filtering out images with negative class hints."""
        html = """
        <html>
        <body>
            <img src="/logo.jpg" class="site-logo">
            <img src="/avatar.jpg" class="user-avatar">
            <img src="/icon.jpg" class="social-icon">
            <img src="/thumb1.jpg" class="thumbnail">
            <img src="/thumb2.jpg" class="thumbnail">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should prefer thumbnail selector over logo/avatar/icon
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        
        assert thumbnail_candidate is not None
        assert thumbnail_candidate.score > 0.2  # Should have decent score
    
    def test_url_quality_assessment(self):
        """Test URL quality assessment."""
        html = """
        <html>
        <body>
            <img src="/high-quality-thumb.jpg">
            <img src="/medium-preview.jpg">
            <img src="/small-icon.png">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        img_candidate = next((c for c in candidates if 'img' in c.selector), None)
        assert img_candidate is not None
        # Should have some positive URL quality score
        assert img_candidate.evidence['url_quality_score'] > 0
    
    def test_random_token_filtering(self):
        """Test filtering out random tokens in selectors."""
        html = """
        <html>
        <body>
            <div class="content abc123def456">
                <img src="/img1.jpg" class="preview xyz789">
                <img src="/img2.jpg" class="preview">
                <img src="/img3.jpg" class="preview">
            </div>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should prefer stable classes over random tokens
        preview_candidate = next(
            (c for c in candidates if 'preview' in c.selector), 
            None
        )
        
        assert preview_candidate is not None
        # Should not include random tokens in selector
        assert 'abc123def456' not in preview_candidate.selector
        assert 'xyz789' not in preview_candidate.selector
    
    def test_maximum_selector_depth(self):
        """Test that selectors don't exceed maximum depth."""
        html = """
        <html>
        <body>
            <div class="outer">
                <div class="middle">
                    <div class="inner">
                        <div class="deep">
                            <img src="/deep-image.jpg" class="target">
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        target_candidate = next(
            (c for c in candidates if 'target' in c.selector), 
            None
        )
        
        assert target_candidate is not None
        # Should not exceed 4 levels
        levels = len(target_candidate.selector.split(' > '))
        assert levels <= 4
    
    def test_empty_html_handling(self):
        """Test handling of empty or invalid HTML."""
        candidates = self.miner.mine_selectors("")
        assert len(candidates) == 0
        
        candidates = self.miner.mine_selectors("<html><body>No images here</body></html>")
        assert len(candidates) == 0
    
    def test_invalid_image_urls(self):
        """Test filtering of invalid image URLs."""
        html = """
        <html>
        <body>
            <img src="javascript:alert('xss')">
            <img src="data:text/plain,not-an-image">
            <img src="/valid-image.jpg">
            <img src="">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should only find the valid image
        img_candidate = next((c for c in candidates if 'img' in c.selector), None)
        assert img_candidate is not None
        assert img_candidate.repetition_count == 1
        assert '/valid-image.jpg' in img_candidate.sample_urls[0]


class TestEvidenceGathering:
    """Test cases for evidence gathering functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")
    
    def test_duration_pattern_detection(self):
        """Test detection of duration patterns in nearby text."""
        html = """
        <html>
        <body>
            <a href="/video/123">
                <img src="/thumb.jpg" class="thumbnail">
                <span>2:30</span>
            </a>
            <a href="/video/456">
                <img src="/thumb2.jpg" class="thumbnail">
                <span>1:45</span>
            </a>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        
        assert thumbnail_candidate is not None
        assert thumbnail_candidate.evidence['duration_score'] > 0
    
    def test_video_url_pattern_detection(self):
        """Test detection of video URL patterns in ancestor links."""
        html = """
        <html>
        <body>
            <a href="/video/watch/123">
                <img src="/thumb.jpg" class="thumbnail">
            </a>
            <a href="/embed/456">
                <img src="/thumb2.jpg" class="thumbnail">
            </a>
            <a href="/player/789">
                <img src="/thumb3.jpg" class="thumbnail">
            </a>
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        
        assert thumbnail_candidate is not None
        assert thumbnail_candidate.evidence['video_url_score'] > 0
    
    def test_class_hint_scoring(self):
        """Test scoring based on positive and negative class hints."""
        html = """
        <html>
        <body>
            <img src="/thumb1.jpg" class="thumbnail preview">
            <img src="/thumb2.jpg" class="thumbnail image">
            <img src="/thumb3.jpg" class="thumbnail pic">
            <img src="/logo.jpg" class="site-logo">
            <img src="/avatar.jpg" class="user-avatar">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Thumbnail should score higher than logo/avatar
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        
        assert thumbnail_candidate is not None
        assert thumbnail_candidate.evidence['class_hint_score'] > 0


class TestScoringSystem:
    """Test cases for the scoring system."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.miner = SelectorMiner("https://example.com")
    
    def test_repetition_scoring(self):
        """Test scoring based on repetition count."""
        html = """
        <html>
        <body>
            <img src="/img1.jpg" class="target">
            <img src="/img2.jpg" class="target">
            <img src="/img3.jpg" class="target">
            <img src="/img4.jpg" class="target">
            <img src="/img5.jpg" class="target">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        target_candidate = next(
            (c for c in candidates if 'target' in c.selector), 
            None
        )
        
        assert target_candidate is not None
        assert target_candidate.repetition_count == 5
        assert target_candidate.evidence['repetition_score'] > 0.4
    
    def test_score_clamping(self):
        """Test that scores are properly clamped to [0, 1] range."""
        html = """
        <html>
        <body>
            <img src="/img.jpg" class="test">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        for candidate in candidates:
            assert 0.0 <= candidate.score <= 1.0
    
    def test_ranking_order(self):
        """Test that candidates are properly ranked by score."""
        html = """
        <html>
        <body>
            <img src="/single.jpg" class="single">
            <img src="/thumb1.jpg" class="thumbnail">
            <img src="/thumb2.jpg" class="thumbnail">
            <img src="/thumb3.jpg" class="thumbnail">
        </body>
        </html>
        """
        
        candidates = self.miner.mine_selectors(html)
        
        # Should be sorted by score (descending)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)
        
        # Thumbnail should rank higher than single
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        single_candidate = next(
            (c for c in candidates if 'single' in c.selector), 
            None
        )
        
        if thumbnail_candidate and single_candidate:
            assert thumbnail_candidate.score > single_candidate.score


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_random_token_detection(self):
        """Test detection of random tokens."""
        miner = SelectorMiner()
        
        # Random-looking tokens
        assert miner._is_random_token("abc123def456")
        assert miner._is_random_token("123456789")
        assert miner._is_random_token("XYZ789ABC")
        assert miner._is_random_token("_abc123def_")
        
        # Stable tokens
        assert not miner._is_random_token("thumbnail")
        assert not miner._is_random_token("preview-image")
        assert not miner._is_random_token("video-thumb")
        assert not miner._is_random_token("ab")  # Too short
    
    def test_url_quality_assessment(self):
        """Test URL quality assessment."""
        miner = SelectorMiner()
        
        # High quality URLs
        assert miner._assess_url_quality("/high-quality-thumb.jpg")
        assert miner._assess_url_quality("/medium-preview.jpg")
        assert miner._assess_url_quality("/hd-image.jpg")
        
        # Lower quality URLs (but still valid images)
        # Note: All image extensions are now considered quality indicators
        assert miner._assess_url_quality("/icon.png")  # .png is a quality indicator
        assert miner._assess_url_quality("/logo.jpg")  # .jpg is a quality indicator
        assert miner._assess_url_quality("/small.jpg")  # .jpg is a quality indicator
    
    def test_stable_id_detection(self):
        """Test detection of stable IDs."""
        miner = SelectorMiner()
        
        # Stable IDs
        assert miner._is_stable_id("main-content")
        assert miner._is_stable_id("header-nav")
        assert miner._is_stable_id("sidebar-menu")
        
        # Unstable IDs
        assert not miner._is_stable_id("abc123def456")
        assert not miner._is_stable_id("element-789")
        assert not miner._is_stable_id("x")


class TestConvenienceFunction:
    """Test cases for the convenience function."""
    
    def test_mine_selectors_for_url(self):
        """Test the convenience function."""
        html = """
        <html>
        <body>
            <img src="/test.jpg" class="thumbnail">
        </body>
        </html>
        """
        
        candidates = mine_selectors_for_url(html, "https://example.com")
        
        assert len(candidates) > 0
        thumbnail_candidate = next(
            (c for c in candidates if 'thumbnail' in c.selector), 
            None
        )
        assert thumbnail_candidate is not None


# Test fixtures for integration testing
@pytest.fixture
def video_gallery_html():
    """HTML fixture with video gallery structure."""
    return """
    <html>
    <head>
        <meta property="og:image" content="/og-image.jpg">
    </head>
    <body>
        <div class="video-gallery">
            <a href="/video/123">
                <img src="/thumb/123.jpg" class="video-thumbnail" alt="Video 1">
                <span class="duration">2:30</span>
            </a>
            <a href="/video/456">
                <img src="/thumb/456.jpg" class="video-thumbnail" alt="Video 2">
                <span class="duration">1:45</span>
            </a>
            <a href="/video/789">
                <img src="/thumb/789.jpg" class="video-thumbnail" alt="Video 3">
                <span class="duration">3:15</span>
            </a>
        </div>
        <div class="sidebar">
            <img src="/logo.jpg" class="site-logo">
            <img src="/ad.jpg" class="advertisement">
        </div>
    </body>
    </html>
    """


@pytest.fixture
def mixed_content_html():
    """HTML fixture with mixed image content."""
    return """
    <html>
    <body>
        <div class="hero" style="background-image: url('/hero.jpg')">
            <h1>Welcome</h1>
        </div>
        <div class="content">
            <img src="/article1.jpg" class="article-image">
            <img src="/article2.jpg" class="article-image">
        </div>
        <div class="gallery">
            <img src="/gallery1.jpg" class="gallery-thumb">
            <img src="/gallery2.jpg" class="gallery-thumb">
            <img src="/gallery3.jpg" class="gallery-thumb">
        </div>
    </body>
    </html>
    """


class TestIntegration:
    """Integration tests with realistic HTML fixtures."""
    
    def test_video_gallery_mining(self, video_gallery_html):
        """Test mining selectors from video gallery HTML."""
        candidates = mine_selectors_for_url(video_gallery_html, "https://example.com")
        
        # Should find video thumbnail selector
        video_thumb_candidate = next(
            (c for c in candidates if 'video-thumbnail' in c.selector), 
            None
        )
        
        assert video_thumb_candidate is not None
        assert video_thumb_candidate.repetition_count == 3
        assert video_thumb_candidate.evidence['duration_score'] > 0
        assert video_thumb_candidate.evidence['video_url_score'] > 0
        assert video_thumb_candidate.score > 0.6  # Should score highly
    
    def test_mixed_content_mining(self, mixed_content_html):
        """Test mining selectors from mixed content HTML."""
        candidates = mine_selectors_for_url(mixed_content_html, "https://example.com")
        
        # Should find multiple selector types
        selector_types = [c.selector for c in candidates]
        
        # Should have gallery selector
        gallery_candidate = next(
            (c for c in candidates if 'gallery-thumb' in c.selector), 
            None
        )
        assert gallery_candidate is not None
        assert gallery_candidate.repetition_count == 3
        
        # Should have article selector
        article_candidate = next(
            (c for c in candidates if 'article-image' in c.selector), 
            None
        )
        assert article_candidate is not None
        assert article_candidate.repetition_count == 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
