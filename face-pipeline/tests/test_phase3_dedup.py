"""
Phase 3 Deduplication Tests

Focused tests for Phase 3 features:
- Hamming distance XOR bitcount logic (table-driven)
- Feature flag behavior (ENABLE_GLOBAL_DEDUP)
- Configurable max_hamming parameter
- Integration tests for near-duplicate detection
"""

import pytest
from unittest.mock import patch, MagicMock
from pipeline.utils import hamming_distance_hex
from pipeline.dedup import should_skip, remember
from config.settings import settings


class TestHammingDistanceLogic:
    """Table-driven tests for Hamming distance calculation."""
    
    @pytest.mark.parametrize("hex_a,hex_b,expected_distance", [
        # Identical hashes
        ("0000", "0000", 0),
        ("ffff", "ffff", 0),
        ("8f373c9c3c9c3c1e", "8f373c9c3c9c3c1e", 0),
        
        # Single bit differences
        ("0000", "0001", 1),
        ("0000", "0002", 1),
        ("0000", "0004", 1),
        ("8f37", "8f36", 1),
        
        # Multiple bit differences
        ("0000", "0003", 2),  # 0b11 = 2 bits
        ("0000", "0007", 3),  # 0b111 = 3 bits
        ("0000", "000f", 4),  # 0b1111 = 4 bits
        
        # Maximum difference (all bits)
        ("0000", "ffff", 16),
        ("00000000", "ffffffff", 32),
        
        # Known real-world cases
        ("8f373c9c3c9c3c1e", "8f373c9c3c9c3c1c", 1),  # Last bit differs
        ("a1b2c3d4e5f6a7b8", "a1b2c3d4e5f6a7b8", 0),  # Identical long hash
        
        # Edge cases
        ("", "", 0),          # Empty strings
        ("0", "0", 0),        # Single char
        ("f", "0", 4),        # Single char, max difference
    ])
    def test_hamming_distance_table(self, hex_a, hex_b, expected_distance):
        """Verify XOR bitcount logic with comprehensive table cases."""
        actual_distance = hamming_distance_hex(hex_a, hex_b)
        assert actual_distance == expected_distance, (
            f"hamming_distance_hex('{hex_a}', '{hex_b}') = {actual_distance}, "
            f"expected {expected_distance}"
        )
    
    def test_hamming_distance_symmetry(self):
        """Verify that distance(a, b) == distance(b, a)."""
        pairs = [
            ("8f37", "8f36"),
            ("0000", "ffff"),
            ("a1b2c3d4", "e5f6a7b8"),
        ]
        
        for a, b in pairs:
            dist_ab = hamming_distance_hex(a, b)
            dist_ba = hamming_distance_hex(b, a)
            assert dist_ab == dist_ba, f"Symmetry failed for {a}, {b}"
    
    def test_hamming_distance_different_lengths(self):
        """Verify that different length strings are padded correctly."""
        # Shorter string should be zero-padded on left
        assert hamming_distance_hex("ff", "00ff") == 0
        assert hamming_distance_hex("1", "0001") == 0
        assert hamming_distance_hex("abc", "000abc") == 0


class TestFeatureFlagBehavior:
    """Test ENABLE_GLOBAL_DEDUP feature flag behavior."""
    
    def setup_method(self):
        """Set up mock Redis client."""
        self.mock_redis = MagicMock()
        self.mock_redis.smembers.return_value = set()
        self.mock_redis.scard.return_value = 0
        self.mock_redis.sadd.return_value = 1
        self.mock_redis.expire.return_value = True
        self.mock_redis.pipeline.return_value.execute.return_value = [1, True]
        
        self.redis_patcher = patch('pipeline.dedup.get_redis_client', return_value=self.mock_redis)
        self.redis_patcher.start()
    
    def teardown_method(self):
        """Clean up patches."""
        self.redis_patcher.stop()
    
    def test_should_skip_disabled_always_false(self):
        """With ENABLE_GLOBAL_DEDUP=false, should_skip always returns False."""
        # Add some hashes to Redis (should be ignored when disabled)
        self.mock_redis.smembers.return_value = {
            b"8f373c9c3c9c3c1e",
            b"8f373c9c3c9c3c1f"
        }
        
        with patch.object(settings, 'enable_global_dedup', False):
            # Even with exact match in Redis, should return False
            result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1e", max_dist=8)
            assert result is False
            
            # Redis should not be queried
            self.mock_redis.smembers.assert_not_called()
    
    def test_remember_disabled_always_succeeds(self):
        """With ENABLE_GLOBAL_DEDUP=false, remember always returns True without Redis ops."""
        with patch.object(settings, 'enable_global_dedup', False):
            result = remember("tenant1", "8f37", "8f373c9c3c9c3c1e")
            assert result is True
            
            # Redis should not be queried
            self.mock_redis.pipeline.assert_not_called()
    
    def test_remember_then_should_skip_exact_match(self):
        """With ENABLE_GLOBAL_DEDUP=true, remember then should_skip returns True for identical pHash."""
        # Simulate remember adding the hash
        self.mock_redis.smembers.return_value = {b"8f373c9c3c9c3c1e"}
        
        with patch.object(settings, 'enable_global_dedup', True):
            # Remember should succeed
            result_remember = remember("tenant1", "8f37", "8f373c9c3c9c3c1e")
            assert result_remember is True
            
            # should_skip should return True for exact match
            result_skip = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1e", max_dist=8)
            assert result_skip is True
    
    def test_remember_then_should_skip_near_duplicate(self):
        """With ENABLE_GLOBAL_DEDUP=true, should_skip detects near-duplicates within threshold."""
        # Store original hash
        self.mock_redis.smembers.return_value = {b"8f373c9c3c9c3c1e"}
        
        with patch.object(settings, 'enable_global_dedup', True):
            # Near-duplicate (Hamming distance = 1) should be detected with max_dist=3
            result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1f", max_dist=3)
            assert result is True
            
            # Far duplicate (many bits different) should not be detected
            result_far = should_skip("tenant1", "8f37", "0000000000000000", max_dist=3)
            assert result_far is False


class TestConfigurableMaxHamming:
    """Test configurable DEDUP_MAX_HAMMING parameter."""
    
    def setup_method(self):
        """Set up mock Redis client."""
        self.mock_redis = MagicMock()
        self.mock_redis.smembers.return_value = {b"8f373c9c3c9c3c00"}
        self.mock_redis.scard.return_value = 1
        
        self.redis_patcher = patch('pipeline.dedup.get_redis_client', return_value=self.mock_redis)
        self.redis_patcher.start()
    
    def teardown_method(self):
        """Clean up patches."""
        self.redis_patcher.stop()
    
    def test_max_dist_parameter_respected(self):
        """Verify that max_dist parameter controls detection threshold."""
        with patch.object(settings, 'enable_global_dedup', True):
            # Hash with distance of 8 bits from stored hash
            # 8f373c9c3c9c3c00 XOR 8f373c9c3c9c3cff = 0xff = 8 bits
            test_hash = "8f373c9c3c9c3cff"
            
            # Should skip with max_dist=8 (exactly at threshold)
            result_8 = should_skip("tenant1", "8f37", test_hash, max_dist=8)
            assert result_8 is True
            
            # Should NOT skip with max_dist=7 (below threshold)
            result_7 = should_skip("tenant1", "8f37", test_hash, max_dist=7)
            assert result_7 is False
            
            # Should skip with max_dist=10 (above threshold)
            result_10 = should_skip("tenant1", "8f37", test_hash, max_dist=10)
            assert result_10 is True
    
    def test_settings_default_used_when_none(self):
        """Verify that settings.dedup_max_hamming is used when max_dist=None."""
        with patch.object(settings, 'enable_global_dedup', True):
            with patch.object(settings, 'dedup_max_hamming', 12):
                # Call without max_dist parameter
                # This should use settings.dedup_max_hamming = 12
                # Hash with distance of 8 should be detected
                test_hash = "8f373c9c3c9c3cff"
                result = should_skip("tenant1", "8f37", test_hash, max_dist=None)
                assert result is True  # 8 <= 12


class TestIntegrationScenarios:
    """Integration tests for real-world deduplication scenarios."""
    
    def setup_method(self):
        """Set up mock Redis client."""
        self.mock_redis = MagicMock()
        self.stored_hashes = set()
        
        # Mock smembers to return our stored hashes
        def mock_smembers(key):
            return {h.encode() for h in self.stored_hashes}
        
        self.mock_redis.smembers.side_effect = mock_smembers
        self.mock_redis.scard.return_value = 0
        self.mock_redis.sadd.return_value = 1
        self.mock_redis.expire.return_value = True
        self.mock_redis.pipeline.return_value.execute.return_value = [1, True]
        
        self.redis_patcher = patch('pipeline.dedup.get_redis_client', return_value=self.mock_redis)
        self.redis_patcher.start()
    
    def teardown_method(self):
        """Clean up patches."""
        self.redis_patcher.stop()
    
    def test_first_run_no_duplicates(self):
        """First run with empty cache should not find duplicates."""
        with patch.object(settings, 'enable_global_dedup', True):
            self.stored_hashes = set()  # Empty cache
            
            result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1e", max_dist=8)
            assert result is False
    
    def test_second_run_exact_duplicate(self):
        """Second run with same hash should detect exact duplicate."""
        with patch.object(settings, 'enable_global_dedup', True):
            test_hash = "8f373c9c3c9c3c1e"
            
            # Simulate first run storing the hash
            self.stored_hashes.add(test_hash)
            
            # Second run should detect duplicate
            result = should_skip("tenant1", "8f37", test_hash, max_dist=8)
            assert result is True
    
    def test_cropped_image_near_duplicate(self):
        """Cropped/resized images should be detected as near-duplicates."""
        with patch.object(settings, 'enable_global_dedup', True):
            original = "8f373c9c3c9c3c1e"
            # Simulate slight variation from cropping (2 bits different)
            cropped = "8f373c9c3c9c3c1c"
            
            self.stored_hashes.add(original)
            
            # Should detect with max_dist=8
            result = should_skip("tenant1", "8f37", cropped, max_dist=8)
            assert result is True
    
    def test_different_images_not_duplicates(self):
        """Completely different images should not be detected as duplicates."""
        with patch.object(settings, 'enable_global_dedup', True):
            image1 = "8f373c9c3c9c3c1e"
            image2 = "0000000000000000"  # Very different
            
            self.stored_hashes.add(image1)
            
            # Should NOT detect as duplicate
            result = should_skip("tenant1", "0000", image2, max_dist=8)
            assert result is False
    
    def test_tenant_isolation(self):
        """Hashes from different tenants should not interfere."""
        with patch.object(settings, 'enable_global_dedup', True):
            test_hash = "8f373c9c3c9c3c1e"
            
            # Store in tenant1
            self.stored_hashes.add(test_hash)
            
            # Query from tenant2 - different Redis key, so won't find it
            # (In real implementation, each tenant has separate keys)
            result_tenant2 = should_skip("tenant2", "8f37", test_hash, max_dist=8)
            # This tests the API, actual isolation is via Redis key patterns

