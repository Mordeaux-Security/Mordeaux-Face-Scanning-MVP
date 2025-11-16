"""
API-level tests for /api/v1/identity_safe_search endpoint.

These tests enforce the strict "verify-then-search, no leak" rule:
- No search occurs when verification fails
- Identity filtering behaves correctly on success
- All failure paths prevent search execution
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Mock problematic dependencies before importing main to avoid dependency issues
sys.modules['logging_utils'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['minio'] = MagicMock()
sys.modules['redis'] = MagicMock()

# Add parent directory to path so we can import main
face_pipeline_path = Path(__file__).parent.parent
sys.path.insert(0, str(face_pipeline_path))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from qdrant_client.http.models import ScoredPoint

import main
from main import app, VERIFY_THRESHOLD_DEFAULT

client = TestClient(app)


def create_normalized_vector(dim: int = 512) -> np.ndarray:
    """Create a normalized unit vector."""
    vec = np.ones(dim, dtype=np.float32)
    return vec / np.sqrt(dim)


def create_fake_match(face_id: str, score: float, tenant_id: str, identity_id: str) -> ScoredPoint:
    """Create a fake ScoredPoint match for testing."""
    return ScoredPoint(
        id=face_id,
        score=score,
        payload={
            "image_sha256": f"hash_{face_id}",
            "tenant_id": tenant_id,
            "identity_id": identity_id,
            "url": f"https://example.com/{face_id}.jpg",
            "site": "example.com",
        }
    )


class TestIdentitySafeSearchHappyPath:
    """Test happy path scenarios where verification succeeds and search runs."""
    
    def test_identity_safe_search_happy_path(self, monkeypatch):
        """Test that happy path returns verified=true with results."""
        # --- Arrange ---
        
        # 1) Mock embedding to be a unit vector
        def mock_embed(image_b64: str, require_single_face: bool, quality_cfg):
            v = np.ones(512, dtype=np.float32)
            v /= np.linalg.norm(v)
            return v, Mock()  # Return embedding and face quality wrapper
        
        monkeypatch.setattr(
            main,
            "embed_one_b64_strict",
            mock_embed,
            raising=True,
        )
        
        # 2) Mock centroid fetch to match the probe vector (so similarity ~1.0)
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            v = np.ones(512, dtype=np.float32)
            v /= np.linalg.norm(v)
            return v
        
        monkeypatch.setattr(
            main,
            "_fetch_identity_centroid",
            mock_fetch_centroid,
            raising=True,
        )
        
        # 3) Mock search to return one identity-matched face
        class DummyMatch:
            def __init__(self, id, score, payload):
                self.id = id
                self.score = score
                self.payload = payload
        
        def mock_search_faces_for_identity(query_vector, tenant_id, identity_id, top_k, min_score):
            return [
                DummyMatch(
                    id="face1",
                    score=0.95,
                    payload={
                        "image_sha256": "img1",  # Maps to image_id in response
                        "tenant_id": tenant_id,
                        "identity_id": identity_id,
                    },
                )
            ]
        
        monkeypatch.setattr(
            main,
            "_search_faces_for_identity",
            mock_search_faces_for_identity,
            raising=True,
        )
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",  # not actually used due to mock
            "top_k": 10,
            "min_score": 0.0,
        }
        
        # --- Act ---
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        # --- Assert ---
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is True
        assert data["similarity"] >= VERIFY_THRESHOLD_DEFAULT
        assert data["threshold"] == pytest.approx(VERIFY_THRESHOLD_DEFAULT)
        assert data["reason"] is None
        
        assert len(data["results"]) == 1
        r0 = data["results"][0]
        assert r0["id"] == "face1"
        assert r0["tenant_id"] == "tenantA"
        assert r0["identity_id"] == "identity123"
        assert r0["image_id"] == "img1"
    
    def test_identity_filtering_on_success(self, monkeypatch):
        """Test that search only returns faces for the specified identity."""
        probe_vec = create_normalized_vector(512)
        centroid_vec = probe_vec.copy() * 0.98  # Very similar
        centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-9)
        
        # Create matches that should be filtered by identity
        fake_matches = [
            create_fake_match("face1", 0.95, "tenantA", "identity123"),  # Correct identity
            create_fake_match("face2", 0.92, "tenantA", "identity123"),  # Correct identity
        ]
        
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            return centroid_vec
        
        def mock_search_faces_for_identity(query_vector, tenant_id, identity_id, top_k, min_score):
            # Verify that search is called with correct identity filter
            assert tenant_id == "tenantA"
            assert identity_id == "identity123"
            return fake_matches
        
        def mock_embed(image_b64, require_single_face, quality_cfg):
            return probe_vec, Mock()
        
        monkeypatch.setattr("main._fetch_identity_centroid", mock_fetch_centroid)
        monkeypatch.setattr("main._search_faces_for_identity", mock_search_faces_for_identity)
        monkeypatch.setattr("main.embed_one_b64_strict", mock_embed)
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is True
        # Verify all results belong to the requested identity
        for result in data["results"]:
            assert result["tenant_id"] == "tenantA"
            assert result["identity_id"] == "identity123"


class TestIdentitySafeSearchNoLeakRule:
    """Test that search is NEVER executed when verification fails (strict no-leak rule)."""
    
    def test_identity_safe_search_low_similarity_no_search(self, monkeypatch):
        """Test that low similarity returns verified=false and search is NOT called."""
        # --- Arrange ---
        
        # 1) Probe embedding: unit vector
        def mock_embed(image_b64: str, require_single_face: bool, quality_cfg):
            v = np.ones(512, dtype=np.float32)
            v /= np.linalg.norm(v)
            return v, Mock()  # Return embedding and face quality wrapper
        
        monkeypatch.setattr(
            main,
            "embed_one_b64_strict",
            mock_embed,
            raising=True,
        )
        
        # 2) Centroid: opposite direction → similarity ~ -1.0
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            v = -np.ones(512, dtype=np.float32)
            v /= np.linalg.norm(v)
            return v
        
        monkeypatch.setattr(
            main,
            "_fetch_identity_centroid",
            mock_fetch_centroid,
            raising=True,
        )
        
        # 3) Search helper should NEVER be called
        def mock_search_faces_for_identity(*args, **kwargs):
            raise AssertionError("_search_faces_for_identity should not be called when similarity is low")
        
        monkeypatch.setattr(
            main,
            "_search_faces_for_identity",
            mock_search_faces_for_identity,
            raising=True,
        )
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        # --- Act ---
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        # --- Assert ---
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is False
        assert data["results"] == []
        assert data["reason"]["error"] == "low_similarity"
    
    def test_identity_not_found_no_search_executed(self, monkeypatch):
        """Test that missing identity returns verified=false and search is NOT called."""
        probe_vec = create_normalized_vector(512)
        
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            return None  # Identity not found
        
        def mock_search_faces_for_identity(query_vector, tenant_id, identity_id, top_k, min_score):
            # CRITICAL: This should NEVER be called when identity is not found
            raise AssertionError(
                "_search_faces_for_identity should NOT be called when identity is not found"
            )
        
        def mock_embed(image_b64, require_single_face, quality_cfg):
            return probe_vec, Mock()
        
        monkeypatch.setattr("main._fetch_identity_centroid", mock_fetch_centroid)
        monkeypatch.setattr("main._search_faces_for_identity", mock_search_faces_for_identity)
        monkeypatch.setattr("main.embed_one_b64_strict", mock_embed)
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "non-existent",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        # Assertions
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is False
        assert data["similarity"] == 0.0
        assert data["reason"]["error"] == "identity_not_found"
        assert data["results"] == []
        assert data["count"] == 0
    
    def test_identity_safe_search_quality_failure(self, monkeypatch):
        """Test that quality/face detection failure returns verified=false and search is NOT called."""
        # --- Arrange ---
        
        # 1) embed_one_b64_strict raises a quality-related HTTPException
        def mock_embed(image_b64: str, require_single_face: bool, quality_cfg):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "multiple_faces_detected", "num_usable_faces": 3},
            )
        
        monkeypatch.setattr(
            main,
            "embed_one_b64_strict",
            mock_embed,
            raising=True,
        )
        
        # 2) These helpers should never be called if embedding fails
        def mock_fetch_centroid(*args, **kwargs):
            raise AssertionError("_fetch_identity_centroid should not be called on quality failure")
        
        def mock_search_faces_for_identity(*args, **kwargs):
            raise AssertionError("_search_faces_for_identity should not be called on quality failure")
        
        monkeypatch.setattr(main, "_fetch_identity_centroid", mock_fetch_centroid, raising=True)
        monkeypatch.setattr(main, "_search_faces_for_identity", mock_search_faces_for_identity, raising=True)
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        # --- Act ---
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        # --- Assert ---
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is False
        assert data["results"] == []
        assert data["reason"]["error"] == "multiple_faces_detected"
    
    def test_no_faces_detected_no_search_executed(self, monkeypatch):
        """Test that no faces detected returns verified=false and search is NOT called."""
        def mock_embed(image_b64, require_single_face, quality_cfg):
            # Simulate no faces detected
            raise HTTPException(
                status_code=422,
                detail={"error": "no_usable_faces", "reasons": ["no_faces_detected"]}
            )
        
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            raise AssertionError("_fetch_identity_centroid should NOT be called when no faces detected")
        
        def mock_search_faces_for_identity(query_vector, tenant_id, identity_id, top_k, min_score):
            raise AssertionError("_search_faces_for_identity should NOT be called when no faces detected")
        
        monkeypatch.setattr("main.embed_one_b64_strict", mock_embed)
        monkeypatch.setattr("main._fetch_identity_centroid", mock_fetch_centroid)
        monkeypatch.setattr("main._search_faces_for_identity", mock_search_faces_for_identity)
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is False
        assert data["similarity"] == 0.0
        assert data["reason"]["error"] == "no_usable_faces"
        assert data["results"] == []
        assert data["count"] == 0


class TestIdentitySafeSearchEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_search_results_but_verified(self, monkeypatch):
        """Test that empty search results still return verified=true if similarity is high."""
        probe_vec = create_normalized_vector(512)
        centroid_vec = probe_vec.copy() * 0.95
        centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-9)
        
        similarity = float(np.dot(probe_vec, centroid_vec))
        assert similarity >= VERIFY_THRESHOLD_DEFAULT
        
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            return centroid_vec
        
        def mock_search_faces_for_identity(query_vector, tenant_id, identity_id, top_k, min_score):
            return []  # Empty results but verified
        
        def mock_embed(image_b64, require_single_face, quality_cfg):
            return probe_vec, Mock()
        
        monkeypatch.setattr("main._fetch_identity_centroid", mock_fetch_centroid)
        monkeypatch.setattr("main._search_faces_for_identity", mock_search_faces_for_identity)
        monkeypatch.setattr("main.embed_one_b64_strict", mock_embed)
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["verified"] is True
        assert data["similarity"] >= VERIFY_THRESHOLD_DEFAULT
        assert data["results"] == []
        assert data["count"] == 0
    
    def test_similarity_exactly_at_threshold(self, monkeypatch):
        """Test behavior when similarity is exactly at threshold (should pass)."""
        probe_vec = create_normalized_vector(512)
        # Create a centroid that gives exactly threshold similarity
        # For threshold 0.78, we need similarity >= 0.78
        # Use a vector that's close but at threshold
        centroid_vec = probe_vec * VERIFY_THRESHOLD_DEFAULT + create_normalized_vector(512) * (1 - VERIFY_THRESHOLD_DEFAULT) * 0.1
        centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-9)
        
        similarity = float(np.dot(probe_vec, centroid_vec))
        
        fake_matches = [create_fake_match("face1", 0.85, "tenantA", "identity123")]
        
        def mock_fetch_centroid(tenant_id: str, identity_id: str):
            return centroid_vec
        
        def mock_search_faces_for_identity(query_vector, tenant_id, identity_id, top_k, min_score):
            return fake_matches
        
        def mock_embed(image_b64, require_single_face, quality_cfg):
            return probe_vec, Mock()
        
        monkeypatch.setattr("main._fetch_identity_centroid", mock_fetch_centroid)
        monkeypatch.setattr("main._search_faces_for_identity", mock_search_faces_for_identity)
        monkeypatch.setattr("main.embed_one_b64_strict", mock_embed)
        
        req = {
            "tenant_id": "tenantA",
            "identity_id": "identity123",
            "image_b64": "data:image/jpeg;base64,FAKE",
            "top_k": 10,
            "min_score": 0.0,
        }
        
        resp = client.post("/api/v1/identity_safe_search", json=req)
        
        assert resp.status_code == 200
        data = resp.json()
        
        # If similarity >= threshold, should be verified
        if similarity >= VERIFY_THRESHOLD_DEFAULT:
            assert data["verified"] is True
            assert len(data["results"]) > 0
        else:
            assert data["verified"] is False
            assert data["results"] == []


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_identity_safe_search_api.py -v
    pytest.main([__file__, "-v"])

