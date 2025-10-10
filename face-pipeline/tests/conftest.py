"""
Pytest Configuration and Fixtures

TODO: Implement shared test fixtures
TODO: Add mock face detector/embedder
TODO: Add sample image fixtures
TODO: Add test database/storage fixtures
TODO: Add cleanup hooks
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image_bytes():
    """
    Provide sample image bytes for testing.
    
    TODO: Load or generate sample image
    TODO: Include faces for testing
    """
    pass


@pytest.fixture
def sample_face_crop():
    """
    Provide sample face crop as numpy array.
    
    TODO: Create or load sample face crop
    """
    pass


@pytest.fixture
def sample_embedding():
    """
    Provide sample face embedding vector.
    
    TODO: Generate realistic embedding (512-dim vector)
    """
    pass


@pytest.fixture
def mock_detector():
    """
    Provide mock face detector for testing.
    
    TODO: Create mock detector that returns predictable results
    """
    pass


@pytest.fixture
def mock_embedder():
    """
    Provide mock face embedder for testing.
    
    TODO: Create mock embedder
    """
    pass


@pytest.fixture
def mock_storage():
    """
    Provide mock storage manager for testing.
    
    TODO: Create in-memory or temp file storage mock
    """
    pass


@pytest.fixture
def mock_indexer():
    """
    Provide mock vector indexer for testing.
    
    TODO: Create in-memory vector index mock
    """
    pass


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Provide path to test data directory.
    
    TODO: Create test data directory structure
    """
    pass


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """
    Cleanup temporary files after each test.
    
    TODO: Implement cleanup logic
    """
    yield
    # TODO: Clean up any temporary files

