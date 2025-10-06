import pytest
import asyncio
import os
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set test environment variables
os.environ.update({
    "ENVIRONMENT": "testing",
    "LOG_LEVEL": "debug",
    "POSTGRES_PASSWORD": "test_password",
    "S3_ACCESS_KEY": "test_access_key",
    "S3_SECRET_KEY": "test_secret_key",
    "RATE_LIMIT_PER_MINUTE": "100",
    "RATE_LIMIT_PER_HOUR": "1000",
    "PRESIGNED_URL_TTL": "600",
    "CRAWLED_THUMBS_RETENTION_DAYS": "90",
    "USER_QUERY_IMAGES_RETENTION_HOURS": "24",
    "MAX_IMAGE_SIZE_MB": "10",
    "P95_LATENCY_THRESHOLD_SECONDS": "5.0"
})

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch('app.core.rate_limiter.get_redis_client') as mock:
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_client.pipeline.return_value.__enter__.return_value = MagicMock()
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    with patch('app.core.audit.get_audit_db_pool') as mock:
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_cur = AsyncMock()
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cur
        mock_conn.__aenter__.return_value = mock_conn
        mock_pool.connection.return_value = mock_conn
        mock.return_value = mock_pool
        yield mock_pool

@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    with patch('app.services.storage.save_raw_and_thumb') as mock:
        mock.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
        yield mock

@pytest.fixture
def mock_face_service():
    """Mock face service for testing."""
    with patch('app.services.face.get_face_service') as mock:
        mock_face = MagicMock()
        mock_face.compute_phash.return_value = "test-phash"
        mock_face.detect_and_embed.return_value = [
            {
                "embedding": [0.1] * 512,
                "bbox": [0, 0, 100, 100],
                "det_score": 0.9
            }
        ]
        mock.return_value = mock_face
        yield mock_face

@pytest.fixture
def mock_vector_service():
    """Mock vector service for testing."""
    with patch('app.services.vector.get_vector_client') as mock:
        mock_vec = MagicMock()
        mock_vec.using_pinecone.return_value = False
        mock_vec.search_similar.return_value = [
            {"id": "face-1", "score": 0.9, "metadata": {}}
        ]
        mock.return_value = mock_vec
        yield mock_vec

@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing."""
    with patch('app.core.audit.get_audit_logger') as mock:
        mock_audit = AsyncMock()
        mock.return_value = mock_audit
        yield mock_audit

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def valid_tenant_headers():
    """Valid tenant headers for testing."""
    return {"X-Tenant-ID": "test-tenant-123"}

@pytest.fixture
def invalid_tenant_headers():
    """Invalid tenant headers for testing."""
    return {"X-Tenant-ID": "ab"}  # Too short

@pytest.fixture
def admin_tenant_headers():
    """Admin tenant headers for testing."""
    return {"X-Tenant-ID": "admin-tenant"}

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests that don't have integration marker
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.keywords for keyword in ["cleanup", "performance", "load"]):
            item.add_marker(pytest.mark.slow)
