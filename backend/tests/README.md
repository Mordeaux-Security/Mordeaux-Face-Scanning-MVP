# Test Suite for Mordeaux Face Scanning MVP

This directory contains comprehensive tests for all Non-Functional Requirements (NFRs) and core functionality.

## Test Structure

### Core Test Files

- **`test_tenant_scoping.py`** - Tests for tenant isolation and X-Tenant-ID header validation
- **`test_performance.py`** - Tests for P95 latency tracking and performance monitoring
- **`test_rate_limiting.py`** - Tests for rate limiting and request size validation
- **`test_audit_logging.py`** - Tests for audit logging functionality
- **`test_retention_cleanup.py`** - Tests for data retention and cleanup jobs

### Configuration

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`requirements.txt`** - Testing dependencies
- **`README.md`** - This documentation

## Running Tests

### Install Dependencies

```bash
cd backend
pip install -r tests/requirements.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests only
pytest tests/ -m integration

# Performance tests only
pytest tests/ -m slow

# Specific test file
pytest tests/test_tenant_scoping.py

# Specific test class
pytest tests/test_tenant_scoping.py::TestTenantScoping

# Specific test method
pytest tests/test_tenant_scoping.py::TestTenantScoping::test_missing_tenant_id_returns_400
```

### Run with Coverage

```bash
pytest tests/ --cov=app --cov-report=html
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

## Test Categories

### 1. Tenant Scoping Tests (`test_tenant_scoping.py`)

Tests the complete tenant isolation system:

- **Header Validation**: Tests X-Tenant-ID header requirements and validation
- **Tenant Isolation**: Verifies that different tenants are properly isolated
- **Service Integration**: Tests that tenant_id is passed to all services
- **Storage Isolation**: Verifies tenant-scoped object storage
- **Vector Isolation**: Tests tenant-filtered vector searches
- **Database Isolation**: Confirms tenant-scoped database operations

**Key Test Methods:**
- `test_missing_tenant_id_returns_400()`
- `test_invalid_tenant_id_returns_400()`
- `test_tenant_id_passed_to_services()`
- `test_tenant_isolation_in_search()`
- `test_different_tenants_isolated()`

### 2. Performance Tests (`test_performance.py`)

Tests P95 latency tracking and performance monitoring:

- **Metrics Collection**: Tests performance metrics gathering
- **P95 Calculation**: Verifies P95 latency calculation accuracy
- **Threshold Detection**: Tests P95 threshold exceeded detection
- **Performance Endpoints**: Tests `/metrics` and `/metrics/p95` endpoints
- **Load Testing**: Simulates performance under load
- **Middleware Integration**: Tests performance tracking in middleware

**Key Test Methods:**
- `test_p95_latency_calculation()`
- `test_threshold_exceeded_detection()`
- `test_performance_under_load()`
- `test_metrics_endpoint()`
- `test_p95_metrics_endpoint()`

### 3. Rate Limiting Tests (`test_rate_limiting.py`)

Tests rate limiting and request size validation:

- **Rate Limiter**: Tests Redis-based rate limiting per tenant
- **Request Size**: Tests 10MB file size validation
- **Configuration**: Tests configurable rate limits
- **Endpoint Protection**: Tests rate limiting on all endpoints
- **Error Handling**: Tests proper error responses
- **Integration**: Tests with other middleware

**Key Test Methods:**
- `test_rate_limiter_exceeds_minute_limit()`
- `test_rate_limiting_on_index_face()`
- `test_large_image_size_rejected()`
- `test_size_validation_on_all_endpoints()`
- `test_rate_limiting_configuration()`

### 4. Audit Logging Tests (`test_audit_logging.py`)

Tests comprehensive audit logging:

- **Request Logging**: Tests logging of all API requests
- **Search Logging**: Tests search-specific audit logging
- **Database Operations**: Tests audit log database operations
- **Error Handling**: Tests graceful error handling
- **Middleware Integration**: Tests audit middleware
- **Correlation IDs**: Tests request correlation tracking

**Key Test Methods:**
- `test_log_request()`
- `test_log_search_operation()`
- `test_audit_logging_on_index_face()`
- `test_audit_middleware_logs_all_requests()`
- `test_audit_logging_with_tenant_scoping()`

### 5. Retention & Cleanup Tests (`test_retention_cleanup.py`)

Tests data retention and cleanup functionality:

- **Cleanup Service**: Tests automated cleanup operations
- **Retention Logic**: Tests retention period logic
- **Storage Cleanup**: Tests object deletion from storage
- **Database Cleanup**: Tests audit log cleanup
- **Configuration**: Tests configurable retention periods
- **API Endpoints**: Tests manual cleanup endpoints

**Key Test Methods:**
- `test_cleanup_crawled_thumbnails()`
- `test_cleanup_user_query_images()`
- `test_cleanup_audit_logs()`
- `test_cleanup_endpoint_success()`
- `test_retention_configuration()`

## Test Fixtures

### Shared Fixtures (in `conftest.py`)

- **`test_client`** - FastAPI test client
- **`mock_redis`** - Mock Redis client
- **`mock_database`** - Mock database connection
- **`mock_storage`** - Mock storage service
- **`mock_face_service`** - Mock face detection service
- **`mock_vector_service`** - Mock vector database service
- **`mock_audit_logger`** - Mock audit logger
- **`valid_tenant_headers`** - Valid tenant headers
- **`invalid_tenant_headers`** - Invalid tenant headers
- **`admin_tenant_headers`** - Admin tenant headers

### Test Images

- **`test_image`** - Small test image (100x100 pixels)
- **`large_test_image`** - Large test image (2000x2000 pixels, >10MB)

## Test Environment

### Environment Variables

Tests use a dedicated test environment with the following settings:

```bash
ENVIRONMENT=testing
LOG_LEVEL=debug
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
PRESIGNED_URL_TTL=600
CRAWLED_THUMBS_RETENTION_DAYS=90
USER_QUERY_IMAGES_RETENTION_HOURS=24
MAX_IMAGE_SIZE_MB=10
P95_LATENCY_THRESHOLD_SECONDS=5.0
```

### Mocking Strategy

Tests use comprehensive mocking to isolate functionality:

- **External Services**: Redis, PostgreSQL, MinIO/S3, Qdrant/Pinecone
- **File Operations**: Image processing and storage
- **Network Calls**: External API calls
- **Time Operations**: Date/time calculations for retention

## NFR Compliance Testing

### Performance Requirements

- ✅ **P95 ≤ 5s**: Tests verify P95 latency calculation and threshold detection
- ✅ **Request Size ≤ 10MB**: Tests validate file size limits
- ✅ **Rate Limiting**: Tests verify per-tenant rate limiting

### Security & Access Control

- ✅ **Tenant Scoping**: Tests verify complete tenant isolation
- ✅ **Audit Logging**: Tests verify comprehensive audit trail
- ✅ **Request Validation**: Tests verify input validation

### Data Management

- ✅ **Retention Policies**: Tests verify automated cleanup
- ✅ **Presigned URLs**: Tests verify 10-minute TTL
- ✅ **Configuration**: Tests verify environment-based config

## Continuous Integration

### GitHub Actions (Recommended)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install -r tests/requirements.txt
      - name: Run tests
        run: |
          cd backend
          pytest tests/ --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Local Development

```bash
# Run tests before committing
pytest tests/ --cov=app

# Run specific test category
pytest tests/ -m unit -v

# Run with performance profiling
pytest tests/test_performance.py --benchmark-only
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Failures**: Check that mocks are properly configured
3. **Async Test Issues**: Use `pytest-asyncio` for async tests
4. **Database Connection**: Tests use mocked database connections

### Debug Mode

```bash
# Run with debug output
pytest tests/ -v -s --tb=long

# Run single test with debug
pytest tests/test_tenant_scoping.py::TestTenantScoping::test_missing_tenant_id_returns_400 -v -s
```

## Test Coverage

The test suite provides comprehensive coverage of:

- **All NFR Requirements**: 100% coverage of specified requirements
- **Core Functionality**: All API endpoints and services
- **Error Handling**: Edge cases and error conditions
- **Integration Points**: Service interactions and middleware
- **Configuration**: Environment-based configuration
- **Performance**: Latency tracking and monitoring

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Add proper docstrings and comments
4. Include both positive and negative test cases
5. Test error conditions and edge cases
6. Update this README if adding new test categories
