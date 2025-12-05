# Integration Tests

## Running Tests

Run all tests:
```bash
pytest backend/tests/test_crawler_integration.py
```

Run specific test:
```bash
pytest backend/tests/test_crawler_integration.py::test_http_connections
```

Run with verbose output:
```bash
pytest backend/tests/test_crawler_integration.py -v
```

Run quick tests (excluding slow tests):
```bash
pytest backend/tests/test_crawler_integration.py -v -m "not slow"
```

Run only integration tests:
```bash
pytest backend/tests/test_crawler_integration.py -v -m "integration"
```

## Test Coverage

- HTTP connections
- JavaScript rendering
- Image extraction
- Image processing pipeline
- Raw image storage
- Thumbnail generation
- JSON sidecar metadata
- Processing speed (>= 0.4 img/s)
- Multisite crawl success rate
- Multithreaded processing
- Batch processing
- Memory management
- Selector miner functionality
- Error handling
- Concurrent processing
- Full integration workflow

## Exit Codes

Tests follow pytest conventions:
- 0: All tests passed
- 1: One or more tests failed

## Test Categories

### Fast Tests
- HTTP connections
- JavaScript rendering
- Image extraction
- Multithreaded functionality
- Batch processing
- Memory management
- Error handling
- Concurrent processing

### Slow Tests (marked with @pytest.mark.slow)
- Processing speed validation
- Multisite success rate
- Selector miner functionality

### Integration Tests (marked with @pytest.mark.integration)
- Full integration workflow

## Docker Testing

Run tests in Docker environment:
```bash
make test-integration
make test-quick
```

## Test Data

Tests use real websites for integration testing:
- `https://wikifeet.com` - For image-rich content
- `https://candidteens.net` - For multisite testing
- `https://httpbin.org/html` - For basic HTTP testing
- `https://example.com` - For JavaScript rendering

## Performance Benchmarks

The test suite includes performance validation:
- Minimum processing rate: 0.4 images/second
- 100% success rate for multisite crawls
- Memory management verification
- Concurrent processing validation

