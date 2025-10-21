# Proxy Smoke Tests Documentation

This document describes the comprehensive smoke tests for the Mordeaux Face Scanning MVP proxy setup.

## Overview

The smoke tests verify that the Nginx reverse proxy is correctly routing requests and that all services are accessible through the expected endpoints.

## Test Categories

### 1. Nginx Routing Tests
- **Frontend routing**: `http://localhost/` → Frontend service
- **Backend API routing**: `http://localhost/api/` → Backend service
- **Face Pipeline routing**: `http://localhost/face-pipeline/` → Face Pipeline service

### 2. CORS Headers Tests
- **Preflight requests**: OPTIONS requests are handled correctly
- **CORS headers**: All responses include proper CORS headers
- **Cross-origin requests**: Frontend can make requests to API endpoints

### 3. Port Mapping Tests
- **Direct backend access**: `http://localhost:8000/health`
- **Direct frontend access**: `http://localhost:3000/`
- **Direct pipeline access**: `http://localhost:8001/health`

### 4. API Endpoints Tests
- **Health endpoints**: `/api/health`, `/face-pipeline/health`
- **Ready endpoints**: `/api/ready`, `/face-pipeline/ready`
- **Search endpoints**: `/api/v1/search` (should return 405 for GET)

### 5. Performance Tests
- **Latency requirements**: Health endpoints should respond under 200ms
- **Concurrent requests**: System should handle multiple simultaneous requests
- **Load testing**: Basic load testing with multiple concurrent connections

### 6. Error Handling Tests
- **404 errors**: Non-existent endpoints return proper 404 responses
- **Method not allowed**: Invalid HTTP methods return 405 responses
- **Server errors**: Proper error handling for server issues

## Running the Tests

### Quick Smoke Test (Recommended)
```bash
make smoketest-quick
```

### Comprehensive Smoke Test
```bash
make smoketest
```

### Platform-Specific Tests
```bash
# Windows
make smoketest-win

# Linux/Mac
make smoketest-linux
```

## Test Scripts

### 1. `scripts/quick_smoke_test.ps1`
- **Purpose**: Quick verification of basic connectivity
- **Platform**: Windows PowerShell
- **Duration**: ~30 seconds
- **Tests**: Basic routing, CORS, port mapping, performance

### 2. `scripts/smoke_test.ps1`
- **Purpose**: Comprehensive testing of all proxy functionality
- **Platform**: Windows PowerShell
- **Duration**: ~2 minutes
- **Tests**: All categories with detailed error reporting

### 3. `scripts/smoke_test.sh`
- **Purpose**: Comprehensive testing for Linux/Mac systems
- **Platform**: Bash (Linux/Mac)
- **Duration**: ~2 minutes
- **Tests**: All categories with detailed error reporting

## Expected Results

### Successful Test Run
```
🧪 Quick Smoke Test - Mordeaux Face Scanning MVP
=================================================

🔍 Testing basic connectivity...
Testing: Nginx Main
✅ PASS: Nginx Main - Status: 200
Testing: Backend Health
✅ PASS: Backend Health - Status: 200
Testing: Pipeline Health
✅ PASS: Pipeline Health - Status: 200

🌐 Testing CORS headers...
✅ PASS: CORS headers present

🔌 Testing port mapping...
✅ PASS: Direct Backend - Status: 200
✅ PASS: Direct Frontend - Status: 200
✅ PASS: Direct Pipeline - Status: 200

⚡ Testing performance...
✅ PASS: Health endpoint latency 45ms (under 200ms)

📊 Test Summary
===============
Total tests: 7
Passed: 7
Failed: 0

🎉 All tests passed! The proxy is working correctly.
```

### Failed Test Run
```
❌ FAIL: Backend Health - Error: The remote server returned an error: (500) Internal Server Error.
❌ FAIL: CORS headers missing
❌ FAIL: Direct Backend - Error: Unable to connect to the remote server

📊 Test Summary
===============
Total tests: 7
Passed: 4
Failed: 3

❌ Some tests failed. Please check the configuration.
```

## Troubleshooting

### Common Issues

#### 1. Services Not Running
**Error**: `Docker services are not running`
**Solution**: Run `make start` to start all services

#### 2. Port Conflicts
**Error**: `Service is not accessible on port X`
**Solution**: Check if ports are already in use, stop conflicting services

#### 3. CORS Issues
**Error**: `CORS headers missing`
**Solution**: Verify Nginx configuration includes CORS headers

#### 4. Routing Issues
**Error**: `Expected status 200, got 404`
**Solution**: Check Nginx configuration and service names in docker-compose.yml

#### 5. Performance Issues
**Error**: `Latency exceeds 200ms requirement`
**Solution**: Check system resources, optimize Docker settings

### Debug Commands

```bash
# Check service status
make status

# View logs
make logs

# Check Docker containers
docker-compose ps

# Test individual endpoints
curl -v http://localhost/api/health
curl -v http://localhost/face-pipeline/health
```

## Configuration Requirements

### Nginx Configuration
- CORS headers must be present
- Proper proxy headers for backend services
- OPTIONS request handling for preflight requests

### Docker Services
- All services must be running and healthy
- Proper port mapping in docker-compose.yml
- Service names must match Nginx configuration

### Network Requirements
- Ports 80, 3000, 8000, 8001 must be available
- No firewall blocking local connections
- Docker network connectivity between services

## Performance Benchmarks

### Latency Requirements
- **Health endpoints**: < 200ms
- **API endpoints**: < 500ms
- **Frontend**: < 100ms

### Throughput Requirements
- **Concurrent requests**: 5+ simultaneous requests
- **Request rate**: 10+ requests per second
- **Error rate**: < 1%

## Continuous Integration

The smoke tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Smoke Tests
  run: make smoketest-quick
```

## Monitoring

For production monitoring, consider:
- Automated smoke test runs every 5 minutes
- Alerting on test failures
- Performance monitoring dashboards
- Log aggregation for debugging

## Security Considerations

- CORS headers are configured for development (*)
- In production, restrict CORS origins to specific domains
- Consider adding authentication headers
- Monitor for unusual request patterns

## Maintenance

- Update test scripts when adding new endpoints
- Review performance benchmarks quarterly
- Update CORS configuration as needed
- Monitor test execution times
