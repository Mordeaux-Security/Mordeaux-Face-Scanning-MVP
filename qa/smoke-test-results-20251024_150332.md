# Smoke Test Results - Mordeaux Face Scanning MVP
**Test Execution Time**: 2025-10-24 15:03:32
**Test Duration**: ~2 minutes
**Total Tests**: 8
**Passed**: 6
**Failed**: 2
**Success Rate**: 75%

## Container Status
- **postgres**: Up (healthy)
- **redis**: Up
- **qdrant**: Up
- **minio**: Up (healthy)
- **nginx**: Up
- **frontend**: Up
- **worker-cpu**: Up
- **backend-cpu**: Not running (syntax errors)
- **face-pipeline**: Not running (configuration issues)

## Test Results

### âœ… PASSED Tests
- **Nginx Main**: Status 200 - Latency: 135ms
- **Frontend Direct**: Status 200 - Latency: 48ms
- **CORS Headers**: Present and configured correctly
- **Port Mapping**: Frontend accessible on port 3000
- **Nginx Proxy**: Routing working correctly
- **Container Health**: Core services (postgres, redis, qdrant, minio) healthy

### âŒ FAILED Tests
- **Backend Health**: Service not running due to syntax errors in routes.py and storage.py
- **Face Pipeline Health**: Service not running due to missing configuration attributes

## Latency Metrics
- **Nginx Main**: P50=135ms, P95=135ms, Avg=135ms
- **Frontend Direct**: P50=48ms, P95=48ms, Avg=48ms

## Issues Identified
1. **Backend Service**: Syntax errors in Python files preventing startup
2. **Face Pipeline Service**: Missing configuration attributes in settings.py
3. **Service Dependencies**: Backend and face-pipeline services not accessible

## Recommendations
1. Fix syntax errors in backend/app/api/routes.py and backend/app/services/storage.py
2. Complete face-pipeline configuration in face-pipeline/config/settings.py
3. Restart backend-cpu and face-pipeline services after fixes
4. Re-run integration tests to validate search endpoints

## Next Steps
- Complete service fixes and restart containers
- Test search endpoints with image and vector input
- Validate end-to-end functionality
