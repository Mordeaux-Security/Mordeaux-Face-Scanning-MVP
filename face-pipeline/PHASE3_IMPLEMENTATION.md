# Phase 3 Implementation Guide - ✅ COMPLETE

**📚 See Also:**
- **[PHASE3_RUNBOOK.md](PHASE3_RUNBOOK.md)** - Operations guide for running, testing, and troubleshooting the queue worker
- **[README.md](README.md)** - Developer documentation and API contracts

---

## Overview

This document describes the Phase 3 production-ready features **FULLY IMPLEMENTED** for the face-pipeline service:

- ✅ **Global Deduplication**: Redis-based pHash deduplication (exact + near-duplicate) - OPERATIONAL
- ✅ **Queue Worker**: Redis Streams-based async message processing - OPERATIONAL  
- ✅ **Statistics Tracking**: Real-time Redis counters and timing metrics - OPERATIONAL
- ✅ **Enhanced Readiness**: Comprehensive dependency health checks - OPERATIONAL
- ✅ **Feature Flags**: Environment-based configuration for safe rollout - OPERATIONAL
- ✅ **Retention Scripts**: Production maintenance and cleanup tools - OPERATIONAL
- ✅ **Calibration Scripts**: Threshold optimization and testing tools - OPERATIONAL

## Architecture

### System Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Crawler/API   │───▶│  Redis Streams  │───▶│  Queue Worker   │
│   (Producer)    │    │  (Message Queue) │    │  (Consumer)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MinIO Storage │◀───│  Face Pipeline   │───▶│  Qdrant Vector  │
│   (Artifacts)   │    │  (Processing)    │    │  (Embeddings)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Redis Counters  │
                       │  (Statistics)    │
                       └──────────────────┘
```

### Component Overview

#### 1. Global Deduplication (`pipeline/dedup.py`)

**Purpose**: Prevent duplicate face processing across the entire system using perceptual hashes.

**Architecture**:
- Redis SET storage: `dedup:phash:{prefix}` → set of full pHashes
- Prefix-based sharding (4-char prefixes) for performance
- TTL-based cache management (24h default)
- Thread-safe atomic operations

**Key Functions**:
- `is_duplicate(phash: str) -> bool`: Check if pHash exists globally
- `mark_processed(phash: str)`: Add pHash to global set
- `clear_dedup_cache()`: Admin utility for testing

**Configuration**:
```bash
ENABLE_GLOBAL_DEDUP=true
DEDUP_TTL_SECONDS=86400  # 24 hours
```

#### 2. Statistics Tracking (`pipeline/stats.py`)

**Purpose**: Real-time monitoring of face processing pipeline performance.

**Architecture**:
- Global counters: `stats:global:processed`, `stats:global:rejected`, `stats:global:dup_skipped`
- Tenant counters: `stats:tenant:{tenant_id}:processed`, etc.
- Atomic Redis operations for thread-safety
- Batch processing support

**Key Functions**:
- `increment_processed(count, tenant_id)`: Update processed counter
- `increment_rejected(count, tenant_id)`: Update rejected counter
- `increment_dup_skipped(count, tenant_id)`: Update dup_skipped counter
- `get_stats(tenant_id=None)`: Get current statistics

**API Integration**:
- `GET /api/v1/stats` returns real-time data
- Supports tenant-specific filtering
- Maintains API v0.1 contract

#### 3. Redis Streams Worker (`worker.py`)

**Purpose**: Async processing of face pipeline messages via Redis Streams with single-batch testing support.

**Architecture**:
- Consumer group: Configurable via `REDIS_GROUP_NAME` (default: "pipeline")
- Consumer name: Auto-generated `pipe-{hostname}-{pid}` or configurable via `REDIS_CONSUMER_NAME`
- Batch processing with configurable batch size (default: 8)
- Dead letter queue for failed messages (`{stream}:dlq`)
- Graceful shutdown with signal handling
- **NEW**: `--once` flag for single-batch processing (testing/CI)

**Key Features**:
- Configurable concurrency (default: 5 workers)
- Batch message processing (default: 8 messages, configurable)
- Automatic DLQ for failed messages
- Health monitoring and logging
- **NEW**: Backward-compatible message parsing (supports both "message" and "data" fields)
- **NEW**: Single-batch mode for testing and CI

**Configuration**:
```bash
ENABLE_QUEUE_WORKER=true
REDIS_URL=redis://redis:6379/0
REDIS_STREAM_NAME=face-processing-queue
REDIS_GROUP_NAME=pipeline
REDIS_CONSUMER_NAME=pipe-1
MAX_WORKER_CONCURRENCY=5
WORKER_BATCH_SIZE=10
```

**Usage Examples**:
```bash
# Long-running worker (production)
python worker.py

# Single-batch testing
python worker.py --once

# Custom batch size
python worker.py --max-batch 16

# Single batch with custom size (CI/testing)
python worker.py --once --max-batch 8
```

**Message Format Support**:
- **Preferred**: `{"message": json.dumps(payload)}`
- **Legacy**: `{"data": json.dumps(payload)}` (backward compatible)

#### 4. Enhanced Readiness (`main.py`)

**Purpose**: Comprehensive health checks for all system dependencies.

**Checks**:
- **Models**: InsightFace detector and embedder loading
- **Storage**: MinIO connectivity and bucket listing
- **Vector DB**: Qdrant connectivity and collection access
- **Redis**: Connection and basic operations

**Response Format**:
```json
{
  "ready": true,
  "reason": "ok",
  "checks": {
    "models": true,
    "storage": true,
    "vector_db": true,
    "redis": true
  }
}
```

## Configuration

### Environment Variables

#### Feature Flags
```bash
# Enable/disable features
ENABLE_GLOBAL_DEDUP=true
ENABLE_QUEUE_WORKER=true
```

#### Redis Configuration
```bash
# Redis connection
REDIS_URL=redis://redis:6379/0
REDIS_STREAM_NAME=face-processing-queue
REDIS_GROUP_NAME=pipeline
REDIS_CONSUMER_NAME=pipe-1
# Legacy aliases (backward compatible):
# FACE_STREAM=face-processing-queue
# FACE_GROUP=pipeline
# FACE_CONSUMER=pipe-1
```

#### Worker Configuration
```bash
# Worker settings
MAX_WORKER_CONCURRENCY=5
WORKER_BATCH_SIZE=10
```

#### Dedup Configuration
```bash
# Deduplication settings
DEDUP_TTL_SECONDS=86400  # 24 hours
```

### Docker Compose Integration

The `docker-compose.yml` includes:

```yaml
face-pipeline:
  environment:
    ENABLE_GLOBAL_DEDUP: "true"
    ENABLE_QUEUE_WORKER: "false"  # API service

face-pipeline-worker:
  environment:
    ENABLE_QUEUE_WORKER: "true"
    ENABLE_GLOBAL_DEDUP: "true"
    MAX_WORKER_CONCURRENCY: 5
  command: ["python", "worker.py"]
  restart: unless-stopped
```

## Usage

### 1. Sending Messages to Queue

For backend/crawler integration:

```python
import redis
import json

# Connect to Redis
r = redis.from_url("redis://redis:6379/0")

# Create message
message = {
    "image_sha256": "abc123def456789...",
    "bucket": "raw-images",
    "key": "tenant1/2024/10/image.jpg",
    "tenant_id": "tenant1",
    "site": "example.com",
    "url": "https://example.com/image.jpg",
    "image_phash": "8f373c9c3c9c3c1e",
    "face_hints": None
}

# Send to stream
r.xadd("face-processing-queue", {"data": json.dumps(message)})
```

### 2. Monitoring Statistics

```bash
# Get global statistics
curl -H "X-Tenant-ID: tenant1" http://localhost:8001/api/v1/stats

# Response
{
  "processed": 1250,
  "rejected": 45,
  "dup_skipped": 12
}
```

### 3. Health Monitoring

```bash
# Check readiness
curl http://localhost:8001/ready

# Response (when healthy)
{
  "ready": true,
  "reason": "ok",
  "checks": {
    "models": true,
    "storage": true,
    "vector_db": true,
    "redis": true
  }
}
```

### 4. Admin Operations

```python
# Clear dedup cache (for testing)
from pipeline.dedup import clear_dedup_cache
deleted_keys = clear_dedup_cache()

# Reset statistics (for testing)
from pipeline.stats import reset_stats
reset_stats()  # Global
reset_stats("tenant1")  # Tenant-specific

# Get dedup statistics
from pipeline.dedup import get_dedup_stats
stats = get_dedup_stats()
```

## Testing

### Running Tests

```bash
cd face-pipeline
python -m pytest tests/ -v
```

### Test Coverage

- **test_dedup.py**: Deduplication service tests
- **test_stats.py**: Statistics tracking tests  
- **test_worker.py**: Queue worker tests
- **test_readiness.py**: Health check tests

### Manual Testing

```bash
# Start services
docker-compose up -d redis qdrant minio face-pipeline

# Test API endpoints
curl http://localhost:8001/ready
curl http://localhost:8001/api/v1/stats

# Start worker
docker-compose up face-pipeline-worker

# Send test message
python -c "
import redis, json
r = redis.from_url('redis://localhost:6379/0')
r.xadd('face-processing-queue', {'data': json.dumps({
    'image_sha256': 'test123',
    'bucket': 'raw-images', 
    'key': 'test/image.jpg',
    'tenant_id': 'test',
    'site': 'example.com',
    'url': 'https://example.com/test.jpg',
    'image_phash': '8f373c9c3c9c3c1e',
    'face_hints': None
})})
"
```

## Rollout Strategy

### Phase 1: Deploy with Flags Disabled
```bash
ENABLE_GLOBAL_DEDUP=false
ENABLE_QUEUE_WORKER=false
```
- Validate existing API endpoints work
- No breaking changes to current functionality

### Phase 2: Enable Deduplication
```bash
ENABLE_GLOBAL_DEDUP=true
```
- Test dedup functionality
- Monitor Redis memory usage
- Verify stats accuracy

### Phase 3: Enable Queue Worker
```bash
ENABLE_QUEUE_WORKER=true
```
- Start worker service
- Send test messages
- Monitor processing throughput

### Phase 4: Scale and Monitor
- Scale worker replicas as needed
- Monitor Redis memory and Qdrant indexing
- Set up alerting for health checks

## Troubleshooting

### Common Issues

#### 1. Redis Connection Errors
```bash
# Check Redis connectivity
redis-cli -h redis -p 6379 ping

# Check Redis memory usage
redis-cli -h redis -p 6379 info memory
```

#### 2. Worker Not Processing Messages
```bash
# Check worker logs
docker-compose logs face-pipeline-worker

# Check Redis streams
redis-cli -h redis -p 6379 xinfo stream face-processing-queue
```

#### 3. Readiness Check Failures
```bash
# Check individual dependencies
curl http://localhost:8001/ready

# Check MinIO
curl http://localhost:9000/minio/health/live

# Check Qdrant
curl http://localhost:6333/collections
```

#### 4. Statistics Not Updating
```bash
# Check Redis counters
redis-cli -h redis -p 6379 get stats:global:processed

# Check worker processing
docker-compose logs face-pipeline-worker | grep "Successfully processed"
```

### Performance Monitoring

#### Redis Memory Usage
```bash
# Monitor dedup cache size
redis-cli -h redis -p 6379 memory usage dedup:phash:8f37

# Monitor stats counters
redis-cli -h redis -p 6379 get stats:global:processed
```

#### Worker Performance
```bash
# Monitor worker throughput
docker-compose logs face-pipeline-worker | grep "processed message"

# Check queue length
redis-cli -h redis -p 6379 xlen face-processing-queue
```

## Security Considerations

### Redis Security
- Use Redis AUTH if needed
- Consider network isolation
- Monitor for unusual activity

### Message Validation
- All messages validated before processing
- Malformed messages moved to DLQ
- No raw data exposure in logs

### Statistics Privacy
- Tenant-scoped statistics
- No sensitive data in counters
- Admin operations logged

## Latest Features Implemented

### Queue Worker Enhancements (Latest)

**Single-Batch Testing Mode**:
- Added `--once` flag for processing one batch then exit
- Perfect for testing and CI environments
- Configurable batch size via `--max-batch` parameter

**Backward-Compatible Message Parsing**:
- Supports both "message" (preferred) and "data" (legacy) field formats
- Prefers "message" when both fields exist
- Maintains compatibility with existing message producers

**Configurable Group/Consumer Names**:
- `REDIS_GROUP_NAME` (default: "pipeline", alias: FACE_GROUP)
- `REDIS_CONSUMER_NAME` (auto-generated or configurable, alias: FACE_CONSUMER)
- Auto-generated consumer names prevent conflicts in distributed deployments

**Enhanced Error Handling**:
- Improved DLQ handling for unparsable messages
- Better logging and error reporting
- Graceful handling of malformed message fields

### Qdrant Payload Indexes (Latest)

**Performance Optimization**:
- Added payload indexes for `tenant_id`, `p_hash_prefix`, and `site` fields
- Uses KEYWORD schema for optimal filter performance
- Idempotent index creation with graceful error handling
- Significantly improves query performance for filtered searches

### Processor Metrics Integration (Latest)

**Comprehensive Metrics Tracking**:
- Added lightweight timing metrics using Redis hashes
- Tracks: `download_ms`, `decode_ms`, `detect_ms`, `align_ms`, `embed_ms`
- Counters: `images_total`, `faces_detected`, `faces_accepted`, `faces_rejected`, `faces_dup_skipped`
- Context manager `timer()` for automatic timing
- `snapshot()` function for combined metrics retrieval

**API Integration**:
- Enhanced `/api/v1/stats` endpoint with timing metrics
- Backward-compatible response format
- Real-time performance monitoring

## Future Enhancements

### Planned Improvements
1. **PostgreSQL Backfill**: Periodic sync of Redis stats to PostgreSQL for historical analysis
2. **Dynamic Feature Flags**: Redis-based configuration for runtime toggles
3. **RabbitMQ Migration**: Optional migration from Redis Streams to RabbitMQ
4. **Metrics Export**: Prometheus metrics for monitoring
5. **Batch Processing**: Optimized batch operations for high throughput

### Scalability Considerations
- Redis clustering for high availability
- Worker auto-scaling based on queue length
- Qdrant sharding for large datasets
- MinIO distributed mode for storage

## API Contract Compliance

All Phase 3 changes maintain API v0.1 contract:

- ✅ `/api/v1/stats` returns real data (no schema change)
- ✅ `/ready` enhanced but backward compatible
- ✅ No changes to search or faces endpoints
- ✅ All existing functionality preserved
- ✅ Feature flags allow safe rollout

## Support

For issues or questions:

1. Check logs: `docker-compose logs face-pipeline-worker`
2. Verify configuration: Environment variables and feature flags
3. Test connectivity: All dependency health checks
4. Monitor resources: Redis memory, worker CPU usage
5. Review documentation: This guide and API documentation

---

## ✅ PHASE 3 COMPLETION STATUS

### **IMPLEMENTATION STATUS: 100% COMPLETE**

All Phase 3 features have been **FULLY IMPLEMENTED** and are **PRODUCTION READY**:

#### ✅ Core Features - OPERATIONAL
- **Global Deduplication**: Redis-based pHash system with exact and near-duplicate detection
- **Queue Worker**: Redis Streams-based async processing with error handling
- **Statistics Tracking**: Real-time Redis counters and comprehensive timing metrics
- **Enhanced Health Checks**: Full dependency monitoring for all services
- **Feature Flags**: Environment-based configuration for safe deployment

#### ✅ Production Tools - OPERATIONAL  
- **Retention Scripts**: `retention_dry_run.py` for MinIO cleanup analysis
- **Calibration Scripts**: `calibrate_threshold.py` for similarity threshold optimization
- **Comprehensive Metrics**: Real-time timing and counter instrumentation
- **Error Handling**: Graceful degradation and retry logic

#### ✅ Testing & Validation - COMPLETE
- **Integration Tests**: End-to-end testing with all dependencies
- **Script Validation**: Retention and calibration scripts tested with real data
- **Performance Testing**: Load testing and optimization completed
- **Documentation**: Complete API and deployment documentation

### **DEPLOYMENT STATUS: READY FOR PRODUCTION**

- ✅ **All Dependencies**: Installed and fully functional
- ✅ **Configuration**: Environment-based with feature flags
- ✅ **Monitoring**: Comprehensive health checks and metrics
- ✅ **Documentation**: Complete implementation and usage guides
- ✅ **Testing**: Validated with real data and production scenarios

---

**Last Updated**: Phase 3 Implementation Complete  
**Version**: 0.1.0  
**Status**: ✅ PRODUCTION READY - FULLY IMPLEMENTED
