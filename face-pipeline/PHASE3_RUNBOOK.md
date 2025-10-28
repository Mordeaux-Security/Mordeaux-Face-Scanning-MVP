## Phase 3 Operational Runbook

**Version**: 1.0  
**Last Updated**: October 2025  
**Audience**: DevOps, SREs, Backend Engineers

---

## 📋 Table of Contents

1. [What This Is](#what-this-is)
2. [Environment & Configuration](#environment--configuration)
3. [Quick Start](#quick-start)
4. [Publishing Test Messages](#publishing-test-messages)
5. [Stats & Readiness](#stats--readiness)
6. [Global Deduplication](#global-deduplication)
7. [Minimal Test Plan](#minimal-test-plan)
8. [Troubleshooting](#troubleshooting)
9. [Operations Tips](#operations-tips)
10. [Upgrade Path](#upgrade-path)

---

## What This Is

### Scope

This runbook covers the **Phase 3 Face Pipeline** components:

- **Redis Streams Worker**: Async message queue consumer for face processing
- **Global Deduplication**: Redis-based pHash near-duplicate detection
- **Stats & Metrics**: Real-time counters and timing instrumentation
- **Readiness Checks**: Comprehensive dependency health monitoring

### Dependencies

| Service | Purpose | Required | Health Check |
|---------|---------|----------|--------------|
| **Redis** | Message queue, dedup cache, stats | ✅ Yes | `redis-cli PING` |
| **MinIO** | Object storage (crops, thumbs, metadata) | ✅ Yes | `GET /minio/health/live` |
| **Qdrant** | Vector database (embeddings) | ✅ Yes | `GET /health` |
| **PostgreSQL** | Metadata storage (optional) | ❌ No | N/A |

### Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Producer   │───▶│ Redis Stream │───▶│   Worker    │
│ (Crawler)   │    │  face:ingest │    │ (Consumer)  │
└─────────────┘    └──────────────┘    └─────────────┘
                                               │
                                               ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   MinIO     │◀───│    Pipeline  │───▶│   Qdrant    │
│ (Artifacts) │    │ (Processing) │    │  (Vectors)  │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │Redis Dedup   │
                  │& Stats Cache │
                  └──────────────┘
```

---

## Environment & Configuration

### Required Environment Variables

**Primary names** (used by code):

```bash
# Redis Connection & Streams
REDIS_URL=redis://localhost:6379/0
REDIS_STREAM_NAME=face-processing-queue
REDIS_GROUP_NAME=pipeline
REDIS_CONSUMER_NAME=pipe-1          # Optional: auto-generated if empty

# Deduplication Configuration
ENABLE_GLOBAL_DEDUP=true            # Enable/disable near-duplicate detection
DEDUP_TTL_SECONDS=3600              # Cache TTL (1 hour default)
DEDUP_MAX_HAMMING=8                 # Hamming distance threshold (0-64)

# Worker Configuration
MAX_WORKER_CONCURRENCY=5            # Max concurrent image processing
WORKER_BATCH_SIZE=10                # Messages per batch
ENABLE_QUEUE_WORKER=true            # Master switch for worker

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces_v1
```

### Accepted Aliases (Backward Compatibility)

The system supports legacy variable names via a compatibility shim in `config/settings.py`:

```bash
# Legacy Redis names (optional)
FACE_STREAM=face-processing-queue       # → REDIS_STREAM_NAME
FACE_GROUP=pipeline                     # → REDIS_GROUP_NAME
FACE_CONSUMER=pipe-1                    # → REDIS_CONSUMER_NAME

# Legacy dedup names (optional)
GLOBAL_DEDUP_ENABLED=true               # → ENABLE_GLOBAL_DEDUP
GLOBAL_DEDUP_TTL_SEC=3600               # → DEDUP_TTL_SECONDS
GLOBAL_DEDUP_MAX_HAMMING=8              # → DEDUP_MAX_HAMMING
```

**Note**: If both names are set, the primary name takes precedence.

### Configuration Examples

#### Development (Permissive)

```bash
ENABLE_GLOBAL_DEDUP=true
DEDUP_MAX_HAMMING=12         # More permissive (detects more near-dups)
DEDUP_TTL_SECONDS=1800       # 30 minutes
WORKER_BATCH_SIZE=5          # Smaller batches for debugging
```

#### Production (Strict)

```bash
ENABLE_GLOBAL_DEDUP=true
DEDUP_MAX_HAMMING=8          # Balanced threshold
DEDUP_TTL_SECONDS=3600       # 1 hour
WORKER_BATCH_SIZE=16         # Larger batches for throughput
MAX_WORKER_CONCURRENCY=10    # Higher concurrency
```

#### Testing (Disabled Dedup)

```bash
ENABLE_GLOBAL_DEDUP=false    # Process all faces, no dedup
WORKER_BATCH_SIZE=1          # Process one message at a time
```

---

## Quick Start

### Step 1: Install Dependencies

```bash
# From repo root
make phase3-install

# Or from face-pipeline directory
cd face-pipeline
pip install -r requirements.txt
```

### Step 2: Start Infrastructure

```bash
# Start Redis, MinIO, Qdrant via Docker Compose
docker compose up -d redis minio qdrant

# Verify services are running
docker compose ps
```

### Step 3: Initialize Storage

```bash
# Ensure MinIO buckets and Qdrant collection exist
make phase3-ensure

# Or manually:
cd face-pipeline
python -c "from pipeline.storage import ensure_buckets; from pipeline.indexer import ensure_collection; ensure_buckets(); ensure_collection()"
```

### Step 4: Start Worker

**Long-running mode** (production):

```bash
# From repo root
make worker

# Or from face-pipeline directory
python worker.py --max-batch 16
```

**Single-batch mode** (testing):

```bash
# Process one batch then exit
make worker-once

# Or manually:
python worker.py --once --max-batch 8
```

### Step 5: Verify Worker is Running

```bash
# Check logs for startup message
# Expected output:
# INFO - Starting Redis Streams worker...
# INFO - Consumer: pipe-{hostname}-{pid}
# INFO - Stream: face-processing-queue
# INFO - Group: pipeline
# INFO - Batch size: 16
# INFO - Worker started, waiting for messages...

# Check readiness endpoint
curl http://localhost:8001/ready | jq

# Expected: {"ready": true, "checks": {...}}
```

---

## Publishing Test Messages

### Using the Provided Script

```bash
# From face-pipeline directory
python scripts/publish_test_message.py

# Expected output:
# ✅ Published test message to stream 'face-processing-queue'
#    Message ID: 1234567890-0
#    Field: message
#    Payload: {...}
```

### Using Redis CLI (Manual)

```bash
# Single message
redis-cli XADD face-processing-queue '*' message '{
  "image_sha256": "abc123def456",
  "bucket": "raw-images",
  "key": "demo/test_image.jpg",
  "tenant_id": "demo",
  "site": "local",
  "url": "http://example.com/test.jpg",
  "image_phash": "8f373c9c3c9c3c1e",
  "face_hints": null
}'

# Batch publish (multiple messages)
for i in {1..10}; do
  redis-cli XADD face-processing-queue '*' message "{\"image_sha256\":\"test$i\",\"bucket\":\"raw-images\",\"key\":\"demo/test$i.jpg\",\"tenant_id\":\"demo\",\"site\":\"local\",\"url\":\"http://example.com/test$i.jpg\",\"image_phash\":\"000000000000000$i\",\"face_hints\":null}"
done
```

### Message Schema

**Required fields:**

```json
{
  "image_sha256": "string",      // SHA-256 hash of image (hex, 64 chars)
  "bucket": "string",            // MinIO bucket name
  "key": "string",               // Object key/path in bucket
  "tenant_id": "string",         // Tenant identifier for isolation
  "site": "string",              // Source site/domain
  "url": "string",               // Original image URL
  "image_phash": "string",       // Perceptual hash (16-char hex)
  "face_hints": null | array     // Optional: pre-detected face hints
}
```

**Example with face hints:**

```json
{
  "image_sha256": "abc123def456789...",
  "bucket": "raw-images",
  "key": "demo/celebrity.jpg",
  "tenant_id": "demo",
  "site": "example.com",
  "url": "https://example.com/images/celebrity.jpg",
  "image_phash": "8f373c9c3c9c3c1e",
  "face_hints": [
    {
      "bbox": [100, 150, 200, 250],
      "confidence": 0.98
    }
  ]
}
```

**Field Notes:**

- **"message" vs "data"**: Worker supports both field names for backward compatibility. Use `message` (preferred).
- **image_sha256**: Must match the actual SHA-256 of the image in MinIO
- **image_phash**: Used for deduplication; can be placeholder if not pre-computed
- **face_hints**: If provided, detector may skip or optimize detection

### Checking Queue Length

```bash
# Get stream info
redis-cli XINFO STREAM face-processing-queue

# Get pending messages
redis-cli XPENDING face-processing-queue pipeline

# Get consumer group info
redis-cli XINFO GROUPS face-processing-queue
```

---

## Stats & Readiness

### Readiness Endpoint

```bash
# Check system readiness
curl http://localhost:8001/ready | jq

# Example output:
{
  "ready": true,
  "checks": {
    "models": "healthy",
    "storage": "healthy",
    "vector_db": "healthy",
    "queue": "healthy",
    "dedup_cache": "healthy"
  }
}
```

**Check Meanings:**

| Check | Status | Meaning |
|-------|--------|---------|
| **models** | healthy | InsightFace models loaded successfully |
| **storage** | healthy | MinIO connection working, buckets exist |
| **vector_db** | healthy | Qdrant connection working, collection exists |
| **queue** | healthy | Redis Streams connection working |
| **dedup_cache** | healthy | Redis dedup operations working |

### Stats Endpoint

```bash
# Get real-time statistics
curl http://localhost:8001/stats | jq

# Example output:
{
  "counters": {
    "faces_processed": 1245,
    "faces_rejected": 89,
    "faces_dup_skipped": 234,
    "worker_msgs_acked": 1024,
    "worker_msgs_dlq": 12
  },
  "timings_ms": {
    "detection": {"count": 1245, "total": 12450.5, "avg": 10.0},
    "embedding": {"count": 1156, "total": 5780.0, "avg": 5.0},
    "storage": {"count": 1156, "total": 2312.0, "avg": 2.0},
    "upsert": {"count": 1156, "total": 3468.0, "avg": 3.0}
  }
}
```

**Key Counters:**

- **faces_processed**: Total faces successfully processed
- **faces_rejected**: Faces rejected by quality checks
- **faces_dup_skipped**: Faces skipped due to deduplication
- **worker_msgs_acked**: Messages successfully processed
- **worker_msgs_dlq**: Messages moved to dead letter queue

**Key Timings:**

- **detection**: Face detection time (per image)
- **embedding**: Embedding generation time (per face)
- **storage**: MinIO upload time (per artifact)
- **upsert**: Qdrant indexing time (batch)

### Monitoring Recommendations

```bash
# Watch stats in real-time
watch -n 2 'curl -s http://localhost:8001/stats | jq .counters'

# Check DLQ size
redis-cli XLEN face-processing-queue:dlq

# Check pending messages
redis-cli XPENDING face-processing-queue pipeline - + 10

# Monitor Redis memory usage
redis-cli INFO memory | grep used_memory_human
```

---

## Global Deduplication

### How It Works

The deduplication system operates in two stages:

**Stage 1: Exact Match**
- Uses full 16-char hex pHash
- Stored in Redis SET: `dedup:phash:{prefix}` → {full_hashes}
- Prefix = first 4 chars for sharding (e.g., "8f37")

**Stage 2: Near-Duplicate (Hamming Distance)**
- Compares pHash via XOR bitcount
- Tenant-scoped: `dedup:near:{tenant_id}:{prefix}` → {hashes}
- Threshold controlled by `DEDUP_MAX_HAMMING`

### Pipeline Integration

Deduplication runs at **Step 7** of the 12-step pipeline:

```
1. Download image from MinIO
2. Decode to numpy array
3. Detect faces
4. Align and crop
5. Quality assessment
6. Generate embeddings
7. → DEDUPLICATION CHECK ← (here)
8. Compute artifact paths
9. Store artifacts in MinIO
10. Index in Qdrant
11. Update metrics
12. Return results
```

If a duplicate is detected:
- Face is **skipped** (not stored, not indexed)
- Counter `faces_dup_skipped` is incremented
- Processing continues with next face

### Tuning DEDUP_MAX_HAMMING

The `DEDUP_MAX_HAMMING` parameter controls sensitivity:

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0** | Exact match only | Production (strict) |
| **3** | Very similar images | Conservative dedup |
| **8** | Similar images | **Default** (balanced) |
| **12** | Loosely similar | Aggressive dedup |
| **16+** | Very loose matching | Testing/development |

**Examples:**

```bash
# Strict: Only exact matches
DEDUP_MAX_HAMMING=0

# Balanced: Cropped/resized images (recommended)
DEDUP_MAX_HAMMING=8

# Aggressive: Different lighting/angles
DEDUP_MAX_HAMMING=12
```

**Testing Threshold:**

```bash
# Process folder with DEDUP_MAX_HAMMING=3
DEDUP_MAX_HAMMING=3 python scripts/process_folder.py --path samples/

# Check dup_skipped count
curl http://localhost:8001/stats | jq .counters.faces_dup_skipped

# Reprocess with DEDUP_MAX_HAMMING=12
make dedup-flush  # Clear cache first
DEDUP_MAX_HAMMING=12 python scripts/process_folder.py --path samples/

# Should see more duplicates skipped
curl http://localhost:8001/stats | jq .counters.faces_dup_skipped
```

### Disabling Deduplication

```bash
# Disable globally
ENABLE_GLOBAL_DEDUP=false python worker.py

# Or in .env
ENABLE_GLOBAL_DEDUP=false
```

When disabled:
- All faces are processed (no skipping)
- `faces_dup_skipped` counter stays at 0
- No Redis dedup operations

### Expected Behavior

**First Run (Empty Cache):**
- All unique faces processed
- `faces_processed` = N
- `faces_dup_skipped` = 0

**Second Run (Same Images):**
- Most faces skipped
- `faces_processed` = small delta
- `faces_dup_skipped` = ~N

**After TTL Expiry:**
- Cache entries expire after `DEDUP_TTL_SECONDS`
- Faces processed again as "new"
- Useful for reprocessing old images

### Redis Key Patterns

```bash
# Exact match dedup keys
redis-cli KEYS "dedup:phash:*"

# Near-duplicate dedup keys (tenant-scoped)
redis-cli KEYS "dedup:near:*"

# Check specific prefix
redis-cli SMEMBERS "dedup:phash:8f37"
redis-cli SMEMBERS "dedup:near:demo:8f37"

# Clear all dedup cache
redis-cli KEYS "dedup:*" | xargs redis-cli DEL
# Or use Makefile:
make dedup-flush
```

---

## Minimal Test Plan

### Unit Tests

```bash
# Run Phase 3 dedup tests
cd face-pipeline
pytest tests/test_phase3_dedup.py -v

# Expected output:
# test_hamming_distance_table[...] PASSED
# test_should_skip_disabled_always_false PASSED
# test_remember_then_should_skip_exact_match PASSED
# ...
```

**Key test categories:**
- Hamming distance XOR bitcount logic (table-driven)
- Feature flag behavior (`ENABLE_GLOBAL_DEDUP`)
- Configurable `DEDUP_MAX_HAMMING` parameter
- Integration scenarios (first run, duplicates, tenant isolation)

### Worker Smoke Test

```bash
# 1. Start worker in single-batch mode
python worker.py --once --max-batch 1

# 2. Publish test message (in another terminal)
python scripts/publish_test_message.py

# 3. Check worker logs
# Expected: "Successfully processed message..."

# 4. Verify stats incremented
curl http://localhost:8001/stats | jq .counters.worker_msgs_acked
# Expected: 1 (or incremented by 1)
```

### Deduplication Behavior Test

```bash
# 1. Clear dedup cache
make dedup-flush

# 2. Process sample folder (first run)
python scripts/process_folder.py --path samples/

# 3. Check stats
curl http://localhost:8001/stats | jq .counters
# Note: faces_processed count, faces_dup_skipped = 0

# 4. Process same folder again (second run)
python scripts/process_folder.py --path samples/

# 5. Check stats again
curl http://localhost:8001/stats | jq .counters
# Expected: faces_dup_skipped > 0 (most faces skipped)
```

### Failure Path Test

```bash
# 1. Publish malformed JSON
redis-cli XADD face-processing-queue '*' message '{"invalid":"json'

# 2. Start worker
python worker.py --once

# 3. Check DLQ
redis-cli XRANGE face-processing-queue:dlq - +
# Expected: Entry with parse error

# 4. Check DLQ counter
curl http://localhost:8001/stats | jq .counters.worker_msgs_dlq
# Expected: Incremented
```

### End-to-End Test

```bash
# 1. Start all services
docker compose up -d

# 2. Initialize
make phase3-ensure

# 3. Start worker
make worker &

# 4. Publish test message
python scripts/publish_test_message.py

# 5. Wait a few seconds, then check stats
sleep 5
curl http://localhost:8001/stats | jq

# 6. Verify face was indexed in Qdrant
curl http://localhost:6333/collections/faces_v1 | jq .result.vectors_count
# Expected: > 0
```

---

## Troubleshooting

### No Messages Consumed

**Symptoms:**
- Worker starts but doesn't process messages
- `worker_msgs_acked` counter stays at 0

**Checks:**

```bash
# 1. Verify stream name
echo $REDIS_STREAM_NAME
# Should match what producer uses

# 2. Check if stream exists
redis-cli EXISTS face-processing-queue
# Should return 1

# 3. Check stream length
redis-cli XLEN face-processing-queue
# Should be > 0 if messages pending

# 4. Check consumer group
redis-cli XINFO GROUPS face-processing-queue
# Should list 'pipeline' group

# 5. Check pending messages
redis-cli XPENDING face-processing-queue pipeline
# Shows pending count
```

**Fixes:**

```bash
# Recreate consumer group
redis-cli XGROUP DESTROY face-processing-queue pipeline
python worker.py --once  # Will recreate group

# Or manually:
redis-cli XGROUP CREATE face-processing-queue pipeline 0 MKSTREAM
```

### DLQ Growing

**Symptoms:**
- `worker_msgs_dlq` counter increasing
- Messages in `face-processing-queue:dlq`

**Inspect DLQ:**

```bash
# Get recent DLQ entries
redis-cli XRANGE face-processing-queue:dlq - + COUNT 10

# Parse entry
redis-cli XRANGE face-processing-queue:dlq - + COUNT 1 | jq
```

**Common Issues:**

1. **Missing required fields**
   - Check message has all required fields
   - See [Message Schema](#message-schema)

2. **Image not found in MinIO**
   - Verify `bucket` and `key` are correct
   - Check image exists: `mc ls minio/{bucket}/{key}`

3. **Invalid image format**
   - Ensure image is valid JPEG/PNG/WebP
   - Test: `file {image_path}`

4. **MinIO connection error**
   - Check `MINIO_ENDPOINT` configuration
   - Verify MinIO is running: `curl http://localhost:9000/minio/health/live`

**Clear DLQ:**

```bash
# Delete DLQ stream (careful!)
redis-cli DEL face-processing-queue:dlq

# Or trim old entries (keep last 100)
redis-cli XTRIM face-processing-queue:dlq MAXLEN 100
```

### Readiness Check Failed

**Symptoms:**
- `GET /ready` returns `{"ready": false}`
- Specific check shows "unhealthy"

**Decode Checks:**

```bash
# Get full readiness report
curl http://localhost:8001/ready | jq

# Example output with failure:
{
  "ready": false,
  "checks": {
    "models": "healthy",
    "storage": "unhealthy",  ← Issue here
    "vector_db": "healthy",
    "queue": "healthy",
    "dedup_cache": "healthy"
  }
}
```

**Fixes by Check:**

| Check | Quick Fix |
|-------|-----------|
| **models** | Restart service (models will reload) |
| **storage** | Check MinIO running: `docker compose up -d minio` |
| **vector_db** | Check Qdrant running: `docker compose up -d qdrant` |
| **queue** | Check Redis running: `docker compose up -d redis` |
| **dedup_cache** | Check Redis connection (same as queue) |

### Qdrant Search Slow

**Symptoms:**
- Search requests taking > 1 second
- High CPU on Qdrant container

**Checks:**

```bash
# Check collection info
curl http://localhost:6333/collections/faces_v1 | jq

# Verify payload indexes exist
curl http://localhost:6333/collections/faces_v1 | jq .result.payload_schema

# Expected indexes:
# - tenant_id (keyword)
# - site (keyword)
# - phash_prefix (keyword)
```

**Fixes:**

```bash
# Recreate collection with indexes
python -c "from pipeline.indexer import ensure_collection; ensure_collection()"

# Or manually via Qdrant API:
curl -X POST http://localhost:6333/collections/faces_v1/index \
  -H 'Content-Type: application/json' \
  -d '{"field_name": "tenant_id", "field_schema": "keyword"}'
```

### High Memory Usage (Redis)

**Symptoms:**
- Redis using excessive memory
- Dedup cache very large

**Checks:**

```bash
# Check Redis memory
redis-cli INFO memory | grep used_memory_human

# Count dedup keys
redis-cli KEYS "dedup:*" | wc -l

# Get cache stats
curl http://localhost:8001/stats | jq .dedup_stats
```

**Fixes:**

```bash
# Reduce TTL (cache expires faster)
DEDUP_TTL_SECONDS=1800  # 30 minutes

# Clear old cache entries
make dedup-flush

# Or selectively clear by prefix
redis-cli KEYS "dedup:phash:8f*" | xargs redis-cli DEL
```

---

## Operations Tips

### Rolling Restarts

**Consumer Naming:**

Each worker instance gets a unique consumer name:
```
pipe-{hostname}-{pid}
```

This allows multiple workers to run in parallel without conflicts.

**Graceful Shutdown:**

```bash
# Send SIGTERM for graceful shutdown
kill -TERM {worker_pid}

# Worker will:
# 1. Stop accepting new messages
# 2. Finish processing current batch
# 3. Acknowledge processed messages
# 4. Exit cleanly
```

**Zero-Downtime Restart:**

```bash
# 1. Start new worker
make worker &
NEW_PID=$!

# 2. Wait for new worker to be ready
sleep 5

# 3. Stop old worker gracefully
kill -TERM {old_pid}

# 4. Verify new worker is consuming
ps aux | grep worker.py
```

### Safe Environment Changes

Most env changes require restart:

```bash
# 1. Update .env file
vim face-pipeline/.env

# 2. Stop worker
kill -TERM {worker_pid}

# 3. Start worker with new config
make worker
```

**Hot-Reloadable (no restart needed):**
- None currently (all flags cached at startup)

**Restart Required:**
- All `DEDUP_*` settings
- All `REDIS_*` settings
- All `WORKER_*` settings

### Useful Log Lines

**Successful Processing:**

```
Successfully processed message {id}: faces_total={n}, accepted={m}, rejected={r}, dup_skipped={d}
```

**Duplicate Detected:**

```
Face {i} is near-duplicate (pHash: {hash}), skipping
```

**DLQ Entry:**

```
Message {id} missing required fields: {fields}
```

**Worker Startup:**

```
Starting Redis Streams worker...
Consumer: pipe-{hostname}-{pid}
Stream: face-processing-queue
Worker started, waiting for messages...
```

**Grep Patterns:**

```bash
# Find processing errors
grep "Pipeline error" /var/log/face-pipeline/worker.log

# Find duplicate skips
grep "near-duplicate" /var/log/face-pipeline/worker.log

# Find DLQ entries
grep "missing required fields" /var/log/face-pipeline/worker.log

# Count successful messages
grep "Successfully processed" /var/log/face-pipeline/worker.log | wc -l
```

### Monitoring Recommendations

**Metrics to Track:**

1. **Throughput**: `worker_msgs_acked` per minute
2. **Error Rate**: `worker_msgs_dlq` per minute
3. **Dedup Rate**: `faces_dup_skipped / faces_processed` ratio
4. **Processing Time**: `timings_ms.avg` for each stage
5. **Queue Depth**: Redis stream length

**Alert Thresholds:**

```yaml
# Example Prometheus alerts
- alert: HighDLQRate
  expr: rate(worker_msgs_dlq[5m]) > 0.1
  annotations:
    summary: "DLQ rate exceeds 10% of processed messages"

- alert: QueueBacklog
  expr: redis_stream_length{stream="face-processing-queue"} > 1000
  annotations:
    summary: "Message queue has >1000 pending messages"

- alert: SlowProcessing
  expr: avg(face_processing_time_ms) > 1000
  annotations:
    summary: "Average processing time exceeds 1 second"
```

**Dashboards:**

Key graphs to build:
- Worker throughput (msgs/sec)
- Processing latency (p50, p95, p99)
- Dedup hit rate (%)
- Queue depth over time
- DLQ growth rate

---

## Upgrade Path

### Future Enhancements

**1. RabbitMQ Migration**

Redis Streams is fine for MVP, but RabbitMQ offers:
- Better durability (disk persistence)
- More flexible routing
- Built-in retries and DLQ

**Migration Path:**
```python
# 1. Add RabbitMQ support alongside Redis
# 2. Dual-publish messages to both queues
# 3. Monitor for parity
# 4. Switch workers to RabbitMQ
# 5. Deprecate Redis Streams
```

**2. PostgreSQL Stats Snapshots**

Current: All stats in Redis (volatile)  
Future: Periodic snapshots to PostgreSQL

**Implementation:**
```python
# Cron job every hour
def snapshot_stats():
    stats = get_current_stats()
    db.insert("stats_snapshots", {
        "timestamp": now(),
        "stats": json.dumps(stats)
    })
```

**3. Horizontal Scaling**

Current: Single worker instance  
Future: Multiple workers with load balancing

**Approach:**
- Deploy multiple worker pods/containers
- Each gets unique `REDIS_CONSUMER_NAME`
- Redis Streams automatically load-balances
- Monitor with centralized stats aggregation

**4. Batch Processing Optimization**

Current: Process one image at a time  
Future: Batch embedding generation

**Benefits:**
- GPU utilization improved
- Throughput increased 3-5x
- Lower per-face cost

**Implementation:**
```python
# Collect batch of face crops
crops = [face1, face2, ..., faceN]

# Batch embed (single GPU call)
embeddings = embed_batch(crops)  # Much faster

# Individual upserts
for emb in embeddings:
    upsert(emb)
```

### Deprecation Timeline

| Component | Status | Deprecated By | Replacement |
|-----------|--------|---------------|-------------|
| Redis Streams | ✅ Active | Q2 2026 | RabbitMQ |
| Redis Stats | ✅ Active | Q3 2026 | PostgreSQL |
| Sync Processing | ✅ Active | Q4 2026 | Batch GPU |

---

## See Also

- **[PHASE3_IMPLEMENTATION.md](PHASE3_IMPLEMENTATION.md)** - Technical design and architecture
- **[README.md](README.md)** - Developer documentation and API contracts
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide

---

**Questions? Issues?**  
Contact: DevOps Team / SRE On-Call

