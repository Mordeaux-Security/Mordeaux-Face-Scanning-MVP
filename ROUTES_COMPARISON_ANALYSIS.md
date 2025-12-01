# Routes.py Comparison: Main vs Branch

**Analysis Date:** 2025-01-27  
**Branch:** `gpu-worker-no-3-3-1-1`  
**Comparison:** `backend/app/api/routes.py`

---

## Executive Summary

The two branches have **completely different API architectures**:

- **This Branch:** Full-featured REST API (886 lines) with direct service integration
- **Main Branch:** Proxy-based API (296 lines) that forwards to face-pipeline service

**Critical Finding:** The new crawler **does NOT depend on API routes** - it uses services directly (storage_manager, redis_manager, GPU worker). However, merging will require choosing one API architecture or supporting both.

---

## Route Comparison

### This Branch Routes (Full-Featured API)

| Method | Path | Handler | Services Used | Purpose |
|--------|------|---------|---------------|---------|
| `POST` | `/index_face` | `index_face()` | `face`, `storage`, `vector`, `audit`, `webhook` | Index face embeddings |
| `POST` | `/search_face` | `search_face()` | `face`, `storage`, `vector`, `audit`, `webhook` | Search similar faces |
| `POST` | `/compare_face` | `compare_face()` | `face`, `cache`, `vector`, `audit`, `webhook` | Compare without storing |
| `GET` | `/images/{bucket}/{key:path}` | `serve_image()` | `storage` | Serve images from storage |
| `POST` | `/admin/cleanup` | `run_cleanup()` | `cleanup`, `audit` | Manual cleanup jobs |
| `POST` | `/batch/index` | `create_batch_index_job()` | `batch`, `audit` | Create batch indexing job |
| `GET` | `/batch/{batch_id}/status` | `get_batch_status()` | `batch` | Get batch job status |
| `GET` | `/batch/list` | `list_batch_jobs()` | `batch` | List batch jobs |
| `DELETE` | `/batch/{batch_id}` | `cancel_batch_job()` | `batch` | Cancel batch job |
| `POST` | `/batch/cleanup` | `cleanup_old_batches()` | `batch` | Cleanup old batches |
| `POST` | `/webhooks/register` | `register_webhook()` | `webhook` | Register webhook |
| `GET` | `/webhooks/list` | `list_webhooks()` | `webhook` | List webhooks |
| `DELETE` | `/webhooks/unregister` | `unregister_webhook()` | `webhook` | Unregister webhook |
| `POST` | `/webhooks/test` | `test_webhook()` | `webhook` | Test webhook |
| `GET` | `/webhooks/stats` | `get_webhook_stats()` | `webhook` | Webhook statistics |

**Total:** 14 routes

**Key Features:**
- Direct service integration (no proxy)
- Comprehensive error handling with custom exceptions
- Audit logging for all operations
- Webhook notifications
- Batch processing support
- Cache integration for compare_face

---

### Main Branch Routes (Proxy-Based API)

| Method | Path | Handler | Services Used | Purpose |
|--------|------|---------|---------------|---------|
| `GET` | `/api/v1/health` | `api_health()` | None | Health check |
| `POST` | `/api/v1/search` | `search_passthrough()` | `httpx` (proxy) | Proxy search to face-pipeline |
| `POST` | `/api/v1/identity_safe_search` | `identity_safe_search_passthrough()` | `httpx` (proxy) | Identity-safe search proxy |
| `POST` | `/api/v1/ingest` | `ingest_now()` | `redis`, `minio` | Enqueue image to Redis stream |
| `POST` | `/api/v1/ingest/batch` | `ingest_batch()` | `redis`, `minio` | Batch enqueue to Redis stream |
| `GET` | `/api/v1/debug/ingest-config` | `debug_ingest_config()` | None | Debug endpoint |

**Total:** 6 routes

**Key Features:**
- Proxy pattern (forwards to face-pipeline service)
- Redis stream-based ingestion
- Identity-safe search (verification-first flow)
- Simpler architecture
- Direct MinIO/Redis access for ingest

---

## Detailed Route Analysis

### 1. Face Operations

#### This Branch: Direct Service Calls
```python
# POST /index_face
- Calls: get_face_service(), save_raw_and_thumb_async(), get_vector_client()
- Returns: indexed count, phash, thumb_url, vector_backend
- Features: Audit logging, webhook notifications

# POST /search_face  
- Calls: get_face_service(), save_raw_and_thumb_async(), get_vector_client()
- Returns: faces_found, phash, thumb_url, results, vector_backend
- Features: Configurable top_k/threshold, audit logging, webhooks

# POST /compare_face
- Calls: get_cache_service(), get_face_service(), get_vector_client()
- Returns: phash, faces_found, results, vector_backend, message
- Features: Caching layer, search-only (no storage)
```

#### Main Branch: Proxy to Face-Pipeline
```python
# POST /api/v1/search
- Proxies to: {PIPELINE_URL}/api/v1/search
- Returns: Pipeline response (passthrough)
- Features: Simple proxy, no local processing
```

**Impact:** Main branch doesn't have direct face operations - everything goes through face-pipeline service.

---

### 2. Batch Processing

#### This Branch: Full Batch Management
```python
# POST /batch/index
- Creates batch job via get_batch_processor()
- Background task processing
- Returns: batch_id, total_images, status, message

# GET /batch/{batch_id}/status
- Gets batch status from batch_processor
- Tenant isolation
- Returns: Full batch status with metrics

# GET /batch/list
- Lists all batches for tenant
- Pagination support
- Returns: batches, total_count, limit, offset, has_more

# DELETE /batch/{batch_id}
- Cancels in-progress batches
- Tenant validation
- Returns: cancellation status

# POST /batch/cleanup
- Cleans up old completed batches
- Age-based cleanup
- Returns: cleaned_count
```

#### Main Branch: Redis Stream Ingest
```python
# POST /api/v1/ingest
- Enqueues single item to Redis stream
- Computes SHA256/phash from MinIO object
- Returns: message_id, enqueued payload

# POST /api/v1/ingest/batch
- Batch enqueue to Redis stream
- Pipeline-based for efficiency
- Dry-run support
- Returns: per-item status with message_ids
```

**Key Difference:**
- **This Branch:** Batch job management with status tracking, cancellation, cleanup
- **Main Branch:** Simple Redis stream ingestion (fire-and-forget pattern)

**Crawler Impact:** The new crawler uses Redis streams directly (via `redis_manager`), so it's compatible with main's ingest pattern. However, this branch's batch management could be useful for tracking crawler jobs.

---

### 3. Image Serving

#### This Branch:
```python
# GET /images/{bucket}/{key:path}
- Serves images from storage via get_object_from_storage()
- StreamingResponse for efficiency
- Error handling with 404
```

#### Main Branch:
- **No image serving endpoint** - likely handled by nginx or face-pipeline

**Impact:** This branch has explicit image serving, main relies on external service.

---

### 4. Webhooks

#### This Branch: Full Webhook System
```python
# POST /webhooks/register
# GET /webhooks/list
# DELETE /webhooks/unregister
# POST /webhooks/test
# GET /webhooks/stats
```
- Complete webhook management
- Event types: face.indexed, face.searched, face.compared, batch.*
- HMAC signature support
- Statistics tracking

#### Main Branch:
- **No webhook system**

**Impact:** This branch has webhook notifications, main doesn't.

---

### 5. Identity-Safe Search (Main Only)

#### Main Branch:
```python
# POST /api/v1/identity_safe_search
- Proxies to face-pipeline
- Verification-first flow (prevents cross-facial recognition)
- Returns: verified, similarity, threshold, results
```

**This Branch:**
- **No identity-safe search endpoint**

**Impact:** Main has security feature for identity verification, this branch doesn't.

---

## Crawler Dependencies

### How the New Crawler Works

The new crawler (`backend/new_crawler/`) **does NOT use API routes**. Instead:

1. **Direct Service Integration:**
   - `storage_manager.py` - Direct MinIO/S3 access
   - `redis_manager.py` - Direct Redis queue access
   - `gpu_interface.py` - Direct GPU worker communication

2. **No HTTP API Calls:**
   - Crawler doesn't call `/index_face`, `/search_face`, etc.
   - Crawler doesn't use `/batch/index` or `/api/v1/ingest`
   - Crawler uses services directly, not through API layer

3. **Storage Flow:**
   ```
   Crawler → storage_manager → MinIO/S3
   Crawler → redis_manager → Redis queues
   Crawler → gpu_interface → GPU worker (HTTP, but not via API routes)
   ```

### Routes the Crawler Could Use (Future)

While the crawler doesn't currently use API routes, these could be useful:

1. **`/api/v1/ingest` (Main)** - Could be used to enqueue crawled images
2. **`/api/v1/ingest/batch` (Main)** - Batch enqueue for crawler results
3. **`/batch/index` (This Branch)** - Could track crawler batch jobs
4. **`/batch/{batch_id}/status` (This Branch)** - Monitor crawler progress

**Current State:** Crawler bypasses API layer entirely.

---

## Merge Impact Analysis

### 🔴 Critical Conflicts

1. **API Architecture Mismatch**
   - **This Branch:** Direct service calls, full-featured API
   - **Main:** Proxy pattern, simpler API
   - **Decision Required:** Choose architecture or support both

2. **Route Path Differences**
   - **This Branch:** `/index_face`, `/search_face`, `/compare_face`
   - **Main:** `/api/v1/search`, `/api/v1/identity_safe_search`
   - **Impact:** Frontend/client code will break if routes change

3. **Service Integration**
   - **This Branch:** Uses `get_face_service()`, `get_vector_client()`, etc.
   - **Main:** Proxies to face-pipeline (different service layer)
   - **Impact:** Different error handling, response formats

### 🟡 Medium Risk

4. **Batch Processing**
   - **This Branch:** Job-based batch management
   - **Main:** Redis stream ingestion
   - **Compatibility:** Both use Redis, but different patterns
   - **Recommendation:** Support both patterns

5. **Image Serving**
   - **This Branch:** `/images/{bucket}/{key:path}` endpoint
   - **Main:** No endpoint (likely nginx)
   - **Impact:** May need to preserve this branch's endpoint

### 🟢 Low Risk (Crawler Not Affected)

6. **Webhooks (This Branch Only)**
   - New feature, doesn't conflict
   - Can be preserved

7. **Identity-Safe Search (Main Only)**
   - New security feature
   - Should be preserved from main

---

## Recommendations

### Option 1: Preserve Both APIs (Recommended)

**Strategy:** Support both API patterns with route prefixes

```python
# This branch's routes (keep)
router.post("/index_face", ...)
router.post("/search_face", ...)
router.post("/compare_face", ...)

# Main's routes (add)
router.post("/api/v1/search", ...)  # Proxy to face-pipeline
router.post("/api/v1/identity_safe_search", ...)
router.post("/api/v1/ingest", ...)
router.post("/api/v1/ingest/batch", ...)

# Shared routes
router.get("/images/{bucket}/{key:path}", ...)  # From this branch
router.get("/api/v1/health", ...)  # From main
```

**Pros:**
- No breaking changes
- Supports both frontend patterns
- Gradual migration path

**Cons:**
- More routes to maintain
- Potential confusion

### Option 2: Migrate to Proxy Pattern (Main's Approach)

**Strategy:** Replace direct service calls with face-pipeline proxy

**Changes Required:**
- Remove direct `get_face_service()` calls
- Add proxy to face-pipeline for all face operations
- Update batch processing to use Redis streams
- Remove webhook system (or move to face-pipeline)

**Pros:**
- Simpler architecture
- Consistent with main
- Service separation

**Cons:**
- Loses webhook functionality
- Loses batch job management
- Requires face-pipeline to support all features

### Option 3: Migrate to Direct Service Pattern (This Branch's Approach)

**Strategy:** Keep this branch's full-featured API, add main's features

**Changes Required:**
- Add `/api/v1/identity_safe_search` using direct services
- Add `/api/v1/ingest` endpoints (can use existing batch system)
- Keep all webhook functionality
- Update face-pipeline to use direct service calls

**Pros:**
- More features (webhooks, batch management)
- Better error handling
- Audit logging

**Cons:**
- More complex
- Requires updating face-pipeline integration

---

## Specific Merge Instructions

### Keep from This Branch:
1. ✅ **All face operation routes** (`/index_face`, `/search_face`, `/compare_face`)
   - More feature-rich
   - Better error handling
   - Audit logging

2. ✅ **Batch management routes** (`/batch/*`)
   - Useful for tracking crawler jobs
   - Better than fire-and-forget pattern

3. ✅ **Webhook routes** (`/webhooks/*`)
   - New feature, no conflicts

4. ✅ **Image serving** (`/images/{bucket}/{key:path}`)
   - Explicit endpoint is useful

### Adopt from Main:
1. ✅ **`/api/v1/identity_safe_search`**
   - Security feature
   - Can be implemented using this branch's services

2. ✅ **`/api/v1/ingest` and `/api/v1/ingest/batch`**
   - Useful for crawler integration
   - Can coexist with batch management

3. ✅ **`/api/v1/health`**
   - Simple health check

### Implementation Notes:

1. **Identity-Safe Search:**
   ```python
   # Add to this branch's routes.py
   @router.post("/api/v1/identity_safe_search")
   async def identity_safe_search(request: Request, req: IdentitySafeSearchReq):
       # Use get_face_service() and get_vector_client()
       # Implement verification-first logic
       # Return IdentitySafeSearchResp
   ```

2. **Ingest Endpoints:**
   ```python
   # Can use existing redis_manager or add new endpoints
   @router.post("/api/v1/ingest")
   async def ingest_now(req: IngestRequest):
       # Use redis_manager to enqueue
       # Or create new ingest service
   ```

3. **Route Compatibility:**
   - Keep `/index_face`, `/search_face` for backward compatibility
   - Add `/api/v1/search` as alias or proxy (if needed)
   - Document both patterns

---

## Testing Checklist

After merge, test:

- [ ] `/index_face` works with new crawler's storage
- [ ] `/search_face` returns correct results
- [ ] `/compare_face` uses cache correctly
- [ ] `/api/v1/ingest` can enqueue crawler images
- [ ] `/api/v1/ingest/batch` handles large batches
- [ ] `/api/v1/identity_safe_search` verifies correctly
- [ ] `/batch/index` creates jobs correctly
- [ ] `/batch/{batch_id}/status` tracks crawler progress
- [ ] `/images/{bucket}/{key:path}` serves crawler images
- [ ] Webhooks fire for crawler-indexed faces

---

## Summary

**Crawler Impact:** ✅ **NONE** - The new crawler doesn't use API routes, so route changes won't break it.

**Merge Risk:** 🔴 **HIGH** - API architectures are fundamentally different, requiring careful reconciliation.

**Recommendation:** 
1. **Preserve this branch's routes** (more features)
2. **Add main's routes** (`/api/v1/*` prefix)
3. **Implement identity-safe search** using this branch's services
4. **Support both patterns** during transition period

**Next Steps:**
1. Review with team: which API pattern to standardize on
2. Create merge branch to test route compatibility
3. Update frontend if route paths change
4. Test crawler integration with both API patterns

---

*Analysis complete. Ready for merge planning.*


