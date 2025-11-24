# Dev/Admin Features - What's Missing & What You Need

## Current Situation

### What You Have
✅ Database tables for audit logs (`audit_logs`, `search_audit_logs`)  
✅ Database tables for images and faces  
✅ Backend services for data export and dashboard  
✅ Mock-based Dev UI (Phase 0-7)

### What's Missing
❌ **Audit logging is DISABLED** (see `backend/app/core/audit.py` lines 36, 67)  
❌ **No API endpoints to query audit data** from frontend  
❌ **No Dev/Admin UI** to browse real user submissions  
❌ **No search history viewer**  
❌ **No user activity tracker**  

---

## What You SHOULD Have Access To

### 1. Search History Viewer
**Purpose**: View all searches performed by users

**Features Needed**:
- List recent searches (last 24h, 7d, 30d)
- Filter by tenant_id
- Filter by date range
- Sort by recency
- View search details (query image, results, parameters)
- Click to view full search results

**API Endpoint Needed**:
```
GET /api/v1/admin/searches?tenant_id={}&start_date={}&end_date={}&limit={}
```

**Response**:
```json
{
  "searches": [
    {
      "search_id": "search-uuid-123",
      "tenant_id": "demo-tenant",
      "operation_type": "search_face",
      "face_count": 1,
      "result_count": 15,
      "vector_backend": "qdrant",
      "created_at": "2025-11-14T12:00:00Z",
      "query_image_url": "https://..."
    }
  ],
  "total": 150,
  "page": 1
}
```

---

### 2. User Activity Tracker
**Purpose**: View activity for a specific user/tenant

**Features Needed**:
- List all searches by user
- Show upload activity
- Show API usage stats
- Timeline view of activity
- Filter by activity type

**API Endpoint Needed**:
```
GET /api/v1/admin/users/{tenant_id}/activity?start_date={}&end_date={}
```

**Response**:
```json
{
  "tenant_id": "demo-tenant",
  "period": {
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-14T23:59:59Z"
  },
  "stats": {
    "total_searches": 150,
    "total_uploads": 45,
    "total_api_calls": 500,
    "avg_results_per_search": 12.5
  },
  "recent_activity": [
    {
      "timestamp": "2025-11-14T12:00:00Z",
      "type": "search",
      "operation": "search_face",
      "result_count": 15,
      "status": "success"
    }
  ]
}
```

---

### 3. Search Details Viewer
**Purpose**: View full details of a specific search

**Features Needed**:
- View query image
- View all results with scores
- View search parameters (top_k, threshold)
- View timing metrics
- See which backend was used (Qdrant, etc.)

**API Endpoint Needed**:
```
GET /api/v1/admin/searches/{search_id}
```

**Response**:
```json
{
  "search_id": "search-uuid-123",
  "tenant_id": "demo-tenant",
  "query": {
    "image_url": "https://...",
    "uploaded_at": "2025-11-14T12:00:00Z",
    "top_k": 50,
    "threshold": 0.75
  },
  "results": [
    {
      "face_id": "face-uuid-456",
      "score": 0.95,
      "metadata": {...}
    }
  ],
  "stats": {
    "face_count": 1,
    "result_count": 15,
    "process_time": 0.245
  }
}
```

---

### 4. Upload/Image Browser
**Purpose**: Browse all uploaded images

**Features Needed**:
- List all uploads
- Filter by tenant
- Filter by date
- View image details
- See which searches used this image
- View extracted faces

**API Endpoint Needed**:
```
GET /api/v1/admin/images?tenant_id={}&start_date={}&end_date={}&limit={}
```

**Response**:
```json
{
  "images": [
    {
      "image_id": "image-uuid-123",
      "tenant_id": "demo-tenant",
      "object_key": "demo-tenant/query-abc123.jpg",
      "bucket_name": "raw-images",
      "phash": "a1b2c3d4e5f6g7h8",
      "width": 1920,
      "height": 1080,
      "created_at": "2025-11-14T12:00:00Z",
      "face_count": 1
    }
  ],
  "total": 450,
  "page": 1
}
```

---

### 5. API Audit Log Viewer
**Purpose**: View all API requests for debugging

**Features Needed**:
- List all API requests
- Filter by tenant
- Filter by endpoint
- Filter by status code (errors only, etc.)
- View request details (method, path, response time)
- Search by request_id

**API Endpoint Needed**:
```
GET /api/v1/admin/audit-logs?tenant_id={}&status_code={}&start_date={}&end_date={}&limit={}
```

**Response**:
```json
{
  "logs": [
    {
      "log_id": "log-uuid-123",
      "request_id": "req-abc123",
      "tenant_id": "demo-tenant",
      "method": "POST",
      "path": "/api/v1/search",
      "status_code": 200,
      "process_time": 0.245,
      "user_agent": "Mozilla/5.0...",
      "ip_address": "192.168.1.1",
      "created_at": "2025-11-14T12:00:00Z"
    }
  ],
  "total": 1000,
  "page": 1
}
```

---

### 6. System Dashboard
**Purpose**: Overview of system health and usage

**Features Needed**:
- Total searches today/week/month
- Total users/tenants active
- Error rate metrics
- Performance metrics (avg response time)
- Storage usage
- Most active tenants
- Recent errors

**API Endpoint Available** (already exists):
```
GET /dashboard/overview
```

---

## Implementation Plan

### Step 1: Enable Audit Logging
**File**: `backend/app/core/audit.py`

Remove the early returns on lines 36 and 67 to enable logging:
```python
# Line 36 - REMOVE THIS:
# return

# Line 67 - REMOVE THIS:
# return
```

### Step 2: Create Admin API Endpoints
**File**: `backend/app/api/admin_routes.py` (NEW)

Add endpoints for:
- GET /api/v1/admin/searches
- GET /api/v1/admin/searches/{search_id}
- GET /api/v1/admin/users/{tenant_id}/activity
- GET /api/v1/admin/images
- GET /api/v1/admin/audit-logs

### Step 3: Create Admin UI Page
**File**: `frontend/src/pages/AdminPage.tsx` (NEW)

Add sections for:
- Search History Table
- User Activity Viewer
- Image Browser
- Audit Log Viewer
- System Dashboard

### Step 4: Update Dev UI
**File**: `frontend/src/pages/SearchDevPage.tsx`

Add ability to:
- Load real search by ID (not just mock data)
- Link to admin page
- View search history for current tenant

---

## Admin UI Wireframe

```
┌─────────────────────────────────────────────────────────────┐
│  ADMIN DASHBOARD                                            │
├─────────────────────────────────────────────────────────────┤
│  Tabs: [Search History] [Users] [Images] [Audit Logs]      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SEARCH HISTORY                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Filters: Tenant [All] | Date [Last 7 days] | Status    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Timestamp        │ Tenant      │ Faces │ Results │ ✓   │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ 2025-11-14 12:00 │ demo-tenant │   1   │   15    │ ✓   │ │
│  │ 2025-11-14 11:55 │ demo-tenant │   1   │    8    │ ✓   │ │
│  │ 2025-11-14 11:50 │ user-123    │   1   │    0    │ ✗   │ │
│  │ 2025-11-14 11:45 │ demo-tenant │   2   │   25    │ ✓   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  [Load More] [Export CSV]                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Actions Needed

### 1. Enable Audit Logging (5 minutes)
```bash
# Edit backend/app/core/audit.py
# Remove the "return" statements on lines 36 and 67
```

### 2. Test Audit Logging (5 minutes)
```bash
# Make a search request
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"

# Check database
docker exec -it mordeaux-face-scanning-mvp-api-1 psql -U postgres -d mordeaux -c "SELECT * FROM search_audit_logs LIMIT 5;"
```

### 3. Create Admin Routes (30 minutes)
Create new file `backend/app/api/admin_routes.py` with admin endpoints

### 4. Create Admin UI (60 minutes)
Create new file `frontend/src/pages/AdminPage.tsx` with admin interface

---

## What You'll Be Able to Do

Once implemented, you'll have:

✅ **View all searches** performed across the system  
✅ **Track user activity** per tenant/user  
✅ **Browse uploaded images** and see their usage  
✅ **View API audit logs** for debugging  
✅ **Monitor system health** and performance  
✅ **Search by search_id** to see historical results  
✅ **Filter and export** data for analysis  
✅ **Real-time stats** on system usage  

---

## Priority Order

1. **HIGH**: Enable audit logging (5 min)
2. **HIGH**: Create search history API endpoint (15 min)
3. **HIGH**: Create admin page with search history viewer (30 min)
4. **MEDIUM**: Add user activity tracker (15 min)
5. **MEDIUM**: Add image browser (15 min)
6. **LOW**: Add detailed audit log viewer (20 min)
7. **LOW**: Polish UI and add export features (30 min)

**Total Time**: ~2-3 hours for full admin interface

---

## Current Status

❌ **NOT IMPLEMENTED** - You're looking at a mock UI right now  
🔧 **BACKEND READY** - Database tables exist, just need API endpoints  
📝 **LOGGING DISABLED** - Need to enable in `audit.py`  
🎨 **UI NEEDS BUILD** - Need to create admin pages  

Would you like me to implement these missing pieces for you?


