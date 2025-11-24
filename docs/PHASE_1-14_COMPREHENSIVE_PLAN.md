# Mordeaux Face Scanning MVP — Comprehensive Implementation Plan
## Phases 1-14 + Admin/Dev Interface

**Document Version:** 1.0  
**Last Updated:** November 15, 2025  
**Status:** Phase 8 Complete, Phase 9-14 In Planning

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1-8: Completed Features](#phase-1-8-completed-features)
3. [Phase 9-14: Planned Features](#phase-9-14-planned-features)
4. [Admin/Dev Interface](#admindev-interface)
5. [Implementation Timeline](#implementation-timeline)
6. [Technical Stack](#technical-stack)
7. [Testing Strategy](#testing-strategy)

---

## Overview

This document provides a complete roadmap for the Mordeaux Face Scanning MVP's frontend development, covering:

- **Phases 1-8:** UI foundations, query submission, results rendering, filters, pagination, and safe external links (✅ COMPLETE)
- **Phases 9-14:** Performance optimization, security hardening, observability, accessibility, backend integration, and UAT (🔄 IN PROGRESS)
- **Admin/Dev Interface:** Tools for monitoring user activity, search history, audit logs, and system health (📋 PLANNED)

### Project Goals

1. Build a robust, user-friendly face search interface for developers
2. Ensure security, privacy, and performance at scale
3. Provide comprehensive admin tools for monitoring and debugging
4. Maintain accessibility and responsiveness standards
5. Integrate seamlessly with real backend APIs

---

## Phase 1-8: Completed Features

### Phase 1 — User Journeys & Wireframes ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ User journey flowcharts for key scenarios
- ✅ Wireframes for main UI screens
- ✅ Component hierarchy diagram
- ✅ State management design
- ✅ URL routing structure

#### Key Decisions
- Single-page app using React + Vite
- Dev-only route at `/dev/search`
- URL-based state for filters/pagination
- Mock-first development approach

**Documentation:** [PHASE_1_USER_JOURNEYS_WIREFRAMES.md](./PHASE_1_USER_JOURNEYS_WIREFRAMES.md)

---

### Phase 2 — Basic Layout & Styles ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ Design tokens (colors, spacing, typography)
- ✅ CSS variables in `tokens.css`
- ✅ Base layout structure
- ✅ Responsive grid system
- ✅ Theme consistency

#### Files Created
```
frontend/
├── src/
│   ├── tokens.css         # Design system tokens
│   ├── App.css           # Global app styles
│   └── pages/
│       └── SearchDevPage.css
```

---

### Phase 3 — Query Image Upload ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ File upload component with drag-and-drop
- ✅ Image preview with metadata
- ✅ File validation (type, size)
- ✅ URL input alternative
- ✅ Loading states

#### Components
- `QueryImage.tsx` - Display uploaded query image
- Image upload form in `SearchDevPage.tsx`

#### Features
- Drag-and-drop file upload
- File type validation (JPG, PNG, WebP)
- Size limit validation
- URL-based upload option
- Thumbnail preview
- Metadata display (dimensions, file size)

---

### Phase 4 — Mock Data & Server ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ Mock server with FastAPI
- ✅ `/api/v1/search` endpoint
- ✅ Realistic mock data generation
- ✅ 50+ mock face records
- ✅ Query parameter support

#### Mock Server Features
```python
# mock-server/app.py
@app.post("/api/v1/search")
async def search_faces(
    image: UploadFile,
    top_k: int = 50,
    threshold: float = 0.75
):
    # Returns realistic mock search results
    return {
        "hits": [...],
        "query_faces": [...]
    }
```

**Documentation:** [mock-server/QUICK_START.md](../mock-server/QUICK_START.md)

---

### Phase 5 — Query Image Safety ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ `SafeImage.tsx` component with 12 security rules
- ✅ URL validation and sanitization
- ✅ Domain whitelist enforcement
- ✅ Presigned URL detection
- ✅ Retry logic with exponential backoff
- ✅ Size limit enforcement
- ✅ Security event logging

#### Security Rules
1. ✅ Validate URL format
2. ✅ Domain whitelist
3. ✅ HTTPS enforcement (except localhost)
4. ✅ No referrer leakage (`referrerPolicy="no-referrer"`)
5. ✅ Cross-origin isolation
6. ✅ Presigned URL detection
7. ✅ XSS prevention
8. ✅ Fallback on error
9. ✅ Timeout handling
10. ✅ Retry logic (3 attempts, exponential backoff)
11. ✅ Image load event tracking
12. ✅ Size validation

**Documentation:** [docs/IMAGE_SAFETY_RULES.md](./IMAGE_SAFETY_RULES.md)

---

### Phase 6 — Results Rendering ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ `ResultCard.tsx` - Grid view component
- ✅ `ResultListItem.tsx` - List view component
- ✅ `ScoreBadge.tsx` - Visual score indicator
- ✅ `DistanceChip.tsx` - Distance metric display
- ✅ `BBoxOverlay.tsx` - Bounding box visualization
- ✅ Grid/List view toggle
- ✅ Empty state component
- ✅ Error state component
- ✅ Loading state component

#### Components Created
```
frontend/src/components/
├── ResultCard.tsx          # Card view for grid
├── ResultCard.css
├── ResultListItem.tsx      # Row view for list
├── ResultListItem.css
├── ScoreBadge.tsx         # Score visualization
├── ScoreBadge.css
├── DistanceChip.tsx       # Distance metric
├── DistanceChip.css
├── BBoxOverlay.tsx        # Bounding box overlay
├── BBoxOverlay.css
├── EmptyState.tsx         # No results UI
├── EmptyState.css
├── ErrorState.tsx         # Error display
├── ErrorState.css
├── LoadingState.tsx       # Loading spinner
└── LoadingState.css
```

#### Features
- Two view modes: Grid and List
- Hover effects reveal bounding boxes
- Copy face ID to clipboard
- View source link
- Score visualization (color-coded)
- Distance metric display
- Quality score indicator
- Responsive layout

**Documentation:** [docs/PHASE_6_RESULTS_RENDERING_COMPLETE.md](./PHASE_6_RESULTS_RENDERING_COMPLETE.md)

---

### Phase 7 — Filters, Pagination, URL Sync ✅
**Completed:** November 14, 2025

#### Deliverables
- ✅ `MinScoreSlider.tsx` - Visual score filter
- ✅ `Pagination.tsx` - Full pagination controls
- ✅ `useUrlState.ts` - URL state synchronization hook
- ✅ Site filter dropdown
- ✅ Deep-linking support
- ✅ Browser back/forward navigation
- ✅ QA test script with 15 test cases

#### URL Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `minScore` | number (0-1) | 0 | Minimum similarity score |
| `site` | string | '' | Filter by domain |
| `page` | integer (≥1) | 1 | Current page number |
| `pageSize` | integer | 25 | Results per page |
| `view` | 'grid' \| 'list' | 'grid' | Display mode |

#### Example URLs
```
# Default state
/dev/search

# Filtered by score
/dev/search?minScore=0.8

# Full state
/dev/search?minScore=0.75&site=example.com&page=2&pageSize=50&view=list
```

#### Features
- Real-time filter updates
- Debounced slider (300ms)
- URL state persistence
- Copy URL button
- Reset filters button
- Filter summary display
- Pagination info ("Showing 1-25 of 100 results")
- Page size selector (10, 25, 50, 100)
- Jump to page input
- First/Last page buttons

**Documentation:** [docs/PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md](./PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md)

**QA Script:** [docs/QA_SCRIPT_PHASE_7.md](./QA_SCRIPT_PHASE_7.md)

---

### Phase 8 — Source/Storage Actions (Safe External Links) ✅
**Completed:** November 15, 2025

#### Deliverables
- ✅ `SafeLink.tsx` - Secure external link component
- ✅ `StorageChip.tsx` - Storage provider indicator
- ✅ `linkAudit.ts` - URL safety validation utility
- ✅ Link audit logging
- ✅ Storage provider detection (MinIO, S3, External)
- ✅ Protocol validation
- ✅ Domain whitelist enforcement

#### Security Features
1. ✅ Block `javascript:` and `data:` URLs
2. ✅ HTTPS enforcement for external links
3. ✅ `rel="noreferrer noopener nofollow"` for all external links
4. ✅ Domain whitelist validation
5. ✅ URL sanitization for logging (remove sensitive params)
6. ✅ Structured console logging for security events
7. ✅ Graceful handling of invalid URLs

#### Components Created
```
frontend/src/
├── components/
│   ├── SafeLink.tsx        # Secure external link
│   ├── SafeLink.css
│   ├── StorageChip.tsx     # Storage provider badge
│   └── StorageChip.css
└── utils/
    └── linkAudit.ts        # URL validation & sanitization
```

#### Storage Detection
Automatically detects and displays storage provider:
- 🗄️ MinIO
- ☁️ S3 (AWS)
- 🌐 External (whitelisted)
- ❓ Unknown

#### Integrated Into
- `ResultCard.tsx` - Source links and storage chips
- `ResultListItem.tsx` - Source links and storage chips
- `QueryImage.tsx` - Full resolution links
- `SafeImage.tsx` - Uses storage detection utility

#### Usage Example
```tsx
import SafeLink from './components/SafeLink';
import StorageChip from './components/StorageChip';

// Safe external link
<SafeLink href={url}>
  🔗 View Source
</SafeLink>

// Storage provider indicator
<StorageChip provider="minio" />
```

#### Acceptance Criteria
- ✅ All external links pass safety checklist
- ✅ Broken links don't crash UI
- ✅ Links open in new tab with security attributes
- ✅ Storage provider correctly detected
- ✅ Console logs structured security events
- ✅ URL sanitization removes sensitive parameters

---

## Phase 9-14: Planned Features

### Phase 9 — Performance Hardening 🔄
**Status:** In Progress  
**Goal:** Smooth at 2,000+ results

#### Deliverables
- [ ] Virtualized list/grid rendering
  - Using `react-window` or `react-virtualized`
  - Only render visible items
  - Smooth scrolling performance
- [ ] Chunked rendering strategy
  - Progressive loading of results
  - Render in batches (e.g., 50 at a time)
- [ ] Lazy image loading
  - Intersection Observer API
  - Load images as they enter viewport
- [ ] Memoization strategy
  - Memoize expensive computations
  - Prevent unnecessary re-renders
  - Use `useMemo` and `React.memo`
- [ ] Abortable fetch
  - Cancel pending requests on new search
  - Clean up resources properly
- [ ] Debounce/throttle optimizations
  - Already in place for slider (300ms)
  - Apply to other interactive elements
- [ ] Bundle size analysis
  - Webpack bundle analyzer
  - Code splitting
  - Dynamic imports
- [ ] Image decode time monitoring
  - Track image loading performance
  - Alert on slow loads

#### Acceptance Criteria
- ✅ First interactive under 2 seconds with 2,000 results
- ✅ Scroll remains smooth (60fps)
- ✅ Memory usage stays reasonable
- ✅ No jank during filtering/pagination
- ✅ Bundle size < 500KB gzipped

#### Implementation Plan
1. Install `react-window` for virtualization
2. Create virtualized versions of ResultCard/ResultListItem
3. Implement Intersection Observer for lazy images
4. Add performance monitoring hooks
5. Profile and optimize hot paths
6. Code split large components

**Estimated Time:** 6-8 hours

---

### Phase 10 — Security/Privacy (Dev-Only Guardrails) 📋
**Status:** Planned  
**Goal:** Ensure dev page can't leak data

#### Deliverables
- [ ] Dev-only route guard
  - Environment flag (e.g., `VITE_DEV_MODE=true`)
  - Feature flag check
  - Auth role check (optional)
  - Redirect non-dev users
- [ ] Data redaction system
  - Hide sensitive metadata by default
  - "Reveal for dev" toggle button
  - Redact internal IDs, IP addresses, tokens
  - Sanitize URLs in logs
- [ ] PII protection
  - No PII in query params beyond IDs
  - No raw URLs logged in production
  - Clear data retention policies
- [ ] Audit documentation
  - Document what data is visible
  - Document redaction rules
  - Document access controls

#### Redaction Rules
```typescript
// Example redaction config
const REDACTION_CONFIG = {
  fields: {
    ip_address: 'masked',      // 192.168.1.1 → 192.168.x.x
    internal_id: 'hidden',     // Hide completely
    presigned_url: 'sanitized', // Remove signature params
    email: 'partial',          // user@example.com → u***@example.com
  },
  revealForDev: true, // Devs can toggle reveal
};
```

#### Acceptance Criteria
- ✅ Feature is invisible to non-dev roles
- ✅ Redaction rules verified in QA
- ✅ No PII in URL params
- ✅ Production logs sanitized
- ✅ Documentation complete

**Estimated Time:** 4-6 hours

---

### Phase 11 — Observability & Diagnostics 📋
**Status:** Planned  
**Goal:** Make debugging easy

#### Deliverables
- [ ] Structured console logs (dev only)
  - Use log levels: INFO, WARN, ERROR
  - Consistent format with timestamps
  - Context-aware logging
- [ ] Performance timing marks
  - Mark key milestones (search start, first render, etc.)
  - Use `performance.mark()` and `performance.measure()`
  - Display in dev tools
- [ ] Error counters
  - Track image load failures
  - Track API failures
  - Track validation errors
- [ ] Event payload viewer
  - Collapsible JSON viewer for single item
  - Inspect full search result payload
  - Copy to clipboard
- [ ] Debug panel (dev only)
  - Toggle visibility
  - Show performance metrics
  - Show error counts
  - Show current state

#### Structured Logging Example
```typescript
// utils/logger.ts
export const logger = {
  info: (event: string, payload: any) => {
    if (import.meta.env.DEV) {
      console.info(`[${new Date().toISOString()}] [INFO] ${event}`, payload);
    }
  },
  warn: (event: string, payload: any) => {
    console.warn(`[${new Date().toISOString()}] [WARN] ${event}`, payload);
  },
  error: (event: string, payload: any) => {
    console.error(`[${new Date().toISOString()}] [ERROR] ${event}`, payload);
  },
};
```

#### Acceptance Criteria
- ✅ Devs can diagnose failing images/links in < 2 minutes with logs
- ✅ Performance metrics visible in console
- ✅ Error patterns easily identifiable
- ✅ Debug panel useful and non-intrusive

**Estimated Time:** 4-5 hours

---

### Phase 12 — Accessibility & Responsiveness QA 📋
**Status:** Planned  
**Goal:** Ship something humane to use

#### Deliverables
- [ ] Keyboard navigation
  - Tab through all interactive elements
  - Enter/Space to activate
  - Arrow keys for pagination
  - Escape to close modals
- [ ] Focus outlines
  - Visible focus indicators
  - Skip to content link
- [ ] ARIA attributes
  - Labels for all controls
  - Roles for custom components
  - Live regions for dynamic content
- [ ] Color contrast check
  - WCAG AA compliance (4.5:1 for text)
  - Test with contrast checker
- [ ] Responsive breakpoints
  - Mobile (< 640px)
  - Tablet (640px - 1024px)
  - Desktop (> 1024px)
- [ ] Touch targets
  - Minimum 40x40px for all buttons
  - Adequate spacing between elements
- [ ] Low-DPI and high-DPI testing
- [ ] Screen reader testing

#### Tools
- axe DevTools
- Lighthouse accessibility audit
- NVDA or JAWS for screen reader testing
- Chrome DevTools responsive mode

#### Acceptance Criteria
- ✅ Meets WCAG AA for core interactions
- ✅ Touch targets ≥ 40px
- ✅ Keyboard navigation fully functional
- ✅ Screen reader friendly
- ✅ Responsive across all breakpoints

**Estimated Time:** 6-8 hours

---

### Phase 13 — Backend Integration (Behind a Flag) 📋
**Status:** Planned  
**Goal:** Swap mocks → real API without breakage

#### Deliverables
- [ ] Configurable API base
  - Environment variable `VITE_API_BASE_URL`
  - Default to mock server in dev
  - Switch to real API in production
- [ ] Error taxonomy mapping
  - Map HTTP codes to user-friendly messages
  - 400 → "Invalid request"
  - 404 → "Not found"
  - 500 → "Server error, please try again"
- [ ] Presigned URL expiry handling
  - Detect expired URLs
  - Request fresh URLs automatically
  - Retry with new URL
- [ ] Retry strategy
  - Exponential backoff (already in SafeImage)
  - Max 3 retries
  - Different strategies for different error types
- [ ] Feature flag system
  - `USE_REAL_API` flag
  - Toggle in UI for testing
  - Graceful fallback to mocks
- [ ] API client wrapper
  - Centralize all API calls
  - Handle auth headers (X-Tenant-ID, etc.)
  - Request/response logging
  - Error handling

#### API Client Example
```typescript
// utils/apiClient.ts
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = {
  async search(formData: FormData, options: SearchOptions) {
    const response = await fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: {
        'X-Tenant-ID': options.tenantId || 'demo-tenant',
      },
      body: formData,
    });

    if (!response.ok) {
      throw new APIError(response.status, await response.text());
    }

    return await response.json();
  },
};
```

#### Acceptance Criteria
- ✅ Live data parity with mocks on same searches
- ✅ No console errors
- ✅ Graceful error handling
- ✅ Retry logic works correctly
- ✅ Feature flag toggles between mock/real API

**Estimated Time:** 6-8 hours

---

### Phase 14 — UAT Script, Checklist, and Handoff 📋
**Status:** Planned  
**Goal:** Freeze scope and document

#### Deliverables
- [ ] UAT steps document
  - Happy path scenarios
  - Edge case scenarios
  - Error scenarios
  - Performance scenarios
- [ ] Runbook for devs
  - How to test a search
  - How to share a deep link
  - What to do when images fail
  - How to toggle feature flags
  - How to read logs
- [ ] Done-Definition checklist
  - All phases 1-13 complete
  - All acceptance criteria met
  - All tests passing
  - Documentation complete
  - Code reviewed
- [ ] Archive mocks
  - Tag mock server version
  - Document mock data structure
  - Preserve for future reference
- [ ] Ship behind feature flag
  - Deploy to staging
  - Test with real users
  - Gradual rollout

#### UAT Script Structure
```markdown
# UAT Test Case Template

## TC-001: Basic Face Search
**Priority:** High
**Preconditions:** User has access to /dev/search

### Steps
1. Navigate to /dev/search
2. Upload a face image
3. Click "Search"
4. Verify results appear

### Expected Results
- Results appear within 3 seconds
- At least 1 result returned
- Images load correctly
- Source links work

### Pass/Fail Criteria
- [ ] All expected results met
- [ ] No console errors
- [ ] Performance within target
```

#### Acceptance Criteria
- ✅ You (stakeholder) sign off
- ✅ All UAT cases pass
- ✅ Runbook is clear and complete
- ✅ Mocks archived
- ✅ Feature shipped behind flag

**Estimated Time:** 4-6 hours

---

## Admin/Dev Interface

### Overview
In addition to the search dev page, we need comprehensive admin tools to monitor user activity, debug issues, and analyze system usage.

### Current Status
❌ **NOT IMPLEMENTED**  
🔧 **BACKEND READY** - Database tables exist, just need API endpoints  
📝 **LOGGING DISABLED** - Need to enable in `audit.py`  
🎨 **UI NEEDS BUILD** - Need to create admin pages

### Required Components

#### 1. Search History Viewer
**Purpose:** View all searches performed by users

**Features:**
- List recent searches (last 24h, 7d, 30d)
- Filter by tenant_id
- Filter by date range
- Sort by recency, result count, status
- View search details (query image, results, parameters)
- Click to view full search results
- Export to CSV

**API Endpoint Needed:**
```
GET /api/v1/admin/searches?tenant_id={}&start_date={}&end_date={}&limit={}
```

**UI Components:**
```
frontend/src/pages/admin/
├── SearchHistoryPage.tsx
├── SearchHistoryTable.tsx
├── SearchDetailsModal.tsx
└── SearchFilters.tsx
```

---

#### 2. User Activity Tracker
**Purpose:** View activity for a specific user/tenant

**Features:**
- List all searches by user
- Show upload activity
- Show API usage stats
- Timeline view of activity
- Filter by activity type
- Export activity report

**API Endpoint Needed:**
```
GET /api/v1/admin/users/{tenant_id}/activity?start_date={}&end_date={}
```

**UI Components:**
```
frontend/src/pages/admin/
├── UserActivityPage.tsx
├── ActivityTimeline.tsx
└── UsageStats.tsx
```

---

#### 3. Search Details Viewer
**Purpose:** View full details of a specific search

**Features:**
- View query image
- View all results with scores
- View search parameters (top_k, threshold)
- View timing metrics
- See which backend was used (Qdrant, etc.)
- Replay search with same params

**API Endpoint Needed:**
```
GET /api/v1/admin/searches/{search_id}
```

**UI Components:**
```
frontend/src/pages/admin/
├── SearchDetailPage.tsx
└── SearchReplayButton.tsx
```

---

#### 4. Upload/Image Browser
**Purpose:** Browse all uploaded images

**Features:**
- List all uploads
- Filter by tenant
- Filter by date
- View image details
- See which searches used this image
- View extracted faces
- Download image

**API Endpoint Needed:**
```
GET /api/v1/admin/images?tenant_id={}&start_date={}&end_date={}&limit={}
```

**UI Components:**
```
frontend/src/pages/admin/
├── ImageBrowserPage.tsx
├── ImageGrid.tsx
└── ImageDetailModal.tsx
```

---

#### 5. API Audit Log Viewer
**Purpose:** View all API requests for debugging

**Features:**
- List all API requests
- Filter by tenant
- Filter by endpoint
- Filter by status code (errors only, etc.)
- View request details (method, path, response time)
- Search by request_id
- Export logs

**API Endpoint Needed:**
```
GET /api/v1/admin/audit-logs?tenant_id={}&status_code={}&start_date={}&end_date={}&limit={}
```

**UI Components:**
```
frontend/src/pages/admin/
├── AuditLogPage.tsx
└── AuditLogTable.tsx
```

---

#### 6. System Dashboard
**Purpose:** Overview of system health and usage

**Features:**
- Total searches today/week/month
- Total users/tenants active
- Error rate metrics
- Performance metrics (avg response time)
- Storage usage
- Most active tenants
- Recent errors

**API Endpoint:** (Already exists)
```
GET /api/v1/dashboard/overview
```

**UI Components:**
```
frontend/src/pages/admin/
├── DashboardPage.tsx
├── MetricsCard.tsx
├── ActivityChart.tsx
└── RecentErrorsList.tsx
```

---

### Admin UI Navigation

```
┌─────────────────────────────────────────────────────────────┐
│  MORDEAUX ADMIN                               [demo-tenant] │
├─────────────────────────────────────────────────────────────┤
│  Navigation:                                                │
│  [Dashboard] [Search History] [Users] [Images] [Audit Logs] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DASHBOARD                                                  │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Total Searches   │  │ Active Tenants   │               │
│  │                  │  │                  │               │
│  │      1,234       │  │        12        │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Avg Response Time│  │ Error Rate       │               │
│  │                  │  │                  │               │
│  │    245ms         │  │     0.5%         │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                              │
│  RECENT ACTIVITY (Last 24h)                                │
│  ┌────────────────────────────────────────────────────────┐│
│  │ [Chart: Searches per hour]                             ││
│  └────────────────────────────────────────────────────────┘│
│                                                              │
│  MOST ACTIVE TENANTS                                        │
│  ┌────────────────────────────────────────────────────────┐│
│  │ 1. demo-tenant       │ 450 searches                    ││
│  │ 2. user-123          │ 234 searches                    ││
│  │ 3. test-user         │ 156 searches                    ││
│  └────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Implementation Steps

#### Step 1: Enable Audit Logging (5 min)
```python
# File: backend/app/core/audit.py
# Remove early returns on lines 36 and 67

# Before:
async def log_audit_event(...):
    return  # ← REMOVE THIS

# After:
async def log_audit_event(...):
    # ... actual logging code runs
```

#### Step 2: Create Admin API Endpoints (2-3 hours)
```python
# File: backend/app/api/admin_routes.py (NEW)

from fastapi import APIRouter, Depends, Query
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

@router.get("/searches")
async def list_searches(
    tenant_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(50, le=500)
):
    """List all searches with filters"""
    pass

@router.get("/searches/{search_id}")
async def get_search_details(search_id: str):
    """Get full details of a specific search"""
    pass

@router.get("/users/{tenant_id}/activity")
async def get_user_activity(
    tenant_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get activity for a specific user/tenant"""
    pass

@router.get("/images")
async def list_images(
    tenant_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(50, le=500)
):
    """List all uploaded images"""
    pass

@router.get("/audit-logs")
async def list_audit_logs(
    tenant_id: Optional[str] = None,
    status_code: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(50, le=500)
):
    """List API audit logs"""
    pass
```

#### Step 3: Create Admin UI Pages (4-6 hours)
```tsx
// File: frontend/src/pages/AdminPage.tsx (NEW)

import { useState } from 'react';
import { Tabs, Tab } from '../components/Tabs';
import DashboardTab from './admin/DashboardTab';
import SearchHistoryTab from './admin/SearchHistoryTab';
import UsersTab from './admin/UsersTab';
import ImagesTab from './admin/ImagesTab';
import AuditLogsTab from './admin/AuditLogsTab';

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="admin-page">
      <header className="admin-header">
        <h1>Mordeaux Admin</h1>
        <div className="tenant-selector">
          <select>
            <option value="">All Tenants</option>
            <option value="demo-tenant">demo-tenant</option>
          </select>
        </div>
      </header>

      <Tabs activeTab={activeTab} onChange={setActiveTab}>
        <Tab id="dashboard" label="Dashboard">
          <DashboardTab />
        </Tab>
        <Tab id="searches" label="Search History">
          <SearchHistoryTab />
        </Tab>
        <Tab id="users" label="Users">
          <UsersTab />
        </Tab>
        <Tab id="images" label="Images">
          <ImagesTab />
        </Tab>
        <Tab id="audit" label="Audit Logs">
          <AuditLogsTab />
        </Tab>
      </Tabs>
    </div>
  );
}
```

#### Step 4: Add Navigation to Admin Page (15 min)
```tsx
// File: frontend/src/App.tsx

import AdminPage from './pages/AdminPage';

<Routes>
  <Route path="/dev/search" element={<SearchDevPage />} />
  <Route path="/admin" element={<AdminPage />} />  {/* NEW */}
  <Route path="/" element={<Navigate to="/dev/search" replace />} />
</Routes>
```

---

### Admin Feature Priority

1. **HIGH**: Enable audit logging (5 min)
2. **HIGH**: Dashboard overview (1 hour)
3. **HIGH**: Search history viewer (2 hours)
4. **MEDIUM**: User activity tracker (1.5 hours)
5. **MEDIUM**: Image browser (1.5 hours)
6. **LOW**: Detailed audit log viewer (2 hours)
7. **LOW**: Export features (1 hour)

**Total Time**: ~8-10 hours for full admin interface

---

## Implementation Timeline

### Already Complete (Phase 1-8)
**Time Invested:** ~20-25 hours  
**Status:** ✅ All acceptance criteria met

### Remaining Work (Phase 9-14 + Admin)

| Phase | Description | Estimated Time | Priority |
|-------|-------------|----------------|----------|
| Phase 9 | Performance Hardening | 6-8 hours | HIGH |
| Phase 10 | Security/Privacy | 4-6 hours | HIGH |
| Phase 11 | Observability | 4-5 hours | MEDIUM |
| Phase 12 | Accessibility QA | 6-8 hours | MEDIUM |
| Phase 13 | Backend Integration | 6-8 hours | HIGH |
| Phase 14 | UAT & Handoff | 4-6 hours | HIGH |
| Admin UI | Full Admin Interface | 8-10 hours | HIGH |

**Total Remaining:** ~38-51 hours (~5-7 days)

### Parallel Work Opportunities
- Phase 9 + Phase 11 can be done together (performance + observability)
- Phase 10 + Phase 12 can be done together (security + accessibility)
- Admin UI can be built in parallel with Phase 9-11

**Optimized Timeline:** ~4-5 days with parallel work

---

## Technical Stack

### Frontend
- **Framework:** React 18+ with TypeScript
- **Build Tool:** Vite 5.4+
- **Routing:** React Router v6
- **Styling:** CSS Modules + Design Tokens
- **State:** URL state + React hooks
- **Testing:** (TBD - Phase 14)

### Backend (Mock Server)
- **Framework:** FastAPI
- **Runtime:** Python 3.11+
- **Server:** Uvicorn

### Backend (Real API)
- **Framework:** FastAPI
- **Database:** PostgreSQL
- **Vector DB:** Qdrant
- **Cache:** Redis
- **Storage:** MinIO / S3

### DevOps
- **Containerization:** Docker
- **Orchestration:** Docker Compose
- **Reverse Proxy:** Nginx

---

## Testing Strategy

### Unit Tests (Phase 14)
- Component tests with React Testing Library
- Utility function tests
- Hook tests

### Integration Tests (Phase 14)
- API integration tests
- End-to-end user flows
- URL state synchronization tests

### Manual QA
- ✅ Phase 7 QA Script (15 test cases)
- [ ] Phase 12 Accessibility QA
- [ ] Phase 14 UAT Script

### Performance Tests (Phase 9)
- Load 2,000 results and verify smooth scrolling
- Bundle size analysis
- Image loading performance
- Memory profiling

---

## Success Metrics

### Performance
- ✅ First interactive < 2 seconds
- ✅ Smooth scrolling at 2,000 results (60fps)
- ✅ Bundle size < 500KB gzipped
- ✅ Image load time < 1 second (P95)

### Security
- ✅ All external links pass safety checklist
- ✅ No PII in URL params
- ✅ Production logs sanitized
- ✅ Dev-only features properly gated

### Accessibility
- ✅ WCAG AA compliance
- ✅ Keyboard navigation functional
- ✅ Screen reader friendly
- ✅ Touch targets ≥ 40px

### User Experience
- ✅ Deep-linking works reliably
- ✅ Error messages are helpful
- ✅ Loading states are clear
- ✅ Empty states are informative

---

## Next Steps

1. ✅ **Complete Phase 8** (Done!)
2. 🔄 **Begin Phase 9**: Install `react-window`, implement virtualization
3. 📋 **Document Phase 8**: Create completion report
4. 🔄 **Plan Admin UI**: Finalize requirements and mockups
5. 🔄 **Enable Audit Logging**: Quick 5-minute fix in `audit.py`

---

## Related Documentation

- [Phase 1: User Journeys](./PHASE_1_USER_JOURNEYS_WIREFRAMES.md)
- [Phase 6: Results Rendering](./PHASE_6_RESULTS_RENDERING_COMPLETE.md)
- [Phase 7: Filters & Pagination](./PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md)
- [QA Script (Phase 7)](./QA_SCRIPT_PHASE_7.md)
- [Image Safety Rules](./IMAGE_SAFETY_RULES.md)
- [Dev Admin Features Needed](../DEV_ADMIN_FEATURES_NEEDED.md)

---

**Document Status:** Living document, updated as phases complete  
**Maintainer:** Development Team  
**Review Cycle:** After each phase completion



