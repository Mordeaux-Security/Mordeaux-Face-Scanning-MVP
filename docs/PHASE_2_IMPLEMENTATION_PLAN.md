# Phase 2 — Implementation Plan: Functional Prototype

## Document Purpose

This document provides a detailed implementation plan for building the dev-only search results visualization page based on approved Phase 1 wireframes.

**Status**: Planning  
**Phase**: 2 - Functional Prototype  
**Prerequisites**: Phase 0 & Phase 1 approved  
**Target Audience**: Development Team

---

## 1. Component Breakdown

### 1.1 Core Components

#### Component: PageController
**Responsibility**: Main application controller, orchestrates all components  
**Key Methods**:
- `init()` - Initialize page, parse URL params, load data
- `loadSearchById(searchId)` - Load search results by search ID
- `loadLatestByUser(userId)` - Load latest search for user
- `loadByUploadId(uploadId)` - Load search by upload ID
- `handleError(error)` - Global error handler
- `showLoading()` / `hideLoading()` - Loading state management

**State**:
```javascript
{
  journey: 'searchId' | 'userId' | 'uploadId',
  searchData: {...},
  filteredResults: [...],
  currentPage: 1,
  pageSize: 25,
  filters: {
    minScore: 0.75,
    site: 'all',
    dateRange: null
  }
}
```

---

#### Component: QueryPanel
**Responsibility**: Display query image and metadata  
**Key Methods**:
- `render(queryData)` - Render query panel with image and metadata
- `loadImage(url)` - Load query image with placeholder
- `renderMetadata(data)` - Display query parameters and info

**Props**:
```javascript
{
  queryImage: string,        // Presigned URL
  uploadTime: string,
  tenantId: string,
  topK: number,
  threshold: number,
  vectorBackend: string
}
```

**HTML Structure**:
```html
<div id="queryPanel" class="query-panel">
  <div class="query-image">
    <img id="queryImage" />
  </div>
  <div class="query-metadata">
    <div class="metadata-item">...</div>
  </div>
</div>
```

---

#### Component: MatchGrid
**Responsibility**: Display search results in grid or list layout  
**Key Methods**:
- `render(results, viewMode)` - Render results in grid or list
- `renderCard(match)` - Render individual match card
- `setupVirtualScroll()` - Initialize virtual scrolling
- `setupLazyLoading()` - Initialize lazy image loading
- `handleCardHover(card)` - Show bbox overlay on hover
- `handleCardClick(match)` - Handle card click actions

**Props**:
```javascript
{
  results: Array<Match>,
  viewMode: 'grid' | 'list',
  onViewSource: (url) => void,
  onCopyId: (id) => void
}
```

**Match Data Structure**:
```javascript
{
  face_id: string,
  score: number,           // 0.0 - 1.0
  payload: {
    site: string,
    url: string,
    ts: string,
    bbox: [x, y, w, h],
    p_hash: string,
    quality: number
  },
  thumb_url: string,       // Presigned URL
  image_url: string        // Presigned URL
}
```

---

#### Component: FilterPanel
**Responsibility**: Client-side filtering controls  
**Key Methods**:
- `render(currentFilters)` - Render filter controls
- `onScoreChange(value)` - Handle similarity slider change
- `onSiteChange(site)` - Handle site dropdown change
- `onDateChange(range)` - Handle date range change
- `applyFilters()` - Apply filters to results
- `resetFilters()` - Reset all filters to defaults

**Props**:
```javascript
{
  minScore: number,
  sites: Array<{name: string, count: number}>,
  dateRange: {from: Date, to: Date} | null,
  onChange: (filters) => void
}
```

**Filter Logic**:
```javascript
function applyFilters(results, filters) {
  return results.filter(match => {
    // Score filter
    if (match.score < filters.minScore) return false;
    
    // Site filter
    if (filters.site !== 'all' && match.payload.site !== filters.site) {
      return false;
    }
    
    // Date range filter
    if (filters.dateRange) {
      const matchDate = new Date(match.payload.ts);
      if (matchDate < filters.dateRange.from || 
          matchDate > filters.dateRange.to) {
        return false;
      }
    }
    
    return true;
  });
}
```

---

#### Component: PaginationControl
**Responsibility**: Pagination UI and logic  
**Key Methods**:
- `render(currentPage, totalPages, pageSize)` - Render pagination controls
- `onPageChange(page)` - Handle page number click
- `onPageSizeChange(size)` - Handle page size change
- `generatePageNumbers(current, total)` - Calculate visible page numbers

**Props**:
```javascript
{
  currentPage: number,
  totalPages: number,
  pageSize: number,
  totalResults: number,
  onPageChange: (page) => void,
  onPageSizeChange: (size) => void
}
```

**Pagination Logic**:
```javascript
function paginate(results, page, pageSize) {
  const start = (page - 1) * pageSize;
  const end = start + pageSize;
  return results.slice(start, end);
}

function getTotalPages(totalResults, pageSize) {
  return Math.ceil(totalResults / pageSize);
}
```

---

#### Component: DebugPanel
**Responsibility**: Show API response and debugging info  
**Key Methods**:
- `render(debugData)` - Render debug panel
- `toggle()` - Expand/collapse panel
- `formatJSON(data)` - Format JSON with syntax highlighting
- `copyJSON()` - Copy JSON to clipboard
- `downloadJSON()` - Download JSON as file

**Props**:
```javascript
{
  apiResponse: object,
  timingMetrics: {
    apiDuration: number,
    ttfb: number,
    imageLoadTime: number
  },
  queryParams: object,
  isExpanded: boolean
}
```

---

#### Component: ErrorState
**Responsibility**: Display error messages  
**Key Methods**:
- `render(errorType, message)` - Render error state
- `getErrorSuggestions(errorType)` - Get helpful suggestions

**Error Types**:
- `search_not_found` - Search ID not found
- `search_expired` - Search results expired
- `user_no_searches` - No searches for user
- `upload_not_found` - Upload ID not found
- `api_error` - Generic API error
- `network_error` - Network/connectivity error

---

#### Component: EmptyState
**Responsibility**: Display "no results" message  
**Key Methods**:
- `render(context)` - Render empty state with suggestions
- `getSuggestions(filters)` - Generate helpful suggestions

---

#### Component: LoadingState
**Responsibility**: Display loading skeleton  
**Key Methods**:
- `render(loadingType)` - Render appropriate loading skeleton
- `show()` / `hide()` - Toggle loading state

**Loading Types**:
- `page` - Full page loading
- `query` - Query panel loading
- `results` - Results grid loading

---

### 1.2 Utility Modules

#### Module: APIClient
**Responsibility**: API communication  
**Methods**:
```javascript
class APIClient {
  constructor(baseURL) { ... }
  
  // Journey A: Get search by ID
  async getSearchById(searchId) {
    // GET /api/v1/search/{searchId}/results
    // or POST /api/v1/search with cached params
  }
  
  // Journey B: Get latest search for user
  async getLatestSearch(userId) {
    // GET /api/v1/search/latest?userId={userId}
  }
  
  // Journey C: Get search by upload ID
  async getSearchByUpload(uploadId) {
    // GET /api/v1/uploads/{uploadId}/search
  }
  
  // Trigger new search from upload
  async triggerSearch(uploadId) {
    // POST /api/v1/uploads/{uploadId}/search
  }
  
  // Refresh expired presigned URLs
  async refreshPresignedUrls(faceIds) {
    // POST /api/v1/search/refresh-urls
    // Body: { face_ids: [...] }
  }
}
```

---

#### Module: URLManager
**Responsibility**: URL parsing and manipulation  
**Methods**:
```javascript
class URLManager {
  // Parse URL parameters
  static parseParams() {
    const params = new URLSearchParams(window.location.search);
    return {
      searchId: params.get('id'),
      userId: params.get('userId'),
      uploadId: params.get('uploadId'),
      page: parseInt(params.get('page')) || 1
    };
  }
  
  // Update URL without reload
  static updateParam(key, value) {
    const url = new URL(window.location);
    url.searchParams.set(key, value);
    window.history.pushState({}, '', url);
  }
  
  // Determine journey type
  static getJourneyType(params) {
    if (params.searchId) return 'searchId';
    if (params.userId) return 'userId';
    if (params.uploadId) return 'uploadId';
    return 'default';
  }
}
```

---

#### Module: ImageManager
**Responsibility**: Image loading and presigned URL handling  
**Methods**:
```javascript
class ImageManager {
  // Load image with retry
  static async loadImage(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
      try {
        return await this._loadImagePromise(url);
      } catch (error) {
        if (i === retries - 1) throw error;
        await this._delay(1000 * (i + 1));
      }
    }
  }
  
  // Check if URL is expired (HTTP 403/404)
  static async isExpired(url) {
    try {
      const response = await fetch(url, { method: 'HEAD' });
      return response.status === 403 || response.status === 404;
    } catch {
      return true;
    }
  }
  
  // Setup lazy loading with Intersection Observer
  static setupLazyLoading(selector) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          observer.unobserve(img);
        }
      });
    });
    
    document.querySelectorAll(selector).forEach(img => {
      observer.observe(img);
    });
  }
}
```

---

#### Module: VirtualScroller
**Responsibility**: Virtual scrolling for large result sets  
**Methods**:
```javascript
class VirtualScroller {
  constructor(container, itemHeight, renderItem) {
    this.container = container;
    this.itemHeight = itemHeight;
    this.renderItem = renderItem;
    this.visibleRange = { start: 0, end: 0 };
  }
  
  init(items) {
    this.items = items;
    this.setupScrollListener();
    this.render();
  }
  
  setupScrollListener() {
    this.container.addEventListener('scroll', 
      this.debounce(() => this.updateVisibleRange(), 100)
    );
  }
  
  updateVisibleRange() {
    const scrollTop = this.container.scrollTop;
    const containerHeight = this.container.clientHeight;
    
    const start = Math.floor(scrollTop / this.itemHeight);
    const end = Math.ceil((scrollTop + containerHeight) / this.itemHeight);
    
    this.visibleRange = { start, end };
    this.render();
  }
  
  render() {
    const visible = this.items.slice(
      this.visibleRange.start, 
      this.visibleRange.end
    );
    
    // Render only visible items
    this.container.innerHTML = visible.map(this.renderItem).join('');
  }
}
```

---

#### Module: ClipboardHelper
**Responsibility**: Copy to clipboard functionality  
**Methods**:
```javascript
class ClipboardHelper {
  static async copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      // Fallback for older browsers
      return this.fallbackCopy(text);
    }
  }
  
  static fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    const success = document.execCommand('copy');
    document.body.removeChild(textarea);
    return success;
  }
}
```

---

#### Module: ToastNotification
**Responsibility**: Show toast messages  
**Methods**:
```javascript
class ToastNotification {
  static show(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
  
  static success(message) { this.show(message, 'success'); }
  static error(message) { this.show(message, 'error'); }
  static info(message) { this.show(message, 'info'); }
}
```

---

## 2. API Integration Plan

### 2.1 API Endpoints

#### Endpoint Mapping

**Primary Endpoint** (Face Pipeline Service - Port 8001):
```
POST /api/v1/search/file
- Upload image + search in one request
- Query params: tenant_id, top_k, threshold
- Returns: SearchResponse with hits
```

**Alternative Endpoint** (Backend Service - Port 8000):
```
POST /api/compare_face
- Similar to above
- Query params: top_k, threshold
- Headers: X-Tenant-ID
- Returns: Response with results
```

**Additional Endpoints** (To be implemented or mocked):
```
GET /api/v1/search/{searchId}/results
- Retrieve search results by ID
- Returns: Cached search results

GET /api/v1/search/latest?userId={userId}
- Get latest search for user
- Returns: Latest SearchResponse or redirect

GET /api/v1/uploads/{uploadId}/search
- Get search results for upload
- Returns: SearchResponse or upload status

POST /api/v1/uploads/{uploadId}/search
- Trigger search for existing upload
- Returns: New SearchResponse

POST /api/v1/search/refresh-urls
- Refresh expired presigned URLs
- Body: { face_ids: [...] }
- Returns: Updated URLs
```

---

### 2.2 API Request Flow

#### Journey A: Load by searchId

```javascript
async function loadSearchById(searchId) {
  try {
    showLoading();
    
    // Call API
    const response = await apiClient.getSearchById(searchId);
    
    // Validate response
    if (!response || !response.hits) {
      throw new Error('Invalid response format');
    }
    
    // Store data
    state.searchData = response;
    state.filteredResults = response.hits;
    
    // Render components
    queryPanel.render(response.query);
    matchGrid.render(state.filteredResults);
    debugPanel.render(response);
    
    hideLoading();
  } catch (error) {
    hideLoading();
    
    if (error.status === 404) {
      errorState.render('search_not_found', searchId);
    } else if (error.status === 410) {
      errorState.render('search_expired', searchId);
    } else {
      errorState.render('api_error', error.message);
    }
  }
}
```

---

#### Journey B: Load latest by userId

```javascript
async function loadLatestByUser(userId) {
  try {
    showLoading();
    
    const response = await apiClient.getLatestSearch(userId);
    
    if (!response || response.count === 0) {
      // No searches found
      emptyState.render({
        title: 'No Searches Found',
        message: `No searches found for user ${userId}`,
        suggestions: ['Upload new image to search']
      });
      hideLoading();
      return;
    }
    
    // If API returns redirect, follow it
    if (response.redirectTo) {
      window.location.href = `/search?id=${response.redirectTo}`;
      return;
    }
    
    // Otherwise, display inline
    state.searchData = response;
    state.filteredResults = response.hits;
    
    queryPanel.render(response.query);
    matchGrid.render(state.filteredResults);
    debugPanel.render(response);
    
    hideLoading();
  } catch (error) {
    hideLoading();
    errorState.render('user_no_searches', userId);
  }
}
```

---

#### Journey C: Load by uploadId

```javascript
async function loadByUploadId(uploadId) {
  try {
    showLoading();
    
    const response = await apiClient.getSearchByUpload(uploadId);
    
    if (response.status === 'no_search') {
      // Upload exists but no search performed
      showUploadDetails(response.upload);
      showTriggerSearchButton(uploadId);
      hideLoading();
      return;
    }
    
    // Search exists, display results
    state.searchData = response;
    state.filteredResults = response.hits;
    
    queryPanel.render(response.query);
    matchGrid.render(state.filteredResults);
    debugPanel.render(response);
    
    hideLoading();
  } catch (error) {
    hideLoading();
    
    if (error.status === 404) {
      errorState.render('upload_not_found', uploadId);
    } else {
      errorState.render('api_error', error.message);
    }
  }
}
```

---

### 2.3 Presigned URL Refresh Strategy

```javascript
async function refreshExpiredUrls() {
  const expiredMatches = state.filteredResults.filter(match => {
    return match._urlExpired === true;
  });
  
  if (expiredMatches.length === 0) return;
  
  const faceIds = expiredMatches.map(m => m.face_id);
  
  try {
    const response = await apiClient.refreshPresignedUrls(faceIds);
    
    // Update URLs in state
    response.urls.forEach(updated => {
      const match = state.filteredResults.find(
        m => m.face_id === updated.face_id
      );
      if (match) {
        match.thumb_url = updated.thumb_url;
        match.image_url = updated.image_url;
        match._urlExpired = false;
      }
    });
    
    // Re-render affected cards
    matchGrid.render(state.filteredResults);
    
    ToastNotification.success('Image URLs refreshed');
  } catch (error) {
    ToastNotification.error('Failed to refresh URLs');
  }
}

// Auto-check for expired URLs every 5 minutes
setInterval(() => {
  checkAndRefreshUrls();
}, 5 * 60 * 1000);
```

---

### 2.4 Error Handling Strategy

```javascript
class APIError extends Error {
  constructor(status, message, details) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function handleAPICall(apiFunction, errorContext) {
  try {
    const response = await apiFunction();
    return response;
  } catch (error) {
    // Log error
    console.error(`API Error [${errorContext}]:`, error);
    
    // Map error to user-friendly message
    if (error.status === 404) {
      throw new APIError(404, 'Resource not found', errorContext);
    } else if (error.status === 429) {
      throw new APIError(429, 'Rate limit exceeded. Please wait.', errorContext);
    } else if (error.status >= 500) {
      throw new APIError(500, 'Server error. Please try again.', errorContext);
    } else if (!navigator.onLine) {
      throw new APIError(0, 'Network error. Check connection.', errorContext);
    } else {
      throw new APIError(
        error.status || 0, 
        error.message || 'Unknown error', 
        errorContext
      );
    }
  }
}
```

---

## 3. File Structure

```
frontend/
├── search.html                 # Main page (single file or split)
├── css/
│   ├── main.css               # Global styles
│   ├── components/
│   │   ├── query-panel.css
│   │   ├── match-grid.css
│   │   ├── filters.css
│   │   ├── pagination.css
│   │   └── debug-panel.css
│   └── states/
│       ├── loading.css
│       ├── error.css
│       └── empty.css
├── js/
│   ├── app.js                 # Main application entry
│   ├── components/
│   │   ├── PageController.js
│   │   ├── QueryPanel.js
│   │   ├── MatchGrid.js
│   │   ├── FilterPanel.js
│   │   ├── PaginationControl.js
│   │   ├── DebugPanel.js
│   │   ├── ErrorState.js
│   │   ├── EmptyState.js
│   │   └── LoadingState.js
│   ├── utils/
│   │   ├── APIClient.js
│   │   ├── URLManager.js
│   │   ├── ImageManager.js
│   │   ├── VirtualScroller.js
│   │   ├── ClipboardHelper.js
│   │   └── ToastNotification.js
│   └── config.js              # Configuration constants
└── assets/
    └── icons/                 # Optional: SVG icons
```

**Alternative: Single File Approach**
```
frontend/
└── search.html               # All-in-one file
    ├── <style> ... </style>  # Embedded CSS
    └── <script> ... </script> # Embedded JS
```

---

## 4. State Management

### 4.1 Global State Object

```javascript
const state = {
  // Journey context
  journey: null,              // 'searchId' | 'userId' | 'uploadId'
  journeyParams: {},          // { searchId, userId, uploadId }
  
  // Search data
  searchData: null,           // Full API response
  filteredResults: [],        // Filtered results
  displayResults: [],         // Paginated results
  
  // UI state
  viewMode: 'grid',           // 'grid' | 'list'
  isLoading: false,
  error: null,
  
  // Pagination
  currentPage: 1,
  pageSize: 25,
  totalResults: 0,
  totalPages: 0,
  
  // Filters
  filters: {
    minScore: 0.75,
    site: 'all',
    dateRange: null
  },
  
  // Available sites for filter dropdown
  availableSites: [],
  
  // Debug panel
  debugExpanded: false,
  timingMetrics: {
    apiDuration: 0,
    ttfb: 0,
    imageLoadTime: 0
  }
};
```

---

### 4.2 State Update Pattern

```javascript
function setState(updates) {
  // Merge updates into state
  Object.assign(state, updates);
  
  // Trigger re-render
  render();
}

function updateFilters(newFilters) {
  // Apply filters
  state.filteredResults = applyFilters(
    state.searchData.hits, 
    newFilters
  );
  
  // Reset to page 1
  state.currentPage = 1;
  
  // Update pagination
  state.totalResults = state.filteredResults.length;
  state.totalPages = getTotalPages(state.totalResults, state.pageSize);
  
  // Paginate filtered results
  state.displayResults = paginate(
    state.filteredResults,
    state.currentPage,
    state.pageSize
  );
  
  // Re-render
  render();
}

function changePage(page) {
  state.currentPage = page;
  
  // Paginate
  state.displayResults = paginate(
    state.filteredResults,
    state.currentPage,
    state.pageSize
  );
  
  // Update URL
  URLManager.updateParam('page', page);
  
  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
  
  // Re-render
  render();
}
```

---

## 5. Development Phases

### Phase 2.1: Basic Structure (Week 1)
- [ ] Create HTML structure
- [ ] Implement basic CSS layout
- [ ] Set up component skeleton (empty render methods)
- [ ] Implement URL parsing and routing
- [ ] Create mock data for testing

**Deliverable**: Static page with layout, no functionality

---

### Phase 2.2: Core Components (Week 1-2)
- [ ] Implement QueryPanel component
- [ ] Implement MatchGrid component (grid view)
- [ ] Implement basic rendering
- [ ] Add loading states
- [ ] Test with mock data

**Deliverable**: Static page showing query and results

---

### Phase 2.3: Filtering & Pagination (Week 2)
- [ ] Implement FilterPanel component
- [ ] Implement PaginationControl component
- [ ] Add client-side filtering logic
- [ ] Add pagination logic
- [ ] Connect filters to results

**Deliverable**: Functional filtering and pagination

---

### Phase 2.4: API Integration (Week 2-3)
- [ ] Implement APIClient utility
- [ ] Connect to real API endpoints
- [ ] Handle API responses
- [ ] Implement error handling
- [ ] Add retry logic

**Deliverable**: Real data loading from API

---

### Phase 2.5: Advanced Features (Week 3)
- [ ] Implement list view toggle
- [ ] Add bbox overlay on hover
- [ ] Implement virtual scrolling
- [ ] Add lazy image loading
- [ ] Implement presigned URL refresh

**Deliverable**: Full feature set working

---

### Phase 2.6: Polish & Debug (Week 3-4)
- [ ] Implement DebugPanel component
- [ ] Add all error/empty states
- [ ] Implement toast notifications
- [ ] Add keyboard shortcuts (optional)
- [ ] Performance optimization
- [ ] Cross-browser testing

**Deliverable**: Production-ready prototype

---

## 6. Configuration

### 6.1 Config File

```javascript
// js/config.js
const CONFIG = {
  // API Configuration
  API: {
    BASE_URL: 'http://localhost:8001',
    ENDPOINTS: {
      SEARCH_FILE: '/api/v1/search/file',
      SEARCH_BY_ID: '/api/v1/search/{id}/results',
      LATEST_SEARCH: '/api/v1/search/latest',
      UPLOAD_SEARCH: '/api/v1/uploads/{id}/search',
      REFRESH_URLS: '/api/v1/search/refresh-urls'
    },
    TIMEOUT: 30000,           // 30 seconds
    RETRY_COUNT: 3
  },
  
  // Default values
  DEFAULTS: {
    TENANT_ID: 'demo-tenant',
    TOP_K: 50,
    THRESHOLD: 0.75,
    PAGE_SIZE: 25,
    VIEW_MODE: 'grid'
  },
  
  // UI Configuration
  UI: {
    GRID_COLUMNS: {
      DESKTOP: 5,
      TABLET: 3,
      MOBILE: 1
    },
    PAGE_SIZE_OPTIONS: [10, 25, 50, 100],
    VIRTUAL_SCROLL: {
      ENABLED: true,
      ITEM_HEIGHT: 230,        // Grid card height
      BUFFER_SIZE: 10          // Extra items to render
    },
    LAZY_LOAD: {
      ENABLED: true,
      ROOT_MARGIN: '200px'     // Start loading 200px before visible
    }
  },
  
  // Performance
  PERFORMANCE: {
    DEBOUNCE_DELAY: 300,       // ms
    THROTTLE_DELAY: 100,       // ms
    IMAGE_LOAD_TIMEOUT: 5000,  // ms
    URL_REFRESH_INTERVAL: 300000  // 5 minutes
  },
  
  // Score thresholds for color coding
  SCORE_THRESHOLDS: {
    HIGH: 0.80,   // Green
    MEDIUM: 0.60  // Yellow (below = red)
  },
  
  // Presigned URL TTL warning threshold
  URL_EXPIRY_WARNING: 60000,   // 1 minute before expiry
  
  // Debug mode
  DEBUG: true
};
```

---

## 7. Performance Optimization Plan

### 7.1 Virtual Scrolling
- Only render visible items + buffer
- Target: Smooth 60 FPS scrolling with 500+ results
- Use `requestAnimationFrame` for scroll updates

### 7.2 Lazy Loading
- Load images only when entering viewport
- Use Intersection Observer API
- Placeholder images during load

### 7.3 Debouncing
- Filter inputs debounced (300ms)
- Scroll events throttled (100ms)
- Resize events debounced (300ms)

### 7.4 Caching
- Cache filtered results
- Cache paginated slices
- Cache presigned URLs (with expiry tracking)

### 7.5 Bundle Size
- Single HTML file: < 100KB (gzipped)
- Or separate files: HTML (10KB) + CSS (20KB) + JS (50KB)
- No external dependencies initially

---

## 8. Testing Approach

### 8.1 Manual Testing Checklist
- [ ] Journey A: Load by searchId (valid)
- [ ] Journey A: Load by searchId (not found)
- [ ] Journey A: Load by searchId (expired)
- [ ] Journey B: Load by userId (has searches)
- [ ] Journey B: Load by userId (no searches)
- [ ] Journey C: Load by uploadId (has search)
- [ ] Journey C: Load by uploadId (no search)
- [ ] Journey C: Load by uploadId (not found)
- [ ] Filter by similarity score
- [ ] Filter by site
- [ ] Filter by date range
- [ ] Pagination (next/prev)
- [ ] Pagination (page numbers)
- [ ] Change page size
- [ ] View source link
- [ ] Copy face ID
- [ ] Toggle grid/list view
- [ ] Expand debug panel
- [ ] Copy JSON from debug panel
- [ ] Expired URL handling
- [ ] Error states display correctly
- [ ] Empty states display correctly
- [ ] Loading states display correctly

### 8.2 Browser Testing
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

### 8.3 Responsive Testing
- [ ] Desktop (1920x1080)
- [ ] Laptop (1366x768)
- [ ] Tablet (768x1024)
- [ ] Mobile (375x667) - if in scope

---

## 9. Acceptance Criteria

### AC1: Core Functionality
- [ ] All three user journeys work correctly
- [ ] Query panel displays image and metadata
- [ ] Results grid displays all matches
- [ ] Filters work (score, site, date)
- [ ] Pagination works correctly
- [ ] Debug panel shows API response

### AC2: Performance
- [ ] Page loads in < 2 seconds (with mock data)
- [ ] Virtual scrolling maintains 60 FPS
- [ ] Filters apply in < 100ms
- [ ] Images load progressively (lazy loading works)

### AC3: Error Handling
- [ ] All error states display correctly
- [ ] Empty states display correctly
- [ ] Network errors handled gracefully
- [ ] Expired URLs detected and handled

### AC4: User Experience
- [ ] UI matches approved wireframes
- [ ] All controls are intuitive
- [ ] Loading states prevent confusion
- [ ] Toast notifications provide feedback

### AC5: Code Quality
- [ ] Code is modular (components separated)
- [ ] Code is commented
- [ ] No console errors
- [ ] Config externalized

---

## 10. Mock Data Strategy

### 10.1 Mock Search Response

```javascript
// js/mockData.js
const MOCK_SEARCH_RESPONSE = {
  query: {
    tenant_id: "demo-tenant",
    search_mode: "image",
    top_k: 50,
    threshold: 0.75,
    uploaded_at: "2024-01-15T14:32:05Z"
  },
  hits: [
    {
      face_id: "face-001",
      score: 0.952,
      payload: {
        site: "example.com",
        url: "https://example.com/images/photo1.jpg",
        ts: "2024-01-15T10:30:00Z",
        bbox: [100, 150, 200, 250],
        p_hash: "a1b2c3d4e5f6g7h8",
        quality: 0.92
      },
      thumb_url: "https://minio.local/thumbnails/demo-tenant/face-001_thumb.jpg",
      image_url: "https://minio.local/thumbnails/demo-tenant/face-001_thumb.jpg"
    },
    // ... more results
  ],
  count: 156
};

// Generate more mock data
function generateMockResults(count = 100) {
  const sites = ['example.com', 'test-site.org', 'demo-site.net'];
  const results = [];
  
  for (let i = 0; i < count; i++) {
    results.push({
      face_id: `face-${String(i).padStart(3, '0')}`,
      score: 0.95 - (i * 0.005),  // Decreasing scores
      payload: {
        site: sites[i % sites.length],
        url: `https://${sites[i % sites.length]}/image-${i}.jpg`,
        ts: new Date(Date.now() - i * 3600000).toISOString(),
        bbox: [100 + i, 150 + i, 200, 250],
        p_hash: `hash${i}`,
        quality: 0.85 + Math.random() * 0.15
      },
      thumb_url: `https://via.placeholder.com/150?text=Face+${i}`,
      image_url: `https://via.placeholder.com/150?text=Face+${i}`
    });
  }
  
  return results;
}
```

---

## 11. Next Steps

After Phase 2 planning approval:

1. **Set up development environment**
   - Create file structure
   - Set up local server (optional: `python -m http.server`)
   - Configure API proxy if needed

2. **Start with Phase 2.1**
   - Create HTML structure
   - Basic CSS layout
   - Component skeleton

3. **Iterate through phases 2.2 - 2.6**
   - Build one component at a time
   - Test with mock data first
   - Integrate with real API incrementally

4. **Weekly check-ins**
   - Demo progress
   - Get feedback
   - Adjust as needed

---

## Sign-off

### Tech Lead
- **Name**: _______________________
- **Date**: _______________________
- **Signature**: "I approve this implementation plan and confirm it's technically sound."

### Development Team
- **Name**: _______________________
- **Date**: _______________________
- **Signature**: "We understand the plan and can execute it within the proposed timeline."

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-10  
**Prerequisites**: Phase 0 & Phase 1 approved  
**Next Phase**: Phase 2 Implementation (Phases 2.1-2.6)  
**Estimated Timeline**: 3-4 weeks

