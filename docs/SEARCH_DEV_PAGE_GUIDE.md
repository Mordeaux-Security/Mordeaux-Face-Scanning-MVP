# Dev Search Page - Complete User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features Overview](#features-overview)
3. [User Journeys](#user-journeys)
4. [Controls & Actions](#controls--actions)
5. [Debug Panel](#debug-panel)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [API Integration](#api-integration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)
10. [Developer Notes](#developer-notes)

---

## Quick Start

### Access the Page

1. **Via Docker** (if running):
   ```
   http://localhost:3000/search-dev.html
   ```

2. **Via Local Server**:
   ```bash
   cd frontend
   python -m http.server 3000
   # Visit: http://localhost:3000/search-dev.html
   ```

3. **Direct File** (for testing):
   ```bash
   open frontend/search-dev.html
   ```

### First Use

1. Open the page (default shows mock data with 156 results)
2. Try adjusting the similarity slider
3. Switch between Grid and List views
4. Click page numbers to navigate
5. Expand the Debug Panel to see API response

---

## Features Overview

### 🖼️ Query Panel
- **Query Image**: Shows the image you searched with
- **Metadata**: Upload time, tenant, parameters, backend
- **Always Visible**: Stays at the top for reference

### 📊 Match Grid View
- **Responsive Grid**: 5 columns (desktop), 3 (tablet), 1 (mobile)
- **Score Badges**: Color-coded (green/yellow/red)
- **BBox Overlay**: Hover to see face bounding box
- **Quick Actions**: View source, copy ID

### 📋 Match List View
- **Detailed View**: More metadata per result
- **Better for Analysis**: Shows all fields
- **Face IDs**: Easily visible
- **Quality Scores**: Displayed prominently

### 🔍 Client-Side Filtering
- **No API Calls**: Instant filtering
- **Similarity Slider**: 0.00 - 1.00 range
- **Site Dropdown**: Filter by specific sites
- **Auto-Update**: Results update live

### 📄 Pagination
- **Page Sizes**: 10, 25, 50, 100 results per page
- **Smart Navigation**: Prev/Next + page numbers
- **Shows Progress**: "Showing X-Y of Z results"
- **Smooth Scrolling**: Auto-scroll to top

### 🐛 Debug Panel
- **Expandable**: Click header to open/close
- **Timing Metrics**: API duration, TTFB, load times
- **Full Response**: JSON with syntax highlighting
- **Export Options**: Copy or download JSON

---

## User Journeys

### Journey A: Search by ID

**URL Pattern**: `?id={searchId}`

**Example**:
```
http://localhost:3000/search-dev.html?id=search-abc-123
```

**When to Use**:
- You have a search ID from logs/audit
- Reviewing historical searches
- Debugging specific search results

**What Happens**:
1. Page loads with search ID in header
2. API call to fetch search results (or mock data)
3. Query panel shows query metadata
4. Results displayed in grid

---

### Journey B: Latest Search by User

**URL Pattern**: `?userId={userId}`

**Example**:
```
http://localhost:3000/search-dev.html?userId=user-456
```

**When to Use**:
- Checking latest activity for a user
- Debugging user issues
- QA testing user flows

**What Happens**:
1. Page loads with user ID in header
2. API fetches latest search for user
3. Displays most recent search results
4. Or redirects to specific search ID

---

### Journey C: Search by Upload

**URL Pattern**: `?uploadId={uploadId}`

**Example**:
```
http://localhost:3000/search-dev.html?uploadId=upload-789
```

**When to Use**:
- Tracing upload → search flow
- Debugging upload processing
- Verifying upload results

**What Happens**:
1. Page loads with upload ID in header
2. API fetches search results for upload
3. Shows upload metadata + results
4. If no search: Shows option to trigger

---

### Journey D: Default (Mock Data)

**URL Pattern**: No parameters

**Example**:
```
http://localhost:3000/search-dev.html
```

**When to Use**:
- Testing the page
- Exploring features
- Development/debugging

**What Happens**:
1. Page loads with mock data (156 results)
2. All features available
3. Warning toast shows "Using mock data"
4. Perfect for testing UI/UX

---

## Controls & Actions

### Similarity Slider

**Location**: Top controls bar  
**Range**: 0.00 - 1.00  
**Default**: 0.75

**How to Use**:
1. Drag slider left (lower threshold) or right (higher threshold)
2. Current value shows next to slider
3. Results filter automatically (debounced 300ms)
4. Lower = more results, Higher = fewer but better matches

**Tips**:
- Start at 0.75 for good matches
- Lower to 0.50 for more results
- Raise to 0.85+ for only very close matches

---

### Site Filter

**Location**: Top controls bar  
**Options**: All Sites + auto-detected sites  
**Default**: All Sites

**How to Use**:
1. Click dropdown to see available sites
2. Select a specific site to filter
3. Results update instantly
4. Shows count per site (when implemented)

**Tips**:
- Use to focus on specific sources
- Combine with similarity filter
- "All Sites" shows everything

---

### View Toggle

**Location**: Top controls bar (right side)  
**Options**: Grid, List  
**Default**: Grid

**Grid View**:
- Best for: Browsing many results
- Shows: Thumbnails in responsive grid
- Actions: Quick view/copy buttons

**List View**:
- Best for: Detailed analysis
- Shows: All metadata fields
- Actions: Full action buttons

---

### Pagination

**Location**: Below results  
**Controls**: Prev, Next, Page Numbers, Page Size

**Page Size**:
- Options: 10, 25, 50, 100
- Default: 25
- Resets to page 1 when changed

**Navigation**:
- **Prev/Next**: Move one page
- **Page Numbers**: Jump to specific page
- **Ellipsis (...)**: Indicates hidden pages

---

### Actions

#### View Source
**Button**: 🔗 Source  
**What it Does**: Opens original source URL in new tab  
**When Disabled**: URL not available

#### Copy ID
**Button**: 📋 Copy  
**What it Does**: Copies face_id to clipboard  
**Feedback**: Toast notification

#### Copy JSON
**Button**: 📋 Copy JSON (in Debug Panel)  
**What it Does**: Copies full API response to clipboard

#### Download JSON
**Button**: 💾 Download JSON (in Debug Panel)  
**What it Does**: Downloads API response as .json file

---

## Debug Panel

### Opening the Debug Panel

1. Scroll to bottom of page
2. Click "🐛 Debug Information" header
3. Panel expands to show contents
4. Click again to collapse

### Timing Metrics

**API Duration**: Total time for API call  
**TTFB**: Time to First Byte from server  
**Image Load Time**: Average image load time

**What's Good**:
- API Duration: < 500ms
- TTFB: < 200ms
- Image Load: < 200ms per image

### API Response

**Shows**: Full JSON response from API  
**Format**: Syntax highlighted, scrollable  
**Max Height**: 400px (scrolls if longer)

**What You See**:
- Query parameters
- All hits/results
- Metadata for each result
- Count and other info

### Export Options

**Copy JSON**: Quick copy to clipboard  
**Download JSON**: Save as file with timestamp

**Use Cases**:
- Share results with team
- Archive search results
- Debug API responses
- Compare searches

---

## Keyboard Shortcuts

### Navigation
- **Arrow Left**: Previous page (when implemented)
- **Arrow Right**: Next page (when implemented)
- **Home**: First page (when implemented)
- **End**: Last page (when implemented)

### Actions
- **Escape**: Close modals/panels (when implemented)
- **Ctrl/Cmd + C**: Copy selected (when implemented)

### Views
- **G**: Switch to Grid view (when implemented)
- **L**: Switch to List view (when implemented)

*Note: Keyboard shortcuts are placeholders for future implementation*

---

## API Integration

### API Endpoints Used

#### Search by ID
```
GET /api/v1/search/{searchId}/results
Headers: X-Tenant-ID
```

#### Latest Search by User
```
GET /api/v1/search/latest?userId={userId}
Headers: X-Tenant-ID
```

#### Search by Upload
```
GET /api/v1/uploads/{uploadId}/search
Headers: X-Tenant-ID
```

### Fallback Behavior

If API is not available:
1. Page attempts API call first
2. On failure: Falls back to mock data
3. Shows warning toast: "Using mock data (API not available)"
4. All features work with mock data

### Timeout Handling

- **Default Timeout**: 30 seconds
- **Behavior**: Falls back to mock data on timeout
- **User Feedback**: Toast notification

---

## Troubleshooting

### Issue: Page Shows No Results

**Possible Causes**:
1. Filters are too restrictive
2. No matches in database
3. API error

**Solutions**:
1. Lower similarity threshold (try 0.50)
2. Select "All Sites" in site filter
3. Check Debug Panel for API errors
4. Click "Reset Filters" button

---

### Issue: Images Not Loading

**Possible Causes**:
1. Presigned URLs expired (> 10 minutes old)
2. Network connectivity
3. CORS issues

**Solutions**:
1. Refresh the page to get new URLs
2. Check browser console for errors
3. Verify API is accessible

---

### Issue: Slow Performance

**Possible Causes**:
1. Too many results displayed
2. Large images
3. Browser memory

**Solutions**:
1. Reduce page size (try 25 or 10)
2. Use Grid view instead of List view
3. Close other tabs/applications
4. Refresh the page

---

### Issue: API Not Responding

**Possible Causes**:
1. API service down
2. Network issues
3. Incorrect configuration

**Solutions**:
1. Check if Docker containers are running
2. Verify API_BASE_URL in config
3. Page will fall back to mock data automatically

---

## Performance Tips

### For Best Performance

1. **Use Grid View**: Lighter than List view
2. **Reduce Page Size**: 25 or fewer results
3. **Filter Early**: Apply filters before browsing
4. **Close Debug Panel**: When not needed
5. **Use Modern Browser**: Chrome/Firefox/Edge latest

### Virtual Scrolling (Coming Soon)

For 500+ results:
- Renders only visible items
- Maintains smooth 60 FPS
- Reduces memory usage
- Automatic when enabled

---

## Developer Notes

### Mock Data

**Generated Data**:
- 156 total results
- Scores: 0.95 down to 0.40
- 4 different sites (rotating)
- Realistic timestamps (hourly)
- Placeholder images

**Timing Metrics** (Simulated):
- API Duration: 245ms
- TTFB: 120ms
- Image Load: 85ms

### Configuration

Edit `CONFIG` object in HTML:

```javascript
const CONFIG = {
  API: {
    BASE_URL: 'http://localhost:8001',  // Change this
    TIMEOUT: 30000
  },
  DEFAULTS: {
    TENANT_ID: 'demo-tenant',  // Change this
    TOP_K: 50,
    THRESHOLD: 0.75,
    PAGE_SIZE: 25
  }
};
```

### State Management

All state in `state` object:

```javascript
state = {
  journey: 'searchId' | 'userId' | 'uploadId' | 'default',
  searchData: { query, hits, count },
  allResults: [...],      // All results
  filteredResults: [...], // After filters
  displayResults: [...],  // Current page
  viewMode: 'grid' | 'list',
  currentPage: 1,
  pageSize: 25,
  filters: { minScore, site }
};
```

### Adding Features

1. **New Filter**: Add to `filters` object, create UI, update `applyFilters()`
2. **New Action**: Add button in card, create handler function
3. **New View Mode**: Add toggle button, create render function

---

## Best Practices

### For Developers

1. **Always Check Debug Panel**: See actual API responses
2. **Use Mock Data First**: Test UI before API integration
3. **Monitor Performance**: Check timing metrics
4. **Test All Journeys**: Verify searchId, userId, uploadId flows
5. **Test Error States**: Try invalid IDs to see error handling

### For Testers

1. **Test Filters**: Try all combinations
2. **Test Pagination**: Navigate all pages
3. **Test Both Views**: Grid and List
4. **Test Actions**: View Source, Copy ID
5. **Test Responsiveness**: Resize browser window

### For Users

1. **Start with Default**: Get familiar with mock data
2. **Use Filters**: Narrow down results efficiently
3. **Check Debug Panel**: Understand API responses
4. **Copy IDs**: Save face_ids for reference
5. **Export JSON**: Archive important searches

---

## Support & Resources

### Documentation Files

- **Phase 0**: `docs/PHASE_0_DEV_SEARCH_PAGE.md` - Requirements
- **Phase 1**: `docs/PHASE_1_USER_JOURNEYS_WIREFRAMES.md` - Design
- **Phase 2**: `docs/PHASE_2_IMPLEMENTATION_PLAN.md` - Technical plan
- **Complete**: `docs/PHASE_2_IMPLEMENTATION_COMPLETE.md` - Implementation summary

### Test Suite

Access at: `http://localhost:3000/search-dev-test.html`

Runs automated tests for:
- Unit tests (filtering, pagination, etc.)
- Integration tests (API, error handling)
- UI tests (rendering, controls)
- Performance tests (timing benchmarks)

### API Documentation

See: `docs/api.md` for full API reference

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-10  
**Page Version**: Search Dev v1.0  
**Status**: Complete and Ready for Use

