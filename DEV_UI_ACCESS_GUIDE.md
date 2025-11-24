# 🎨 Dev UI Access Guide - What You Have Access To

**Generated**: November 14, 2025  
**Status**: ✅ All Services Running

---

## 🌐 Access URLs

### Primary Development Interface

**🎯 Main Dev Page** (React App - Phase 0-7):
- **URL**: http://localhost:5173/dev/search
- **Status**: ✅ Running
- **Description**: Full-featured React application with all Phase 0-7 features

**Alternative Access** (via Docker Nginx):
- **URL**: http://localhost (when Docker is running)
- **Status**: ✅ Running
- **Description**: Production build served through Nginx

---

## 🎯 What You Have Access To in the Dev UI

### 1. Search Dev Page (`/dev/search`)

**URL**: http://localhost:5173/dev/search

#### Features Available

##### 📊 Query Panel
- **Query Image Display**: Shows the uploaded/search image
- **Image Metadata**: Displays image information
- **Bounding Box Overlay**: Visual face detection overlay
- **Always Visible**: Stays at top for reference

##### 🔍 Filters Panel (Phase 7)

**Min Score Slider**:
- **Range**: 0-100% (0.0 to 1.0)
- **Default**: 0 (shows all results)
- **Visual Feedback**: Gradient visualization
- **Debouncing**: 300ms delay for smooth UX
- **URL Sync**: Updates URL parameter `minScore`

**Site Filter**:
- **Dropdown**: Filter by source site
- **Options**: All detected sites from results
- **Default**: "All Sites"
- **URL Sync**: Updates URL parameter `site`

**Results Counter**:
- **Display**: "X of Y results"
- **Updates**: Real-time as filters change
- **Location**: Below filters

**Reset Filters Button**:
- **Action**: Resets all filters to defaults
- **Clears**: minScore, site filter
- **Resets**: Page to 1

##### 📄 Results Display

**Grid View** (Default):
- **Layout**: Responsive grid (5 columns desktop, 3 tablet, 1 mobile)
- **Cards**: Each result in a card with:
  - Thumbnail image
  - Score badge (color-coded)
  - Site label
  - Quick actions (View Source, Copy ID)
  - Bounding box overlay on hover
- **Best For**: Browsing many results quickly

**List View**:
- **Layout**: Vertical list with more details
- **Details**: Shows all metadata per result
- **Features**: 
  - Face ID
  - Score
  - Quality metrics
  - Timestamp
  - Full metadata
- **Best For**: Detailed analysis

**View Toggle**:
- **Buttons**: Grid/List toggle buttons
- **Location**: Top right controls
- **URL Sync**: Updates URL parameter `view`
- **Persistent**: Remembers preference in URL

##### 📊 Pagination (Phase 7)

**Page Navigation**:
- **First Page**: Jump to page 1
- **Previous**: Go to previous page
- **Next**: Go to next page
- **Last Page**: Jump to last page
- **Page Numbers**: Click to jump to specific page
- **Ellipsis**: Shows when pages are hidden

**Page Size Control**:
- **Options**: 10, 25, 50, 100 results per page
- **Default**: 25
- **Location**: Pagination controls
- **URL Sync**: Updates URL parameter `pageSize`
- **Auto-Reset**: Resets to page 1 when changed

**Page Info**:
- **Display**: "Showing 1-25 of 100 results"
- **Updates**: Real-time as filters/pagination change

**Jump to Page**:
- **Input**: Direct page number entry
- **Validation**: Ensures valid page number
- **Action**: Navigates to specified page

##### 🔗 URL State Synchronization (Phase 7)

**Deep-Linking**:
- **All State in URL**: Filters, pagination, view mode
- **Shareable**: Copy/paste URLs to share exact state
- **Persistent**: State persists across page reloads
- **Browser History**: Works with back/forward buttons

**URL Parameters**:
```
/dev/search?minScore=0.75&site=example.com&page=2&pageSize=50&view=list
```

**Parameters Available**:
- `minScore`: 0-1 (minimum match score)
- `site`: domain string (filter by site)
- `page`: ≥1 (current page number)
- `pageSize`: 10, 25, 50, 100 (results per page)
- `view`: 'grid' or 'list' (view mode)

**Copy URL Button**:
- **Location**: Header/controls area
- **Action**: Copies current URL to clipboard
- **Use Case**: Share exact search state

##### 🎨 UI Components (All Phases)

**Result Cards** (Phase 6):
- **SafeImage**: Image loading with error handling
- **ScoreBadge**: Color-coded match scores
- **DistanceChip**: Distance metrics display
- **BBoxOverlay**: Face bounding box visualization
- **Quality Indicators**: Image quality metrics

**State Components**:
- **LoadingState**: Loading spinner and message
- **EmptyState**: No results message
- **ErrorState**: Error messages with retry

**Query Image** (Phase 6):
- **Upload Display**: Shows uploaded image
- **Metadata**: Image information
- **Bounding Box**: Face detection overlay

##### 🐛 Debug Features

**Browser Console**:
- **Logs**: API calls, state changes, errors
- **Performance**: Timing metrics
- **Network**: Request/response details

**React DevTools** (if installed):
- **Component Tree**: Inspect React components
- **State Inspection**: View component state
- **Props**: See component props
- **Performance**: Profile component renders

---

### 2. Mock Server Interface

**URL**: http://localhost:8000/docs

#### API Documentation (Swagger UI)
- **Interactive API Docs**: Full Swagger UI interface
- **Try It Out**: Test API endpoints directly
- **Request/Response**: See full API schemas
- **Authentication**: See required headers

#### Mock Server Endpoints

**Health Check**:
- **URL**: http://localhost:8000/api/v1/health
- **Method**: GET
- **Response**: Service status and configuration

**Search Endpoint**:
- **URL**: http://localhost:8000/api/v1/search
- **Method**: POST
- **Description**: Face search with image upload
- **Fixtures**: tiny, medium, large, edge_cases, errors

**Search by ID**:
- **URL**: http://localhost:8000/api/v1/search-by-id
- **Method**: GET
- **Description**: Retrieve search results by ID

**Mock Fixtures**:
- **URL**: http://localhost:8000/mock/fixtures
- **Method**: GET
- **Description**: List available fixture datasets

**Mock Config**:
- **URL**: http://localhost:8000/mock/config
- **Method**: GET/POST
- **Description**: Configure mock server behavior

---

### 3. Docker Services (via Nginx)

**URL**: http://localhost

#### Available Services

**Frontend** (Production Build):
- **URL**: http://localhost
- **Description**: Production React build served by Nginx

**Backend API**:
- **URL**: http://localhost/api
- **Description**: Main backend API service
- **Health**: http://localhost/api/healthz

**Face Pipeline**:
- **URL**: http://localhost/pipeline
- **Description**: Face recognition pipeline service
- **Health**: http://localhost/pipeline/api/v1/health

**MinIO Console**:
- **URL**: http://localhost:9001
- **Credentials**: minioadmin / minioadmin
- **Description**: Object storage management console
- **Purpose**: Manage image storage buckets

**Qdrant Dashboard**:
- **URL**: http://localhost:6333/dashboard
- **Description**: Vector database management
- **Purpose**: View face embeddings and collections
- **Health**: http://localhost:6333/readyz

---

## 🎯 What You Can Do

### 1. Search Face Images

**Upload Image**:
- Upload a face image to search
- See query image displayed
- View bounding box overlay

**View Results**:
- See matching faces in grid or list view
- Filter by match score
- Filter by source site
- Navigate through pages

**Interact with Results**:
- Click to view source image
- Copy face IDs
- See match scores
- View metadata

### 2. Filter & Navigate

**Filter Results**:
- Adjust min score slider (0-100%)
- Select site from dropdown
- See filtered result count
- Reset filters easily

**Navigate Pages**:
- Change page size (10, 25, 50, 100)
- Navigate pages (First, Prev, Next, Last)
- Jump to specific page
- See page information

### 3. Customize View

**View Modes**:
- Switch between Grid and List views
- Grid: Quick browsing
- List: Detailed analysis

**URL Sharing**:
- Copy URL with current state
- Share exact search state
- Deep-link to specific results

### 4. Test & Debug

**Mock Data**:
- Test with different fixture datasets
- Test error scenarios
- Test edge cases

**API Testing**:
- Use Swagger UI to test API
- See request/response formats
- Test different endpoints

**Debug Tools**:
- Browser console for logs
- React DevTools for components
- Network tab for API calls

---

## 📋 Quick Reference

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Dev UI** | http://localhost:5173/dev/search | Main development interface |
| **Mock Server** | http://localhost:8000/docs | API documentation |
| **Production** | http://localhost | Production build |
| **MinIO** | http://localhost:9001 | Object storage console |
| **Qdrant** | http://localhost:6333/dashboard | Vector database dashboard |

### Features Summary

✅ **Query Image Display**  
✅ **Results Grid/List View**  
✅ **Min Score Filtering** (0-100%)  
✅ **Site Filtering**  
✅ **Pagination** (First/Prev/Next/Last)  
✅ **Page Size Control** (10/25/50/100)  
✅ **URL State Sync** (deep-linking)  
✅ **Copy URL**  
✅ **Reset Filters**  
✅ **Bounding Box Overlays**  
✅ **Score Badges**  
✅ **Loading/Error/Empty States**  
✅ **Responsive Design**  

---

## 🚀 Getting Started

### 1. Open Dev UI

```
http://localhost:5173/dev/search
```

### 2. Try Features

1. **Adjust Min Score Slider**: See results filter in real-time
2. **Select Site Filter**: Filter by specific source site
3. **Switch Views**: Toggle between Grid and List
4. **Navigate Pages**: Use pagination controls
5. **Change Page Size**: Try different page sizes
6. **Copy URL**: Share current state
7. **Reset Filters**: Clear all filters

### 3. Test URL Sync

1. Adjust filters and pagination
2. Check URL updates automatically
3. Copy URL and open in new tab
4. Verify state is preserved
5. Use browser back/forward buttons

### 4. Explore API

1. Open http://localhost:8000/docs
2. Try API endpoints
3. Test with different fixtures
4. See request/response formats

---

## 🎨 UI Features by Phase

### Phase 0: Dev Search Page
- ✅ Basic page structure
- ✅ Mock data display
- ✅ Responsive layout

### Phase 4: Non-Functional Shell
- ✅ Loading states
- ✅ Empty states
- ✅ Error states
- ✅ Safe image loading

### Phase 5: Image Safety
- ✅ Safe image component
- ✅ Error handling
- ✅ Fallback images

### Phase 6: Results Rendering
- ✅ Result cards
- ✅ List items
- ✅ Bounding box overlays
- ✅ Score badges
- ✅ Query image display

### Phase 7: Filters & Pagination
- ✅ Min score slider
- ✅ Site filter
- ✅ Pagination controls
- ✅ URL state sync
- ✅ Copy URL button
- ✅ Reset filters

---

## 📊 Mock Data

**Available Fixtures**:
- **tiny**: 10 results (quick testing)
- **medium**: 200 results (default)
- **large**: 2000 results (stress testing)
- **edge_cases**: 15 results (boundary testing)
- **errors**: 20 results (error handling)

**Current Dev UI**: Uses 100 mock results with scores from 0.95 to 0.15

---

## 🔧 Configuration

### Frontend Configuration

**Vite Dev Server**:
- Port: 5173
- Hot Module Replacement: Enabled
- Source Maps: Enabled
- Proxy: Configured for API calls

### Mock Server Configuration

**Default Settings**:
- Port: 8000
- Default Fixture: medium
- Latency Simulation: Enabled (50-300ms)
- Error Rate: 0%

**Change Settings**:
```bash
curl -X POST http://localhost:8000/mock/config \
  -H "Content-Type: application/json" \
  -d '{"default_fixture": "large", "simulate_latency": false}'
```

---

## 🎉 Summary

**You have access to**:
- ✅ Full-featured React Dev UI (Phase 0-7)
- ✅ All filtering and pagination features
- ✅ URL state synchronization
- ✅ Grid and List view modes
- ✅ Mock server with API documentation
- ✅ Docker services (Nginx, API, Pipeline, MinIO, Qdrant)
- ✅ All Phase 0-7 UI components
- ✅ Debug tools and console logs

**Everything is ready for development and testing!** 🚀

---

**Last Updated**: November 14, 2025  
**Status**: All services running and accessible


