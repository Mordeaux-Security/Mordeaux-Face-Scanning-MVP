# Developer Runbook
## Mordeaux Face Scanning MVP - Frontend

**Version:** 1.0  
**Last Updated:** November 15, 2025

---

## Quick Start

### Prerequisites
- Node.js 18+ and npm 9+
- Git
- Modern browser (Chrome/Firefox/Safari/Edge)

### Initial Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd Mordeaux-Face-Scanning-MVP

# 2. Install frontend dependencies
cd frontend
npm install

# 3. Configure environment
cp .env.example .env
# Edit .env if needed

# 4. Start development server
npm run dev
```

### Access Points
- **Dev Search Page:** http://localhost:5173/dev/search
- **Test Page:** http://localhost:5173/test
- **Mock API:** http://localhost:8000/docs

---

## Environment Configuration

### `.env` Variables

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_REAL_API=false

# Dev Mode (required for dev features)
VITE_DEV_MODE=true

# Feature Flags
VITE_ENABLE_VIRTUALIZATION=true
VITE_ENABLE_DEBUG_PANEL=true
VITE_ENABLE_REDACTION_TOGGLE=true

# Tenant
VITE_DEFAULT_TENANT_ID=demo-tenant

# Performance
VITE_MAX_RESULTS=2000
VITE_DEFAULT_PAGE_SIZE=25
```

### Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `VITE_DEV_MODE` | Enable dev-only features | `true` |
| `VITE_USE_REAL_API` | Use real backend vs mock | `false` |
| `VITE_ENABLE_VIRTUALIZATION` | Allow virtualization toggle | `true` |
| `VITE_ENABLE_DEBUG_PANEL` | Show debug panel | `true` |
| `VITE_ENABLE_REDACTION_TOGGLE` | Allow data reveal | `true` |

---

## Development Workflow

### Running the Dev Server
```bash
cd frontend
npm run dev

# Server starts at http://localhost:5173
```

### Building for Production
```bash
npm run build

# Output in dist/
```

### Preview Production Build
```bash
npm run preview

# Serves from dist/
```

---

## Testing a Search

### Manual Test with Mock Data

1. Navigate to http://localhost:5173/dev/search
2. Click "Show Results" in demo controls
3. Verify 100 mock results appear
4. Test filters and pagination

### With Real Image (Mock Server)

1. Ensure mock server running
2. (Future) Upload image via form
3. View results

---

## Features Guide

### Phase 1-8: Core Features

**Grid/List View**
- Toggle between card grid and list
- State syncs to URL (`?view=grid|list`)

**Filters**
- Min Score Slider: Filter by similarity (0-1)
- Site Filter: Filter by domain
- Both sync to URL

**Pagination**
- Navigate pages with controls
- Change page size (10/25/50/100)
- State syncs to URL

**Safe Links**
- All external links validated
- Blocked: `javascript:`, `data:` URLs
- Security attributes applied

**Storage Indicators**
- Shows MinIO/S3/External chips
- Helps identify data source

### Phase 9: Performance

**Virtualization**
- Toggle in UI: "Use Virtualization"
- Renders all results in viewport
- Smooth at 2,000+ results
- Disables pagination

**Performance Monitoring**
- Open console for metrics
- `[Performance]` logs show timing
- Tracks render/filter duration

### Phase 10: Security

**Dev Route Guard**
- Protects `/dev/*` routes
- Checks `VITE_DEV_MODE` flag
- Can use localStorage override:
  ```js
  localStorage.setItem('ENABLE_DEV_MODE', 'true')
  ```

**Data Redaction**
- Toggle: "🔓 Reveal Sensitive Data"
- Masks IPs, emails, tokens
- Dev-only feature

### Phase 11: Observability

**Debug Panel**
- Click button at bottom-right
- Tabs: Metrics, Events, Logs
- Export logs to JSON
- Clear all data

**Structured Logging**
- Check console for logs
- Format: `[timestamp] [level] [context] event`
- Levels: DEBUG, INFO, WARN, ERROR

### Phase 12: Accessibility

**Keyboard Navigation**
- Tab through interactive elements
- Enter/Space to activate
- Skip link (Tab from top)

**Screen Reader Support**
- All images have alt text
- ARIA labels on controls
- Live regions for updates

**Responsive**
- Mobile: 1 column
- Tablet: 2 columns
- Desktop: 3-4 columns

### Phase 13: Backend

**API Client**
- Located: `src/utils/apiClient.ts`
- Supports mock and real APIs
- Auto-retry on 5xx errors
- Configurable timeout/retry

**Switching APIs**
```bash
# Use real API
VITE_USE_REAL_API=true
VITE_API_BASE_URL=https://api.mordeaux.com

# Use mock API
VITE_USE_REAL_API=false
VITE_API_BASE_URL=http://localhost:8000
```

---

## Troubleshooting

### White Screen / Not Loading

**Check:**
1. Console for errors
2. `npm install` completed
3. `.env` configured
4. Port 5173 available

**Fix:**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Clear cache
rm -rf .vite
npm run dev
```

### Images Not Loading

**Check:**
1. Mock server running (`http://localhost:8000/docs`)
2. URLs in console (SafeImage logs)
3. Network tab for 404s

**Fix:**
- Mock images use `minio.example.com` (not real)
- These will show placeholder/error states
- Expected behavior in dev

### Filters Not Working

**Check:**
1. URL updates when filter changes
2. Console for `[Performance] filter-duration`
3. Result count updates

**Fix:**
- Clear URL params: navigate to `/dev/search`
- Click "Reset Filters"

### Virtualization Laggy

**Check:**
1. Result count (< 2,000 is best)
2. Browser performance tab
3. Console for warnings

**Fix:**
- Disable virtualization toggle
- Use pagination mode
- Close other tabs

---

## Sharing a Deep Link

### Create Shareable URL

1. Set desired filters/pagination
2. Click "📋 Copy URL" button
3. Share URL with teammate

### Example URLs

```
# Filtered by score
http://localhost:5173/dev/search?minScore=0.8

# Specific page
http://localhost:5173/dev/search?page=3&pageSize=50

# Combined
http://localhost:5173/dev/search?minScore=0.75&site=example.com&page=2&view=list
```

### URL Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `minScore` | number (0-1) | 0 | Min similarity |
| `site` | string | '' | Filter by domain |
| `page` | integer | 1 | Page number |
| `pageSize` | integer | 25 | Results per page |
| `view` | grid\|list | grid | Display mode |

---

## What to Do When Images Fail

### Diagnosis

1. Open Debug Panel → Events tab
2. Check "Image Loads (Error)" counter
3. Open console, look for `[SafeImage] IMAGE_LOAD_FAILED`

### Common Causes

**Mock URLs (Expected)**
- URLs like `minio.example.com` are not real
- Will show placeholder/fallback
- Normal in dev mode

**Network Issues**
- Check mock server running
- Verify CORS settings
- Check browser network tab

**Invalid URLs**
- SafeImage blocks `javascript:`, `data:` URLs
- Check `[SafeLink] LINK_BLOCKED` logs

### Fixes

**For Mock Data:**
- Use local placeholder images
- Update mock URLs to `/placeholder.jpg`
- SafeImage has automatic fallback

**For Real Data:**
- Verify presigned URLs not expired
- Check S3/MinIO permissions
- Retry logic will attempt 3x

---

## Performance Tips

### For Large Result Sets

1. Enable virtualization toggle
2. Check console for performance metrics
3. Monitor memory in Debug Panel

### For Slow Filters

1. Debouncing reduces updates (300ms)
2. Check `[Performance] filter-duration`
3. Memoization prevents re-renders

### For Smooth Scrolling

1. Use virtualization for 1,000+ results
2. Lazy loading reduces initial load
3. Check FPS in Debug Panel (target: 60)

---

## Common Tasks

### Add New Filter

1. Update `useUrlState` in SearchDevPage
2. Add UI control (slider/select)
3. Wire onChange to `setUrlState`
4. Update memoized filter logic

### Add New Component

1. Create in `src/components/`
2. Add CSS file
3. Export from component
4. Import and use in page

### Add New Hook

1. Create in `src/hooks/`
2. Follow existing patterns
3. Document usage
4. Export from hook

### Add New Page

1. Create in `src/pages/`
2. Add route in `App.tsx`
3. Wrap in `DevRouteGuard` if dev-only

---

## Deployment

### Behind Feature Flag

**Production Setup:**
```bash
# Build with dev mode disabled
VITE_DEV_MODE=false npm run build

# Deploy to staging
# Enable flag for specific users
```

**Gradual Rollout:**
1. Deploy to staging with flag off
2. Enable for internal team
3. Monitor metrics
4. Enable for 10% users
5. Monitor and expand

---

## Support

### Documentation
- [Phase 1-14 Comprehensive Plan](./PHASE_1-14_COMPREHENSIVE_PLAN.md)
- [Phase 9 Performance](./PHASE_9_PERFORMANCE_HARDENING_COMPLETE.md)
- [Phase 9-11 Summary](./PHASE_9-11_IMPLEMENTATION_SUMMARY.md)
- [UAT Script](./UAT_SCRIPT_PHASES_1-14.md)

### Debugging
- Check Debug Panel (bottom-right)
- Review console logs
- Export logs for analysis
- Check network tab

### Contact
- Team: _____________________
- Slack: _____________________
- Docs: _____________________

---

**Runbook Version:** 1.0  
**Maintained by:** Development Team  
**Review Cycle:** Monthly



