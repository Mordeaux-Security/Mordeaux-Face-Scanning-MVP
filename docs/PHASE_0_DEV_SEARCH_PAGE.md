# Phase 0 — Dev Search Page: Outcome + Scope

## Document Purpose

This document defines the scope, success metrics, and acceptance criteria for a **dev-only** search results visualization page that displays query images alongside search hits with fast filtering, safe links, and debugging information.

**Status**: Draft for Sign-off  
**Target Audience**: Development Team  
**Environment**: Development/Testing Only (Not Production)

---

## 1. One-Page Brief

### Purpose

Create a developer-focused single-page application (SPA) that visualizes face search results by displaying:
- The user's uploaded query image
- Matching face results (thumbnails) in a scrollable grid
- Client-side filtering controls (similarity threshold, site filter, date range)
- Debugging information (API response, timing metrics, metadata)
- Safe links to source images (presigned URLs with expiry handling)

### In Scope

✅ **Core Features**
- Display query image prominently at top of page
- Render search hits as thumbnail grid (up to 500 results)
- Client-side filtering by:
  - Similarity score (threshold slider)
  - Source site (dropdown)
  - Date range (timestamp filter)
- Show debugging panel with:
  - Full API response (JSON)
  - Request timing (TTFB, total duration)
  - Query parameters used
  - Result count and pagination info
  - Vector backend (Qdrant/Pinecone)
- Safe link handling:
  - Presigned URLs for thumbnails
  - Automatic refresh of expired URLs
  - Source URL display (read-only, with sanitization)
- Performance optimizations:
  - Virtual scrolling for large result sets
  - Lazy image loading
  - Debounced filter inputs

✅ **Technical Requirements**
- Single HTML page with embedded JavaScript (no build step required)
- Uses existing API endpoints (`/api/v1/search/file` or `/api/compare_face`)
- Responsive design (works on desktop, tablet)
- No external dependencies (vanilla JS) or minimal dependencies (optional: lightweight UI library)

### Out of Scope

❌ **Explicitly Excluded**
- Production deployment (dev-only)
- User authentication/authorization
- Multi-tenant UI (single tenant assumed)
- Real-time updates (static results display)
- Export functionality (CSV, JSON download)
- Image editing/cropping
- Batch search (single query at a time)
- Persistent state (no local storage)
- Advanced filtering (fuzzy search, regex)
- Accessibility features (WCAG compliance)
- Mobile optimization (desktop-first)
- Internationalization (English only)

### Success Metrics

#### Performance Targets
- **Time to Interactive (TTI)**: < 2 seconds (with mock data, 10 results)
- **Render Performance**: Render 500 results without jank (60 FPS scrolling)
- **Filter Response**: < 100ms for client-side filtering (500 results)
- **Image Load Time**: < 500ms per thumbnail (lazy loaded)

#### Functional Targets
- **Accuracy**: Display all metadata fields correctly (site, url, ts, bbox, p_hash, quality)
- **Safety**: All image URLs are presigned (no raw storage URLs exposed)
- **Debugging**: Full API response visible in expandable panel
- **Error Handling**: Graceful error messages for API failures, expired URLs

#### User Experience Targets
- **Usability**: Developers can filter results in < 3 clicks
- **Visibility**: Query image and results visible above the fold (no scrolling required for initial view)
- **Feedback**: Clear loading states, error messages, and empty states

---

## 2. Glossary

| Term | Definition |
|------|------------|
| **Query Image** | The user-uploaded image used to search for similar faces in the database |
| **Result / Hit** | A single matching face from the database, returned as a search result |
| **Similarity / Score** | Cosine similarity score (0.0-1.0) indicating how similar a result is to the query face |
| **Distance** | Inverse of similarity (1 - score), used internally by vector databases |
| **Face BBox / BBox** | Face bounding box coordinates `[x, y, width, height]` indicating face location in source image |
| **Source / Site** | The domain/website where the source image was originally found (e.g., "example.com") |
| **Storage** | Object storage system (MinIO or AWS S3) where images and thumbnails are stored |
| **Presigned URL** | Temporary, time-limited URL (TTL: 10 minutes) that provides secure access to stored images |
| **Thumbnail** | Scaled-down version of an image (256px longest side) used for display in search results |
| **Metadata** | Additional information about a face result: site, url, ts, bbox, p_hash, quality |
| **Vector Backend** | The vector database used for similarity search (Qdrant or Pinecone) |
| **TTI (Time to Interactive)** | Time from page load until user can interact with the page (filters, clicks) |
| **Jank** | Visual stuttering or frame drops during scrolling or animations |

---

## 3. Risks & Mitigations

### Risk 1: Presigned URL Expiry

**Description**: Presigned URLs expire after 10 minutes. If a user keeps the page open for > 10 minutes, thumbnails will fail to load.

**Impact**: High — Broken images in search results

**Mitigation**:
- Implement URL expiry detection (check response status codes)
- Automatically refresh expired URLs by calling API endpoint to regenerate presigned URLs
- Show placeholder/error state for expired images
- Add visual indicator (expiry timer) for URLs approaching expiry
- **Fallback**: Display "Image expired" message with option to refresh

### Risk 2: Oversized Payloads

**Description**: Large result sets (500+ results) may cause:
- Slow API responses (> 5 seconds)
- Large JSON payloads (> 5 MB)
- Browser memory issues

**Impact**: Medium — Poor performance, potential browser crashes

**Mitigation**:
- Implement pagination on API side (limit to 100 results per page)
- Use virtual scrolling to render only visible results
- Lazy load images (load thumbnails as user scrolls)
- Implement result limiting in UI (max 500 results displayed)
- Add loading indicators for pagination
- **Fallback**: Show warning if result count > 500, suggest filtering

### Risk 3: PII Exposure

**Description**: Source URLs or metadata may contain sensitive information (personally identifiable information).

**Impact**: High — Privacy violation, compliance issues

**Mitigation**:
- Rely on API-side filtering (only allowed fields: site, url, ts, bbox, p_hash, quality)
- Sanitize URLs before display (remove query parameters, truncate long URLs)
- Add warning banner: "Dev-only: Contains test data"
- Never log or store PII in client-side code
- **Fallback**: Mask URLs in debugging panel (show only domain)

### Risk 4: API Rate Limiting

**Description**: Multiple rapid searches or URL refresh requests may hit rate limits.

**Impact**: Low — Temporary unavailability

**Mitigation**:
- Implement request debouncing (wait 300ms before submitting search)
- Cache API responses (session-level cache)
- Show rate limit error messages clearly
- Implement exponential backoff for retries
- **Fallback**: Disable search button during rate limit, show retry option

### Risk 5: Browser Compatibility

**Description**: Modern JavaScript features may not work in older browsers.

**Impact**: Low — Dev-only page, modern browsers expected

**Mitigation**:
- Test in Chrome, Firefox, Safari (latest versions)
- Use polyfills for critical features (if needed)
- Document browser requirements
- **Fallback**: Show "Browser not supported" message for IE/older browsers

### Risk 6: Performance Degradation with Large Results

**Description**: Rendering 500+ thumbnails may cause browser slowdown.

**Impact**: Medium — Poor user experience

**Mitigation**:
- Implement virtual scrolling (render only visible items)
- Use `requestAnimationFrame` for smooth scrolling
- Lazy load images (Intersection Observer API)
- Debounce filter inputs (avoid re-rendering on every keystroke)
- **Fallback**: Limit initial display to 50 results, add "Load More" button

---

## 4. Acceptance Criteria

### AC1: Scope Agreement
- [ ] Product Owner agrees that the scope (in/out) is correct
- [ ] Development team confirms technical feasibility
- [ ] Success metrics are documented and measurable

### AC2: Success Metrics Validation
- [ ] TTI < 2 seconds (measured with mock data, 10 results)
- [ ] Render 500 results without jank (60 FPS scrolling verified)
- [ ] Filter response < 100ms (measured with 500 results)
- [ ] All thumbnails load within 500ms (lazy loaded)

### AC3: Functional Requirements
- [ ] Query image displays correctly at top of page
- [ ] Search hits render as thumbnail grid
- [ ] Client-side filtering works (similarity, site, date)
- [ ] Debugging panel shows full API response
- [ ] Presigned URLs are used for all images
- [ ] Expired URLs are handled gracefully (refresh or error state)

### AC4: Risk Mitigations
- [ ] Presigned URL expiry handling implemented
- [ ] Pagination/virtual scrolling for large result sets
- [ ] PII sanitization in place (URL masking in debug panel)
- [ ] Rate limiting handled (debouncing, error messages)
- [ ] Performance optimizations implemented (lazy loading, virtual scrolling)

### AC5: Documentation
- [ ] README with setup instructions
- [ ] API endpoint documentation (which endpoint to use)
- [ ] Browser requirements documented
- [ ] Known limitations documented

### AC6: Sign-off
- [ ] **Product Owner Sign-off**: "I agree the scope is correct and the success metrics are reasonable."
- [ ] **Tech Lead Sign-off**: "I confirm this is technically feasible and aligns with our architecture."
- [ ] **Dev Team Sign-off**: "We understand the requirements and can deliver within the defined scope."

---

## Sign-off Section

### Product Owner
- **Name**: _______________________
- **Date**: _______________________
- **Signature**: "I agree the scope is correct and the success metrics are reasonable."

### Tech Lead
- **Name**: _______________________
- **Date**: _______________________
- **Signature**: "I confirm this is technically feasible and aligns with our architecture."

### Development Team
- **Name**: _______________________
- **Date**: _______________________
- **Signature**: "We understand the requirements and can deliver within the defined scope."

---

## Appendix: API Endpoints

### Recommended Endpoint
- **Face Pipeline**: `POST /api/v1/search/file` (face-pipeline service, port 8001)
- **Backend API**: `POST /api/compare_face` (backend service, port 8000)

### Response Format
```json
{
  "query": {
    "tenant_id": "demo-tenant",
    "search_mode": "image",
    "top_k": 10,
    "threshold": 0.75
  },
  "hits": [
    {
      "face_id": "face-123",
      "score": 0.95,
      "payload": {
        "site": "example.com",
        "url": "https://example.com/image.jpg",
        "ts": "2024-01-01T12:00:00Z",
        "bbox": [100, 150, 200, 250],
        "p_hash": "a1b2c3d4e5f6g7h8",
        "quality": 0.92
      },
      "thumb_url": "https://minio.example.com/thumbnails/...?X-Amz-Algorithm=...",
      "image_url": "https://minio.example.com/thumbnails/...?X-Amz-Algorithm=..."
    }
  ],
  "count": 10
}
```

### Presigned URL TTL
- **Default**: 10 minutes (600 seconds)
- **Max**: 10 minutes (enforced by API)
- **Refresh**: Call API endpoint to regenerate URLs for expired images

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-10  
**Next Review**: After Phase 0 sign-off

