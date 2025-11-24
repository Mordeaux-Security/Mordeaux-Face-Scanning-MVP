# Phase 5 - Query Image & Safety Rules

## Overview

Phase 5 adds **secure image rendering** with comprehensive safety rules for the query panel and all images.

---

## Quick Start

### 1. View the Implementation

```bash
npm run dev
# Open: http://localhost:3000/dev/search
```

The query panel now shows a **real image component** with:
- ✅ Security validation
- ✅ Loading states
- ✅ Error handling
- ✅ Retry logic
- ✅ Fallback placeholders

### 2. Key Components

#### SafeImage (Core Security)

```tsx
import SafeImage from './components/SafeImage';

<SafeImage
  src={presignedUrl}               // Validated URL
  alt="Query face"
  referrerPolicy="no-referrer"     // No referrer leakage
  crossOrigin="anonymous"          // CORS isolation
  fallback={<Placeholder />}       // Required fallback
  onError={(err) => console.log(err)}
/>
```

#### QueryImage (Query Panel)

```tsx
import QueryImage from './components/QueryImage';

<QueryImage
  thumbnailUrl={thumbUrl}          // Presigned thumbnail URL
  fullResolutionUrl={fullUrl}      // Presigned full res URL
  alt="Query face"
  size={150}
  metadata={{
    fileName: 'query.jpg',
    fileSize: 245678,
    dimensions: { width: 1024, height: 1024 }
  }}
/>
```

---

## Security Rules

### ✅ Enforced Rules

1. **Presigned URLs Only** - Time-limited access with signatures
2. **Whitelisted Domains** - Only approved storage domains
3. **HTTPS Only** - Secure connections (except localhost)
4. **No Referrer Leakage** - `referrerPolicy="no-referrer"`
5. **Cross-Origin Isolation** - `crossOrigin="anonymous"`
6. **No Inline Content** - No data URIs or blob URLs
7. **Noreferrer Links** - `rel="noreferrer noopener"` on external links
8. **Always Fallback** - Graceful degradation
9. **Retry Logic** - 2 retries with exponential backoff
10. **Size Limits** - Max 2048x2048, 5MB

**Full Documentation**: `../docs/IMAGE_SAFETY_RULES.md`

---

## Image States

### Loading
```
┌───────────┐
│           │
│    ⏳     │  ← Spinner
│  Loading  │
│           │
└───────────┘
```

### Loaded
```
┌───────────┐
│           │
│  [Image]  │  ← Actual image
│           │
└───────────┘
🔍 View Full Resolution
```

### Error
```
┌───────────┐
│           │
│    ⚠️     │  ← Warning icon
│   Error   │
└───────────┘
```

### Fallback
```
┌───────────┐
│           │
│    🖼️     │  ← Placeholder icon
│Unavailable│
└───────────┘
```

---

## Retry Logic

**Timeline**:
```
0ms:    Load attempt #1 → Fails
1000ms: Load attempt #2 (retry after 1s) → Fails
3000ms: Load attempt #3 (retry after 2s) → Fails
3000ms: Show fallback
```

**Configuration**:
- Max Retries: 2
- Initial Delay: 1 second
- Backoff Factor: 2x (exponential)
- Timeout: 10 seconds per attempt

---

## Domain Whitelist

**Allowed Domains**:
```typescript
// Production
'minio.mordeaux.com'
'storage.mordeaux.com'
's3.amazonaws.com'

// Development
'localhost'
'127.0.0.1'
'minio.local'
'minio.example.com'

// Dev Placeholders
'via.placeholder.com'
'i.pravatar.cc'
```

**To Add Domain**:
Edit `frontend/src/components/SafeImage.tsx`:
```typescript
const ALLOWED_IMAGE_DOMAINS = [
  // ... existing domains
  'your-new-domain.com',  // Add here
];
```

---

## Testing

### Manual Tests

1. **Valid URL** ✅
   ```
   https://minio.example.com/...?X-Amz-Signature=mock
   → Image loads successfully
   ```

2. **Invalid Domain** ✅
   ```
   https://evil.com/image.jpg
   → Fallback shown, error logged
   ```

3. **Non-HTTPS** ✅
   ```
   http://example.com/image.jpg
   → Fallback shown (except localhost)
   ```

4. **404 Image** ✅
   ```
   https://minio.example.com/nonexistent.jpg
   → 2 retries, then fallback
   ```

### Check Security

**Browser Console**:
```javascript
// Check image attributes
const img = document.querySelector('.safe-image');
console.log(img.referrerPolicy);  // Should be "no-referrer"
console.log(img.crossOrigin);     // Should be "anonymous"

// Check links
const link = document.querySelector('.safe-link');
console.log(link.rel);            // Should be "noreferrer noopener"
```

---

## Integration with API

### Expected API Response

```json
{
  "query": {
    "image_url": "https://minio.example.com/thumbnails/tenant/face-123_thumb.jpg?X-Amz-Signature=...",
    "image_url_full": "https://minio.example.com/images/tenant/face-123.jpg?X-Amz-Signature=...",
    "image_metadata": {
      "file_name": "query.jpg",
      "file_size": 245678,
      "width": 1024,
      "height": 1024
    }
  }
}
```

### Usage in Component

```tsx
// From API response
const queryData = response.query;

<QueryImage
  thumbnailUrl={queryData.image_url}
  fullResolutionUrl={queryData.image_url_full}
  alt="Query face"
  size={150}
  metadata={{
    fileName: queryData.image_metadata.file_name,
    fileSize: queryData.image_metadata.file_size,
    dimensions: {
      width: queryData.image_metadata.width,
      height: queryData.image_metadata.height
    }
  }}
/>
```

---

## Troubleshooting

### Images Not Loading

**Check**:
1. ✓ Domain is whitelisted
2. ✓ URL is HTTPS (or localhost)
3. ✓ Presigned URL not expired
4. ✓ CORS headers on server

**Debug**:
```javascript
// Enable debug mode
localStorage.setItem('DEBUG_SAFE_IMAGE', 'true');

// Check browser console for security warnings
```

### Fallback Always Shown

**Possible Causes**:
- Domain not whitelisted → Add to `ALLOWED_IMAGE_DOMAINS`
- Non-HTTPS URL → Use HTTPS or localhost
- Expired presigned URL → Get fresh URL from API
- CORS error → Check server CORS headers

---

## Files

```
src/components/
├── SafeImage.tsx          # Core security component
├── SafeImage.css          # Styles
├── QueryImage.tsx         # Query panel image
└── QueryImage.css         # Styles

docs/
└── IMAGE_SAFETY_RULES.md  # Full documentation
```

---

## What's Different from Phase 4?

**Phase 4** (Non-functional):
```tsx
<div className="query-image-placeholder">
  📸 Query Image
</div>
```

**Phase 5** (Functional with security):
```tsx
<QueryImage
  thumbnailUrl={presignedUrl}      // Real URL
  fullResolutionUrl={fullUrl}      // Real URL
  alt="Query face"
  size={150}
  // + Security validation
  // + Loading states
  // + Error handling
  // + Retry logic
/>
```

---

## Next Steps

### Phase 6: Match Grid Images
- Apply `SafeImage` to result thumbnails
- Add lazy loading for off-screen images
- Implement thumbnail cache

### Phase 7: API Integration
- Connect to real backend
- Handle presigned URL refresh
- Add image preloading

---

## Quick Reference

### DO ✅
- Use `SafeImage` for all user images
- Provide alt text
- Include fallback
- Use presigned URLs
- Whitelist domains
- Use HTTPS

### DON'T ❌
- Use data URIs for user images
- Skip `referrerPolicy="no-referrer"`
- Use non-whitelisted domains
- Forget fallback handling
- Skip error handling

---

**Phase**: 5 - Query Image & Safety  
**Status**: ✅ Complete  
**Security**: High  
**Next**: Phase 6 - Match Grid Images

**Full Docs**: `../docs/PHASE_5_QUERY_IMAGE_SAFETY_COMPLETE.md`

