/**
 * SafeImage Component - Phase 5
 * ==============================
 * 
 * Secure image renderer with safety rules enforcement.
 * 
 * Security Features:
 * - Presigned URL validation
 * - Domain whitelist
 * - HTTPS enforcement
 * - No referrer leakage
 * - Fallback on error
 * - Retry logic
 * 
 * See: docs/IMAGE_SAFETY_RULES.md
 */

import { useState, useEffect, useRef, ReactNode } from 'react';
import './SafeImage.css';

// Whitelisted domains (Rule 2)
const ALLOWED_IMAGE_DOMAINS = [
  // Production storage
  'minio.mordeaux.com',
  'storage.mordeaux.com',
  's3.amazonaws.com',
  
  // Development/Testing
  'localhost',
  '127.0.0.1',
  'minio.local',
  'minio.example.com', // Mock server domain
  
  // Placeholder services (dev only)
  'via.placeholder.com',
  'i.pravatar.cc',
  'randomuser.me',
];

// Retry configuration (Rule 10)
const RETRY_CONFIG = {
  maxRetries: 2,
  initialDelay: 1000,  // 1 second
  backoffFactor: 2,
  timeout: 10000,      // 10 seconds
};

// Size limits (Rule 12)
const SIZE_LIMITS = {
  maxWidth: 2048,
  maxHeight: 2048,
};

/**
 * Validate if URL is from allowed domain (Rule 2)
 */
function isDomainAllowed(url: string): boolean {
  try {
    const { hostname } = new URL(url);
    
    return ALLOWED_IMAGE_DOMAINS.some(allowed => {
      // Exact match
      if (hostname === allowed) return true;
      
      // Subdomain match (e.g., *.amazonaws.com)
      if (hostname.endsWith(`.${allowed}`)) return true;
      
      return false;
    });
  } catch {
    return false;
  }
}

/**
 * Validate if URL is secure (Rule 3)
 */
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

/**
 * Check if URL looks like a presigned URL (Rule 1)
 */
function isPresignedUrl(url: string): boolean {
  try {
    const urlObj = new URL(url);
    const params = urlObj.searchParams;
    
    // Check for common presigned URL parameters
    return (
      params.has('X-Amz-Signature') ||  // AWS S3
      params.has('sig') ||                // Azure
      params.has('token') ||              // Generic
      params.has('expires') ||            // Generic
      params.has('X-Amz-Expires')        // AWS S3
    );
  } catch {
    return false;
  }
}

/**
 * Validate URL safety
 */
function validateUrl(url: string): { valid: boolean; reason?: string } {
  // Rule 6: Reject inline content
  if (url.startsWith('data:') || url.startsWith('blob:')) {
    return { valid: false, reason: 'Inline content not allowed' };
  }
  
  // Rule 3: HTTPS check
  if (!isSecureUrl(url)) {
    return { valid: false, reason: 'Non-HTTPS URL' };
  }
  
  // Rule 2: Domain whitelist
  if (!isDomainAllowed(url)) {
    return { valid: false, reason: 'Domain not whitelisted' };
  }
  
  // Rule 1: Presigned URL recommendation (warning only)
  if (!isPresignedUrl(url)) {
    console.warn('[SafeImage] URL is not presigned:', url);
  }
  
  return { valid: true };
}

/**
 * Log security event (Rule 13)
 */
function logSecurityEvent(type: string, url: string, error?: string) {
  const event = {
    timestamp: new Date().toISOString(),
    type,
    url: sanitizeUrl(url),
    error,
  };
  
  console.warn('[SafeImage Security]', event);
  
  // In production, send to logging service
  // sendToLogger(event);
}

/**
 * Sanitize URL for logging (remove sensitive parameters)
 */
function sanitizeUrl(url: string): string {
  try {
    const urlObj = new URL(url);
    // Remove signature and token parameters
    urlObj.searchParams.delete('X-Amz-Signature');
    urlObj.searchParams.delete('sig');
    urlObj.searchParams.delete('token');
    return urlObj.toString();
  } catch {
    return '[invalid-url]';
  }
}

/**
 * Sleep utility for retry backoff
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

interface SafeImageProps {
  src: string;
  alt: string;
  className?: string;
  fallback?: ReactNode;
  onError?: (error: Error) => void;
  onLoad?: () => void;
  width?: number | string;
  height?: number | string;
}

export default function SafeImage({
  src,
  alt,
  className = '',
  fallback,
  onError,
  onLoad,
  width,
  height,
}: SafeImageProps) {
  const [imageState, setImageState] = useState<'loading' | 'loaded' | 'error'>('loading');
  const [retryCount, setRetryCount] = useState(0);
  const imgRef = useRef<HTMLImageElement>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    // Validate URL on mount
    const validation = validateUrl(src);
    
    if (!validation.valid) {
      logSecurityEvent('INVALID_URL', src, validation.reason);
      setImageState('error');
      onError?.(new Error(validation.reason || 'Invalid URL'));
      return;
    }

    // Reset state when src changes
    setImageState('loading');
    setRetryCount(0);

    // Cleanup timeout on unmount
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [src]);

  const handleLoad = () => {
    // Rule 12: Validate image size
    if (imgRef.current) {
      const { naturalWidth, naturalHeight } = imgRef.current;
      
      if (naturalWidth > SIZE_LIMITS.maxWidth || naturalHeight > SIZE_LIMITS.maxHeight) {
        console.warn(
          `[SafeImage] Image too large: ${naturalWidth}x${naturalHeight} ` +
          `(max: ${SIZE_LIMITS.maxWidth}x${SIZE_LIMITS.maxHeight})`
        );
      }
    }

    setImageState('loaded');
    onLoad?.();
  };

  const handleError = () => {
    const error = new Error(`Image failed to load: ${src}`);
    
    // Rule 10: Retry logic
    if (retryCount < RETRY_CONFIG.maxRetries) {
      const delay = RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffFactor, retryCount);
      
      logSecurityEvent('IMAGE_LOAD_FAILED', src, `Retry ${retryCount + 1}/${RETRY_CONFIG.maxRetries}`);
      
      retryTimeoutRef.current = setTimeout(() => {
        setRetryCount(retryCount + 1);
        setImageState('loading');
      }, delay);
    } else {
      // All retries exhausted
      logSecurityEvent('IMAGE_LOAD_FAILED', src, 'All retries exhausted');
      setImageState('error');
      onError?.(error);
    }
  };

  // Rule 8: Show fallback on error
  if (imageState === 'error') {
    if (fallback) {
      return <>{fallback}</>;
    }
    
    // Default fallback
    return (
      <div 
        className={`safe-image-fallback ${className}`}
        role="img"
        aria-label={alt}
        style={{ width, height }}
      >
        <span className="fallback-icon" aria-hidden="true">🖼️</span>
        <span className="fallback-text">Image unavailable</span>
      </div>
    );
  }

  // Append retry parameter to bust cache
  const imageUrl = retryCount > 0 
    ? `${src}${src.includes('?') ? '&' : '?'}retry=${retryCount}`
    : src;

  return (
    <>
      {imageState === 'loading' && (
        <div 
          className={`safe-image-loading ${className}`}
          style={{ width, height }}
          aria-label="Loading image"
        >
          <div className="loading-spinner"></div>
        </div>
      )}
      
      <img
        ref={imgRef}
        src={imageUrl}
        alt={alt}
        className={`safe-image ${className} ${imageState === 'loaded' ? 'loaded' : 'loading'}`}
        onLoad={handleLoad}
        onError={handleError}
        // Rule 4: No referrer leakage
        referrerPolicy="no-referrer"
        // Rule 5: Cross-origin isolation
        crossOrigin="anonymous"
        width={width}
        height={height}
        // Prevent loading attribute conflicts
        loading="eager"
        style={{ display: imageState === 'loaded' ? 'block' : 'none' }}
      />
    </>
  );
}

