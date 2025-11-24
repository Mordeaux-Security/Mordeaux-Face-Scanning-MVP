/**
 * LazyImage - Phase 9 Performance Hardening
 * ==========================================
 * 
 * Lazy-loading image component using Intersection Observer.
 * Only loads images when they enter the viewport.
 * 
 * Features:
 * - Lazy loading with Intersection Observer
 * - Placeholder while not loaded
 * - Smooth fade-in on load
 * - Error handling
 */

import { useState } from 'react';
import { useLazyImage } from '../hooks/useLazyImage';
import './LazyImage.css';

interface LazyImageProps {
  src: string;
  alt: string;
  className?: string;
  width?: number | string;
  height?: number | string;
  placeholder?: React.ReactNode;
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

export default function LazyImage({
  src,
  alt,
  className = '',
  width,
  height,
  placeholder,
  onLoad,
  onError,
}: LazyImageProps) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const { ref, shouldLoad } = useLazyImage(src);

  const handleLoad = () => {
    setLoaded(true);
    onLoad?.();
  };

  const handleError = () => {
    setError(true);
    onError?.(new Error(`Failed to load image: ${src}`));
  };

  return (
    <div
      className={`lazy-image-container ${className}`}
      style={{ width, height }}
    >
      {/* Placeholder */}
      {!loaded && !error && (
        <div className="lazy-image-placeholder">
          {placeholder || (
            <div className="lazy-image-skeleton" aria-hidden="true">
              <div className="skeleton-shimmer"></div>
            </div>
          )}
        </div>
      )}

      {/* Actual Image */}
      {shouldLoad && !error && (
        <img
          ref={ref}
          src={src}
          alt={alt}
          className={`lazy-image ${loaded ? 'loaded' : 'loading'}`}
          onLoad={handleLoad}
          onError={handleError}
          loading="lazy"
          decoding="async"
          style={{ width, height }}
        />
      )}

      {/* Error State */}
      {error && (
        <div className="lazy-image-error" aria-label="Image failed to load">
          <span className="error-icon" aria-hidden="true">⚠️</span>
          <span className="error-text">Failed to load</span>
        </div>
      )}
    </div>
  );
}


