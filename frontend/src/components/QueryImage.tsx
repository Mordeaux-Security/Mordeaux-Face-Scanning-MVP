/**
 * QueryImage Component - Phase 5
 * ===============================
 * 
 * Query image renderer for the query panel.
 * Uses SafeImage with query-specific features.
 * 
 * Features:
 * - Safe image loading with security rules
 * - Full resolution link
 * - Loading/error/fallback states
 * - Image metadata display
 */

import { useState } from 'react';
import SafeImage from './SafeImage';
import SafeLink from './SafeLink';
import './QueryImage.css';

interface QueryImageProps {
  /** Thumbnail URL (presigned) */
  thumbnailUrl?: string;
  
  /** Full resolution URL (presigned) */
  fullResolutionUrl?: string;
  
  /** Query metadata */
  metadata?: {
    uploadedAt?: string;
    fileName?: string;
    fileSize?: number;
    dimensions?: { width: number; height: number };
  };
  
  /** Alt text */
  alt?: string;
  
  /** Size (default: 150px) */
  size?: number;
}

export default function QueryImage({
  thumbnailUrl,
  fullResolutionUrl,
  metadata,
  alt = 'Query image',
  size = 150,
}: QueryImageProps) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  const handleError = (error: Error) => {
    console.error('[QueryImage] Failed to load:', error);
    setImageError(true);
  };

  const handleLoad = () => {
    setImageLoaded(true);
  };

  // Fallback if no thumbnail URL provided
  if (!thumbnailUrl) {
    return (
      <div className="query-image-container">
        <div 
          className="query-image-placeholder"
          style={{ width: size, height: size }}
          role="img"
          aria-label={alt}
        >
          <span className="placeholder-icon" aria-hidden="true">📸</span>
          <span className="placeholder-text">No query image</span>
        </div>
      </div>
    );
  }

  return (
    <div className="query-image-container">
      {/* Image */}
      <div className="query-image-wrapper" style={{ width: size, height: size }}>
        <SafeImage
          src={thumbnailUrl}
          alt={alt}
          className="query-image"
          width={size}
          height={size}
          onError={handleError}
          onLoad={handleLoad}
          fallback={
            <div 
              className="query-image-error"
              style={{ width: size, height: size }}
            >
              <span className="error-icon" aria-hidden="true">⚠️</span>
              <span className="error-text">Image failed to load</span>
            </div>
          }
        />
        
        {/* Loading indicator overlay */}
        {!imageLoaded && !imageError && (
          <div className="query-image-loading-overlay">
            <div className="loading-spinner-small"></div>
          </div>
        )}
      </div>

      {/* Full Resolution Link */}
      {fullResolutionUrl && imageLoaded && (
        <div className="query-image-actions">
          <SafeLink href={fullResolutionUrl} className="view-full-res-link">
            🔍 View Full Resolution
          </SafeLink>
        </div>
      )}

      {/* Metadata (optional) */}
      {metadata && (
        <div className="query-image-metadata">
          {metadata.fileName && (
            <div className="metadata-item">
              <span className="metadata-label">File:</span>
              <span className="metadata-value" title={metadata.fileName}>
                {metadata.fileName}
              </span>
            </div>
          )}
          
          {metadata.fileSize && (
            <div className="metadata-item">
              <span className="metadata-label">Size:</span>
              <span className="metadata-value">
                {formatFileSize(metadata.fileSize)}
              </span>
            </div>
          )}
          
          {metadata.dimensions && (
            <div className="metadata-item">
              <span className="metadata-label">Dimensions:</span>
              <span className="metadata-value">
                {metadata.dimensions.width} × {metadata.dimensions.height}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Error message */}
      {imageError && (
        <div className="query-image-error-message">
          <span role="alert">
            ⚠️ Could not load query image. Please try refreshing the page.
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Format file size for display
 */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  } else {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}

