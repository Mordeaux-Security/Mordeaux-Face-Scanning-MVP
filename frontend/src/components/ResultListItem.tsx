/**
 * ResultListItem Component - Phase 6
 * ===================================
 * 
 * List view item for search results.
 * Horizontal layout with more metadata visible.
 * 
 * Features:
 * - Larger thumbnail
 * - Extended metadata display
 * - BBox overlay on hover
 * - Actions
 */

import { useMemo, useState } from 'react';
import SafeImage from './SafeImage';
import ScoreBadge from './ScoreBadge';
import DistanceChip from './DistanceChip';
import BBoxOverlay, { useImageDimensions } from './BBoxOverlay';
import './ResultListItem.css';
import { SearchHit } from './ResultCard';
import SafeLink from './SafeLink';
import StorageChip from './StorageChip';
import { detectStorageProvider } from '../utils/linkAudit';

interface ResultListItemProps {
  hit: SearchHit;
  showDistance?: boolean;
  onCopyId?: (faceId: string) => void;
}

export default function ResultListItem({
  hit,
  showDistance = false,
  onCopyId,
}: ResultListItemProps) {
  const [showBBox, setShowBBox] = useState(false);
  const [imageDimensions, imgRef] = useImageDimensions();
  
  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(hit.face_id);
      onCopyId?.(hit.face_id);
    } catch (error) {
      console.error('[ResultListItem] Failed to copy ID:', error);
    }
  };
  
  // Format timestamp (full format for list view)
  const timestamp = new Date(hit.payload.ts).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  
  const distance = 1 - hit.score;
  const storageInfo = useMemo(
    () => detectStorageProvider(hit.thumb_url || hit.payload.url),
    [hit.thumb_url, hit.payload.url]
  );
  
  return (
    <article 
      className="result-list-item"
      onMouseEnter={() => setShowBBox(true)}
      onMouseLeave={() => setShowBBox(false)}
      role="listitem"
    >
      {/* Thumbnail */}
      <div className="result-list-item__thumbnail">
        <SafeImage
          src={hit.thumb_url}
          alt={`Face match from ${hit.payload.site}`}
          className="result-list-item__image"
          width={100}
          height={100}
          fallback={
            <div className="result-list-item__image-fallback">
              <span aria-hidden="true">👤</span>
            </div>
          }
        />
        
        {/* BBox Overlay */}
        {showBBox && imageDimensions.width > 0 && (
          <BBoxOverlay
            bbox={hit.payload.bbox}
            imageDimensions={imageDimensions}
            showOnHover={false}
            showCoordinates={true}
          />
        )}
      </div>
      
      {/* Content */}
      <div className="result-list-item__content">
        {/* Header */}
        <div className="result-list-item__header">
          <ScoreBadge score={hit.score} showIcon={true} size="small" />
          
          <div className="result-list-item__site">
          <strong>Site:</strong> {hit.payload.site}
          </div>
        {storageInfo && <StorageChip storage={storageInfo} />}
        </div>
        
        {/* Metadata */}
        <div className="result-list-item__metadata">
          <div className="metadata-row">
            <span className="metadata-label">Face ID:</span>
            <span className="metadata-value" title={hit.face_id}>
              {hit.face_id}
            </span>
          </div>
          
          <div className="metadata-row">
            <span className="metadata-label">Timestamp:</span>
            <span className="metadata-value">{timestamp}</span>
          </div>
          
          {hit.payload.quality !== undefined && (
            <div className="metadata-row">
              <span className="metadata-label">Quality:</span>
              <span className="metadata-value">{hit.payload.quality.toFixed(3)}</span>
            </div>
          )}
          
          {hit.payload.p_hash && (
            <div className="metadata-row">
              <span className="metadata-label">P-Hash:</span>
              <span className="metadata-value">{hit.payload.p_hash}</span>
            </div>
          )}
          
          {showDistance && (
            <div className="metadata-row">
              <span className="metadata-label">Distance:</span>
              <span className="metadata-value">
                <DistanceChip distance={distance} type="cosine" size="small" />
              </span>
            </div>
          )}
        </div>
      </div>
      
      {/* Actions */}
      <div className="result-list-item__actions">
        <SafeLink href={hit.payload.url} className="result-list-item__action-button">
          Open Source
        </SafeLink>
        <button 
          className="result-list-item__action-button"
          onClick={handleCopyId}
          type="button"
          aria-label="Copy face ID"
        >
          📋 Copy ID
        </button>
        <button 
          className="result-list-item__action-button"
          type="button"
          aria-label="View details"
        >
          🔍 Details
        </button>
      </div>
    </article>
  );
}

