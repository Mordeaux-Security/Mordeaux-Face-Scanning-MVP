/**
 * ResultCard Component - Phase 6
 * ===============================
 * 
 * Grid view card for search results.
 * 
 * Features:
 * - Thumbnail with SafeImage
 * - Score badge
 * - Optional distance chip
 * - BBox overlay on hover
 * - Actions (view source, copy ID)
 */

import { useMemo, useState } from 'react';
import SafeImage from './SafeImage';
import ScoreBadge from './ScoreBadge';
import DistanceChip from './DistanceChip';
import BBoxOverlay, { useImageDimensions } from './BBoxOverlay';
import SafeLink from './SafeLink';
import StorageChip from './StorageChip';
import { detectStorageProvider } from '../utils/linkAudit';
import './ResultCard.css';

export interface SearchHit {
  face_id: string;
  score: number;
  payload: {
    site: string;
    url: string;
    ts: string;
    bbox: [number, number, number, number];
    p_hash?: string;
    quality?: number;
  };
  thumb_url: string;
}

interface ResultCardProps {
  hit: SearchHit;
  showDistance?: boolean;
  onCopyId?: (faceId: string) => void;
}

export default function ResultCard({
  hit,
  showDistance = false,
  onCopyId,
}: ResultCardProps) {
  const [showBBox, setShowBBox] = useState(false);
  const [imageDimensions, imgRef] = useImageDimensions();
  
  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(hit.face_id);
      onCopyId?.(hit.face_id);
    } catch (error) {
      console.error('[ResultCard] Failed to copy ID:', error);
    }
  };
  
  // Format timestamp
  const timestamp = new Date(hit.payload.ts).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  
  // Calculate cosine distance from score (for optional display)
  const distance = 1 - hit.score;
  const storageInfo = useMemo(
    () => detectStorageProvider(hit.thumb_url || hit.payload.url),
    [hit.thumb_url, hit.payload.url]
  );
  
  return (
    <article 
      className="result-card"
      onMouseEnter={() => setShowBBox(true)}
      onMouseLeave={() => setShowBBox(false)}
      role="listitem"
    >
      {/* Thumbnail */}
      <div className="result-card__thumbnail">
        <SafeImage
          src={hit.thumb_url}
          alt={`Face match from ${hit.payload.site}`}
          className="result-card__image"
          fallback={
            <div className="result-card__image-fallback">
              <span aria-hidden="true">👤</span>
            </div>
          }
        />
        
        {/* BBox Overlay (shown on hover) */}
        {showBBox && imageDimensions.width > 0 && (
          <BBoxOverlay
            bbox={hit.payload.bbox}
            imageDimensions={imageDimensions}
            showOnHover={false}
            showCoordinates={true}
          />
        )}
        
        {/* Score Badge */}
        <div className="result-card__score">
          <ScoreBadge score={hit.score} showIcon={false} />
        </div>
      </div>
      
      {/* Metadata */}
      <div className="result-card__meta">
        <div className="result-card__site" title={hit.payload.site}>
          📍 {hit.payload.site}
        </div>
        <div className="result-card__timestamp" title={hit.payload.ts}>
          🕒 {timestamp}
        </div>
        
        {storageInfo && (
          <div className="result-card__storage">
            <StorageChip storage={storageInfo} />
          </div>
        )}
        
        {/* Optional Distance */}
        {showDistance && (
          <div className="result-card__distance">
            <DistanceChip distance={distance} type="cosine" />
          </div>
        )}
        
        {/* Quality Score (if available) */}
        {hit.payload.quality !== undefined && (
          <div className="result-card__quality" title={`Quality: ${hit.payload.quality.toFixed(2)}`}>
            ⚡ {(hit.payload.quality * 100).toFixed(0)}%
          </div>
        )}
      </div>
      
      {/* Actions */}
      <div className="result-card__actions">
        <SafeLink href={hit.payload.url} className="result-card__action">
          Open Source
        </SafeLink>
        <button 
          className="result-card__action" 
          onClick={handleCopyId}
          type="button"
          aria-label="Copy face ID"
        >
          📋 Copy ID
        </button>
      </div>
    </article>
  );
}

