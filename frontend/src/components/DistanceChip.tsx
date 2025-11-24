/**
 * DistanceChip Component - Phase 6
 * =================================
 * 
 * Optional distance/similarity metric chip.
 * Shows cosine distance or other similarity metrics.
 */

import './DistanceChip.css';

interface DistanceChipProps {
  /** Distance value */
  distance: number;
  
  /** Distance type */
  type?: 'cosine' | 'euclidean' | 'manhattan';
  
  /** Size variant */
  size?: 'small' | 'medium';
  
  /** Show label */
  showLabel?: boolean;
}

/**
 * Format distance value
 */
function formatDistance(distance: number, type: string): string {
  // Cosine distance is typically 0-2 (lower is more similar)
  // For display, we might want to show it as similarity (1 - distance/2)
  if (type === 'cosine') {
    return distance.toFixed(4);
  }
  
  // Euclidean and Manhattan distances
  return distance.toFixed(2);
}

/**
 * Get distance label
 */
function getDistanceLabel(type: string): string {
  switch (type) {
    case 'cosine':
      return 'Cosine Distance';
    case 'euclidean':
      return 'Euclidean Distance';
    case 'manhattan':
      return 'Manhattan Distance';
    default:
      return 'Distance';
  }
}

export default function DistanceChip({
  distance,
  type = 'cosine',
  size = 'small',
  showLabel = false,
}: DistanceChipProps) {
  const formattedDistance = formatDistance(distance, type);
  const label = getDistanceLabel(type);
  const title = `${label}: ${formattedDistance}`;
  
  return (
    <div 
      className={`distance-chip distance-chip--${size}`}
      title={title}
      aria-label={title}
    >
      {showLabel && (
        <span className="distance-chip__label">{label}:</span>
      )}
      <span className="distance-chip__value">{formattedDistance}</span>
    </div>
  );
}

