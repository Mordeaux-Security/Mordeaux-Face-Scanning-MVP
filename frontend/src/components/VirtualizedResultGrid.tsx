/**
 * VirtualizedResultGrid - Phase 9 Performance Hardening
 * ======================================================
 * 
 * Virtualized grid view using react-window for optimal performance
 * with large result sets (2,000+ items).
 * 
 * Features:
 * - Only renders visible items
 * - Smooth scrolling performance
 * - Memory-efficient
 * - Responsive column layout
 */

import { FixedSizeGrid as Grid } from 'react-window';
import { SearchHit } from './ResultCard';
import MemoizedResultCard from './MemoizedResultCard';
import './VirtualizedResultGrid.css';

interface VirtualizedResultGridProps {
  results: SearchHit[];
  onCopyId: (faceId: string) => void;
  showDistance?: boolean;
}

// Calculate grid dimensions based on viewport
const getGridDimensions = () => {
  const width = window.innerWidth;
  
  // Responsive column count
  let columnCount: number;
  let cardWidth: number;
  
  if (width < 640) {
    // Mobile: 1 column
    columnCount = 1;
    cardWidth = Math.min(width - 32, 400); // Max 400px
  } else if (width < 1024) {
    // Tablet: 2 columns
    columnCount = 2;
    cardWidth = 280;
  } else if (width < 1536) {
    // Desktop: 3 columns
    columnCount = 3;
    cardWidth = 300;
  } else {
    // Large desktop: 4 columns
    columnCount = 4;
    cardWidth = 320;
  }
  
  return { columnCount, cardWidth };
};

const CARD_HEIGHT = 380; // Height of ResultCard
const GAP = 16; // Gap between cards

export default function VirtualizedResultGrid({
  results,
  onCopyId,
  showDistance = false,
}: VirtualizedResultGridProps) {
  const { columnCount, cardWidth } = getGridDimensions();
  const rowCount = Math.ceil(results.length / columnCount);
  
  // Calculate container dimensions
  const containerWidth = typeof window !== 'undefined' ? window.innerWidth - 32 : 1200;
  const containerHeight = typeof window !== 'undefined' ? window.innerHeight - 400 : 800;
  
  // Cell renderer
  const Cell = ({ columnIndex, rowIndex, style }: any) => {
    const index = rowIndex * columnCount + columnIndex;
    
    // Return empty cell if out of bounds
    if (index >= results.length) {
      return null;
    }
    
    const hit = results[index];
    
    return (
      <div
        style={{
          ...style,
          padding: GAP / 2,
          boxSizing: 'border-box',
        }}
      >
        <MemoizedResultCard
          hit={hit}
          showDistance={showDistance}
          onCopyId={onCopyId}
        />
      </div>
    );
  };
  
  // Handle empty state
  if (results.length === 0) {
    return (
      <div className="virtualized-grid-empty">
        <p>No results to display</p>
      </div>
    );
  }
  
  return (
    <div className="virtualized-result-grid" role="list">
      <Grid
        columnCount={columnCount}
        columnWidth={cardWidth + GAP}
        height={containerHeight}
        rowCount={rowCount}
        rowHeight={CARD_HEIGHT + GAP}
        width={containerWidth}
        className="virtualized-grid"
      >
        {Cell}
      </Grid>
    </div>
  );
}

