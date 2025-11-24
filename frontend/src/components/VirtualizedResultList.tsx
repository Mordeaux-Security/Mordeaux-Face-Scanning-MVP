/**
 * VirtualizedResultList - Phase 9 Performance Hardening
 * ======================================================
 * 
 * Virtualized list view using react-window for optimal performance
 * with large result sets (2,000+ items).
 * 
 * Features:
 * - Only renders visible items
 * - Smooth scrolling performance
 * - Memory-efficient
 */

import { FixedSizeList as List } from 'react-window';
import { SearchHit } from './ResultCard';
import MemoizedResultListItem from './MemoizedResultListItem';
import './VirtualizedResultList.css';

interface VirtualizedResultListProps {
  results: SearchHit[];
  onCopyId: (faceId: string) => void;
  showDistance?: boolean;
}

const ITEM_HEIGHT = 180; // Height of each list item

export default function VirtualizedResultList({
  results,
  onCopyId,
  showDistance = true,
}: VirtualizedResultListProps) {
  // Calculate container dimensions
  const containerHeight = typeof window !== 'undefined' ? window.innerHeight - 400 : 800;
  const containerWidth = typeof window !== 'undefined' ? window.innerWidth - 32 : 1200;
  
  // Row renderer
  const Row = ({ index, style }: any) => {
    const hit = results[index];
    
    return (
      <div style={style}>
        <MemoizedResultListItem
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
      <div className="virtualized-list-empty">
        <p>No results to display</p>
      </div>
    );
  }
  
  return (
    <div className="virtualized-result-list" role="list">
      <List
        height={containerHeight}
        itemCount={results.length}
        itemSize={ITEM_HEIGHT}
        width={containerWidth}
        className="virtualized-list"
      >
        {Row}
      </List>
    </div>
  );
}

