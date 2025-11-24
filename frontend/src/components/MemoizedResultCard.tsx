/**
 * MemoizedResultCard - Phase 9 Performance Hardening
 * ===================================================
 * 
 * Memoized wrapper around ResultCard to prevent unnecessary re-renders.
 * Only re-renders when props actually change.
 */

import { memo } from 'react';
import ResultCard, { SearchHit } from './ResultCard';

interface MemoizedResultCardProps {
  hit: SearchHit;
  showDistance?: boolean;
  onCopyId?: (faceId: string) => void;
}

// Custom comparison function for memo
const areEqual = (
  prevProps: MemoizedResultCardProps,
  nextProps: MemoizedResultCardProps
) => {
  return (
    prevProps.hit.face_id === nextProps.hit.face_id &&
    prevProps.hit.score === nextProps.hit.score &&
    prevProps.showDistance === nextProps.showDistance &&
    prevProps.hit.thumb_url === nextProps.hit.thumb_url
  );
};

const MemoizedResultCard = memo(ResultCard, areEqual);

export default MemoizedResultCard;


