/**
 * MemoizedResultListItem - Phase 9 Performance Hardening
 * =======================================================
 * 
 * Memoized wrapper around ResultListItem to prevent unnecessary re-renders.
 * Only re-renders when props actually change.
 */

import { memo } from 'react';
import ResultListItem from './ResultListItem';
import { SearchHit } from './ResultCard';

interface MemoizedResultListItemProps {
  hit: SearchHit;
  showDistance?: boolean;
  onCopyId?: (faceId: string) => void;
}

// Custom comparison function for memo
const areEqual = (
  prevProps: MemoizedResultListItemProps,
  nextProps: MemoizedResultListItemProps
) => {
  return (
    prevProps.hit.face_id === nextProps.hit.face_id &&
    prevProps.hit.score === nextProps.hit.score &&
    prevProps.showDistance === nextProps.showDistance &&
    prevProps.hit.thumb_url === nextProps.hit.thumb_url
  );
};

const MemoizedResultListItem = memo(ResultListItem, areEqual);

export default MemoizedResultListItem;


