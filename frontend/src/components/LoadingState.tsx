/**
 * LoadingState Component - Phase 4
 * =================================
 * 
 * Loading state placeholder.
 * No business logic - just visual presentation.
 */

import './LoadingState.css';

interface LoadingStateProps {
  message?: string;
}

export default function LoadingState({ message = 'Loading search results...' }: LoadingStateProps) {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-spinner" aria-hidden="true">
        <div className="spinner"></div>
      </div>
      <p className="loading-message">{message}</p>
      <div className="loading-skeleton">
        {/* Skeleton placeholders matching match grid */}
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} className="skeleton-card">
            <div className="skeleton-image"></div>
            <div className="skeleton-text skeleton-text-lg"></div>
            <div className="skeleton-text skeleton-text-sm"></div>
            <div className="skeleton-text skeleton-text-sm"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

