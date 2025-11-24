/**
 * EmptyState Component - Phase 4
 * ===============================
 * 
 * Empty state placeholder for no results.
 * No business logic - just visual presentation.
 */

import './EmptyState.css';

interface EmptyStateProps {
  title?: string;
  message?: string;
  suggestions?: string[];
  onAction?: () => void;
  actionLabel?: string;
}

export default function EmptyState({
  title = 'No Matches Found',
  message = 'No faces found matching your search criteria.',
  suggestions = [
    'Lower the similarity threshold',
    'Remove site filters',
    'Try a different query image'
  ],
  onAction,
  actionLabel = 'Adjust Filters'
}: EmptyStateProps) {
  return (
    <div className="empty-state" role="status" aria-live="polite">
      <div className="empty-icon" aria-hidden="true">
        🔍
      </div>
      
      <h2 className="empty-title">{title}</h2>
      
      <p className="empty-message">{message}</p>
      
      {suggestions && suggestions.length > 0 && (
        <div className="empty-suggestions">
          <h3 className="suggestions-title">Suggestions:</h3>
          <ul className="suggestions-list">
            {suggestions.map((suggestion, index) => (
              <li key={index} className="suggestion-item">
                • {suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {onAction && actionLabel && (
        <button 
          className="empty-action-button"
          onClick={onAction}
          type="button"
        >
          ⚙️ {actionLabel}
        </button>
      )}
    </div>
  );
}

