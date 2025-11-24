/**
 * ErrorState Component - Phase 4
 * ===============================
 * 
 * Error state placeholder.
 * No business logic - just visual presentation.
 */

import './ErrorState.css';

interface ErrorStateProps {
  title?: string;
  message?: string;
  suggestions?: string[];
  onRetry?: () => void;
  retryLabel?: string;
}

export default function ErrorState({
  title = 'Search Not Found',
  message = 'The search ID could not be found.',
  suggestions = [
    'Search results have expired',
    'Invalid search ID',
    'Search was deleted'
  ],
  onRetry,
  retryLabel = 'Try Again'
}: ErrorStateProps) {
  return (
    <div className="error-state" role="alert" aria-live="assertive">
      <div className="error-icon" aria-hidden="true">
        ⚠️
      </div>
      
      <h2 className="error-title">{title}</h2>
      
      <p className="error-message">{message}</p>
      
      {suggestions && suggestions.length > 0 && (
        <div className="error-suggestions">
          <h3 className="suggestions-title">Possible reasons:</h3>
          <ul className="suggestions-list">
            {suggestions.map((suggestion, index) => (
              <li key={index} className="suggestion-item">
                • {suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}
      
      <div className="error-actions">
        {onRetry && retryLabel && (
          <button 
            className="error-retry-button"
            onClick={onRetry}
            type="button"
          >
            🔄 {retryLabel}
          </button>
        )}
        
        <button 
          className="error-secondary-button"
          onClick={() => window.location.href = '/dev/search'}
          type="button"
        >
          📤 Upload New Image
        </button>
      </div>
    </div>
  );
}

