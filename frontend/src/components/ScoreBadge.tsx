/**
 * ScoreBadge Component - Phase 6
 * ===============================
 * 
 * Color-coded similarity score badge.
 * 
 * Score Thresholds:
 * - High (≥ 80%): Green
 * - Medium (60-79%): Yellow
 * - Low (< 60%): Red
 */

import './ScoreBadge.css';

interface ScoreBadgeProps {
  /** Similarity score (0-1) */
  score: number;
  
  /** Display format */
  format?: 'percentage' | 'decimal';
  
  /** Size variant */
  size?: 'small' | 'medium' | 'large';
  
  /** Show icon */
  showIcon?: boolean;
}

/**
 * Get badge variant based on score
 */
function getScoreVariant(score: number): 'high' | 'medium' | 'low' {
  if (score >= 0.80) return 'high';
  if (score >= 0.60) return 'medium';
  return 'low';
}

/**
 * Get icon based on score
 */
function getScoreIcon(score: number): string {
  if (score >= 0.80) return '🟢';
  if (score >= 0.60) return '🟡';
  return '🔴';
}

/**
 * Get accessibility label
 */
function getScoreLabel(score: number): string {
  const percentage = (score * 100).toFixed(1);
  const variant = getScoreVariant(score);
  const confidence = variant === 'high' ? 'high' : variant === 'medium' ? 'medium' : 'low';
  return `${percentage}% similarity, ${confidence} confidence`;
}

export default function ScoreBadge({
  score,
  format = 'percentage',
  size = 'medium',
  showIcon = false,
}: ScoreBadgeProps) {
  // Validate score
  if (score < 0 || score > 1) {
    console.warn(`[ScoreBadge] Invalid score: ${score} (must be 0-1)`);
    return null;
  }
  
  const variant = getScoreVariant(score);
  const icon = getScoreIcon(score);
  const label = getScoreLabel(score);
  
  // Format score
  const displayValue = format === 'percentage' 
    ? `${(score * 100).toFixed(1)}%`
    : score.toFixed(3);
  
  return (
    <div 
      className={`score-badge score-badge--${variant} score-badge--${size}`}
      role="status"
      aria-label={label}
      title={label}
    >
      {showIcon && (
        <span className="score-badge__icon" aria-hidden="true">
          {icon}
        </span>
      )}
      <span className="score-badge__value">
        {displayValue}
      </span>
    </div>
  );
}

