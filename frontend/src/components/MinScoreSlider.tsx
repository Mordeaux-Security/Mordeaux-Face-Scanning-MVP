/**
 * MinScoreSlider Component - Phase 7
 * ===================================
 * 
 * Slider control for filtering results by minimum similarity score.
 * 
 * Features:
 * - Range: 0-100%
 * - Step: 1%
 * - Visual feedback
 * - Debounced updates
 * - Keyboard accessible
 */

import { useState, useEffect, useRef } from 'react';
import './MinScoreSlider.css';

interface MinScoreSliderProps {
  value: number;              // Current value (0-1)
  onChange: (value: number) => void;
  min?: number;               // Minimum value (0-1)
  max?: number;               // Maximum value (0-1)
  step?: number;              // Step size (0-1)
  debounceMs?: number;        // Debounce delay in ms
  showLabel?: boolean;
  disabled?: boolean;
}

export default function MinScoreSlider({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  debounceMs = 300,
  showLabel = true,
  disabled = false,
}: MinScoreSliderProps) {
  const [localValue, setLocalValue] = useState(value);
  const debounceTimerRef = useRef<NodeJS.Timeout>();
  
  // Sync with external value changes
  useEffect(() => {
    setLocalValue(value);
  }, [value]);
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    setLocalValue(newValue);
    
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    // Debounce the onChange callback
    debounceTimerRef.current = setTimeout(() => {
      onChange(newValue);
    }, debounceMs);
  };
  
  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);
  
  const percentage = Math.round(localValue * 100);
  const fillPercentage = ((localValue - min) / (max - min)) * 100;
  
  return (
    <div className={`min-score-slider ${disabled ? 'disabled' : ''}`}>
      {showLabel && (
        <div className="slider-header">
          <label htmlFor="min-score-slider" className="slider-label">
            Min Score:
          </label>
          <span className="slider-value" aria-live="polite">
            {percentage}%
          </span>
        </div>
      )}
      
      <div className="slider-container">
        <input
          type="range"
          id="min-score-slider"
          className="slider-input"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleChange}
          disabled={disabled}
          aria-label={`Minimum similarity score: ${percentage}%`}
          aria-valuemin={min * 100}
          aria-valuemax={max * 100}
          aria-valuenow={percentage}
        />
        
        {/* Visual fill indicator */}
        <div 
          className="slider-fill"
          style={{ width: `${fillPercentage}%` }}
          aria-hidden="true"
        />
        
        {/* Tick marks */}
        <div className="slider-ticks" aria-hidden="true">
          <span className="tick" style={{ left: '0%' }}>0</span>
          <span className="tick" style={{ left: '25%' }}>25</span>
          <span className="tick" style={{ left: '50%' }}>50</span>
          <span className="tick" style={{ left: '75%' }}>75</span>
          <span className="tick" style={{ left: '100%' }}>100</span>
        </div>
      </div>
      
      {/* Result count hint */}
      <div className="slider-hint">
        Filters results below {percentage}% similarity
      </div>
    </div>
  );
}





