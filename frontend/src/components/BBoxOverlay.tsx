/**
 * BBoxOverlay Component - Phase 6
 * ================================
 * 
 * Renders bounding box overlay on images.
 * Converts normalized coordinates to CSS percentages.
 * 
 * BBox Format: [x, y, width, height] in pixels (absolute coordinates)
 * Image dimensions needed to convert to percentages.
 * 
 * Tolerance: ±2% alignment accuracy
 */

import { useState, useEffect, useRef } from 'react';
import './BBoxOverlay.css';

interface BBoxOverlayProps {
  /** Bounding box coordinates [x, y, width, height] in pixels */
  bbox: [number, number, number, number];
  
  /** Original image dimensions */
  imageDimensions: { width: number; height: number };
  
  /** Show overlay on hover only */
  showOnHover?: boolean;
  
  /** Color of the overlay */
  color?: string;
  
  /** Show coordinates tooltip */
  showCoordinates?: boolean;
}

/**
 * Convert absolute pixel coordinates to CSS percentages
 * 
 * Input: BBox [x, y, width, height] in pixels
 * Output: { left, top, width, height } in percentages
 * 
 * Formula:
 * - left% = (x / imageWidth) * 100
 * - top% = (y / imageHeight) * 100
 * - width% = (bboxWidth / imageWidth) * 100
 * - height% = (bboxHeight / imageHeight) * 100
 */
function bboxToPercentages(
  bbox: [number, number, number, number],
  imageDimensions: { width: number; height: number }
): { left: string; top: string; width: string; height: string } {
  const [x, y, width, height] = bbox;
  const { width: imgWidth, height: imgHeight } = imageDimensions;
  
  // Prevent division by zero
  if (imgWidth === 0 || imgHeight === 0) {
    return { left: '0%', top: '0%', width: '0%', height: '0%' };
  }
  
  // Convert to percentages (2 decimal places for precision)
  const left = ((x / imgWidth) * 100).toFixed(2);
  const top = ((y / imgHeight) * 100).toFixed(2);
  const bboxWidth = ((width / imgWidth) * 100).toFixed(2);
  const bboxHeight = ((height / imgHeight) * 100).toFixed(2);
  
  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${bboxWidth}%`,
    height: `${bboxHeight}%`,
  };
}

/**
 * Validate BBox coordinates
 */
function validateBBox(
  bbox: [number, number, number, number],
  imageDimensions: { width: number; height: number }
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const [x, y, width, height] = bbox;
  const { width: imgWidth, height: imgHeight } = imageDimensions;
  
  // Check for negative values
  if (x < 0) errors.push('X coordinate is negative');
  if (y < 0) errors.push('Y coordinate is negative');
  if (width <= 0) errors.push('Width must be positive');
  if (height <= 0) errors.push('Height must be positive');
  
  // Check if bbox exceeds image bounds
  if (x + width > imgWidth) {
    errors.push('BBox exceeds image width');
  }
  if (y + height > imgHeight) {
    errors.push('BBox exceeds image height');
  }
  
  // Check minimum size (at least 1% of image)
  const minSize = Math.min(imgWidth, imgHeight) * 0.01;
  if (width < minSize || height < minSize) {
    errors.push('BBox too small (< 1% of image)');
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

export default function BBoxOverlay({
  bbox,
  imageDimensions,
  showOnHover = false,
  color = 'rgba(102, 126, 234, 0.8)', // Primary color with transparency
  showCoordinates = false,
}: BBoxOverlayProps) {
  const [isVisible, setIsVisible] = useState(!showOnHover);
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[] }>({
    valid: true,
    errors: [],
  });
  
  useEffect(() => {
    // Validate BBox on mount and when props change
    const result = validateBBox(bbox, imageDimensions);
    setValidation(result);
    
    if (!result.valid) {
      console.warn('[BBoxOverlay] Invalid BBox:', result.errors);
    }
  }, [bbox, imageDimensions]);
  
  // Don't render if invalid
  if (!validation.valid) {
    return null;
  }
  
  // Convert to CSS percentages
  const position = bboxToPercentages(bbox, imageDimensions);
  
  const handleMouseEnter = () => {
    if (showOnHover) {
      setIsVisible(true);
    }
  };
  
  const handleMouseLeave = () => {
    if (showOnHover) {
      setIsVisible(false);
    }
  };
  
  const [x, y, width, height] = bbox;
  const coordinatesText = `[${x}, ${y}, ${width}, ${height}]`;
  
  return (
    <div
      className={`bbox-overlay ${isVisible ? 'visible' : 'hidden'}`}
      style={{
        position: 'absolute',
        left: position.left,
        top: position.top,
        width: position.width,
        height: position.height,
        border: `2px dashed ${color}`,
        borderRadius: 'var(--radius-sm)',
        pointerEvents: 'none',
        boxSizing: 'border-box',
        transition: 'opacity var(--transition-fast)',
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      aria-hidden="true"
    >
      {/* Coordinates tooltip */}
      {showCoordinates && isVisible && (
        <div className="bbox-coordinates-tooltip">
          {coordinatesText}
        </div>
      )}
    </div>
  );
}

/**
 * Hook to get image dimensions from img element
 * 
 * Usage:
 * const [dimensions, imgRef] = useImageDimensions();
 * <img ref={imgRef} ... />
 * <BBoxOverlay imageDimensions={dimensions} ... />
 */
export function useImageDimensions(): [
  { width: number; height: number },
  React.RefObject<HTMLImageElement>
] {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const imgRef = useRef<HTMLImageElement>(null);
  
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    
    const updateDimensions = () => {
      setDimensions({
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height,
      });
    };
    
    // Update on load
    if (img.complete) {
      updateDimensions();
    } else {
      img.addEventListener('load', updateDimensions);
      return () => img.removeEventListener('load', updateDimensions);
    }
  }, []);
  
  return [dimensions, imgRef];
}

