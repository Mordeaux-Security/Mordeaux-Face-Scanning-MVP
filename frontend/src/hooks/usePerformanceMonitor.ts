/**
 * usePerformanceMonitor - Phase 9 Performance Hardening
 * ======================================================
 * 
 * Hook for monitoring component performance using the Performance API.
 * Tracks render times, paint times, and custom marks.
 * 
 * Features:
 * - Performance marks and measures
 * - Render time tracking
 * - Memory usage monitoring (if available)
 * - Dev-only logging
 */

import { useEffect, useRef, useCallback } from 'react';

interface PerformanceMetrics {
  renderTime: number;
  paintTime: number;
  memoryUsage?: {
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  };
}

export function usePerformanceMonitor(componentName: string, enabled: boolean = true) {
  const renderStartTimeRef = useRef<number>(0);
  const renderCountRef = useRef<number>(0);

  // Mark render start
  useEffect(() => {
    if (!enabled || !performance) return;

    renderStartTimeRef.current = performance.now();
    renderCountRef.current++;

    const markName = `${componentName}-render-${renderCountRef.current}`;
    performance.mark(markName);
  });

  // Mark render end and measure
  useEffect(() => {
    if (!enabled || !performance) return;

    const renderEndTime = performance.now();
    const renderDuration = renderEndTime - renderStartTimeRef.current;

    const markName = `${componentName}-render-${renderCountRef.current}`;
    const measureName = `${componentName}-render-duration-${renderCountRef.current}`;

    try {
      performance.measure(measureName, markName);
    } catch (error) {
      // Mark might not exist yet
    }

    // Log in dev mode only
    if (import.meta.env.DEV && renderDuration > 16) {
      // Warn if render takes longer than one frame (16ms at 60fps)
      console.warn(
        `[Performance] ${componentName} render #${renderCountRef.current} took ${renderDuration.toFixed(2)}ms`
      );
    }
  });

  // Get current metrics
  const getMetrics = useCallback((): PerformanceMetrics | null => {
    if (!enabled || !performance) return null;

    const metrics: PerformanceMetrics = {
      renderTime: performance.now() - renderStartTimeRef.current,
      paintTime: 0,
    };

    // Get paint timing if available
    const paintEntries = performance.getEntriesByType('paint');
    const firstPaint = paintEntries.find((entry) => entry.name === 'first-paint');
    if (firstPaint) {
      metrics.paintTime = firstPaint.startTime;
    }

    // Get memory info if available (Chrome only)
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      metrics.memoryUsage = {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit,
      };
    }

    return metrics;
  }, [enabled, componentName]);

  // Mark a custom event
  const mark = useCallback(
    (eventName: string) => {
      if (!enabled || !performance) return;

      const markName = `${componentName}-${eventName}`;
      performance.mark(markName);

      if (import.meta.env.DEV) {
        console.log(`[Performance] Mark: ${markName}`);
      }
    },
    [enabled, componentName]
  );

  // Measure between two marks
  const measure = useCallback(
    (eventName: string, startMark: string, endMark?: string) => {
      if (!enabled || !performance) return null;

      const measureName = `${componentName}-${eventName}`;
      const startMarkName = `${componentName}-${startMark}`;
      const endMarkName = endMark ? `${componentName}-${endMark}` : undefined;

      try {
        performance.measure(measureName, startMarkName, endMarkName);

        const measure = performance.getEntriesByName(measureName, 'measure')[0];
        if (measure && import.meta.env.DEV) {
          console.log(
            `[Performance] ${measureName}: ${measure.duration.toFixed(2)}ms`
          );
        }

        return measure?.duration || null;
      } catch (error) {
        console.warn(`[Performance] Failed to measure ${measureName}:`, error);
        return null;
      }
    },
    [enabled, componentName]
  );

  // Clear all marks and measures for this component
  const clearMetrics = useCallback(() => {
    if (!enabled || !performance) return;

    performance.getEntriesByName(componentName).forEach((entry) => {
      performance.clearMarks(entry.name);
      performance.clearMeasures(entry.name);
    });
  }, [enabled, componentName]);

  return {
    getMetrics,
    mark,
    measure,
    clearMetrics,
  };
}


