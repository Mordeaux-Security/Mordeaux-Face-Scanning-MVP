/**
 * useLazyImage - Phase 9 Performance Hardening
 * ==============================================
 * 
 * Hook for lazy loading images using Intersection Observer API.
 * Only loads images when they enter the viewport.
 * 
 * Features:
 * - Intersection Observer for viewport detection
 * - Configurable threshold and root margin
 * - Automatic cleanup on unmount
 */

import { useEffect, useRef, useState } from 'react';

interface UseLazyImageOptions {
  threshold?: number;
  rootMargin?: string;
}

export function useLazyImage(
  src: string,
  options: UseLazyImageOptions = {}
): {
  ref: React.RefObject<HTMLImageElement>;
  isInView: boolean;
  shouldLoad: boolean;
} {
  const { threshold = 0.1, rootMargin = '50px' } = options;
  
  const ref = useRef<HTMLImageElement>(null);
  const [isInView, setIsInView] = useState(false);
  const [shouldLoad, setShouldLoad] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    // Check if IntersectionObserver is supported
    if (!('IntersectionObserver' in window)) {
      // Fallback: load immediately if not supported
      setShouldLoad(true);
      setIsInView(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsInView(true);
            setShouldLoad(true);
            // Once loaded, we don't need to observe anymore
            observer.unobserve(element);
          }
        });
      },
      {
        threshold,
        rootMargin,
      }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [src, threshold, rootMargin]);

  return { ref, isInView, shouldLoad };
}


