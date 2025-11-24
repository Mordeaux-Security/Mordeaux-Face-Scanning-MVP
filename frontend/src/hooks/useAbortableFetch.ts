/**
 * useAbortableFetch - Phase 9 Performance Hardening
 * ==================================================
 * 
 * Hook for making abortable fetch requests.
 * Automatically cancels pending requests when component unmounts
 * or when a new request is initiated.
 * 
 * Features:
 * - AbortController for request cancellation
 * - Automatic cleanup on unmount
 * - Cancel previous requests on new requests
 * - Loading and error state management
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface FetchState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

export function useAbortableFetch<T = any>() {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // Fetch function
  const fetchData = useCallback(
    async (url: string, options: RequestInit = {}) => {
      // Cancel any pending requests
      cleanup();

      // Create new AbortController
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setState({ data: null, loading: true, error: null });

      try {
        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Only update state if this request wasn't aborted
        if (!controller.signal.aborted) {
          setState({ data, loading: false, error: null });
        }

        return data;
      } catch (error: any) {
        // Ignore abort errors
        if (error.name === 'AbortError') {
          console.log('[useAbortableFetch] Request aborted');
          return null;
        }

        // Only update state if this request wasn't aborted
        if (!controller.signal.aborted) {
          setState({ data: null, loading: false, error });
        }

        throw error;
      }
    },
    [cleanup]
  );

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    ...state,
    fetchData,
    abort: cleanup,
  };
}


