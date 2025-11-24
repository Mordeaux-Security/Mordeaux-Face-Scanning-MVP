/**
 * useUrlState Hook - Phase 7
 * ===========================
 * 
 * Custom hook for synchronizing component state with URL search parameters.
 * Enables deep-linking: copy/paste URL restores exact view.
 * 
 * Features:
 * - Bidirectional sync (state ↔ URL)
 * - Type-safe getters/setters
 * - Default values
 * - Browser history integration
 * - Works across page reloads
 * 
 * Usage:
 * ```tsx
 * const [state, setState] = useUrlState({
 *   page: { default: 1, parse: parseInt },
 *   minScore: { default: 0, parse: parseFloat },
 *   view: { default: 'grid', parse: (v) => v },
 * });
 * 
 * // Read: state.page
 * // Write: setState({ page: 2 })
 * ```
 */

import { useSearchParams } from 'react-router-dom';
import { useMemo, useCallback } from 'react';

/**
 * Type definition for state configuration
 */
type StateConfig<T> = {
  [K in keyof T]: {
    default: T[K];
    parse: (value: string) => T[K];
    serialize?: (value: T[K]) => string;
  };
};

/**
 * Custom hook for URL state synchronization
 */
export function useUrlState<T extends Record<string, any>>(
  config: StateConfig<T>
): [T, (updates: Partial<T>) => void, () => void] {
  const [searchParams, setSearchParams] = useSearchParams();
  
  /**
   * Read current state from URL
   */
  const state = useMemo(() => {
    const result = {} as T;
    
    for (const key in config) {
      const { default: defaultValue, parse } = config[key];
      const urlValue = searchParams.get(key);
      
      if (urlValue !== null) {
        try {
          result[key] = parse(urlValue);
        } catch (error) {
          console.warn(`[useUrlState] Failed to parse "${key}": ${urlValue}`, error);
          result[key] = defaultValue;
        }
      } else {
        result[key] = defaultValue;
      }
    }
    
    return result;
  }, [searchParams, config]);
  
  /**
   * Update state and sync to URL
   */
  const setState = useCallback((updates: Partial<T>) => {
    setSearchParams((prev) => {
      const newParams = new URLSearchParams(prev);
      
      for (const key in updates) {
        const value = updates[key];
        const { default: defaultValue, serialize } = config[key];
        
        // Remove param if value equals default (keep URL clean)
        if (value === defaultValue) {
          newParams.delete(key);
        } else {
          // Serialize value
          const serialized = serialize 
            ? serialize(value)
            : String(value);
          
          newParams.set(key, serialized);
        }
      }
      
      return newParams;
    });
  }, [setSearchParams, config]);
  
  /**
   * Reset all state to defaults
   */
  const resetState = useCallback(() => {
    setSearchParams(new URLSearchParams());
  }, [setSearchParams]);
  
  return [state, setState, resetState];
}

/**
 * Common parsers for URL state
 */
export const urlParsers = {
  string: (v: string) => v,
  number: (v: string) => {
    const parsed = parseFloat(v);
    return isNaN(parsed) ? 0 : parsed;
  },
  int: (v: string) => {
    const parsed = parseInt(v, 10);
    return isNaN(parsed) ? 0 : parsed;
  },
  boolean: (v: string) => v === 'true' || v === '1',
  array: (v: string) => v.split(',').filter(Boolean),
  json: <T,>(v: string): T | null => {
    try {
      return JSON.parse(v);
    } catch {
      return null;
    }
  },
};

/**
 * Common serializers for URL state
 */
export const urlSerializers = {
  array: (v: any[]) => v.join(','),
  json: (v: any) => JSON.stringify(v),
  boolean: (v: boolean) => v ? '1' : '0',
};

/**
 * Utility: Get URL with current state (for copy/paste)
 */
export function getCurrentUrl(): string {
  return window.location.href;
}

/**
 * Utility: Copy current URL to clipboard
 */
export async function copyCurrentUrl(): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(window.location.href);
    return true;
  } catch (error) {
    console.error('[useUrlState] Failed to copy URL:', error);
    return false;
  }
}





