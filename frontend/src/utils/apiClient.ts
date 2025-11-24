/**
 * apiClient - Phase 13 Backend Integration
 * =========================================
 * 
 * Centralized API client for making requests to the backend.
 * Supports both mock and real API backends with feature flag.
 * 
 * Features:
 * - Configurable API base URL
 * - Request/response interceptors
 * - Error handling and retry logic
 * - Auth headers (X-Tenant-ID)
 * - Logging and metrics
 */

import { logger } from './logger';

// API configuration
export const API_CONFIG = {
  // Use environment variable or default to mock server
  baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  
  // Feature flag: Use real API vs mock
  useRealApi: import.meta.env.VITE_USE_REAL_API === 'true' || false,
  
  // Default headers
  defaultHeaders: {
    'Content-Type': 'application/json',
  },
  
  // Timeout (ms)
  timeout: 30000,
  
  // Retry configuration
  retry: {
    maxAttempts: 3,
    initialDelay: 1000,
    maxDelay: 5000,
    backoffFactor: 2,
  },
};

/**
 * HTTP error taxonomy
 */
export class APIError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public data?: any
  ) {
    super(`API Error ${status}: ${statusText}`);
    this.name = 'APIError';
  }

  /**
   * Get user-friendly error message
   */
  getUserMessage(): string {
    switch (this.status) {
      case 400:
        return 'Invalid request. Please check your input.';
      case 401:
        return 'Unauthorized. Please log in again.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return 'Resource not found.';
      case 408:
        return 'Request timeout. Please try again.';
      case 429:
        return 'Too many requests. Please slow down.';
      case 500:
        return 'Server error. Please try again later.';
      case 502:
        return 'Bad gateway. The server is temporarily unavailable.';
      case 503:
        return 'Service unavailable. Please try again later.';
      case 504:
        return 'Gateway timeout. Please try again.';
      default:
        return 'An error occurred. Please try again.';
    }
  }
}

/**
 * Sleep helper for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Make HTTP request with retry logic
 */
async function makeRequest(
  url: string,
  options: RequestInit,
  attempt: number = 1
): Promise<Response> {
  const { retry } = API_CONFIG;

  try {
    const response = await fetch(url, options);

    // Retry on 5xx errors (server errors)
    if (response.status >= 500 && attempt < retry.maxAttempts) {
      const delay = Math.min(
        retry.initialDelay * Math.pow(retry.backoffFactor, attempt - 1),
        retry.maxDelay
      );

      logger.warn('API_RETRY', {
        url,
        status: response.status,
        attempt,
        delay,
      });

      await sleep(delay);
      return makeRequest(url, options, attempt + 1);
    }

    return response;
  } catch (error: any) {
    // Retry on network errors
    if (attempt < retry.maxAttempts) {
      const delay = Math.min(
        retry.initialDelay * Math.pow(retry.backoffFactor, attempt - 1),
        retry.maxDelay
      );

      logger.warn('API_NETWORK_RETRY', {
        url,
        error: error.message,
        attempt,
        delay,
      });

      await sleep(delay);
      return makeRequest(url, options, attempt + 1);
    }

    throw error;
  }
}

/**
 * API Client
 */
export class APIClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor(baseUrl?: string, headers?: HeadersInit) {
    this.baseUrl = baseUrl || API_CONFIG.baseUrl;
    this.defaultHeaders = headers || API_CONFIG.defaultHeaders;
  }

  /**
   * Make API request
   */
  async request<T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const startTime = performance.now();

    logger.info('API_REQUEST_START', {
      method: options.method || 'GET',
      endpoint,
      url,
    });

    try {
      const response = await makeRequest(url, {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
        },
      });

      const duration = performance.now() - startTime;

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const error = new APIError(response.status, response.statusText, errorData);

        logger.error('API_REQUEST_ERROR', {
          endpoint,
          status: response.status,
          statusText: response.statusText,
          duration,
          data: errorData,
        });

        throw error;
      }

      const data = await response.json();

      logger.info('API_REQUEST_SUCCESS', {
        endpoint,
        status: response.status,
        duration,
      });

      return data;
    } catch (error: any) {
      const duration = performance.now() - startTime;

      if (!(error instanceof APIError)) {
        logger.error('API_REQUEST_FAILED', {
          endpoint,
          error: error.message,
          duration,
        });
      }

      throw error;
    }
  }

  /**
   * GET request
   */
  async get<T = any>(endpoint: string, params?: Record<string, any>): Promise<T> {
    const queryString = params
      ? '?' + new URLSearchParams(params).toString()
      : '';

    return this.request<T>(`${endpoint}${queryString}`, {
      method: 'GET',
    });
  }

  /**
   * POST request
   */
  async post<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data instanceof FormData ? data : JSON.stringify(data),
      headers: data instanceof FormData ? {} : this.defaultHeaders,
    });
  }

  /**
   * PUT request
   */
  async put<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE request
   */
  async delete<T = any>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    });
  }

  /**
   * Add custom header
   */
  setHeader(key: string, value: string) {
    this.defaultHeaders = {
      ...this.defaultHeaders,
      [key]: value,
    };
  }

  /**
   * Set tenant ID header
   */
  setTenantId(tenantId: string) {
    this.setHeader('X-Tenant-ID', tenantId);
  }
}

/**
 * Default API client instance
 */
export const apiClient = new APIClient();

/**
 * Search API endpoints
 */
export const searchAPI = {
  /**
   * Search for faces
   */
  async searchFaces(formData: FormData, topK: number = 50, threshold: number = 0.75) {
    // Add query params
    formData.append('top_k', topK.toString());
    formData.append('threshold', threshold.toString());

    return apiClient.post('/api/v1/search', formData);
  },

  /**
   * Get search by ID
   */
  async getSearch(searchId: string) {
    return apiClient.get(`/api/v1/searches/${searchId}`);
  },
};

/**
 * Health check API
 */
export const healthAPI = {
  async check() {
    return apiClient.get('/api/v1/health');
  },

  async detailed() {
    return apiClient.get('/api/v1/health/detailed');
  },
};


