import { PolicyClient, Policy, PolicyResponse, PolicyError } from '../policy-client';

// Mock fetch
global.fetch = jest.fn();

describe('PolicyClient', () => {
  let policyClient: PolicyClient;
  const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

  beforeEach(() => {
    policyClient = new PolicyClient('http://test-policy-engine:3004');
    mockFetch.mockClear();
  });

  describe('resolvePolicy', () => {
    it('should resolve policy successfully', async () => {
      const mockPolicy: Policy = {
        tenant_id: '00000000-0000-0000-0000-000000000001',
        search_enabled: true,
        max_search_results: 100,
        allowed_sources: ['camera-1', 'upload-api'],
        retention_days: 30,
        alert_threshold: 0.8,
        features: {
          face_detection: true,
          clustering: true,
          similarity_search: true,
          real_time_alerts: true
        },
        restrictions: {
          max_file_size_mb: 10,
          allowed_file_types: ['jpg', 'jpeg', 'png'],
          rate_limit_per_minute: 60
        },
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z'
      };

      const mockResponse: PolicyResponse = {
        policy: mockPolicy,
        found: true
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await policyClient.resolvePolicy('00000000-0000-0000-0000-000000000001');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-policy-engine:3004/v1/policies/resolve?tenant_id=00000000-0000-0000-0000-000000000001',
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      expect(result).toEqual(mockResponse);
    });

    it('should handle policy not found (404)', async () => {
      const mockError: PolicyError = {
        error: 'Policy not found',
        message: 'No policy found for tenant_id: 00000000-0000-0000-0000-000000000999',
        tenant_id: '00000000-0000-0000-0000-000000000999'
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => mockError
      } as Response);

      const result = await policyClient.resolvePolicy('00000000-0000-0000-0000-000000000999');

      expect(result).toEqual(mockError);
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await policyClient.resolvePolicy('00000000-0000-0000-0000-000000000001');

      expect(result).toEqual({
        error: 'Policy resolution failed',
        message: 'Network error',
        tenant_id: '00000000-0000-0000-0000-000000000001'
      });
    });

    it('should handle server errors (500)', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      } as Response);

      const result = await policyClient.resolvePolicy('00000000-0000-0000-0000-000000000001');

      expect(result).toEqual({
        error: 'Policy resolution failed',
        message: 'Policy engine error: 500 Internal Server Error',
        tenant_id: '00000000-0000-0000-0000-000000000001'
      });
    });
  });

  describe('isSearchAllowed', () => {
    it('should return true when search is enabled', async () => {
      const mockPolicy: Policy = {
        tenant_id: '00000000-0000-0000-0000-000000000001',
        search_enabled: true,
        max_search_results: 100,
        allowed_sources: ['camera-1'],
        retention_days: 30,
        alert_threshold: 0.8,
        features: {
          face_detection: true,
          clustering: true,
          similarity_search: true,
          real_time_alerts: true
        },
        restrictions: {
          max_file_size_mb: 10,
          allowed_file_types: ['jpg'],
          rate_limit_per_minute: 60
        },
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ policy: mockPolicy, found: true })
      } as Response);

      const result = await policyClient.isSearchAllowed('00000000-0000-0000-0000-000000000001');

      expect(result).toBe(true);
    });

    it('should return false when search is disabled', async () => {
      const mockPolicy: Policy = {
        tenant_id: '00000000-0000-0000-0000-000000000002',
        search_enabled: false,
        max_search_results: 0,
        allowed_sources: ['upload-api'],
        retention_days: 7,
        alert_threshold: 0.9,
        features: {
          face_detection: true,
          clustering: false,
          similarity_search: false,
          real_time_alerts: false
        },
        restrictions: {
          max_file_size_mb: 5,
          allowed_file_types: ['jpg'],
          rate_limit_per_minute: 30
        },
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ policy: mockPolicy, found: true })
      } as Response);

      const result = await policyClient.isSearchAllowed('00000000-0000-0000-0000-000000000002');

      expect(result).toBe(false);
    });

    it('should return false when policy resolution fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({
          error: 'Policy not found',
          message: 'No policy found',
          tenant_id: '00000000-0000-0000-0000-000000000999'
        })
      } as Response);

      const result = await policyClient.isSearchAllowed('00000000-0000-0000-0000-000000000999');

      expect(result).toBe(false);
    });

    it('should return false when network error occurs', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await policyClient.isSearchAllowed('00000000-0000-0000-0000-000000000001');

      expect(result).toBe(false);
    });
  });

  describe('getMaxSearchResults', () => {
    it('should return max search results from policy', async () => {
      const mockPolicy: Policy = {
        tenant_id: '00000000-0000-0000-0000-000000000001',
        search_enabled: true,
        max_search_results: 50,
        allowed_sources: ['camera-1'],
        retention_days: 30,
        alert_threshold: 0.8,
        features: {
          face_detection: true,
          clustering: true,
          similarity_search: true,
          real_time_alerts: true
        },
        restrictions: {
          max_file_size_mb: 10,
          allowed_file_types: ['jpg'],
          rate_limit_per_minute: 60
        },
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ policy: mockPolicy, found: true })
      } as Response);

      const result = await policyClient.getMaxSearchResults('00000000-0000-0000-0000-000000000001');

      expect(result).toBe(50);
    });

    it('should return default value when policy resolution fails', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await policyClient.getMaxSearchResults('00000000-0000-0000-0000-000000000001');

      expect(result).toBe(10); // Default fallback
    });
  });

  describe('getAlertThreshold', () => {
    it('should return alert threshold from policy', async () => {
      const mockPolicy: Policy = {
        tenant_id: '00000000-0000-0000-0000-000000000001',
        search_enabled: true,
        max_search_results: 100,
        allowed_sources: ['camera-1'],
        retention_days: 30,
        alert_threshold: 0.7,
        features: {
          face_detection: true,
          clustering: true,
          similarity_search: true,
          real_time_alerts: true
        },
        restrictions: {
          max_file_size_mb: 10,
          allowed_file_types: ['jpg'],
          rate_limit_per_minute: 60
        },
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ policy: mockPolicy, found: true })
      } as Response);

      const result = await policyClient.getAlertThreshold('00000000-0000-0000-0000-000000000001');

      expect(result).toBe(0.7);
    });

    it('should return default value when policy resolution fails', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await policyClient.getAlertThreshold('00000000-0000-0000-0000-000000000001');

      expect(result).toBe(0.8); // Default fallback
    });
  });

  describe('constructor', () => {
    it('should use default URL when none provided', () => {
      const client = new PolicyClient();
      expect(client).toBeInstanceOf(PolicyClient);
    });

    it('should use provided URL', () => {
      const customUrl = 'http://custom-policy-engine:8080';
      const client = new PolicyClient(customUrl);
      expect(client).toBeInstanceOf(PolicyClient);
    });
  });
});
