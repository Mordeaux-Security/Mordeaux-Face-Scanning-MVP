import { createLogger } from '@mordeaux/common';

const logger = createLogger('policy-client');

export interface Policy {
  tenant_id: string;
  search_enabled: boolean;
  max_search_results: number;
  allowed_sources: string[];
  retention_days: number;
  alert_threshold: number;
  features: {
    face_detection: boolean;
    clustering: boolean;
    similarity_search: boolean;
    real_time_alerts: boolean;
  };
  restrictions: {
    max_file_size_mb: number;
    allowed_file_types: string[];
    rate_limit_per_minute: number;
  };
  created_at: string;
  updated_at: string;
}

export interface PolicyResponse {
  policy: Policy;
  found: boolean;
}

export interface PolicyError {
  error: string;
  message: string;
  tenant_id?: string;
}

export class PolicyClient {
  private baseUrl: string;

  constructor(baseUrl: string = process.env.POLICY_ENGINE_URL || 'http://policy-engine:3004') {
    this.baseUrl = baseUrl;
  }

  async resolvePolicy(tenant_id: string): Promise<PolicyResponse | PolicyError> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/policies/resolve?tenant_id=${tenant_id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        if (response.status === 404) {
          const errorData = await response.json() as PolicyError;
          logger.warn('Policy not found', { tenant_id, error: errorData });
          return errorData;
        }
        throw new Error(`Policy engine error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json() as PolicyResponse;
      logger.info('Policy resolved successfully', { 
        tenant_id, 
        search_enabled: data.policy.search_enabled,
        max_search_results: data.policy.max_search_results
      });

      return data;
    } catch (error) {
      logger.error('Failed to resolve policy', error as Error, { tenant_id });
      return {
        error: 'Policy resolution failed',
        message: (error as Error).message,
        tenant_id
      };
    }
  }

  async isSearchAllowed(tenant_id: string): Promise<boolean> {
    try {
      const result = await this.resolvePolicy(tenant_id);
      
      if ('error' in result) {
        logger.warn('Policy resolution failed, denying search', { 
          tenant_id, 
          error: result.error 
        });
        return false;
      }

      const isAllowed = result.policy.search_enabled;
      logger.info('Search permission check', { 
        tenant_id, 
        search_enabled: result.policy.search_enabled,
        allowed: isAllowed
      });

      return isAllowed;
    } catch (error) {
      logger.error('Failed to check search permission', error as Error, { tenant_id });
      return false;
    }
  }

  async getMaxSearchResults(tenant_id: string): Promise<number> {
    try {
      const result = await this.resolvePolicy(tenant_id);
      
      if ('error' in result) {
        logger.warn('Policy resolution failed, using default max results', { 
          tenant_id, 
          error: result.error 
        });
        return 10; // Default fallback
      }

      return result.policy.max_search_results;
    } catch (error) {
      logger.error('Failed to get max search results', error as Error, { tenant_id });
      return 10; // Default fallback
    }
  }

  async getAlertThreshold(tenant_id: string): Promise<number> {
    try {
      const result = await this.resolvePolicy(tenant_id);
      
      if ('error' in result) {
        logger.warn('Policy resolution failed, using default alert threshold', { 
          tenant_id, 
          error: result.error 
        });
        return 0.8; // Default fallback
      }

      return result.policy.alert_threshold;
    } catch (error) {
      logger.error('Failed to get alert threshold', error as Error, { tenant_id });
      return 0.8; // Default fallback
    }
  }
}
