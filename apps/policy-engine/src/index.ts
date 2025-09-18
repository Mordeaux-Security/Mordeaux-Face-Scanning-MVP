import Fastify from 'fastify';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { HealthCheckSchema, ReadyCheckSchema } from '@mordeaux/contracts';

validateEnv();

const logger = createLogger('policy-engine');
const fastify = Fastify({
  logger: {
    level: env.LOG_LEVEL,
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true
      }
    }
  }
});

// Policy interface
interface Policy {
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

// Static policy data (in-memory)
const policies: Record<string, Policy> = {
  '00000000-0000-0000-0000-000000000001': {
    tenant_id: '00000000-0000-0000-0000-000000000001',
    search_enabled: true,
    max_search_results: 100,
    allowed_sources: ['camera-1', 'camera-2', 'upload-api'],
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
      allowed_file_types: ['jpg', 'jpeg', 'png', 'webp'],
      rate_limit_per_minute: 60
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  },
  '00000000-0000-0000-0000-000000000002': {
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
      allowed_file_types: ['jpg', 'jpeg'],
      rate_limit_per_minute: 30
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  },
  '00000000-0000-0000-0000-000000000003': {
    tenant_id: '00000000-0000-0000-0000-000000000003',
    search_enabled: true,
    max_search_results: 50,
    allowed_sources: ['camera-1', 'camera-2', 'camera-3', 'upload-api'],
    retention_days: 90,
    alert_threshold: 0.7,
    features: {
      face_detection: true,
      clustering: true,
      similarity_search: true,
      real_time_alerts: true
    },
    restrictions: {
      max_file_size_mb: 20,
      allowed_file_types: ['jpg', 'jpeg', 'png', 'webp', 'bmp'],
      rate_limit_per_minute: 120
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z'
  }
};

// Health check endpoints
fastify.get('/healthz', {
  schema: {
    description: 'Health check endpoint',
    response: {
      200: HealthCheckSchema
    }
  }
}, async (request, reply) => {
  return {
    status: 'ok',
    timestamp: new Date().toISOString(),
    service: 'policy-engine',
    version: '1.0.0'
  };
});

fastify.get('/readyz', {
  schema: {
    description: 'Readiness check endpoint',
    response: {
      200: ReadyCheckSchema
    }
  }
}, async (request, reply) => {
  return {
    status: 'ready',
    timestamp: new Date().toISOString(),
    service: 'policy-engine',
    checks: {
      policies_loaded: Object.keys(policies).length > 0
    }
  };
});

// Policy resolution endpoint
fastify.get('/v1/policies/resolve', {
  schema: {
    description: 'Resolve policy for a tenant',
    querystring: {
      type: 'object',
      properties: {
        tenant_id: {
          type: 'string',
          format: 'uuid'
        }
      },
      required: ['tenant_id']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          policy: {
            type: 'object',
            properties: {
              tenant_id: { type: 'string', format: 'uuid' },
              search_enabled: { type: 'boolean' },
              max_search_results: { type: 'number' },
              allowed_sources: { type: 'array', items: { type: 'string' } },
              retention_days: { type: 'number' },
              alert_threshold: { type: 'number' },
              features: {
                type: 'object',
                properties: {
                  face_detection: { type: 'boolean' },
                  clustering: { type: 'boolean' },
                  similarity_search: { type: 'boolean' },
                  real_time_alerts: { type: 'boolean' }
                }
              },
              restrictions: {
                type: 'object',
                properties: {
                  max_file_size_mb: { type: 'number' },
                  allowed_file_types: { type: 'array', items: { type: 'string' } },
                  rate_limit_per_minute: { type: 'number' }
                }
              },
              created_at: { type: 'string', format: 'date-time' },
              updated_at: { type: 'string', format: 'date-time' }
            }
          },
          found: { type: 'boolean' }
        },
        required: ['policy', 'found']
      },
      404: {
        type: 'object',
        properties: {
          error: { type: 'string' },
          message: { type: 'string' },
          tenant_id: { type: 'string', format: 'uuid' }
        }
      }
    }
  }
}, async (request, reply) => {
  const { tenant_id } = request.query as { tenant_id: string };

  try {
    const policy = policies[tenant_id];
    
    if (!policy) {
      logger.warn('Policy not found for tenant', { tenant_id });
      reply.code(404);
      return {
        error: 'Policy not found',
        message: `No policy found for tenant_id: ${tenant_id}`,
        tenant_id
      };
    }

    logger.info('Policy resolved successfully', { 
      tenant_id, 
      search_enabled: policy.search_enabled,
      max_search_results: policy.max_search_results
    });

    return {
      policy,
      found: true
    };
  } catch (error) {
    logger.error('Failed to resolve policy', error as Error, { tenant_id });
    reply.code(500);
    return {
      error: 'Internal server error',
      message: 'Failed to resolve policy'
    };
  }
});

// List all policies endpoint (for debugging)
fastify.get('/v1/policies', {
  schema: {
    description: 'List all available policies',
    response: {
      200: {
        type: 'object',
        properties: {
          policies: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                tenant_id: { type: 'string', format: 'uuid' },
                search_enabled: { type: 'boolean' },
                max_search_results: { type: 'number' }
              }
            }
          },
          total: { type: 'number' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const policyList = Object.values(policies).map(policy => ({
      tenant_id: policy.tenant_id,
      search_enabled: policy.search_enabled,
      max_search_results: policy.max_search_results
    }));

    logger.info('Listed all policies', { total: policyList.length });

    return {
      policies: policyList,
      total: policyList.length
    };
  } catch (error) {
    logger.error('Failed to list policies', error as Error);
    reply.code(500);
    return {
      error: 'Internal server error',
      message: 'Failed to list policies'
    };
  }
});

async function start() {
  try {
    await fastify.listen({ 
      port: env.PORT + 4, // Use port 3004
      host: '0.0.0.0' 
    });
    logger.info(`Policy engine service listening on port ${env.PORT + 4}`);
    logger.info(`Loaded ${Object.keys(policies).length} policies`);
  } catch (err) {
    logger.error('Policy engine service failed to start', err);
    process.exit(1);
  }
}

const shutdown = async () => {
  logger.info('Shutting down policy engine service...');
  await fastify.close();
  logger.info('Policy engine service shut down gracefully.');
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

start();
