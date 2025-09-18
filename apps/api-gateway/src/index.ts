import Fastify from 'fastify';
import cors from '@fastify/cors';
import helmet from '@fastify/helmet';
import rateLimit from '@fastify/rate-limit';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { HealthCheckSchema, ReadyCheckSchema } from '@mordeaux/contracts';

// Validate environment
validateEnv();

const logger = createLogger('api-gateway');
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

async function buildApp() {
  // Register plugins
  await fastify.register(cors, {
    origin: true
  });

  await fastify.register(helmet);

  await fastify.register(rateLimit, {
    max: 100,
    timeWindow: '1 minute'
  });

  await fastify.register(swagger, {
    openapi: {
      info: {
        title: 'Mordeaux API Gateway',
        description: 'API Gateway for Mordeaux face protection system',
        version: '1.0.0'
      },
      servers: [
        {
          url: `http://localhost:${env.PORT}`,
          description: 'Development server'
        }
      ]
    }
  });

  await fastify.register(swaggerUi, {
    routePrefix: '/docs',
    uiConfig: {
      docExpansion: 'full',
      deepLinking: false
    }
  });

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
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'api-gateway',
      version: '1.0.0',
      uptime: process.uptime()
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
    // TODO: Check dependencies (auth service, etc.)
    return {
      status: 'ready',
      timestamp: new Date().toISOString(),
      service: 'api-gateway',
      dependencies: {
        auth: 'up',
        search: 'up',
        policy: 'up'
      }
    };
  });

  // API routes
  fastify.register(async function (fastify) {
    // Auth routes
    fastify.register(async function (fastify) {
      fastify.addHook('preHandler', async (request, reply) => {
        // TODO: Implement authentication middleware
        logger.info('Auth route accessed', { url: request.url });
      });

      fastify.post('/auth/login', {
        schema: {
          description: 'User login endpoint',
          tags: ['auth']
        }
      }, async (request, reply) => {
        return { message: 'Login endpoint - TODO: implement' };
      });

      fastify.post('/auth/refresh', {
        schema: {
          description: 'Token refresh endpoint',
          tags: ['auth']
        }
      }, async (request, reply) => {
        return { message: 'Refresh endpoint - TODO: implement' };
      });
    }, { prefix: '/v1' });

    // Search routes
    fastify.register(async function (fastify) {
      fastify.addHook('preHandler', async (request, reply) => {
        // TODO: Implement authentication middleware
        logger.info('Search route accessed', { url: request.url });
      });

      fastify.post('/search/by-image', {
        schema: {
          description: 'Search by image endpoint',
          tags: ['search']
        }
      }, async (request, reply) => {
        return { message: 'Search by image endpoint - TODO: implement' };
      });

      fastify.post('/search/by-vector', {
        schema: {
          description: 'Search by vector endpoint',
          tags: ['search']
        }
      }, async (request, reply) => {
        return { message: 'Search by vector endpoint - TODO: implement' };
      });
    }, { prefix: '/v1' });

    // Policy routes
    fastify.register(async function (fastify) {
      fastify.addHook('preHandler', async (request, reply) => {
        // TODO: Implement authentication middleware
        logger.info('Policy route accessed', { url: request.url });
      });

      fastify.get('/policies/resolve', {
        schema: {
          description: 'Resolve policy for tenant',
          tags: ['policy'],
          querystring: {
            type: 'object',
            properties: {
              tenant_id: { type: 'string', format: 'uuid' }
            },
            required: ['tenant_id']
          }
        }
      }, async (request, reply) => {
        const { tenant_id } = request.query as { tenant_id: string };
        return {
          tenant_id,
          policy: {
            name: 'Default Policy',
            rules: [
              {
                type: 'allow',
                conditions: {},
                actions: ['search', 'view']
              }
            ]
          }
        };
      });
    }, { prefix: '/v1' });

  }, { prefix: '/api' });

  return fastify;
}

async function start() {
  try {
    const app = await buildApp();
    
    await app.listen({ 
      port: env.PORT, 
      host: '0.0.0.0' 
    });

    logger.info(`API Gateway started on port ${env.PORT}`);
  } catch (err) {
    logger.error('Failed to start API Gateway', err as Error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  await fastify.close();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('Received SIGINT, shutting down gracefully');
  await fastify.close();
  process.exit(0);
});

start();
