import Fastify from 'fastify';
import jwt from 'jsonwebtoken';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { HealthCheckSchema, ReadyCheckSchema } from '@mordeaux/contracts';

validateEnv();

const logger = createLogger('auth');
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
    service: 'auth',
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
  return {
    status: 'ready',
    timestamp: new Date().toISOString(),
    service: 'auth',
    dependencies: {
      database: 'up'
    }
  };
});

// Auth endpoints
fastify.post('/v1/auth/issue', {
  schema: {
    description: 'Issue JWT token for development',
    body: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        tenant_id: { type: 'string' }
      },
      required: ['user_id', 'tenant_id']
    }
  }
}, async (request, reply) => {
  const { user_id, tenant_id } = request.body as { user_id: string; tenant_id: string };
  
  const token = jwt.sign(
    { user_id, tenant_id, iat: Math.floor(Date.now() / 1000) },
    env.JWT_SECRET,
    { expiresIn: '1h' }
  );

  logger.info('JWT token issued', { user_id, tenant_id });
  
  return { token };
});

fastify.post('/v1/auth/verify', {
  schema: {
    description: 'Verify JWT token',
    body: {
      type: 'object',
      properties: {
        token: { type: 'string' }
      },
      required: ['token']
    }
  }
}, async (request, reply) => {
  const { token } = request.body as { token: string };
  
  try {
    const decoded = jwt.verify(token, env.JWT_SECRET) as any;
    logger.info('JWT token verified', { user_id: decoded.user_id });
    return { valid: true, payload: decoded };
  } catch (error) {
    logger.warn('JWT token verification failed', { error: (error as Error).message });
    return { valid: false, error: 'Invalid token' };
  }
});

async function start() {
  try {
    await fastify.listen({ 
      port: env.PORT + 1, // Use different port
      host: '0.0.0.0' 
    });

    logger.info(`Auth service started on port ${env.PORT + 1}`);
  } catch (err) {
    logger.error('Failed to start Auth service', err as Error);
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
