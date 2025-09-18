import Fastify from 'fastify';
import { Client } from 'minio';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { HealthCheckSchema, ReadyCheckSchema } from '@mordeaux/contracts';

validateEnv();

const logger = createLogger('ingest-upload');
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

// MinIO client
const minioClient = new Client({
  endPoint: env.MINIO_ENDPOINT.replace('http://', '').replace('https://', ''),
  port: env.MINIO_ENDPOINT.includes('https') ? 443 : 9000,
  useSSL: env.MINIO_ENDPOINT.includes('https'),
  accessKey: env.MINIO_ACCESS_KEY,
  secretKey: env.MINIO_SECRET_KEY,
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
    service: 'ingest-upload',
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
    service: 'ingest-upload',
    dependencies: {
      minio: 'up'
    }
  };
});

// Upload endpoints
fastify.post('/v1/upload/presigned-url', {
  schema: {
    description: 'Generate presigned URL for upload',
    body: {
      type: 'object',
      properties: {
        filename: { type: 'string' },
        content_type: { type: 'string' },
        tenant_id: { type: 'string' }
      },
      required: ['filename', 'content_type', 'tenant_id']
    }
  }
}, async (request, reply) => {
  const { filename, content_type, tenant_id } = request.body as {
    filename: string;
    content_type: string;
    tenant_id: string;
  };

  try {
    const objectName = `raw/${tenant_id}/${Date.now()}-${filename}`;
    const presignedUrl = await minioClient.presignedPutObject('raw', objectName, 60 * 60); // 1 hour

    logger.info('Presigned URL generated', { filename, tenant_id, objectName });

    return {
      presigned_url: presignedUrl,
      object_name: objectName,
      expires_in: 3600
    };
  } catch (error) {
    logger.error('Failed to generate presigned URL', error as Error);
    reply.code(500);
    return { error: 'Failed to generate presigned URL' };
  }
});

async function start() {
  try {
    await fastify.listen({ 
      port: env.PORT + 2, // Use different port
      host: '0.0.0.0' 
    });

    logger.info(`Ingest upload service started on port ${env.PORT + 2}`);
  } catch (err) {
    logger.error('Failed to start Ingest upload service', err as Error);
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
