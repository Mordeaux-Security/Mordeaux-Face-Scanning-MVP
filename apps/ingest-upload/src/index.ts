import Fastify from 'fastify';
import { Client } from 'minio';
import { v4 as uuidv4 } from 'uuid';
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

// Bucket name
const BUCKET_NAME = 'raw';

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
fastify.post('/v1/upload/presign', {
  schema: {
    description: 'Generate presigned URL for upload',
    body: {
      type: 'object',
      properties: {
        filename: { 
          type: 'string',
          minLength: 1,
          maxLength: 255,
          pattern: '^[^/\\\\:*?"<>|]+$' // Basic filename validation
        },
        content_type: { 
          type: 'string',
          minLength: 1,
          maxLength: 100
        },
        tenant_id: { 
          type: 'string',
          format: 'uuid'
        }
      },
      required: ['filename', 'content_type', 'tenant_id']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          presigned_url: { type: 'string', format: 'uri' },
          s3_key_raw: { type: 'string' },
          expires_in: { type: 'number' },
          content_id: { type: 'string', format: 'uuid' }
        },
        required: ['presigned_url', 's3_key_raw', 'expires_in', 'content_id']
      }
    }
  }
}, async (request, reply) => {
  const { filename, content_type, tenant_id } = request.body as {
    filename: string;
    content_type: string;
    tenant_id: string;
  };

  try {
    // TODO: Add real authentication/authorization
    // For now, just validate tenant_id format
    
    // Generate unique content ID and S3 key
    const content_id = uuidv4();
    const timestamp = Date.now();
    const sanitizedFilename = filename.replace(/[^a-zA-Z0-9.-]/g, '_');
    const s3_key_raw = `${tenant_id}/${timestamp}-${content_id}-${sanitizedFilename}`;
    
    // Generate presigned URL (1 hour expiry)
    const presignedUrl = await minioClient.presignedPutObject(
      BUCKET_NAME, 
      s3_key_raw, 
      60 * 60, // 1 hour
      {
        'Content-Type': content_type
      }
    );

    logger.info('Presigned URL generated', { 
      content_id, 
      filename, 
      tenant_id, 
      s3_key_raw,
      content_type 
    });

    return {
      presigned_url: presignedUrl,
      s3_key_raw,
      expires_in: 3600,
      content_id
    };
  } catch (error) {
    logger.error('Failed to generate presigned URL', error as Error, { 
      filename, 
      tenant_id, 
      content_type 
    });
    reply.code(500);
    return { error: 'Failed to generate presigned URL' };
  }
});

fastify.post('/v1/upload/commit', {
  schema: {
    description: 'Commit upload and publish NEW_CONTENT event',
    body: {
      type: 'object',
      properties: {
        content_id: { 
          type: 'string',
          format: 'uuid'
        },
        tenant_id: { 
          type: 'string',
          format: 'uuid'
        },
        source_id: { 
          type: 'string',
          format: 'uuid'
        },
        s3_key_raw: { 
          type: 'string',
          minLength: 1,
          maxLength: 500
        },
        url: { 
          type: 'string',
          format: 'uri',
          maxLength: 1000
        }
      },
      required: ['content_id', 'tenant_id', 'source_id', 's3_key_raw']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          success: { type: 'boolean' },
          message: { type: 'string' },
          content_id: { type: 'string', format: 'uuid' }
        },
        required: ['success', 'message', 'content_id']
      }
    }
  }
}, async (request, reply) => {
  const { content_id, tenant_id, source_id, s3_key_raw, url } = request.body as {
    content_id: string;
    tenant_id: string;
    source_id: string;
    s3_key_raw: string;
    url?: string;
  };

  try {
    // TODO: Add real authentication/authorization
    // TODO: Validate that the file actually exists in MinIO
    // TODO: Get file metadata (size, etc.) from MinIO
    
    // Publish NEW_CONTENT event to orchestrator
    const eventPayload = {
      content_id,
      tenant_id,
      source_id,
      s3_key_raw,
      url: url || '',
      fetch_ts: new Date().toISOString()
    };

    // TODO: Replace with actual event publishing to RabbitMQ
    // For now, just log the event
    logger.info('NEW_CONTENT event would be published', { 
      content_id, 
      tenant_id, 
      source_id, 
      s3_key_raw,
      eventPayload 
    });

    // TODO: Call orchestrator service to publish event
    // const orchestratorUrl = `http://orchestrator:3003/dev/publish`;
    // await fetch(orchestratorUrl, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({
    //     type: 'NEW_CONTENT',
    //     payload: eventPayload
    //   })
    // });

    logger.info('Upload committed successfully', { 
      content_id, 
      tenant_id, 
      source_id, 
      s3_key_raw 
    });

    return {
      success: true,
      message: 'Upload committed and NEW_CONTENT event published',
      content_id
    };
  } catch (error) {
    logger.error('Failed to commit upload', error as Error, { 
      content_id, 
      tenant_id, 
      source_id, 
      s3_key_raw 
    });
    reply.code(500);
    return { 
      success: false, 
      error: 'Failed to commit upload' 
    };
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
