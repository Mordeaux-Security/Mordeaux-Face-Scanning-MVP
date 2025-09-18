import Fastify from 'fastify';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { HealthCheckSchema, ReadyCheckSchema, SearchByVectorRequest, SearchResult } from '@mordeaux/contracts';

validateEnv();

const logger = createLogger('search-api');
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

// Vector index client
const VECTOR_INDEX_URL = process.env.VECTOR_INDEX_URL || 'http://vector-index:3006';

interface VectorIndexQueryResult {
  embedding_id: string;
  score: number;
  meta: {
    content_id: string;
    face_id: string;
  };
}

interface VectorIndexResponse {
  results: VectorIndexQueryResult[];
  total_found: number;
  query_time_ms: number;
}

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
    service: 'search-api',
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
  // Check if vector-index service is reachable
  try {
    const response = await fetch(`${VECTOR_INDEX_URL}/healthz`);
    if (!response.ok) {
      throw new Error(`Vector index service unhealthy: ${response.status}`);
    }
  } catch (error) {
    logger.error('Vector index service not reachable', error as Error);
    reply.code(503);
    return {
      status: 'not ready',
      timestamp: new Date().toISOString(),
      service: 'search-api',
      checks: {
        vector_index: 'unreachable'
      }
    };
  }

  return {
    status: 'ready',
    timestamp: new Date().toISOString(),
    service: 'search-api',
    checks: {
      vector_index: 'ok'
    }
  };
});

// Search endpoints
fastify.post('/search/by-image', {
  schema: {
    description: 'Search by image file (TODO: implement face detection and embedding)',
    consumes: ['multipart/form-data'],
    body: {
      type: 'object',
      properties: {
        image: {
          type: 'string',
          format: 'binary'
        },
        tenant_id: {
          type: 'string',
          format: 'uuid'
        },
        topK: {
          type: 'number',
          minimum: 1,
          maximum: 100,
          default: 10
        }
      },
      required: ['image', 'tenant_id']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                content_id: { type: 'string', format: 'uuid' },
                face_id: { type: 'string', format: 'uuid' },
                score: { type: 'number' },
                thumb_s3_key: { type: 'string' },
                cluster_id: { type: 'string', format: 'uuid' }
              }
            }
          },
          message: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  // TODO: Implement face detection and embedding extraction
  logger.info('Search by image endpoint called (not implemented)', { 
    tenant_id: (request.body as any).tenant_id 
  });

  return {
    results: [],
    message: 'Search by image not implemented yet - requires face detection and embedding extraction'
  };
});

fastify.post('/search/by-vector', {
  schema: {
    description: 'Search by vector similarity',
    body: {
      type: 'object',
      properties: {
        vector: {
          type: 'array',
          items: { type: 'number' },
          minItems: 1,
          maxItems: 2048
        },
        filters: {
          type: 'object',
          properties: {
            tenant_id: { type: 'string', format: 'uuid' },
            site: { type: 'string', maxLength: 100 },
            ts_from: { type: 'string', format: 'date-time' },
            ts_to: { type: 'string', format: 'date-time' }
          }
        },
        topK: {
          type: 'number',
          minimum: 1,
          maximum: 100,
          default: 10
        }
      },
      required: ['vector']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                content_id: { type: 'string', format: 'uuid' },
                face_id: { type: 'string', format: 'uuid' },
                score: { type: 'number' },
                thumb_s3_key: { type: 'string' },
                cluster_id: { type: 'string', format: 'uuid' }
              }
            }
          },
          total_found: { type: 'number' },
          query_time_ms: { type: 'number' }
        }
      }
    }
  }
}, async (request, reply) => {
  const { vector, filters, topK = 10 } = request.body as {
    vector: number[];
    filters?: {
      tenant_id?: string;
      site?: string;
      ts_from?: string;
      ts_to?: string;
    };
    topK?: number;
  };

  try {
    // Determine index namespace from filters
    const index_ns = filters?.tenant_id || 'default';
    
    // Call vector-index service
    const vectorIndexResponse = await fetch(`${VECTOR_INDEX_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        index_ns,
        vector,
        topK
      })
    });

    if (!vectorIndexResponse.ok) {
      throw new Error(`Vector index service error: ${vectorIndexResponse.status}`);
    }

    const vectorResults: VectorIndexResponse = await vectorIndexResponse.json();

    // Transform results to SearchResult format
    const searchResults: SearchResult[] = vectorResults.results.map(result => ({
      content_id: result.meta.content_id,
      face_id: result.meta.face_id,
      score: result.score,
      thumb_s3_key: `thumbs/${result.meta.content_id}/${result.meta.face_id}.jpg`, // TODO: Get from database
      cluster_id: undefined // TODO: Get from database
    }));

    // TODO: Apply additional filters (site, timestamp) by querying database
    // For now, just return the vector similarity results

    logger.info('Vector search completed', {
      vector_length: vector.length,
      index_ns,
      results_returned: searchResults.length,
      query_time_ms: vectorResults.query_time_ms
    });

    return {
      results: searchResults,
      total_found: searchResults.length,
      query_time_ms: vectorResults.query_time_ms
    };
  } catch (error) {
    logger.error('Vector search failed', error as Error, {
      vector_length: vector.length,
      filters
    });
    reply.code(500);
    return {
      error: 'Vector search failed',
      message: (error as Error).message
    };
  }
});

// OpenAPI documentation
fastify.register(require('@fastify/swagger'), {
  openapi: {
    info: {
      title: 'Search API',
      description: 'Search API for face protection system',
      version: '1.0.0'
    },
    servers: [
      {
        url: `http://localhost:${env.PORT + 5}`,
        description: 'Development server'
      }
    ]
  }
});

fastify.register(require('@fastify/swagger-ui'), {
  routePrefix: '/docs',
  uiConfig: {
    docExpansion: 'full',
    deepLinking: false
  }
});

async function start() {
  try {
    await fastify.listen({ 
      port: env.PORT + 5, // Use port 3005
      host: '0.0.0.0' 
    });
    logger.info(`Search API service listening on port ${env.PORT + 5}`);
  } catch (err) {
    logger.error('Search API service failed to start', err);
    process.exit(1);
  }
}

const shutdown = async () => {
  logger.info('Shutting down search API service...');
  await fastify.close();
  logger.info('Search API service shut down gracefully.');
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

start();
