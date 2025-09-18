import Fastify from 'fastify';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { HealthCheckSchema, ReadyCheckSchema } from '@mordeaux/contracts';

validateEnv();

const logger = createLogger('vector-index');
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

// In-memory vector storage
interface VectorEntry {
  embedding_id: string;
  vector: number[];
  meta: {
    content_id: string;
    face_id: string;
  };
  timestamp: number;
}

interface VectorIndex {
  [index_ns: string]: VectorEntry[];
}

const vectorIndex: VectorIndex = {};

// Cosine similarity function
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  if (normA === 0 || normB === 0) {
    return 0;
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
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
    service: 'vector-index',
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
    service: 'vector-index',
    checks: {
      memory: 'ok',
      storage: 'ok'
    }
  };
});

// Vector operations
fastify.post('/upsert', {
  schema: {
    description: 'Upsert a vector into the index',
    body: {
      type: 'object',
      properties: {
        embedding_id: { 
          type: 'string',
          format: 'uuid'
        },
        index_ns: { 
          type: 'string',
          minLength: 1,
          maxLength: 100
        },
        vector: { 
          type: 'array',
          items: { type: 'number' },
          minItems: 1,
          maxItems: 2048
        },
        meta: {
          type: 'object',
          properties: {
            content_id: { 
              type: 'string',
              format: 'uuid'
            },
            face_id: { 
              type: 'string',
              format: 'uuid'
            }
          },
          required: ['content_id', 'face_id']
        }
      },
      required: ['embedding_id', 'index_ns', 'vector', 'meta']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          success: { type: 'boolean' },
          message: { type: 'string' },
          embedding_id: { type: 'string', format: 'uuid' },
          index_ns: { type: 'string' }
        },
        required: ['success', 'message', 'embedding_id', 'index_ns']
      }
    }
  }
}, async (request, reply) => {
  const { embedding_id, index_ns, vector, meta } = request.body as {
    embedding_id: string;
    index_ns: string;
    vector: number[];
    meta: {
      content_id: string;
      face_id: string;
    };
  };

  try {
    // Initialize index namespace if it doesn't exist
    if (!vectorIndex[index_ns]) {
      vectorIndex[index_ns] = [];
    }

    // Check if embedding already exists and remove it
    const existingIndex = vectorIndex[index_ns].findIndex(
      entry => entry.embedding_id === embedding_id
    );
    
    if (existingIndex !== -1) {
      vectorIndex[index_ns].splice(existingIndex, 1);
      logger.info('Removed existing embedding', { embedding_id, index_ns });
    }

    // Add new vector entry
    const vectorEntry: VectorEntry = {
      embedding_id,
      vector,
      meta,
      timestamp: Date.now()
    };

    vectorIndex[index_ns].push(vectorEntry);

    logger.info('Vector upserted successfully', { 
      embedding_id, 
      index_ns, 
      vector_length: vector.length,
      total_vectors: vectorIndex[index_ns].length
    });

    return {
      success: true,
      message: 'Vector upserted successfully',
      embedding_id,
      index_ns
    };
  } catch (error) {
    logger.error('Failed to upsert vector', error as Error, { 
      embedding_id, 
      index_ns 
    });
    reply.code(500);
    return { 
      success: false, 
      error: 'Failed to upsert vector' 
    };
  }
});

fastify.post('/query', {
  schema: {
    description: 'Query vectors by similarity',
    body: {
      type: 'object',
      properties: {
        index_ns: { 
          type: 'string',
          minLength: 1,
          maxLength: 100
        },
        vector: { 
          type: 'array',
          items: { type: 'number' },
          minItems: 1,
          maxItems: 2048
        },
        topK: { 
          type: 'number',
          minimum: 1,
          maximum: 100,
          default: 10
        }
      },
      required: ['index_ns', 'vector']
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
                embedding_id: { type: 'string', format: 'uuid' },
                score: { type: 'number' },
                meta: {
                  type: 'object',
                  properties: {
                    content_id: { type: 'string', format: 'uuid' },
                    face_id: { type: 'string', format: 'uuid' }
                  }
                }
              }
            }
          },
          total_found: { type: 'number' },
          query_time_ms: { type: 'number' }
        },
        required: ['results', 'total_found', 'query_time_ms']
      }
    }
  }
}, async (request, reply) => {
  const { index_ns, vector, topK = 10 } = request.body as {
    index_ns: string;
    vector: number[];
    topK?: number;
  };

  try {
    const startTime = Date.now();
    
    // Check if index namespace exists
    if (!vectorIndex[index_ns]) {
      logger.info('Index namespace not found', { index_ns });
      return {
        results: [],
        total_found: 0,
        query_time_ms: Date.now() - startTime
      };
    }

    const vectors = vectorIndex[index_ns];
    if (vectors.length === 0) {
      logger.info('No vectors in index namespace', { index_ns });
      return {
        results: [],
        total_found: 0,
        query_time_ms: Date.now() - startTime
      };
    }

    // Calculate similarities
    const similarities = vectors.map(entry => {
      try {
        const score = cosineSimilarity(vector, entry.vector);
        return {
          embedding_id: entry.embedding_id,
          score,
          meta: entry.meta
        };
      } catch (error) {
        logger.warn('Failed to calculate similarity', { 
          embedding_id: entry.embedding_id, 
          error: (error as Error).message 
        });
        return {
          embedding_id: entry.embedding_id,
          score: 0,
          meta: entry.meta
        };
      }
    });

    // Sort by score (descending) and take topK
    const sortedResults = similarities
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(result => result.score > 0); // Filter out zero scores

    const queryTime = Date.now() - startTime;

    logger.info('Vector query completed', { 
      index_ns, 
      query_vector_length: vector.length,
      total_vectors: vectors.length,
      results_returned: sortedResults.length,
      query_time_ms: queryTime
    });

    return {
      results: sortedResults,
      total_found: sortedResults.length,
      query_time_ms: queryTime
    };
  } catch (error) {
    logger.error('Failed to query vectors', error as Error, { 
      index_ns, 
      vector_length: vector.length 
    });
    reply.code(500);
    return { 
      error: 'Failed to query vectors' 
    };
  }
});

// Stats endpoint for debugging
fastify.get('/stats', {
  schema: {
    description: 'Get vector index statistics',
    response: {
      200: {
        type: 'object',
        properties: {
          namespaces: {
            type: 'object',
            additionalProperties: {
              type: 'object',
              properties: {
                vector_count: { type: 'number' },
                memory_usage_mb: { type: 'number' }
              }
            }
          },
          total_vectors: { type: 'number' },
          total_namespaces: { type: 'number' }
        }
      }
    }
  }
}, async (request, reply) => {
  const namespaces: Record<string, { vector_count: number; memory_usage_mb: number }> = {};
  let totalVectors = 0;

  for (const [ns, vectors] of Object.entries(vectorIndex)) {
    const vectorCount = vectors.length;
    const memoryUsage = (JSON.stringify(vectors).length / 1024 / 1024); // Rough estimate
    
    namespaces[ns] = {
      vector_count: vectorCount,
      memory_usage_mb: Math.round(memoryUsage * 100) / 100
    };
    
    totalVectors += vectorCount;
  }

  return {
    namespaces,
    total_vectors: totalVectors,
    total_namespaces: Object.keys(vectorIndex).length
  };
});

async function start() {
  try {
    await fastify.listen({ 
      port: env.PORT + 6, // Use port 3006
      host: '0.0.0.0' 
    });
    logger.info(`Vector index service listening on port ${env.PORT + 6}`);
  } catch (err) {
    logger.error('Vector index service failed to start', err);
    process.exit(1);
  }
}

const shutdown = async () => {
  logger.info('Shutting down vector index service...');
  await fastify.close();
  logger.info('Vector index service shut down gracefully.');
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

start();
