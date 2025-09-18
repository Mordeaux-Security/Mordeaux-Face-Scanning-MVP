import Fastify from 'fastify';
import amqp from 'amqplib';
import { createLogger, env, validateEnv } from '@mordeaux/common';
import { 
  HealthCheckSchema, 
  ReadyCheckSchema, 
  EventSchema,
  NewContentEventSchema,
  FacesExtractedEventSchema,
  IndexedEventSchema
} from '@mordeaux/contracts';

validateEnv();

const logger = createLogger('orchestrator');
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

// RabbitMQ connection
let connection: amqp.Connection | null = null;
let channel: amqp.Channel | null = null;

// Event exchange and queue names
const EXCHANGE_NAME = 'events';
const DLQ_NAME = 'events.dlq';
const FACES_EXTRACTED_QUEUE = 'faces_extracted';
const INDEXED_QUEUE = 'indexed';

// Initialize RabbitMQ connection
async function initRabbitMQ() {
  try {
    connection = await amqp.connect(env.RABBITMQ_URL);
    channel = await connection.createChannel();

    // Create exchange
    await channel.assertExchange(EXCHANGE_NAME, 'direct', { durable: true });

    // Create dead letter queue
    await channel.assertQueue(DLQ_NAME, { durable: true });

    // Create queues for orchestrator consumers
    await channel.assertQueue(FACES_EXTRACTED_QUEUE, { 
      durable: true,
      arguments: {
        'x-dead-letter-exchange': '',
        'x-dead-letter-routing-key': DLQ_NAME
      }
    });
    
    await channel.assertQueue(INDEXED_QUEUE, { 
      durable: true,
      arguments: {
        'x-dead-letter-exchange': '',
        'x-dead-letter-routing-key': DLQ_NAME
      }
    });

    // Bind queues to exchange
    await channel.bindQueue(FACES_EXTRACTED_QUEUE, EXCHANGE_NAME, 'FACES_EXTRACTED');
    await channel.bindQueue(INDEXED_QUEUE, EXCHANGE_NAME, 'INDEXED');

    logger.info('RabbitMQ connection initialized');
  } catch (error) {
    logger.error('Failed to initialize RabbitMQ connection', error as Error);
    throw error;
  }
}

// Publish event to RabbitMQ
async function publishEvent(type: string, payload: any) {
  if (!channel) {
    throw new Error('RabbitMQ channel not initialized');
  }

  try {
    // Validate payload against schema
    let validatedPayload;
    switch (type) {
      case 'NEW_CONTENT':
        validatedPayload = NewContentEventSchema.parse(payload);
        break;
      case 'FACES_EXTRACTED':
        validatedPayload = FacesExtractedEventSchema.parse(payload);
        break;
      case 'INDEXED':
        validatedPayload = IndexedEventSchema.parse(payload);
        break;
      default:
        throw new Error(`Unknown event type: ${type}`);
    }

    const message = JSON.stringify({
      type,
      payload: validatedPayload,
      timestamp: new Date().toISOString(),
      source: 'orchestrator'
    });

    const published = channel.publish(
      EXCHANGE_NAME,
      type,
      Buffer.from(message),
      { persistent: true }
    );

    if (published) {
      logger.info('Event published successfully', { type, payload: validatedPayload });
      return { success: true, message: 'Event published successfully' };
    } else {
      throw new Error('Failed to publish event - channel buffer full');
    }
  } catch (error) {
    logger.error('Failed to publish event', error as Error, { type, payload });
    
    // Send to dead letter queue
    try {
      const dlqMessage = JSON.stringify({
        type,
        payload,
        error: (error as Error).message,
        timestamp: new Date().toISOString(),
        source: 'orchestrator'
      });
      
      channel.publish('', DLQ_NAME, Buffer.from(dlqMessage), { persistent: true });
      logger.warn('Event sent to dead letter queue', { type, error: (error as Error).message });
    } catch (dlqError) {
      logger.error('Failed to send to dead letter queue', dlqError as Error);
    }
    
    throw error;
  }
}

// Consumer for FACES_EXTRACTED events
async function consumeFacesExtracted() {
  if (!channel) {
    throw new Error('RabbitMQ channel not initialized');
  }

  await channel.consume(FACES_EXTRACTED_QUEUE, async (msg) => {
    if (!msg) return;

    try {
      const event = JSON.parse(msg.content.toString());
      logger.info('Processing FACES_EXTRACTED event', { event });
      
      // TODO: Implement actual processing logic
      // For now, just log the event
      logger.info('FACES_EXTRACTED event processed (no-op)', { 
        contentId: event.payload.content_id,
        faceCount: event.payload.faces.length 
      });

      channel.ack(msg);
    } catch (error) {
      logger.error('Failed to process FACES_EXTRACTED event', error as Error);
      channel.nack(msg, false, false); // Send to DLQ
    }
  });
}

// Consumer for INDEXED events
async function consumeIndexed() {
  if (!channel) {
    throw new Error('RabbitMQ channel not initialized');
  }

  await channel.consume(INDEXED_QUEUE, async (msg) => {
    if (!msg) return;

    try {
      const event = JSON.parse(msg.content.toString());
      logger.info('Processing INDEXED event', { event });
      
      // TODO: Implement actual processing logic
      // For now, just log the event
      logger.info('INDEXED event processed (no-op)', { 
        contentId: event.payload.content_id,
        embeddingCount: event.payload.embedding_ids.length 
      });

      channel.ack(msg);
    } catch (error) {
      logger.error('Failed to process INDEXED event', error as Error);
      channel.nack(msg, false, false); // Send to DLQ
    }
  });
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
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'orchestrator',
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
    service: 'orchestrator',
    dependencies: {
      rabbitmq: connection ? 'up' : 'down'
    }
  };
});

// Publisher endpoint
fastify.post('/dev/publish', {
  schema: {
    description: 'Publish event to RabbitMQ',
    body: {
      type: 'object',
      properties: {
        type: { 
          type: 'string',
          enum: ['NEW_CONTENT', 'FACES_EXTRACTED', 'INDEXED']
        },
        payload: { type: 'object' }
      },
      required: ['type', 'payload']
    }
  }
}, async (request, reply) => {
  const { type, payload } = request.body as { type: string; payload: any };
  
  try {
    const result = await publishEvent(type, payload);
    return result;
  } catch (error) {
    reply.code(400);
    return { 
      success: false, 
      error: (error as Error).message 
    };
  }
});

async function start() {
  try {
    // Initialize RabbitMQ
    await initRabbitMQ();
    
    // Start consumers
    await consumeFacesExtracted();
    await consumeIndexed();
    
    // Start server
    await fastify.listen({ 
      port: env.PORT + 3, // Use different port
      host: '0.0.0.0' 
    });

    logger.info(`Orchestrator service started on port ${env.PORT + 3}`);
  } catch (err) {
    logger.error('Failed to start Orchestrator service', err as Error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  if (channel) await channel.close();
  if (connection) await connection.close();
  await fastify.close();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('Received SIGINT, shutting down gracefully');
  if (channel) await channel.close();
  if (connection) await connection.close();
  await fastify.close();
  process.exit(0);
});

start();
