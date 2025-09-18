#!/usr/bin/env python3
"""Face processing worker with RabbitMQ consumer."""

import json
import time
import signal
import sys
from typing import Dict, Any
from mordeaux_common import get_logger, setup_logging, load_env, NewContentEvent, ValidationError
import pika

logger = get_logger("face-worker")

class FaceWorker:
    """Face processing worker with RabbitMQ consumer."""
    
    def __init__(self):
        self.env = load_env()
        self.connection = None
        self.channel = None
        self.running = True
        
        # Queue names
        self.exchange_name = 'events'
        self.dlq_name = 'events.dlq'
        self.new_content_queue = 'new_content'
        
    def connect_rabbitmq(self):
        """Connect to RabbitMQ and setup queues."""
        try:
            self.connection = pika.BlockingConnection(
                pika.URLParameters(self.env.rabbitmq_url)
            )
            self.channel = self.connection.channel()
            
            # Declare exchange
            self.channel.exchange_declare(
                exchange=self.exchange_name,
                exchange_type='direct',
                durable=True
            )
            
            # Declare dead letter queue
            self.channel.queue_declare(queue=self.dlq_name, durable=True)
            
            # Declare new content queue with DLQ
            self.channel.queue_declare(
                queue=self.new_content_queue,
                durable=True,
                arguments={
                    'x-dead-letter-exchange': '',
                    'x-dead-letter-routing-key': self.dlq_name
                }
            )
            
            # Bind queue to exchange
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.new_content_queue,
                routing_key='NEW_CONTENT'
            )
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e))
            raise
    
    def validate_event_payload(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate event payload against schema."""
        try:
            if event_type == 'NEW_CONTENT':
                # Validate using Pydantic schema
                validated_event = NewContentEvent(**payload)
                return validated_event.model_dump()
            else:
                raise ValidationError(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error("Payload validation failed", error=str(e), event_type=event_type, payload=payload)
            raise ValidationError(f"Invalid payload for {event_type}: {str(e)}")
    
    def send_to_dlq(self, event_type: str, payload: Dict[str, Any], error: str):
        """Send failed event to dead letter queue."""
        try:
            dlq_message = {
                'type': event_type,
                'payload': payload,
                'error': error,
                'timestamp': time.time(),
                'source': 'face-worker'
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key=self.dlq_name,
                body=json.dumps(dlq_message),
                properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
            )
            
            logger.warn("Event sent to dead letter queue", 
                       event_type=event_type, error=error)
                       
        except Exception as e:
            logger.error("Failed to send to dead letter queue", error=str(e))
    
    def process_new_content(self, payload: Dict[str, Any]):
        """Process NEW_CONTENT event."""
        try:
            # Validate payload
            validated_payload = self.validate_event_payload('NEW_CONTENT', payload)
            
            logger.info("Processing NEW_CONTENT event", 
                       content_id=validated_payload['content_id'],
                       tenant_id=validated_payload['tenant_id'],
                       source_id=validated_payload['source_id'])
            
            # TODO: Implement actual face processing logic
            # For now, just simulate processing
            time.sleep(1)  # Simulate processing time
            
            logger.info("NEW_CONTENT event processed successfully", 
                       content_id=validated_payload['content_id'])
            
        except ValidationError as e:
            logger.error("Validation error in NEW_CONTENT processing", error=str(e))
            raise
        except Exception as e:
            logger.error("Error processing NEW_CONTENT event", error=str(e))
            raise
    
    def process_message(self, ch, method, properties, body):
        """Process a message from the queue."""
        try:
            message = json.loads(body)
            event_type = message.get('type')
            payload = message.get('payload', {})
            
            logger.info("Received message", event_type=event_type, message=message)
            
            if event_type == 'NEW_CONTENT':
                self.process_new_content(payload)
            else:
                logger.warn("Unknown event type received", event_type=event_type)
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Message processed successfully")
            
        except ValidationError as e:
            logger.error("Validation error", error=str(e))
            # Send to DLQ and acknowledge
            self.send_to_dlq(event_type, payload, str(e))
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            # Send to DLQ and acknowledge
            self.send_to_dlq(event_type, payload, str(e))
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    def start_consuming(self):
        """Start consuming messages from the queue."""
        try:
            # Set up consumer
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(
                queue=self.new_content_queue,
                on_message_callback=self.process_message
            )
            
            logger.info("Starting to consume messages from NEW_CONTENT queue")
            self.channel.start_consuming()
            
        except Exception as e:
            logger.error("Error in message consumption", error=str(e))
            raise
    
    def stop_consuming(self):
        """Stop consuming messages."""
        self.running = False
        if self.channel:
            self.channel.stop_consuming()
        logger.info("Stopped consuming messages")
    
    def cleanup(self):
        """Clean up connections."""
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error("Error closing RabbitMQ connection", error=str(e))
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.stop_consuming()
        self.cleanup()
        sys.exit(0)

def main():
    """Main worker function."""
    setup_logging("face-worker")
    
    worker = FaceWorker()
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, worker.signal_handler)
    signal.signal(signal.SIGINT, worker.signal_handler)
    
    try:
        # Connect to RabbitMQ
        worker.connect_rabbitmq()
        
        # Start consuming
        worker.start_consuming()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Worker failed", error=str(e))
    finally:
        worker.cleanup()

if __name__ == "__main__":
    main()
