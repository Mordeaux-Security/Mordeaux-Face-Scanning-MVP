#!/usr/bin/env python3
"""Face processing worker."""

import json
import time
from mordeaux_common import get_logger, setup_logging, load_env
import pika

logger = get_logger("face-worker")


def process_message(ch, method, properties, body):
    """Process a message from the queue."""
    try:
        message = json.loads(body)
        logger.info("Processing message", message=message)
        
        # TODO: Implement actual face processing
        # For now, just log the message
        time.sleep(1)  # Simulate processing
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Message processed successfully")
        
    except Exception as e:
        logger.error("Failed to process message", error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    """Main worker function."""
    setup_logging("face-worker")
    env = load_env()
    
    logger.info("Starting face worker")
    
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(
        pika.URLParameters(env.rabbitmq_url)
    )
    channel = connection.channel()
    
    # Declare queue
    channel.queue_declare(queue='face_processing', durable=True)
    
    # Set up consumer
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue='face_processing',
        on_message_callback=process_message
    )
    
    logger.info("Waiting for messages. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Stopping worker")
        channel.stop_consuming()
        connection.close()


if __name__ == "__main__":
    main()
