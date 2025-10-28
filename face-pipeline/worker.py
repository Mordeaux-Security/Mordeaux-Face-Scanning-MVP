#!/usr/bin/env python3
"""
Redis Streams Queue Worker

Consumes messages from Redis Streams and processes them through the face pipeline.
Implements graceful shutdown, error handling, dead letter queue, and single-batch mode.

Usage:
    python worker.py                    # Long-running worker
    python worker.py --once             # Single-batch mode (for testing)
    python worker.py --max-batch 16     # Custom batch size

Environment Variables:
    ENABLE_QUEUE_WORKER: Enable/disable worker (default: false)
    REDIS_URL: Redis connection URL (default: redis://redis:6379/0)
    REDIS_STREAM_NAME: Stream name (default: face-processing-queue, alias: FACE_STREAM)
    REDIS_GROUP_NAME: Consumer group name (default: pipeline, alias: FACE_GROUP)
    REDIS_CONSUMER_NAME: Consumer name (default: auto-generated, alias: FACE_CONSUMER)
    MAX_WORKER_CONCURRENCY: Max concurrent workers (default: 5)
    WORKER_BATCH_SIZE: Batch size for message consumption (default: 10)
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from typing import Dict, Any, Optional
import redis.asyncio as aioredis
from config.settings import settings
from pipeline.processor import process_image

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
shutdown_event = asyncio.Event()
redis_client: Optional[aioredis.Redis] = None


def get_consumer_name() -> str:
    """
    Get consumer name from settings or generate default.
    
    Returns:
        Consumer name in format: pipe-{hostname}-{pid}
    """
    if settings.redis_consumer_name:
        return settings.redis_consumer_name
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"pipe-{hostname}-{pid}"


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


def _parse_message(fields: Dict[bytes, bytes]) -> Dict[str, Any]:
    """
    Parse message from Redis Stream fields.
    
    Supports both "message" and "data" fields for backward compatibility.
    Prefers "message" if both exist.
    
    Args:
        fields: Redis Stream message fields
        
    Returns:
        Parsed message dict, or empty dict if parsing fails
    """
    # Check for 'message' field first (preferred), then 'data' (backward compat)
    raw = fields.get(b"message") or fields.get(b"data")
    
    if not raw:
        logger.warning("Message has neither 'message' nor 'data' field")
        return {}
    
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from message: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error parsing message: {e}")
        return {}


async def ensure_consumer_group(r: aioredis.Redis):
    """
    Ensure consumer group exists, create if missing.
    
    Args:
        r: Redis client
        
    Raises:
        redis.ResponseError: If creation fails for reasons other than BUSYGROUP
    """
    try:
        # Create stream + group if missing (mkstream=True)
        await r.xgroup_create(
            name=settings.redis_stream_name,
            groupname=settings.redis_group_name,
            id="$",  # Start from end for new groups
            mkstream=True
        )
        logger.info(f"Created consumer group '{settings.redis_group_name}' for stream '{settings.redis_stream_name}'")
    except aioredis.ResponseError as e:
        # BUSYGROUP means it already exists - this is fine
        if "BUSYGROUP" in str(e):
            logger.debug(f"Consumer group '{settings.redis_group_name}' already exists")
        else:
            raise


async def handle_message(
    r: aioredis.Redis,
    stream: bytes,
    message_id: bytes,
    fields: Dict[bytes, bytes]
) -> bool:
    """
    Process a single message through the face pipeline.
    
    Args:
        r: Redis client
        stream: Stream name
        message_id: Redis Stream message ID
        fields: Message fields
        
    Returns:
        True if processing succeeded, False otherwise
    """
    message_id_str = message_id.decode("utf-8")
    
    # Parse message
    msg = _parse_message(fields)
    
    if not msg:
        # Move unparsable message to DLQ
        logger.error(f"Message {message_id_str} has no valid data, moving to DLQ")
        dlq_stream = f"{settings.redis_stream_name}:dlq"
        
        await r.xadd(dlq_stream, {
            "error": "empty_or_unparsable",
            "original_id": message_id_str,
            "fields": json.dumps({
                k.decode(errors='ignore'): v.decode(errors='ignore')
                for k, v in fields.items()
            })
        })
        
        # Acknowledge to remove from pending
        await r.xack(settings.redis_stream_name, settings.redis_group_name, message_id)
        return False
    
    # Validate required fields
    required_fields = [
        "image_sha256", "bucket", "key", "tenant_id",
        "site", "url", "image_phash"
    ]
    
    missing_fields = [f for f in required_fields if f not in msg]
    if missing_fields:
        logger.error(f"Message {message_id_str} missing required fields: {missing_fields}")
        dlq_stream = f"{settings.redis_stream_name}:dlq"
        
        await r.xadd(dlq_stream, {
            "error": f"missing_fields: {', '.join(missing_fields)}",
            "original_id": message_id_str,
            "message": json.dumps(msg)
        })
        
        await r.xack(settings.redis_stream_name, settings.redis_group_name, message_id)
        return False
    
    # Process through face pipeline
    try:
        logger.info(f"Processing message {message_id_str} for image {msg.get('image_sha256')}")
        
        result = process_image(msg)
        
        # Check for processing errors
        if "error" in result:
            logger.error(f"Pipeline error for message {message_id_str}: {result['error']}")
            
            # Move to DLQ
            dlq_stream = f"{settings.redis_stream_name}:dlq"
            await r.xadd(dlq_stream, {
                "error": result["error"],
                "original_id": message_id_str,
                "message": json.dumps(msg)
            })
            
            await r.xack(settings.redis_stream_name, settings.redis_group_name, message_id)
            return False
        
        # Success - log and acknowledge
        counts = result.get("counts", {})
        logger.info(
            f"Successfully processed message {message_id_str}: "
            f"faces_total={counts.get('faces_total', 0)}, "
            f"accepted={counts.get('accepted', 0)}, "
            f"rejected={counts.get('rejected', 0)}, "
            f"dup_skipped={counts.get('dup_skipped', 0)}"
        )
        
        await r.xack(settings.redis_stream_name, settings.redis_group_name, message_id)
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error processing message {message_id_str}: {e}", exc_info=True)
        
        # Move to DLQ
        dlq_stream = f"{settings.redis_stream_name}:dlq"
        await r.xadd(dlq_stream, {
            "error": str(e),
            "original_id": message_id_str,
            "message": json.dumps(msg)
        })
        
        # Acknowledge to remove from pending
        await r.xack(settings.redis_stream_name, settings.redis_group_name, message_id)
        return False


async def run_worker(once: bool = False, max_batch: int = 8):
    """
    Main worker loop - consume and process messages from Redis Streams.
    
    Args:
        once: If True, process one batch then exit (for testing)
        max_batch: Maximum messages to read per batch
    """
    consumer_name = get_consumer_name()
    
    # Connect to Redis
    r = aioredis.from_url(settings.redis_url, decode_responses=False)
    
    try:
        # Ensure consumer group exists
        await ensure_consumer_group(r)
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: shutdown_event.set())
        
        logger.info(f"Worker {consumer_name} started")
        logger.info(f"  Stream: {settings.redis_stream_name}")
        logger.info(f"  Group: {settings.redis_group_name}")
        logger.info(f"  Max batch: {max_batch}")
        logger.info(f"  Once mode: {once}")
        
        processed_count = 0
        
        # Main consumption loop
        while not shutdown_event.is_set():
            try:
                # Read messages from stream
                # Use ">" to get new messages not yet delivered to other consumers
                resp = await r.xreadgroup(
                    groupname=settings.redis_group_name,
                    consumername=consumer_name,
                    streams={settings.redis_stream_name: b">"},
                    count=max_batch,
                    block=5000  # 5 second timeout
                )
                
                if not resp:
                    # No messages - continue or exit if once mode
                    if once:
                        logger.info("No messages in batch, exiting (once mode)")
                        break
                    continue
                
                # Process messages
                # resp is list of (stream, [(id, fields), ...])
                for stream, entries in resp:
                    for message_id, fields in entries:
                        success = await handle_message(r, stream, message_id, fields)
                        if success:
                            processed_count += 1
                
                logger.info(f"Processed batch: {len(entries)} messages (total: {processed_count})")
                
                # Exit after first batch if once mode
                if once:
                    logger.info(f"Completed one batch, exiting (once mode)")
                    break
                
            except aioredis.RedisError as e:
                logger.error(f"Redis error in consumption loop: {e}")
                await asyncio.sleep(5)  # Back off on Redis errors
            except Exception as e:
                logger.error(f"Unexpected error in consumption loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info(f"Worker shutting down (processed {processed_count} messages)")
        
    finally:
        await r.aclose()


async def health_check():
    """Periodic health check and monitoring."""
    while not shutdown_event.is_set():
        try:
            # Check Redis connectivity
            r = aioredis.from_url(settings.redis_url)
            await r.ping()
            await r.aclose()
            
            logger.debug("Health check passed")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        await asyncio.sleep(30)  # Check every 30 seconds


async def main_async(once: bool = False, max_batch: int = 8):
    """
    Main async entry point.
    
    Args:
        once: Single-batch mode
        max_batch: Maximum batch size
    """
    # Check if worker is enabled
    if not settings.enable_queue_worker:
        logger.info("Queue worker is disabled via ENABLE_QUEUE_WORKER=false")
        return
    
    logger.info("Starting Redis Streams worker...")
    logger.info(f"Configuration:")
    logger.info(f"  Redis URL: {settings.redis_url}")
    logger.info(f"  Stream: {settings.redis_stream_name}")
    logger.info(f"  Group: {settings.redis_group_name}")
    logger.info(f"  Consumer: {get_consumer_name()}")
    logger.info(f"  Concurrency: {settings.max_worker_concurrency}")
    logger.info(f"  Batch Size: {max_batch}")
    
    # Test Redis connection
    try:
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)
    
    try:
        if once:
            # Once mode - just run worker, no health checks
            await run_worker(once=True, max_batch=max_batch)
        else:
            # Long-running mode - run health check and worker concurrently
            await asyncio.gather(
                run_worker(once=False, max_batch=max_batch),
                health_check(),
                return_exceptions=True
            )
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Worker shutdown complete")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Redis Streams Queue Worker for Face Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python worker.py                    # Long-running worker
  python worker.py --once             # Process one batch then exit
  python worker.py --max-batch 16     # Custom batch size
  python worker.py --once --max-batch 8  # Single batch with size 8
        """
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one batch then exit (for testing/CI)"
    )
    
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Maximum entries to pull per XREADGROUP call (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Run async main
    try:
        asyncio.run(main_async(once=args.once, max_batch=args.max_batch))
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
