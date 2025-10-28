#!/usr/bin/env python3
"""
Test Message Publisher for Redis Streams Worker

Publishes a test message to the Redis Stream for worker consumption.
Demonstrates both "message" (preferred) and "data" (legacy) field formats.

Usage:
    python scripts/publish_test_message.py
    python scripts/publish_test_message.py --format data  # Use legacy format
"""

import argparse
import json
import os
import redis


def publish_test_message(use_data_field: bool = False):
    """
    Publish a test message to Redis Stream.
    
    Args:
        use_data_field: If True, use "data" field (legacy), else use "message" (preferred)
    """
    # Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    stream_name = os.getenv("REDIS_STREAM_NAME", "face-processing-queue")
    
    r = redis.Redis.from_url(redis_url)
    
    # Test message payload
    payload = {
        "image_sha256": "test123abc456def789",
        "bucket": "raw-images",
        "key": "demo/test_image.jpg",
        "tenant_id": "demo",
        "site": "local",
        "url": "file:///tmp/test.jpg",
        "image_phash": "0000000000000000",
        "face_hints": None
    }
    
    # Choose field name
    field_name = "data" if use_data_field else "message"
    
    # Publish to stream
    message_id = r.xadd(stream_name, {field_name: json.dumps(payload)})
    
    print(f"✅ Published test message to stream '{stream_name}'")
    print(f"   Message ID: {message_id.decode('utf-8')}")
    print(f"   Field: {field_name}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    print(f"\nStream info:")
    
    try:
        stream_info = r.xinfo_stream(stream_name)
        print(f"   Length: {stream_info['length']}")
        print(f"   First entry: {stream_info.get('first-entry', 'N/A')}")
        print(f"   Last entry: {stream_info.get('last-entry', 'N/A')}")
    except Exception as e:
        print(f"   Could not get stream info: {e}")
    
    print(f"\nTo consume this message, run:")
    print(f"   python worker.py --once")


def main():
    parser = argparse.ArgumentParser(
        description="Publish test message to Redis Stream",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--format",
        choices=["message", "data"],
        default="message",
        help="Message field format: 'message' (preferred) or 'data' (legacy)"
    )
    
    args = parser.parse_args()
    
    use_data_field = (args.format == "data")
    
    try:
        publish_test_message(use_data_field)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

