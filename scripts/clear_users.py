#!/usr/bin/env python3
"""
Clear all user emails and identities from Redis.

This script removes all user signup data stored in Redis, allowing you to
re-register with the same emails.

Usage:
    python scripts/clear_users.py [--tenant-id TENANT_ID]

Options:
    --tenant-id: Only clear users for a specific tenant (default: clears all tenants)
"""

import os
import sys
import argparse
import redis
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

def clear_users(tenant_id: str = None):
    """Clear all user emails and identities from Redis."""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        print(f"✅ Connected to Redis: {redis_url}")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return False
    
    # Build pattern to match user keys
    if tenant_id:
        user_pattern = f"user:{tenant_id}:*"
        identity_pattern = f"user-identity:{tenant_id}:*"
        print(f"🔍 Clearing users for tenant: {tenant_id}")
    else:
        user_pattern = "user:*"
        identity_pattern = "user-identity:*"
        print("🔍 Clearing all users (all tenants)")
    
    # Find all matching keys
    user_keys = list(client.keys(user_pattern))
    identity_keys = list(client.keys(identity_pattern))
    
    total_keys = len(user_keys) + len(identity_keys)
    
    if total_keys == 0:
        print("ℹ️  No user keys found to clear")
        return True
    
    print(f"📋 Found {len(user_keys)} user keys and {len(identity_keys)} identity keys")
    
    # Delete all keys
    if user_keys:
        deleted_users = client.delete(*user_keys)
        print(f"✅ Deleted {deleted_users} user keys")
    
    if identity_keys:
        deleted_identities = client.delete(*identity_keys)
        print(f"✅ Deleted {deleted_identities} identity keys")
    
    print(f"🎉 Successfully cleared {total_keys} keys")
    return True

def main():
    parser = argparse.ArgumentParser(description="Clear all user emails from Redis")
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=None,
        help="Only clear users for a specific tenant (default: clears all tenants)"
    )
    args = parser.parse_args()
    
    success = clear_users(tenant_id=args.tenant_id)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

