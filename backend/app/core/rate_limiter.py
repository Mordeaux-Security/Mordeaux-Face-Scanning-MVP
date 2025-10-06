import time
import redis
from typing import Optional
from fastapi import Request, HTTPException, status
import logging
from .config import get_settings
from .metrics import record_rate_limit_violation

logger = logging.getLogger(__name__)

# Redis connection
_redis_client = None

def get_redis_client():
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.redis_client = get_redis_client()
    
    def is_rate_limited(self, tenant_id: str) -> bool:
        """Check if tenant has exceeded rate limits."""
        current_time = int(time.time())
        
        # Check per-minute limit
        minute_key = f"rate_limit:{tenant_id}:minute:{current_time // 60}"
        minute_count = self.redis_client.get(minute_key)
        if minute_count and int(minute_count) >= self.requests_per_minute:
            return True
        
        # Check per-hour limit
        hour_key = f"rate_limit:{tenant_id}:hour:{current_time // 3600}"
        hour_count = self.redis_client.get(hour_key)
        if hour_count and int(hour_count) >= self.requests_per_hour:
            return True
        
        return False
    
    def increment_counter(self, tenant_id: str):
        """Increment rate limit counters for tenant."""
        current_time = int(time.time())
        
        # Increment per-minute counter
        minute_key = f"rate_limit:{tenant_id}:minute:{current_time // 60}"
        pipe = self.redis_client.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)  # Expire after 1 minute
        
        # Increment per-hour counter
        hour_key = f"rate_limit:{tenant_id}:hour:{current_time // 3600}"
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)  # Expire after 1 hour
        
        pipe.execute()

# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_per_hour)
    return _rate_limiter

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to enforce rate limiting per tenant."""
    tenant_id = getattr(request.state, "tenant_id", None)
    
    if not tenant_id:
        # Skip rate limiting if no tenant ID (shouldn't happen due to tenant middleware)
        response = await call_next(request)
        return response
    
    rate_limiter = get_rate_limiter()
    
    # Check if rate limited
    if rate_limiter.is_rate_limited(tenant_id):
        # Record rate limit violation
        record_rate_limit_violation(tenant_id)
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Increment counter
    rate_limiter.increment_counter(tenant_id)
    
    response = await call_next(request)
    return response
