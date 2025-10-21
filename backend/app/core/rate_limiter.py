import time
import redis
from typing import Optional, Tuple
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

class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation for burst capacity."""

    def __init__(self, requests_per_second: float = 10.0, burst_capacity: int = 50):
        self.requests_per_second = requests_per_second
        self.burst_capacity = burst_capacity
        self.redis_client = get_redis_client()

    def _get_bucket_key(self, tenant_id: str) -> str:
        """Get Redis key for tenant's token bucket."""
        return f"rate_limit:token_bucket:{tenant_id}"

    def _get_bucket_data(self, tenant_id: str) -> Tuple[float, float]:
        """Get current bucket state (tokens, last_refill_time)."""
        key = self._get_bucket_key(tenant_id)
        data = self.redis_client.hmget(key, "tokens", "last_refill")

        if data[0] is None or data[1] is None:
            # Initialize bucket
            return float(self.burst_capacity), time.time()

        return float(data[0]), float(data[1])

    def _refill_tokens(self, tenant_id: str, current_time: float) -> float:
        """Refill tokens based on time elapsed and return current token count."""
        key = self._get_bucket_key(tenant_id)
        tokens, last_refill = self._get_bucket_data(tenant_id)

        # Calculate time elapsed since last refill
        time_elapsed = current_time - last_refill

        # Calculate tokens to add (rate * time_elapsed)
        tokens_to_add = time_elapsed * self.requests_per_second

        # Add tokens, but don't exceed burst capacity
        new_tokens = min(self.burst_capacity, tokens + tokens_to_add)

        # Update bucket state in Redis
        pipe = self.redis_client.pipeline()
        pipe.hmset(key, {
            "tokens": new_tokens,
            "last_refill": current_time
        })
        pipe.expire(key, 3600)  # Expire after 1 hour of inactivity
        pipe.execute()

        return new_tokens

    def consume_token(self, tenant_id: str) -> bool:
        """
        Try to consume a token from the bucket.
        Returns True if token was consumed, False if rate limited.
        """
        current_time = time.time()

        # Refill tokens first
        tokens = self._refill_tokens(tenant_id, current_time)

        if tokens >= 1.0:
            # Consume one token
            key = self._get_bucket_key(tenant_id)
            pipe = self.redis_client.pipeline()
            pipe.hincrbyfloat(key, "tokens", -1.0)
            pipe.expire(key, 3600)
            pipe.execute()
            return True

        return False

class LegacyRateLimiter:
    """Legacy rate limiter for per-minute and per-hour limits."""

    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.redis_client = get_redis_client()

    def is_rate_limited(self, tenant_id: str) -> bool:
        """Check if tenant has exceeded legacy rate limits."""
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
        """Increment legacy rate limit counters for tenant."""
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

class RateLimiter:
    """Combined rate limiter with both token bucket and legacy limits."""

    def __init__(self,
                 requests_per_second: float = 10.0,
                 burst_capacity: int = 50,
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000):
        self.token_bucket = TokenBucketRateLimiter(requests_per_second, burst_capacity)
        self.legacy_limiter = LegacyRateLimiter(requests_per_minute, requests_per_hour)

    def is_rate_limited(self, tenant_id: str) -> bool:
        """Check if tenant has exceeded any rate limits."""
        # Check legacy limits first (per-minute and per-hour)
        if self.legacy_limiter.is_rate_limited(tenant_id):
            return True

        # Check token bucket limit
        if not self.token_bucket.consume_token(tenant_id):
            return True

        return False

    def increment_legacy_counters(self, tenant_id: str):
        """Increment legacy rate limit counters."""
        self.legacy_limiter.increment_counter(tenant_id)

# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(
            requests_per_second=settings.rate_limit_requests_per_second,
            burst_capacity=settings.rate_limit_burst_capacity,
            requests_per_minute=settings.rate_limit_per_minute,
            requests_per_hour=settings.rate_limit_per_hour
        )
    return _rate_limiter

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to enforce rate limiting per tenant with burst capacity."""
    tenant_id = getattr(request.state, "tenant_id", None)

    if not tenant_id:
        # Skip rate limiting if no tenant ID (shouldn't happen due to tenant middleware)
        response = await call_next(request)
        return response

    rate_limiter = get_rate_limiter()

    # Check if rate limited (includes both token bucket and legacy limits)
    if rate_limiter.is_rate_limited(tenant_id):
        # Record rate limit violation
        record_rate_limit_violation(tenant_id)

        # Get current settings for error message
        settings = get_settings()

        # Get request ID from request state if available
        request_id = getattr(request.state, "request_id", None)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limited",
                "message": "Too many requests",
                "request_id": request_id,
                "details": {
                    "sustained_rate_per_second": settings.rate_limit_requests_per_second,
                    "burst_capacity": settings.rate_limit_burst_capacity,
                    "per_minute_limit": settings.rate_limit_per_minute,
                    "per_hour_limit": settings.rate_limit_per_hour
                }
            }
        )

    # Increment legacy counters for monitoring
    rate_limiter.increment_legacy_counters(tenant_id)

    response = await call_next(request)
    return response
