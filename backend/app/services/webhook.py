import asyncio
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import logging


from ..core.config import get_settings

logger = logging.getLogger(__name__)

class WebhookEvent:
    """Represents a webhook event."""

    def __init__(self, event_type: str, tenant_id: str, data: Dict[str, Any],
                 request_id: Optional[str] = None, timestamp: Optional[float] = None):
        self.event_type = event_type
        self.tenant_id = tenant_id
        self.data = data
        self.request_id = request_id or f"webhook_{int(time.time() * 1000)}"
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "data": self.data,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "version": "1.0"
        }

class WebhookEndpoint:
    """Represents a webhook endpoint configuration."""

    def __init__(self, url: str, events: List[str], secret: Optional[str] = None,
                 timeout: int = 30, retry_count: int = 3, retry_delay: int = 1):
        self.url = url
        self.events = events
        self.secret = secret
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.created_at = time.time()
        self.last_used = None
        self.success_count = 0
        self.failure_count = 0

    def should_trigger(self, event_type: str) -> bool:
        """Check if this endpoint should be triggered for the given event type."""
        return event_type in self.events

    def generate_signature(self, payload: str) -> str:
        """Generate HMAC signature for webhook payload."""
        if not self.secret:
            return ""

        signature = hmac.new(
            self.secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return f"sha256={signature}"

class WebhookService:
    """Service for managing and sending webhooks."""

    def __init__(self):
        self.settings = get_settings()
        self.endpoints: Dict[str, List[WebhookEndpoint]] = {}  # tenant_id -> endpoints
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for webhook delivery."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def register_webhook(self, tenant_id: str, url: str, events: List[str],
                              secret: Optional[str] = None, **kwargs) -> str:
        """Register a new webhook endpoint for a tenant."""
        async with self._lock:
            if tenant_id not in self.endpoints:
                self.endpoints[tenant_id] = []

            endpoint = WebhookEndpoint(url, events, secret, **kwargs)
            self.endpoints[tenant_id].append(endpoint)

            logger.info(f"Registered webhook for tenant {tenant_id}: {url} (events: {events})")
            return f"webhook_{len(self.endpoints[tenant_id])}"

    async def unregister_webhook(self, tenant_id: str, url: str) -> bool:
        """Unregister a webhook endpoint."""
        async with self._lock:
            if tenant_id not in self.endpoints:
                return False

            original_count = len(self.endpoints[tenant_id])
            self.endpoints[tenant_id] = [
                ep for ep in self.endpoints[tenant_id] if ep.url != url
            ]

            removed = len(self.endpoints[tenant_id]) < original_count
            if removed:
                logger.info(f"Unregistered webhook for tenant {tenant_id}: {url}")

            return removed

    async def list_webhooks(self, tenant_id: str) -> List[Dict[str, Any]]:
        """List all webhook endpoints for a tenant."""
        if tenant_id not in self.endpoints:
            return []

        return [
            {
                "url": ep.url,
                "events": ep.events,
                "has_secret": bool(ep.secret),
                "timeout": ep.timeout,
                "retry_count": ep.retry_count,
                "created_at": ep.created_at,
                "last_used": ep.last_used,
                "success_count": ep.success_count,
                "failure_count": ep.failure_count
            }
            for ep in self.endpoints[tenant_id]
        ]

    async def send_webhook(self, event: WebhookEvent) -> Dict[str, Any]:
        """Send a webhook event to all registered endpoints."""
        if event.tenant_id not in self.endpoints:
            return {"sent": 0, "failed": 0, "results": []}

        endpoints = [
            ep for ep in self.endpoints[event.tenant_id]
            if ep.should_trigger(event.event_type)
        ]

        if not endpoints:
            return {"sent": 0, "failed": 0, "results": []}

        # Send to all matching endpoints concurrently
        tasks = [self._send_to_endpoint(endpoint, event) for endpoint in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        sent = 0
        failed = 0
        detailed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                detailed_results.append({
                    "url": endpoints[i].url,
                    "success": False,
                    "error": str(result)
                })
            else:
                if result["success"]:
                    sent += 1
                else:
                    failed += 1
                detailed_results.append(result)

        logger.info(f"Webhook event {event.event_type} sent to {sent} endpoints, {failed} failed")

        return {
            "sent": sent,
            "failed": failed,
            "results": detailed_results
        }

    async def _send_to_endpoint(self, endpoint: WebhookEndpoint, event: WebhookEvent) -> Dict[str, Any]:
        """Send webhook to a specific endpoint with retry logic."""
        payload = json.dumps(event.to_dict())
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mordeaux-Webhook/1.0",
            "X-Webhook-Event": event.event_type,
            "X-Webhook-Timestamp": str(int(event.timestamp)),
            "X-Webhook-Request-ID": event.request_id
        }

        # Add signature if secret is configured
        if endpoint.secret:
            signature = endpoint.generate_signature(payload)
            headers["X-Webhook-Signature"] = signature

        session = await self._get_session()
        last_error = None

        for attempt in range(endpoint.retry_count + 1):
            try:
                async with session.post(
                    endpoint.url,
                    data=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    endpoint.last_used = time.time()

                    if 200 <= response.status < 300:
                        endpoint.success_count += 1
                        return {
                            "url": endpoint.url,
                            "success": True,
                            "status_code": response.status,
                            "attempt": attempt + 1
                        }
                    else:
                        last_error = f"HTTP {response.status}: {await response.text()}"

            except asyncio.TimeoutError:
                last_error = "Request timeout"
            except Exception as e:
                last_error = str(e)

            # Wait before retry (except on last attempt)
            if attempt < endpoint.retry_count:
                await asyncio.sleep(endpoint.retry_delay * (2 ** attempt))  # Exponential backoff

        # All attempts failed
        endpoint.failure_count += 1
        return {
            "url": endpoint.url,
            "success": False,
            "error": last_error,
            "attempts": endpoint.retry_count + 1
        }

    async def test_webhook(self, tenant_id: str, url: str) -> Dict[str, Any]:
        """Test a webhook endpoint with a test event."""
        test_event = WebhookEvent(
            event_type="test",
            tenant_id=tenant_id,
            data={
                "message": "This is a test webhook from Mordeaux",
                "timestamp": time.time()
            }
        )

        # Create temporary endpoint for testing
        test_endpoint = WebhookEndpoint(url, ["test"], timeout=10, retry_count=1)

        try:
            result = await self._send_to_endpoint(test_endpoint, test_event)
            return {
                "success": result["success"],
                "message": "Test webhook sent successfully" if result["success"] else f"Test failed: {result.get('error', 'Unknown error')}",
                "details": result
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Test webhook failed: {str(e)}",
                "details": {"error": str(e)}
            }

    async def get_webhook_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get webhook statistics for a tenant."""
        if tenant_id not in self.endpoints:
            return {"total_endpoints": 0, "total_success": 0, "total_failures": 0}

        endpoints = self.endpoints[tenant_id]
        total_success = sum(ep.success_count for ep in endpoints)
        total_failures = sum(ep.failure_count for ep in endpoints)

        return {
            "total_endpoints": len(endpoints),
            "total_success": total_success,
            "total_failures": total_failures,
            "success_rate": total_success / (total_success + total_failures) if (total_success + total_failures) > 0 else 0,
            "endpoints": [
                {
                    "url": ep.url,
                    "events": ep.events,
                    "success_count": ep.success_count,
                    "failure_count": ep.failure_count,
                    "last_used": ep.last_used
                }
                for ep in endpoints
            ]
        }

    async def cleanup_old_endpoints(self, max_age_days: int = 30) -> int:
        """Clean up old unused webhook endpoints."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        cleaned_count = 0

        async with self._lock:
            for tenant_id in list(self.endpoints.keys()):
                original_count = len(self.endpoints[tenant_id])
                self.endpoints[tenant_id] = [
                    ep for ep in self.endpoints[tenant_id]
                    if (ep.last_used is None or
                        current_time - ep.last_used < max_age_seconds)
                ]

                cleaned_count += original_count - len(self.endpoints[tenant_id])

                # Remove tenant if no endpoints left
                if not self.endpoints[tenant_id]:
                    del self.endpoints[tenant_id]

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old webhook endpoints")

        return cleaned_count

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

# Global webhook service instance
_webhook_service = None

def get_webhook_service() -> WebhookService:
    """Get webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service

# Webhook event types
WEBHOOK_EVENTS = {
    "face.indexed": "Face successfully indexed",
    "face.searched": "Face search completed",
    "face.compared": "Face comparison completed",
    "batch.created": "Batch job created",
    "batch.completed": "Batch job completed",
    "batch.failed": "Batch job failed",
    "test": "Test webhook event"
}
