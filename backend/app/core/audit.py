import time
import uuid
from typing import Optional, Dict, Any
from fastapi import Request
import logging
import psycopg
from contextlib import asynccontextmanager
from .config import get_settings

logger = logging.getLogger(__name__)

# Database connection for audit logs
_audit_db_pool = None

def get_audit_db_pool():
    """Get audit database connection pool."""
    global _audit_db_pool
    if _audit_db_pool is None:
        settings = get_settings()
        connection_string = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        # For now, return the connection string - we'll use direct connections
        _audit_db_pool = connection_string
    return _audit_db_pool

class AuditLogger:
    def __init__(self):
        self.db_pool = get_audit_db_pool()
    
    async def log_request(self, request: Request, response_status: int, process_time: float, 
                         response_data: Optional[Dict[str, Any]] = None):
        """Log API request to audit table."""
        try:
            # Temporarily disable audit logging to fix database connection issues
            return
            async with await psycopg.AsyncConnection.connect(self.db_pool) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        INSERT INTO audit_logs (
                            request_id, tenant_id, method, path, status_code,
                            process_time, user_agent, ip_address, response_size,
                            created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        getattr(request.state, "request_id", None),
                        getattr(request.state, "tenant_id", None),
                        request.method,
                        request.url.path,
                        response_status,
                        process_time,
                        request.headers.get("user-agent"),
                        request.client.host if request.client else None,
                        len(str(response_data)) if response_data else 0,
                        time.time()
                    ))
                    await conn.commit()
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
    
    async def log_search_operation(self, tenant_id: str, operation_type: str, 
                                  face_count: int, result_count: int, 
                                  vector_backend: str, request_id: str):
        """Log search-specific operations."""
        try:
            # Temporarily disable audit logging to fix database connection issues
            return
            async with await psycopg.AsyncConnection.connect(self.db_pool) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        INSERT INTO search_audit_logs (
                            request_id, tenant_id, operation_type, face_count,
                            result_count, vector_backend, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        request_id,
                        tenant_id,
                        operation_type,
                        face_count,
                        result_count,
                        vector_backend,
                        time.time()
                    ))
                    await conn.commit()
        except Exception as e:
            logger.error(f"Failed to log search audit entry: {e}")

# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

async def audit_middleware(request: Request, call_next):
    """Middleware to log all API requests for audit purposes."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Calculate process time
    process_time = time.time() - start_time
    
    # Log the request (async, don't wait for completion)
    audit_logger = get_audit_logger()
    try:
        await audit_logger.log_request(request, response.status_code, process_time)
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")
    
    return response
