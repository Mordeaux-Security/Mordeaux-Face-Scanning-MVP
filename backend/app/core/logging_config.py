import logging
import sys
import json
from typing import Dict, Any
from datetime import datetime


from .config import get_settings

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add correlation IDs if available
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'tenant_id'):
            log_entry['tenant_id'] = record.tenant_id

        # Add any extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'request_id', 'tenant_id']:
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

class CorrelationFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation IDs to the log record."""
        # These will be set by the middleware
        if not hasattr(record, 'request_id'):
            record.request_id = None
        if not hasattr(record, 'tenant_id'):
            record.tenant_id = None

        return True

def setup_logging():
    """Setup structured logging with correlation IDs."""
    settings = get_settings()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Add structured formatter
    formatter = StructuredFormatter()
    console_handler.setFormatter(formatter)

    # Add correlation filter
    correlation_filter = CorrelationFilter()
    console_handler.addFilter(correlation_filter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("minio").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)

    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured logging configured."""
    return logging.getLogger(name)

def log_with_context(logger: logging.Logger, level: int, message: str,
                    request_id: str = None, tenant_id: str = None, **kwargs):
    """Log a message with correlation context."""
    extra = kwargs.copy()
    if request_id:
        extra['request_id'] = request_id
    if tenant_id:
        extra['tenant_id'] = tenant_id

    logger.log(level, message, extra=extra)

def log_request_start(logger: logging.Logger, request_id: str, tenant_id: str,
                     method: str, path: str, **kwargs):
    """Log request start with correlation context."""
    log_with_context(
        logger, logging.INFO,
        f"Request started: {method} {path}",
        request_id=request_id,
        tenant_id=tenant_id,
        method=method,
        path=path,
        **kwargs
    )

def log_request_end(logger: logging.Logger, request_id: str, tenant_id: str,
                   method: str, path: str, status_code: int, duration: float, **kwargs):
    """Log request end with correlation context."""
    log_with_context(
        logger, logging.INFO,
        f"Request completed: {method} {path} - {status_code} in {duration:.3f}s",
        request_id=request_id,
        tenant_id=tenant_id,
        method=method,
        path=path,
        status_code=status_code,
        duration=duration,
        **kwargs
    )

def log_error(logger: logging.Logger, request_id: str, tenant_id: str,
              error: Exception, **kwargs):
    """Log error with correlation context."""
    log_with_context(
        logger, logging.ERROR,
        f"Error occurred: {str(error)}",
        request_id=request_id,
        tenant_id=tenant_id,
        error_type=type(error).__name__,
        error_message=str(error),
        **kwargs
    )

def log_performance_warning(logger: logging.Logger, request_id: str, tenant_id: str,
                           message: str, **kwargs):
    """Log performance warning with correlation context."""
    log_with_context(
        logger, logging.WARNING,
        f"Performance warning: {message}",
        request_id=request_id,
        tenant_id=tenant_id,
        **kwargs
    )
