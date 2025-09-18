"""Logging utilities for Mordeaux services."""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(service_name: str, log_level: str = "INFO") -> None:
    """Set up structured logging for the service."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(service_name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the service."""
    return structlog.get_logger(service_name)


class Logger:
    """Logger wrapper with request context."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = get_logger(service_name)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, service=self.service_name, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message."""
        context = {"service": self.service_name, **kwargs}
        if error:
            context.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
        self.logger.error(message, **context)
    
    def warn(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warn(message, service=self.service_name, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, service=self.service_name, **kwargs)
