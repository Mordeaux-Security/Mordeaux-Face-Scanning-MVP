"""Mordeaux Common Python Package."""

from .logger import get_logger, setup_logging
from .env import load_env, EnvConfig
from .errors import MordeauxError, ValidationError, ServiceError
from .schemas import BaseModel, EventSchema, NewContentEvent, FacesExtractedEvent, IndexedEvent

__version__ = "1.0.0"
__all__ = [
    "get_logger",
    "setup_logging", 
    "load_env",
    "EnvConfig",
    "MordeauxError",
    "ValidationError", 
    "ServiceError",
    "BaseModel",
    "EventSchema",
    "NewContentEvent",
    "FacesExtractedEvent", 
    "IndexedEvent"
]
