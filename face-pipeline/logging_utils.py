import logging
import json
import os
from pythonjsonlogger import jsonlogger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging():
    """
    Configure root logger to output structured JSON logs.
    Safe for both dev and production.
    """
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    # Remove duplicated handlers if reload happens (e.g. uvicorn reload)
    if logger.handlers:
        return

    handler = logging.StreamHandler()

    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_event(event: str, **fields):
    """
    Central safe logging function for structured logs.
    Does NOT log raw images, embeddings, or file contents.
    """
    safe_fields = {}

    for key, value in fields.items():
        # Prevent logging large or unsafe data
        if key in ("image_b64", "image_bytes", "embedding", "embeddings"):
            safe_fields[key] = "[REDACTED]"
        else:
            # Convert un-serializable types to str if needed
            try:
                json.dumps(value)
                safe_fields[key] = value
            except Exception:
                safe_fields[key] = str(value)

    logging.getLogger("mordeaux").info(
        {"event": event, **safe_fields}
    )

