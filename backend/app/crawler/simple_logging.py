"""
Simple Crawler Logging Enhancement

Adds timestamped log files and reduces terminal noise for better debugging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_crawler_logging(log_dir: str = "crawlerlogs"):
    """Setup simple enhanced logging for crawler operations."""
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"crawler_{timestamp}.log"
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    noisy_loggers = [
        'httpx', 'httpcore', 'urllib3', 'boto3', 'botocore', 
        'minio', 'qdrant_client', 'pinecone', 'uvicorn.access'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return log_file
