"""Environment configuration for Mordeaux services."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """Environment configuration model."""
    
    # Service config
    service_name: str = Field(default="mordeaux-service")
    log_level: str = Field(default="INFO")
    
    # Database
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/mordeaux")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    
    # RabbitMQ
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672")
    
    # MinIO
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    
    # Vector DB
    vector_db_url: str = Field(default="http://localhost:8080")
    
    # JWT
    jwt_secret: str = Field(default="dev-secret-key")


def load_env(env_file: Optional[str] = None) -> EnvConfig:
    """Load environment configuration."""
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    return EnvConfig(
        service_name=os.getenv("SERVICE_NAME", "mordeaux-service"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        database_url=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mordeaux"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672"),
        minio_endpoint=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        minio_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        vector_db_url=os.getenv("VECTOR_DB_URL", "http://localhost:8080"),
        jwt_secret=os.getenv("JWT_SECRET", "dev-secret-key"),
    )
