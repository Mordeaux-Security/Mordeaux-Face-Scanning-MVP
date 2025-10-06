# Deployment Guide with NFR Configuration

## Overview

This guide covers deploying the Mordeaux Face Scanning MVP with all Non-Functional Requirements (NFRs) properly configured.

## Environment Variables

### Required Environment Variables

```bash
# Database
POSTGRES_PASSWORD=your_secure_password

# Storage (MinIO for development)
S3_ACCESS_KEY=your_minio_access_key
S3_SECRET_KEY=your_minio_secret_key

# For production with AWS S3
# S3_ENDPOINT=  # Leave empty for AWS S3
# S3_ACCESS_KEY=your_aws_access_key
# S3_SECRET_KEY=your_aws_secret_key
# S3_REGION=us-east-1

# For production with Pinecone
# PINECONE_API_KEY=your_pinecone_api_key
# ENVIRONMENT=production
```

### NFR Configuration Variables

```bash
# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Presigned URLs
PRESIGNED_URL_TTL=600  # 10 minutes in seconds

# Data Retention
CRAWLED_THUMBS_RETENTION_DAYS=90
USER_QUERY_IMAGES_RETENTION_HOURS=24

# Performance
MAX_IMAGE_SIZE_MB=10
P95_LATENCY_THRESHOLD_SECONDS=5.0

# Logging
LOG_LEVEL=info
```

## Development Deployment

### Using Docker Compose

1. **Create environment file:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

2. **Start services:**
```bash
docker-compose up -d
```

3. **Verify deployment:**
```bash
# Check health
curl http://localhost:8000/healthz/detailed

# Check configuration
curl http://localhost:8000/config
```

### Environment Configuration

For development, the system uses:
- **MinIO** for object storage (S3-compatible)
- **Qdrant** for vector database
- **PostgreSQL** for metadata and audit logs
- **Redis** for rate limiting and caching

## Production Deployment

### AWS S3 + Pinecone Configuration

1. **Set production environment:**
```bash
ENVIRONMENT=production
```

2. **Configure AWS S3:**
```bash
# Remove S3_ENDPOINT to use AWS S3
S3_ENDPOINT=
S3_ACCESS_KEY=your_aws_access_key
S3_SECRET_KEY=your_aws_secret_key
S3_REGION=us-east-1
S3_USE_SSL=true
```

3. **Configure Pinecone:**
```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=faces
```

### Production Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=info

# Database
POSTGRES_HOST=your_postgres_host
POSTGRES_PORT=5432
POSTGRES_DB=mordeaux
POSTGRES_USER=mordeaux
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://your_redis_host:6379/0

# Storage (AWS S3)
S3_REGION=us-east-1
S3_BUCKET_RAW=your-raw-images-bucket
S3_BUCKET_THUMBS=your-thumbnails-bucket
S3_ACCESS_KEY=your_aws_access_key
S3_SECRET_KEY=your_aws_secret_key
S3_USE_SSL=true

# Vector Database (Pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=faces

# NFR Configuration
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
PRESIGNED_URL_TTL=600
CRAWLED_THUMBS_RETENTION_DAYS=90
USER_QUERY_IMAGES_RETENTION_HOURS=24
MAX_IMAGE_SIZE_MB=10
P95_LATENCY_THRESHOLD_SECONDS=5.0

# Celery
CELERY_BROKER_URL=redis://your_redis_host:6379/1
CELERY_RESULT_BACKEND=redis://your_redis_host:6379/2
```

## NFR Compliance Verification

### 1. Tenant Scoping
All API endpoints require `X-Tenant-ID` header:
```bash
curl -H "X-Tenant-ID: tenant123" http://localhost:8000/api/index_face
```

### 2. Rate Limiting
Rate limits are enforced per tenant:
- 60 requests per minute
- 1000 requests per hour

### 3. Request Size Limits
Maximum image size: 10MB
```bash
# Test with large file
curl -H "X-Tenant-ID: tenant123" -F "file=@large_image.jpg" http://localhost:8000/api/index_face
```

### 4. Presigned URLs
All presigned URLs expire in 10 minutes (600 seconds)

### 5. Audit Logging
All requests are logged to `audit_logs` and `search_audit_logs` tables

### 6. Data Retention
- Crawled thumbnails: 90 days
- User query images: 24 hours
- Audit logs: 30 days (configurable)

## Monitoring and Health Checks

### Health Check Endpoints

1. **Basic Health Check:**
```bash
curl http://localhost:8000/healthz
```

2. **Detailed Health Check:**
```bash
curl http://localhost:8000/healthz/detailed
```

3. **Configuration Check:**
```bash
curl http://localhost:8000/config
```

### Performance Monitoring

The system tracks:
- P95 latency (target: ≤5 seconds)
- Request processing times
- Rate limit violations
- Storage and vector DB performance

### Cleanup Jobs

Automated cleanup runs daily via Celery:
```bash
# Manual cleanup trigger
curl -H "X-Tenant-ID: admin" -X POST http://localhost:8000/api/admin/cleanup
```

## Security Considerations

### 1. Tenant Isolation
- All data is scoped by tenant ID
- Vector searches are filtered by tenant
- Storage objects are prefixed with tenant ID

### 2. Rate Limiting
- Per-tenant rate limits prevent abuse
- Redis-based implementation for scalability

### 3. Request Validation
- File size limits prevent resource exhaustion
- Content type validation for image uploads

### 4. Audit Trail
- All operations are logged with tenant context
- Request/response metadata captured

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors:**
   - Check environment variables
   - Verify database connections
   - Ensure Redis is accessible

2. **Rate Limiting Issues:**
   - Check Redis connection
   - Verify rate limit configuration
   - Monitor rate limit counters

3. **Storage Issues:**
   - Verify S3/MinIO credentials
   - Check bucket permissions
   - Ensure presigned URL generation

4. **Vector Database Issues:**
   - Check Qdrant/Pinecone connectivity
   - Verify index configuration
   - Monitor search performance

### Logs and Debugging

Enable debug logging:
```bash
LOG_LEVEL=debug
```

Check application logs:
```bash
docker-compose logs -f backend-cpu
```

## Scaling Considerations

### Horizontal Scaling
- Multiple backend instances behind load balancer
- Shared Redis for rate limiting
- Shared PostgreSQL for audit logs

### Performance Optimization
- Monitor P95 latency metrics
- Optimize vector search queries
- Implement caching for frequent searches

### Storage Optimization
- Use CDN for image serving
- Implement image compression
- Monitor storage costs

## Backup and Recovery

### Database Backups
```bash
# PostgreSQL backup
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER $POSTGRES_DB > backup.sql
```

### Storage Backups
- Enable S3 versioning
- Configure cross-region replication
- Regular backup verification

### Configuration Backups
- Version control environment files
- Document configuration changes
- Test configuration in staging
