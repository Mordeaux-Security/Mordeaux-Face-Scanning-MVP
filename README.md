# Mordeaux Face Protection System

A comprehensive face protection and monitoring system built with TypeScript, Python, and modern infrastructure.

## 🏗️ Architecture

This is a skeleton-only monorepo for a face-protection system with the following services:

### Core Services
- **API Gateway** (Node/TypeScript) - Request routing, rate limiting, OpenAPI docs
- **Auth Service** (Node/TypeScript) - JWT token management for development
- **Upload Service** (Node/TypeScript) - Presigned URL generation for MinIO
- **Search API** (Node/TypeScript) - Image and vector search endpoints
- **Policy Engine** (Node/TypeScript) - Policy resolution service
- **Face Workers** (Python) - Face detection, alignment, embedding, and hashing
- **Admin Console** (React/Vite) - Web UI for system management

### Infrastructure
- **PostgreSQL** - Primary database with comprehensive schema
- **Redis** - Caching and session storage
- **RabbitMQ** - Message queue for async processing
- **MinIO** - S3-compatible object storage
- **Vector Index** - In-memory vector similarity search

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.9+ (for local development)

### One-Command Start

```bash
# Clone and start everything
git clone <repository-url>
cd Mordeaux-Face-Scanning-MVP
make up
```

### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| API Gateway | http://localhost:3000 | Main API endpoint |
| API Documentation | http://localhost:3000/docs | Swagger UI |
| Auth Service | http://localhost:3001 | Authentication |
| Upload Service | http://localhost:3002 | File uploads |
| Orchestrator | http://localhost:3003 | Event publishing & processing |
| Admin Console | http://localhost:3005 | Web admin interface |
| MinIO Console | http://localhost:9001 | Object storage UI |
| RabbitMQ Management | http://localhost:15672 | Message queue UI |
| Vector Index | http://localhost:8080 | Vector search API |

### Health Checks

All services expose health check endpoints:
- `/healthz` - Service health status
- `/readyz` - Service readiness (dependencies)

## 🛠️ Development

### Local Development

```bash
# Start infrastructure only
make dev-up

# Start individual services for development
cd apps/api-gateway && npm run dev
cd apps/auth && npm run dev
# ... etc
```

### Available Commands

```bash
make help          # Show all available commands
make up            # Start all services
make down          # Stop all services
make logs          # Show logs for all services
make ps            # Show running services
make migrate       # Run database migrations
make clean         # Clean up containers and volumes
make build         # Build all services
make urls          # Show service URLs
```

### Project Structure

```
├── apps/                    # Application services
│   ├── api-gateway/        # API Gateway (Fastify)
│   ├── auth/               # Authentication service
│   ├── ingest-upload/      # Upload service
│   ├── crawler-manager/    # Crawler management
│   ├── orchestrator/       # Event orchestration
│   ├── search-api/         # Search endpoints
│   ├── policy-engine/      # Policy resolution
│   ├── admin-console/      # React admin UI
│   └── face-workers/       # Python face processing
├── packages/               # Shared packages
│   ├── contracts/          # TypeScript contracts & schemas
│   ├── common/             # TypeScript utilities
│   └── py-common/          # Python utilities
├── infra/                  # Infrastructure
│   ├── docker-compose.yml  # Service orchestration
│   ├── migrations/         # Database migrations
│   └── vector-index/       # Vector search service
└── docs/                   # Documentation
```

## 📋 API Endpoints

### Authentication
- `POST /v1/auth/issue` - Issue JWT token (dev only)
- `POST /v1/auth/verify` - Verify JWT token

### Search
- `POST /v1/search/by-image` - Search by image (TODO)
- `POST /v1/search/by-vector` - Search by vector (TODO)

### Policy
- `GET /v1/policies/resolve?tenant_id=...` - Resolve tenant policy

### Upload
- `POST /v1/upload/presigned-url` - Generate presigned upload URL

### Event Publishing (Development)
- `POST /dev/publish` - Publish events to RabbitMQ for testing

#### Example: Publish NEW_CONTENT Event
```bash
curl -X POST http://localhost:3003/dev/publish \
  -H "Content-Type: application/json" \
  -d '{
    "type": "NEW_CONTENT",
    "payload": {
      "content_id": "123e4567-e89b-12d3-a456-426614174000",
      "tenant_id": "00000000-0000-0000-0000-000000000001",
      "source_id": "456e7890-e89b-12d3-a456-426614174001",
      "s3_key_raw": "raw/tenant1/2024-01-15/image.jpg",
      "url": "https://example.com/image.jpg",
      "fetch_ts": "2024-01-15T10:30:00Z"
    }
  }'
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your configuration
```

Key configuration options:
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `RABBITMQ_URL` - RabbitMQ connection
- `MINIO_ENDPOINT` - MinIO server URL
- `JWT_SECRET` - JWT signing secret

## 🗄️ Database Schema

The system uses PostgreSQL with the following main tables:
- `tenants` - Multi-tenant organization
- `sources` - Content sources (crawlers, uploads)
- `content` - Processed content items
- `faces` - Detected faces with bounding boxes
- `embeddings` - Face embeddings (512-dimensional vectors)
- `clusters` - Face clusters for grouping
- `policies` - Tenant-specific policies
- `alerts` - System alerts and notifications

## 🔄 Event Flow

The system processes content through the following event flow:

1. **NEW_CONTENT** - New content detected/uploaded
2. **FACES_EXTRACTED** - Faces detected and aligned
3. **INDEXED** - Faces embedded and indexed for search

Events flow through RabbitMQ queues for async processing.

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific service tests
cd apps/api-gateway && npm test
cd apps/face-workers && python -m pytest
```

## 📊 Monitoring

- Health checks: All services expose `/healthz` and `/readyz`
- Logging: Structured JSON logging with request IDs
- Metrics: Prometheus-compatible metrics (TODO)

## 🚀 Next Steps

### Immediate Development Tasks
- [ ] Implement actual face detection algorithms
- [ ] Add real embedding models (FaceNet, ArcFace, etc.)
- [ ] Implement clustering algorithms
- [ ] Add comprehensive test coverage
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and alerting

### Production Readiness
- [ ] Replace development JWT with OIDC integration
- [ ] Add proper secret management
- [ ] Implement rate limiting and WAF rules
- [ ] Add comprehensive error handling
- [ ] Set up production monitoring
- [ ] Add backup and disaster recovery

### Feature Enhancements
- [ ] Real-time face recognition
- [ ] Advanced policy engine
- [ ] Multi-modal search (face + text)
- [ ] Batch processing optimization
- [ ] API versioning and backward compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Add your license here]

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the API documentation at `/docs` endpoint