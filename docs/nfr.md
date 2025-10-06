# Non-Functional Requirements (NFRs)

## Performance Requirements

### P95 End-to-End Latency
- **Requirement**: P95 end-to-end (upload → results render) ≤ 5s on dev data
- **Scope**: Complete request lifecycle from image upload to search results display
- **Measurement**: 95th percentile response time across all search operations
- **Environment**: Development data set (local MinIO + Qdrant setup)

## Security & Access Control

### Presigned URLs TTL
- **Requirement**: Presigned URLs TTL ≤ 10 minutes
- **Scope**: All generated presigned URLs for image access
- **Purpose**: Limit exposure window for temporary access tokens
- **Implementation**: Configurable via environment variables

### Tenant Scoping
- **Requirement**: All endpoints tenant-scoped (header X-Tenant-ID)
- **Scope**: Every API endpoint must validate and scope operations by tenant
- **Header**: `X-Tenant-ID` (required for all requests)
- **Purpose**: Multi-tenant isolation and data segregation

## Rate Limiting & Resource Management

### Rate Limiting
- **Requirement**: Basic rate-limit (per-tenant)
- **Scope**: All API endpoints
- **Implementation**: Per-tenant rate limiting using Redis
- **Default**: Configurable via environment variables

### Request Size Limits
- **Requirement**: ≤ 10MB per image
- **Scope**: All image upload endpoints
- **Purpose**: Prevent resource exhaustion and ensure performance
- **Validation**: File size check before processing

## Audit & Compliance

### Audit Logging
- **Requirement**: Audit logs for every search and response
- **Scope**: All search operations and API responses
- **Data**: Request metadata, tenant ID, timestamps, response status
- **Storage**: Structured logging to PostgreSQL audit table
- **Retention**: Configurable retention period

## Data Retention

### Crawled Thumbnails
- **Requirement**: 90 days default retention
- **Scope**: Thumbnail images from web crawling operations
- **Implementation**: Automated cleanup job
- **Configurable**: Via environment variables

### User Query Images
- **Requirement**: 24 hours max retention
- **Scope**: Images uploaded for search queries (not indexing)
- **Purpose**: Privacy compliance and storage optimization
- **Implementation**: Automated cleanup job

## Configuration Management

### Environment-Based Configuration
- **Requirement**: Config via env; no vendor-specific calls exposed above adapter layer
- **Scope**: All external service integrations
- **Purpose**: Maintain swapability between MinIO/S3 and Qdrant/Pinecone
- **Implementation**: Adapter pattern with environment-driven backend selection

## Architecture Constraints

### Adapter Layer Isolation
- **Requirement**: No vendor-specific calls exposed above adapter layer
- **Scope**: Storage (MinIO/S3) and Vector (Qdrant/Pinecone) services
- **Purpose**: Enable seamless backend swapping
- **Implementation**: Abstract interfaces with concrete implementations

### Backend Swapability
- **Current**: MinIO + Qdrant (development)
- **Target**: S3 + Pinecone (production)
- **Requirement**: Zero code changes required for backend switching
- **Implementation**: Environment variable configuration
