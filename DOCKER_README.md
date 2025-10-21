# 🐳 Docker Deployment Guide

This guide will help you build and deploy the Mordeaux Face Scanning MVP using Docker.

## 📋 Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** (included with Docker Desktop)
- At least **8GB RAM** and **20GB free disk space**
- **Git** (to clone the repository)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Mordeaux-Face-Scanning-MVP
```

### 2. Configure Environment

Copy the example environment file and configure it:

```bash
# Linux/Mac
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

**Important**: Edit the `.env` file with your actual configuration values, especially:
- `POSTGRES_PASSWORD` - Set a secure password
- `S3_ACCESS_KEY` and `S3_SECRET_KEY` - MinIO credentials
- `PINECONE_API_KEY` - If using Pinecone for production

### 3. Build and Start

#### Linux/Mac:
```bash
./build-docker.sh
```

#### Windows PowerShell:
```powershell
.\build-docker.ps1
```

#### Manual Docker Compose:
```bash
docker-compose up --build -d
```

## 🌐 Service URLs

Once started, the following services will be available:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Main web interface |
| **Backend API** | http://localhost:8000 | REST API endpoints |
| **Face Pipeline** | http://localhost:8001 | Face processing service |
| **MinIO Console** | http://localhost:9001 | Object storage management |
| **pgAdmin** | http://localhost:5050 | Database administration |
| **Qdrant** | http://localhost:6333 | Vector database |
| **Nginx (Main)** | http://localhost:80 | Main entry point |

## 🔐 Default Credentials

- **MinIO**: `minioadmin` / `minioadmin`
- **pgAdmin**: `admin@admin.com` / `admin`

## 📊 Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │    Frontend     │    │    Backend      │
│   (Port 80)     │◄──►│   (Port 3000)   │◄──►│   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Face Pipeline  │    │     Worker      │    │    PostgreSQL   │
│   (Port 8001)   │    │   (Celery)      │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Qdrant      │    │      Redis      │    │      MinIO      │
│   (Port 6333)   │    │   (Port 6379)   │    │   (Port 9000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Management Commands

### Using Build Scripts

#### Linux/Mac:
```bash
./build-docker.sh [command]
```

#### Windows PowerShell:
```powershell
.\build-docker.ps1 [command]
```

**Available Commands:**
- `build` / `start` - Build and start all services (default)
- `stop` - Stop all services
- `restart` - Restart all services
- `status` - Show service status
- `logs` - Show recent logs
- `cleanup` - Stop services and clean up resources
- `help` - Show help message

### Using Docker Compose Directly

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart backend-cpu

# Scale worker services
docker-compose up -d --scale worker-cpu=3
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password

# Storage
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Vector Database
QDRANT_URL=http://qdrant:6333
PINECONE_API_KEY=your_pinecone_key

# Face Processing
DETECTOR_MODEL=buffalo_l
MIN_FACE_QUALITY=0.5
```

### Service Configuration

#### Backend (CPU)
- **Image**: Custom build from `./backend`
- **Port**: 8000
- **Workers**: 2 (configurable via `ASGI_WORKERS`)

#### Face Pipeline
- **Image**: Custom build from `./face-pipeline`
- **Port**: 8001
- **Models**: InsightFace buffalo_l

#### Worker
- **Image**: Custom build from `./worker`
- **Queues**: faces, ingest
- **Scalable**: Yes (use `--scale worker-cpu=N`)

## 📈 Monitoring and Logs

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend-cpu

# Last 100 lines
docker-compose logs --tail=100
```

### Health Checks
All services include health checks:
- **Backend**: `GET /health`
- **Face Pipeline**: `GET /health`
- **PostgreSQL**: `pg_isready`
- **MinIO**: `GET /minio/health/live`

### Resource Monitoring
```bash
# Container stats
docker stats

# Service status
docker-compose ps
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Conflicts
If ports are already in use:
```bash
# Check what's using the port
netstat -tulpn | grep :8000

# Stop conflicting services or change ports in docker-compose.yml
```

#### 2. Out of Memory
If containers are killed due to memory:
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or reduce worker count:
docker-compose up -d --scale worker-cpu=1
```

#### 3. Build Failures
```bash
# Clean build
docker-compose down
docker system prune -f
docker-compose up --build -d
```

#### 4. Database Connection Issues
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Debug Mode

Run services in debug mode:
```bash
# Override command for debugging
docker-compose run --rm backend-cpu bash
docker-compose run --rm face-pipeline bash
```

## 🔄 Updates and Maintenance

### Updating Services
```bash
# Pull latest images and rebuild
docker-compose pull
docker-compose up --build -d
```

### Backup Data
```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U mordeaux mordeaux > backup.sql

# Backup volumes
docker run --rm -v mordeaux_pgdata:/data -v $(pwd):/backup alpine tar czf /backup/pgdata.tar.gz -C /data .
```

### Cleanup
```bash
# Remove all containers and volumes
docker-compose down -v

# Remove unused images
docker image prune -f

# Full cleanup
docker system prune -a -f
```

## 🚀 Production Deployment

### Security Considerations
1. **Change default passwords** in `.env`
2. **Use secrets management** for sensitive data
3. **Enable SSL/TLS** with proper certificates
4. **Configure firewall** rules
5. **Use production-grade** PostgreSQL and Redis

### Performance Optimization
1. **Increase worker count** based on CPU cores
2. **Configure resource limits** in docker-compose.yml
3. **Use external databases** for production
4. **Enable Redis persistence**
5. **Configure log rotation**

### Example Production Override
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'
services:
  backend-cpu:
    environment:
      ASGI_WORKERS: 4
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
  
  worker-cpu:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
```

Deploy with:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Celery with Docker](https://docs.celeryproject.org/en/stable/userguide/deployment.html)

## 🆘 Support

If you encounter issues:
1. Check the logs: `docker-compose logs`
2. Verify environment configuration
3. Ensure all prerequisites are met
4. Check Docker Desktop/system resources
5. Review this troubleshooting guide

For additional help, please refer to the main project documentation or create an issue in the repository.
