# 🌐 Complete Localhost Services List

**Generated**: November 14, 2025  
**Status**: Current running and available services

---

## ✅ Currently Running Services

### 1. Frontend Development Server (Vite)

**URL**: `http://localhost:5173`  
**Status**: ✅ **RUNNING**  
**Process**: Node.js (PID: 31992)  
**Command**: `vite`  
**Location**: `frontend/node_modules/.bin/vite.js`

**Accessible Endpoints**:
- **Home**: http://localhost:5173
- **Search Dev Page**: http://localhost:5173/dev/search
- **Network Access**: http://10.82.135.51:5173

**Features Available**:
- ✅ React Application (Phase 0-7)
- ✅ Hot Module Replacement (HMR)
- ✅ Development mode with source maps
- ✅ All Phase 7 features (filters, pagination, URL sync)

**Test**:
```powershell
curl.exe http://localhost:5173
```

---

### 2. Mock Server (FastAPI)

**URL**: `http://localhost:8000`  
**Status**: ✅ **RUNNING**  
**Process**: Python/Uvicorn (PID: 34028)  
**Command**: `uvicorn app:app --host 0.0.0.0 --port 8000`  
**Location**: `mock-server/venv/Scripts/python.exe`

**Accessible Endpoints**:
- **Base API**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/v1/health
- **API Documentation**: http://localhost:8000/docs
- **Search Endpoint**: http://localhost:8000/api/v1/search
- **Search by ID**: http://localhost:8000/api/v1/search-by-id
- **Mock Fixtures**: http://localhost:8000/mock/fixtures
- **Mock Config**: http://localhost:8000/mock/config

**Available Fixtures**:
- `tiny` - 10 results (quick testing)
- `medium` - 200 results (default)
- `large` - 2000 results (stress testing)
- `edge_cases` - 15 results (boundary testing)
- `errors` - 20 results (error handling)

**Configuration**:
- Latency Simulation: Enabled (50-300ms)
- Error Rate: 0%
- Default Fixture: medium

**Test**:
```powershell
curl.exe http://localhost:8000/api/v1/health
```

---

## 🐳 Docker Services (Not Currently Running)

**Docker Status**: ❌ Docker Desktop not running

The following services are configured in `docker-compose.yml` but require Docker to be running:

### 3. Nginx (Frontend Proxy)

**URL**: `http://localhost:80` (when Docker is running)  
**Status**: ⚠️ **NOT RUNNING** (Docker required)  
**Port**: 80  
**Purpose**: Serves frontend production build and proxies API requests

**To Start**:
```powershell
docker-compose up -d
```

---

### 4. Backend API

**URL**: `http://localhost:80/api` (via Nginx when Docker is running)  
**Status**: ⚠️ **NOT RUNNING** (Docker required)  
**Internal Port**: 8000 (exposed only within Docker network)  
**Purpose**: Main backend API service

**Health Endpoint**: `http://localhost:80/api/healthz` (via Nginx)

**To Start**:
```powershell
docker-compose up -d api
```

---

### 5. Face Pipeline Service

**URL**: `http://localhost:80/pipeline` (via Nginx when Docker is running)  
**Status**: ⚠️ **NOT RUNNING** (Docker required)  
**Internal Port**: 8001 (exposed only within Docker network)  
**Purpose**: Face recognition pipeline service

**Health Endpoint**: `http://localhost:80/pipeline/api/v1/health` (via Nginx)

**To Start**:
```powershell
docker-compose up -d face-pipeline
```

---

### 6. MinIO (Object Storage)

**URL**: `http://localhost:9001` (when Docker is running)  
**Status**: ⚠️ **NOT RUNNING** (Docker required)  
**Port**: 9001 (Console UI)  
**Internal Port**: 9000 (API, exposed only within Docker network)  
**Purpose**: S3-compatible object storage for images

**Credentials**:
- Username: `minioadmin`
- Password: `minioadmin`

**Console URL**: http://localhost:9001 (when Docker is running)

**To Start**:
```powershell
docker-compose up -d minio
```

---

### 7. Qdrant (Vector Database)

**URL**: `http://localhost:6333` (when Docker is running)  
**Status**: ⚠️ **NOT RUNNING** (Docker required)  
**Port**: 6333  
**Purpose**: Vector similarity search database for face embeddings

**Health Endpoint**: `http://localhost:6333/readyz` (when Docker is running)

**Dashboard**: http://localhost:6333/dashboard (when Docker is running)

**To Start**:
```powershell
docker-compose up -d qdrant
```

---

### 8. Redis (Cache & Queue)

**URL**: Redis protocol (when Docker is running)  
**Status**: ⚠️ **NOT RUNNING** (Docker required)  
**Port**: 6379 (exposed only within Docker network)  
**Purpose**: Caching and message queue for face processing

**To Start**:
```powershell
docker-compose up -d redis
```

---

## 📊 Service Summary

### Currently Available (2 services)

| Service | Port | Status | URL |
|---------|------|--------|-----|
| Frontend (Vite) | 5173 | ✅ Running | http://localhost:5173 |
| Mock Server | 8000 | ✅ Running | http://localhost:8000 |

### Available via Docker (6 services)

| Service | Port | Status | URL |
|---------|------|--------|-----|
| Nginx | 80 | ⚠️ Requires Docker | http://localhost:80 |
| Backend API | 80/api | ⚠️ Requires Docker | http://localhost:80/api |
| Face Pipeline | 80/pipeline | ⚠️ Requires Docker | http://localhost:80/pipeline |
| MinIO Console | 9001 | ⚠️ Requires Docker | http://localhost:9001 |
| Qdrant | 6333 | ⚠️ Requires Docker | http://localhost:6333 |
| Redis | 6379 | ⚠️ Requires Docker | (Internal only) |

---

## 🚀 Quick Start Commands

### Start Currently Running Services

**Frontend** (already running):
```powershell
cd frontend
npm.cmd run dev
```

**Mock Server** (already running):
```powershell
cd mock-server
.\venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Start Docker Services

**Start All Services**:
```powershell
docker-compose up -d
```

**Start Specific Service**:
```powershell
docker-compose up -d nginx
docker-compose up -d api
docker-compose up -d face-pipeline
docker-compose up -d minio
docker-compose up -d qdrant
docker-compose up -d redis
```

**Check Docker Status**:
```powershell
docker ps
docker-compose ps
```

---

## 🔍 Port Check Commands

### Check All Listening Ports
```powershell
netstat -ano | findstr "LISTENING" | findstr "5173 8000 80 9001 6333"
```

### Check Specific Service
```powershell
# Frontend
netstat -ano | findstr ":5173"

# Mock Server
netstat -ano | findstr ":8000"

# Docker Services (when running)
netstat -ano | findstr ":80 :9001 :6333"
```

### Check Running Processes
```powershell
Get-Process | Where-Object {$_.ProcessName -eq "node" -or $_.ProcessName -eq "python"}
```

---

## 🌐 Network URLs

### Development Mode (Current)

- **Frontend**: http://localhost:5173
- **Search Dev Page**: http://localhost:5173/dev/search
- **Mock Server**: http://localhost:8000
- **Mock API Docs**: http://localhost:8000/docs

### Production Mode (Docker)

- **Frontend**: http://localhost
- **Backend API**: http://localhost/api
- **Face Pipeline**: http://localhost/pipeline
- **MinIO Console**: http://localhost:9001
- **Qdrant Dashboard**: http://localhost:6333/dashboard

---

## 📝 Service Dependencies

### Development Stack (Current)
```
Frontend (5173) → Mock Server (8000)
```

### Full Stack (Docker)
```
Nginx (80) → Frontend (dist)
           → Backend API (8000)
           → Face Pipeline (8001)
           → Redis (6379)
           → MinIO (9000)
           → Qdrant (6333)
```

---

## ✅ Health Check URLs

### Current Services

**Frontend**:
```powershell
curl.exe http://localhost:5173
```

**Mock Server**:
```powershell
curl.exe http://localhost:8000/api/v1/health
```

### Docker Services (when running)

**Nginx**:
```powershell
curl.exe http://localhost
```

**Backend API**:
```powershell
curl.exe http://localhost/api/healthz
```

**Face Pipeline**:
```powershell
curl.exe http://localhost/pipeline/api/v1/health
```

**MinIO**:
```powershell
curl.exe http://localhost:9001
```

**Qdrant**:
```powershell
curl.exe http://localhost:6333/readyz
```

---

## 🎯 Recommended Setup

### For UI Development (Current Setup)
✅ **Frontend**: http://localhost:5173  
✅ **Mock Server**: http://localhost:8000  

This setup is perfect for frontend development with all Phase 0-7 features.

### For Full Stack Development
⚠️ **Start Docker Desktop first**, then:
```powershell
docker-compose up -d
```

This will start all services including:
- Nginx (port 80)
- Backend API
- Face Pipeline
- MinIO (port 9001)
- Qdrant (port 6333)
- Redis

---

## 📋 Summary

**Currently Running**: 2 services
- ✅ Frontend Dev Server (port 5173)
- ✅ Mock Server (port 8000)

**Available via Docker**: 6 services
- ⚠️ Nginx (port 80)
- ⚠️ Backend API
- ⚠️ Face Pipeline
- ⚠️ MinIO (port 9001)
- ⚠️ Qdrant (port 6333)
- ⚠️ Redis (internal)

**Total Available**: 8 services (2 running, 6 via Docker)

---

**Last Updated**: November 14, 2025  
**Status**: Frontend and Mock Server are running and healthy! 🎉


