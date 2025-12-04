# Complete Localhost Setup Guide

## 🚀 "Run All Localhosts" - Complete Command Reference

When you say **"run all localhosts"**, this includes starting **ALL** of the following services:

---

## 1️⃣ Docker Services (Backend Stack)

**Start Command:**
```powershell
docker-compose up -d
# OR
.\start-local.ps1
```

**Services Included:**
- ✅ **Backend API**: http://localhost/api (nginx proxy to port 8000)
- ✅ **Face Pipeline**: Internal port 8001
- ✅ **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- ✅ **Qdrant Vector DB**: http://localhost:6333
- ✅ **Redis**: Internal port 6379
- ✅ **Nginx Reverse Proxy**: http://localhost

**Stop Command:**
```powershell
docker-compose down
```

---

## 2️⃣ Frontend Dev Tools (Dev Server)

**Start Command:**
```powershell
# In a new PowerShell window:
cd frontend
npm.cmd run dev
```

**Or use this one-liner to open in new window:**
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$pwd\frontend'; npm.cmd run dev"
```

**Services Included:**
- ✅ **Vite Dev Server**: http://localhost:5173
- ✅ **Dev Search Page**: http://localhost:5173/dev/search
- ✅ **Enroll Identity Page**: http://localhost:5173/enroll
- ✅ **Verify Search Page**: http://localhost:5173/verify

**Stop Command:**
- Press `Ctrl+C` in the terminal running the dev server
- Or close the PowerShell window

---

## 🎯 "What Are My Localhosts" - Complete List

### Production-Like Services (Docker)
| Service | URL | Credentials |
|---------|-----|-------------|
| Main Frontend (Nginx) | http://localhost | - |
| Backend API | http://localhost/api | - |
| MinIO Console | http://localhost:9001 | minioadmin/minioadmin |
| Qdrant Dashboard | http://localhost:6333/dashboard | - |

### Development Services
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend Dev Server | http://localhost:5173 | Hot-reload development |
| Dev Search Page | http://localhost:5173/dev/search | Search functionality testing |
| Enroll Identity Page | http://localhost:5173/enroll | Identity enrollment |
| Verify Search Page | http://localhost:5173/verify | Identity verification |

### Optional Development Services
| Service | URL | Command | Purpose |
|---------|-----|---------|---------|
| Mock Server | http://localhost:8000 | `cd mock-server; .\start.ps1` | Backend API simulation with fixtures |

---

## 📋 Quick Start - Full Stack

### Windows PowerShell (Recommended)

**Option 1: Step-by-Step**
```powershell
# 1. Start Docker services
docker-compose up -d

# 2. Wait for services to be ready (optional)
Start-Sleep -Seconds 10

# 3. Start frontend dev server in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$pwd\frontend'; npm.cmd run dev"

# 4. Check status
docker-compose ps
netstat -an | findstr ":5173"
```

**Option 2: All-in-One Script**
```powershell
# Start Docker services
.\start-local.ps1

# Start frontend dev server in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$pwd\frontend'; npm.cmd run dev"
```

---

## 🔍 Verify All Services Are Running

### Check Docker Services
```powershell
docker-compose ps
```

Expected output: All services should be "Up" and "healthy" (or "health: starting")

### Check Frontend Dev Server
```powershell
netstat -an | findstr ":5173"
```

Expected output: `TCP    0.0.0.0:5173           0.0.0.0:0              LISTENING`

### Quick Health Check
```powershell
# Docker services
curl http://localhost/api/health -UseBasicParsing

# Frontend dev server (should return HTML)
curl http://localhost:5173 -UseBasicParsing

# Qdrant
curl http://localhost:6333/readyz -UseBasicParsing
```

---

## 🛑 Stop All Localhosts

```powershell
# 1. Stop Docker services
docker-compose down

# 2. Stop frontend dev server
# Find the PowerShell window running "npm.cmd run dev" and press Ctrl+C
# OR kill the Node process:
Get-Process node | Stop-Process -Force
```

---

## 🔧 Troubleshooting

### Docker services won't start
```powershell
# Check if Docker Desktop is running
docker info

# If not, start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait 30 seconds, then try again
Start-Sleep -Seconds 30
docker-compose up -d
```

### Frontend dev server won't start
```powershell
# Check if dependencies are installed
cd frontend
if (!(Test-Path "node_modules")) {
    npm.cmd install
}

# Check if port 5173 is already in use
netstat -an | findstr ":5173"

# If port is in use, kill the process and restart
Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | 
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Start dev server
npm.cmd run dev
```

### Port conflicts
If you get port conflicts:
- **Port 80**: Another web server is running (IIS, Apache, etc.)
- **Port 5173**: Another Vite dev server is running
- **Port 8000**: Mock server might be running
- **Port 3000**: Old frontend-dev Docker service

Solution:
```powershell
# Stop orphan Docker containers
docker-compose down --remove-orphans

# Kill conflicting Node processes
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 📚 Related Documentation

- **Quick Start**: `LOCALHOST_QUICK_START.md`
- **Docker Details**: `DOCKER_README.md`
- **Frontend Dev Guide**: `frontend/README_SEARCH_DEV.md`
- **Phase Documentation**: `docs/PHASE_*.md`
- **API Documentation**: `docs/api.md`

---

## 💡 Pro Tips

1. **Always start Docker services first** before the frontend dev server
2. **Keep the frontend dev server in a separate terminal** so you can see hot-reload logs
3. **Use mock server** (`cd mock-server; .\start.ps1`) for frontend-only development without Docker
4. **Check Docker health** with `docker-compose ps` if APIs aren't responding
5. **Clear browser cache** if you see stale content at http://localhost:5173

---

## ⚙️ Environment Setup

### First Time Setup
```powershell
# Install frontend dependencies
cd frontend
npm.cmd install

# Verify Docker is installed
docker --version
docker-compose --version

# Pull Docker images (optional, speeds up first start)
docker-compose pull
```

---

**Last Updated**: December 4, 2025  
**Maintained by**: Development Team  
**Status**: ✅ All services operational

