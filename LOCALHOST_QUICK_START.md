# 🚀 Localhost Quick Start Guide

## ⚡ Start Development Environment (30 seconds)

### Option A: Frontend with Mock Server (Recommended for UI Dev)

**Terminal 1 - Start Mock Server:**
```powershell
cd mock-server
.\start.ps1
```
✅ Running on: http://localhost:8000

**Terminal 2 - Start Frontend:**
```powershell
cd frontend
npm.cmd run dev
```
✅ Running on: http://localhost:5173
✅ Dev Page: http://localhost:5173/dev/search

---

### Option B: Full Docker Stack

```powershell
.\start-local.ps1
```

**Available Services:**
- 🌐 Frontend: http://localhost
- 🔌 API: http://localhost/api
- 📦 MinIO: http://localhost:9001 (minioadmin/minioadmin)
- 🗄️ Qdrant: http://localhost:6333

**Stop Services:**
```powershell
docker-compose down
```

---

## 🎯 Phase 7 Features Available

- ✅ Min Score Filtering (0-100% slider)
- ✅ Site Filtering (dropdown)
- ✅ Pagination (First/Prev/Next/Last)
- ✅ Page Size Control (10/25/50/100)
- ✅ URL State Sync (deep-linking)
- ✅ Grid/List View Toggle
- ✅ Copy URL Button
- ✅ Reset Filters

**Test URL:**
```
http://localhost:5173/dev/search?minScore=0.75&site=example.com&page=2&pageSize=50
```

---

## 📋 Component Library

All Phase 0-7 components ready:

| Component | Phase | Purpose |
|-----------|-------|---------|
| SafeImage | 5 | Image safety & error handling |
| BBoxOverlay | 6 | Bounding box visualization |
| DistanceChip | 6 | Distance score display |
| ScoreBadge | 6 | Match score badge |
| QueryImage | 6 | Uploaded query image |
| ResultCard | 6 | Grid view result card |
| ResultListItem | 6 | List view result item |
| MinScoreSlider | 7 | Filter slider with debounce |
| Pagination | 7 | Full pagination control |
| EmptyState | 4 | No results state |
| ErrorState | 4 | Error handling |
| LoadingState | 4 | Loading indicator |

**Custom Hook:**
- `useUrlState` - URL state synchronization

---

## 🧪 Quick Tests

### Test Mock Server
```powershell
cd mock-server
.\start.ps1
# Wait for startup, then:
curl http://localhost:8000/api/v1/health
```

### Test Frontend
```powershell
cd frontend
npm.cmd run dev
# Open: http://localhost:5173/dev/search
```

### Test Docker
```powershell
docker info
docker-compose config
```

---

## 📝 File Locations

**Phase Documentation:**
- `docs/PHASE_0_DEV_SEARCH_PAGE.md` through `docs/PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md`

**Frontend Source:**
- `frontend/src/components/` - All UI components
- `frontend/src/pages/SearchDevPage.tsx` - Main dev page
- `frontend/src/hooks/useUrlState.ts` - URL state hook

**Quick Reference:**
- `frontend/README_PHASE_7.md` - Latest features
- `frontend/QUICK_START_PHASE_4.md` - Getting started
- `mock-server/QUICK_START.md` - Mock server setup

---

## 💡 Pro Tips

**PowerShell Script Issues?**
Use `npm.cmd` instead of `npm`:
```powershell
npm.cmd install
npm.cmd run dev
```

**Port Already in Use?**
```powershell
# Mock server on different port
cd mock-server
python -c "from app import app; import uvicorn; uvicorn.run(app, port=8001)"

# Frontend on different port
cd frontend
npm.cmd run dev -- --port 5174
```

**Hot Module Reload Not Working?**
```powershell
cd frontend
npm.cmd run dev -- --force
```

---

## 🔍 Troubleshooting

### Mock Server Won't Start
```powershell
cd mock-server
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Frontend Build Errors
```powershell
cd frontend
rm -rf node_modules package-lock.json
npm.cmd install
npm.cmd run dev
```

### Docker Services Not Starting
```powershell
docker-compose down -v
docker-compose up --build
```

---

## 🎯 Current Status

✅ Repository synced with GitHub (origin/main)  
✅ All Phase 0-7 UI work preserved  
✅ Dependencies installed and verified  
✅ Docker v28.4.0 ready  
✅ Python 3.14.0 ready  
✅ Node.js & npm ready  
✅ All localhost operations functional  

**You're ready to develop!** 🚀

---

## 📚 Full Documentation

- Integration Report: `INTEGRATION_SUCCESS_REPORT.md`
- Phase 7 Complete: `docs/PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md`
- QA Test Script: `docs/QA_SCRIPT_PHASE_7.md`
- Mock Server Guide: `mock-server/README.md`

---

**Need Help?** Check the full documentation or the integration report for detailed information.


