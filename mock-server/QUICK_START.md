# Mock Server Quick Start Guide

## ⚡ Get Started in 60 Seconds

### Step 1: Install Dependencies (10 seconds)

```bash
cd mock-server
pip install -r requirements.txt
```

### Step 2: Start Server (5 seconds)

**Windows**:
```powershell
.\start.ps1
```

**Linux/Mac**:
```bash
./start.sh
```

**Or directly**:
```bash
python app.py
```

### Step 3: Verify It's Running (5 seconds)

Open in browser: http://localhost:8000/docs

Or test with curl:
```bash
curl http://localhost:8000/api/v1/health
```

### Step 4: Use in Frontend (30 seconds)

Update your frontend API configuration:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

Done! 🎉

---

## 📊 Available Fixtures

Quick test different datasets by adding `?fixture=NAME` to search endpoint:

| Fixture | Count | Purpose | Example |
|---------|-------|---------|---------|
| `tiny` | 10 | Quick UI testing | `?fixture=tiny` |
| `medium` | 200 | **Default** development | `?fixture=medium` |
| `large` | 2000 | Stress testing | `?fixture=large` |
| `edge_cases` | 15 | Boundary testing | `?fixture=edge_cases` |
| `errors` | 20 | Error handling | `?fixture=errors` |

---

## 🧪 Test Scenarios

### Basic Search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"
```

### Large Dataset
```bash
curl -X POST "http://localhost:8000/api/v1/search?fixture=large" \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg" \
  -F "top_k=50"
```

### No Results
```bash
curl -X POST "http://localhost:8000/api/v1/search?error_scenario=no_results" \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"
```

### All Broken URLs
```bash
curl -X POST "http://localhost:8000/api/v1/search?fixture=errors" \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"
```

---

## ⚙️ Configuration

### Disable Latency (Faster Testing)
```bash
curl -X POST http://localhost:8000/mock/config \
  -H "Content-Type: application/json" \
  -d '{"simulate_latency": false}'
```

### Use Large Dataset by Default
```bash
curl -X POST http://localhost:8000/mock/config \
  -H "Content-Type: application/json" \
  -d '{"default_fixture": "large"}'
```

### Add Random Errors (5% chance)
```bash
curl -X POST http://localhost:8000/mock/config \
  -H "Content-Type: application/json" \
  -d '{"error_rate": 0.05}'
```

---

## 📚 Documentation

- **Full Documentation**: See `README.md`
- **API Docs**: http://localhost:8000/docs (auto-generated)
- **Fixtures Info**: `curl http://localhost:8000/mock/fixtures`
- **Phase 3 Complete**: See `../docs/PHASE_3_MOCK_SERVER_COMPLETE.md`

---

## 🐛 Troubleshooting

**Port 8000 already in use?**
```bash
# Use different port
python -c "from app import app; import uvicorn; uvicorn.run(app, port=8001)"
```

**Dependencies not installed?**
```bash
pip install fastapi uvicorn python-multipart
```

**Server not responding?**
```bash
# Check if running
curl http://localhost:8000/api/v1/health
```

---

## ✅ Quick Verification

Run test suite to verify everything works:

```bash
python test_mock_server.py
```

Expected: **9/9 tests passed** 🎉

---

## 🎯 Next Steps

1. ✅ Start mock server
2. ✅ Update frontend to use `http://localhost:8000`
3. ✅ Test with different fixtures
4. ✅ Test error scenarios
5. 🚀 Develop frontend independently!

When backend is ready, just change the URL back to real API.

---

**Questions?** See full documentation in `README.md` or `../docs/PHASE_3_MOCK_SERVER_COMPLETE.md`

