# Crawler Setup Checklist

This document outlines everything needed to run crawls successfully.

## ✅ Completed

- [x] Redis is running on port 6379
- [x] Python dependencies installed
- [x] Playwright installed (v1.48.0)
- [x] Docker Compose configuration created

## ⚠️ Required Setup

### 1. Environment Configuration (.env file)
**Status:** Missing - needs to be created

```bash
cd diabetes-crawler
cp .env.example .env
# Edit .env with your settings (defaults should work for local dev)
```

### 2. MinIO Storage Service
**Status:** Not running - port 9000 is in use

The crawler requires MinIO (or S3) for storing images and metadata. Options:

**Option A: Use existing MinIO** (if you have one running on port 9000)
- Update `.env` to point to your existing MinIO instance

**Option B: Start MinIO in docker-compose** (if port 9000 is free)
```bash
./start-services.sh
```

**Option C: Use different ports** (if port 9000 is taken)
- Edit `docker-compose.yml` to use different ports
- Update `.env` accordingly

**Option D: Use AWS S3** (production)
- Set `s3_endpoint`, `s3_access_key`, `s3_secret_key` in `.env`

### 3. Playwright Browsers
**Status:** Not installed - needed for JavaScript rendering

Playwright requires browser binaries to be downloaded:

```bash
source activate.sh
playwright install chromium
# Or install all browsers: playwright install
```

### 4. Sites to Crawl
**Status:** Has example URLs - update with real targets

Edit `sites.txt` with URLs you want to crawl (one per line).

## 🔧 Optional but Recommended

### 5. GPU Worker (for face detection)
**Status:** Optional - has CPU fallback

The crawler can work without a GPU worker (uses CPU fallback), but GPU is much faster.

- If you have a GPU worker, set `gpu_worker_url` in `.env`
- If not, the crawler will automatically use CPU fallback
- You can disable GPU worker entirely: `gpu_worker_enabled=false`

### 6. Qdrant Vector Database
**Status:** Optional - only if vectorization is enabled

Required only if `vectorization_enabled=true` in `.env`:

- Install Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- Or set `vectorization_enabled=false` to skip vector storage

## 📋 Quick Start Commands

```bash
# 1. Create environment file
cp .env.example .env

# 2. Install Playwright browsers
source activate.sh
playwright install chromium

# 3. Start services (Redis + MinIO)
./start-services.sh

# 4. Edit sites.txt with your URLs

# 5. Run health check
python -m diabetes_crawler.main --health-check

# 6. Start crawling
python -m diabetes_crawler.main --sites-file sites.txt
```

## 🔍 Verify Setup

Run the health check to verify everything is configured:

```bash
source activate.sh
python -m diabetes_crawler.main --health-check
```

This will check:
- ✅ Redis connectivity
- ✅ MinIO/S3 connectivity (if configured)
- ✅ GPU worker availability (optional)
- ⚠️ Will show warnings for missing optional services

## 🚨 Common Issues

### "Port 9000 already in use"
- Another MinIO instance is running
- Either use that instance or change ports in docker-compose.yml

### "GPU worker not available"
- This is fine - CPU fallback will be used automatically
- Performance will be slower but functional

### "Playwright browser not found"
- Run: `playwright install chromium`

### "Redis connection failed"
- Check Redis is running: `docker-compose ps redis`
- Verify `redis_url` in `.env` matches your setup

## 📊 Minimum Requirements

To run basic crawls, you need:
1. ✅ Redis (running)
2. ⚠️ MinIO/S3 (for storage)
3. ⚠️ .env file (configuration)
4. ⚠️ Playwright browsers (for JS rendering)
5. ⚠️ Sites to crawl

Everything else is optional!

