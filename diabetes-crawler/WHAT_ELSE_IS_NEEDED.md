# What Else is Needed for Crawls to Work?

## ✅ Already Set Up

1. **Redis** - Running on port 6379 ✓
2. **MinIO** - Accessible on port 9000 ✓ (there's already a MinIO instance running)
3. **Playwright Browsers** - Installed ✓
4. **Python Dependencies** - All installed ✓
5. **.env file** - Created from template ✓

## ⚠️ Still Need to Configure

### 1. Update `.env` File (if needed)

The `.env` file has been created with defaults. Check and update these settings:

**Critical Settings:**
- `redis_url` - Should be `redis://localhost:6379/0` for local dev
- `s3_endpoint` - Should match your MinIO instance (default: `http://localhost:9000`)
- `s3_access_key` / `s3_secret_key` - Your MinIO credentials (default: `MINIOADMIN`)

**Optional Settings:**
- `gpu_worker_enabled` - Set to `false` if you don't have a GPU worker (CPU fallback will be used)
- `vectorization_enabled` - Set to `false` if you don't have Qdrant running

### 2. Update `sites.txt` with Real URLs

Currently has example URLs:
```
https://example.com/models
https://another-example.com/gallery
```

Replace with real sites you want to crawl (one URL per line).

### 3. Verify MinIO Buckets Exist

The crawler needs these buckets:
- `raw-images` (or your configured `s3_bucket_raw`)
- `thumbnails` (or your configured `s3_bucket_thumbs`)

You can create them via MinIO console (http://localhost:9001) or the buckets will be created automatically on first use.

### 4. Optional: Start GPU Worker (for faster processing)

If you have a GPU worker available:
- Set `gpu_worker_url` in `.env` (e.g., `http://localhost:8765`)
- Ensure GPU worker is running

If not available, the crawler will automatically use CPU fallback (slower but functional).

### 5. Optional: Start Qdrant (for vector storage)

Only needed if `vectorization_enabled=true`:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Or set `vectorization_enabled=false` in `.env` to skip vector storage.

## 🚀 Ready to Test?

Run a health check to verify everything:

```bash
cd diabetes-crawler
source activate.sh
python -m diabetes_crawler.main --health-check
```

This will:
- ✅ Check Redis connectivity
- ✅ Check MinIO/S3 connectivity
- ⚠️ Check GPU worker (optional, will warn if unavailable)
- ⚠️ Show any configuration issues

## 📝 Minimum Requirements Summary

**Required:**
1. ✅ Redis (running)
2. ✅ MinIO/S3 (for storage) 
3. ✅ .env configuration file
4. ✅ Sites to crawl (sites.txt)

**Optional:**
- GPU Worker (CPU fallback available)
- Qdrant (only if vectorization enabled)
- Playwright browsers (already installed)

## 🎯 Next Steps

1. **Edit `sites.txt`** with real URLs
2. **Verify `.env` settings** match your setup
3. **Run health check**: `python -m diabetes_crawler.main --health-check`
4. **Start crawling**: `python -m diabetes_crawler.main --sites-file sites.txt`

That's it! The crawler should be ready to go. 🎉

