# Dev3 Context - Crawler Feature

## Key Files
- `backend/app/services/crawler.py` - Main crawler service
- `backend/app/services/face.py` - Face detection service
- `backend/app/services/storage.py` - MinIO/S3 storage
- `backend/scripts/crawl_images.py` - CLI interface
- `backend/app/services/cache.py` - Redis + PostgreSQL caching

## Pipeline Overview
1. **Discover**: Smart CSS selectors find images (data-mediumthumb, js-videoThumb, etc.)
2. **Download**: Stream images with early abort and validation
3. **Detect**: Multi-scale face detection with enhancement and early exit
4. **Store**: Content-addressed storage with Blake3 hashing for deduplication
5. **Cache**: Hybrid Redis/PostgreSQL caching prevents reprocessing

## Test Protocol

1. **Rebuild Backend**
   ```bash
   make down && make up
   ```

2. **Initial Crawl**
   ```bash
   make crawl URL=https://www.pornhub.com REQUIRE_FACE=false CROP_FACES=true CRAWL_MODE=site MAX_PAGES=5 MAX_TOTAL_IMAGES=100
   ```

3. **Check MinIO Storage** (localhost:9001)
   - Files in `raw-images` bucket: `default/{hash[:2]}/{hash}.jpg`
   - Files in `thumbnails` bucket: `default/{hash[:2]}/{hash}_thumb.jpg`

4. **Verify Accuracy**
   ```bash
   make download-both
   ```
   Check `MORDEAUX-Face-Scanning-MVP/flat/thumbnails/` - most should be faces

5. **Test Cache Hits**
   ```bash
   make crawl URL=https://www.pornhub.com REQUIRE_FACE=false CROP_FACES=true CRAWL_MODE=site MAX_PAGES=1 MAX_TOTAL_IMAGES=10
   ```
   Should show cache hits and few new images

6. **Reset Caches**
   ```bash
   make reset-both
   ```

7. **Verify Cache Miss**
   ```bash
   make crawl URL=https://www.pornhub.com REQUIRE_FACE=false CROP_FACES=true CRAWL_MODE=site MAX_PAGES=1 MAX_TOTAL_IMAGES=10
   ```
   Should show 0 cache hits
