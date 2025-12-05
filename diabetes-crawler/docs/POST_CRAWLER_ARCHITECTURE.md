# Post Crawler Architecture Analysis

## Overview
This document explains how the crawler system has been adapted from image crawling to diabetes post crawling, focusing on `crawler_worker.py` and the overall architecture.

**Current Mode**: Post-focused crawler with image extraction and GPU processing **DISABLED**.

---

## System Architecture Flow

```
SiteTask (Redis) 
    ↓
CrawlerWorker (crawler_worker.py)
    ↓ [uses]
SelectorMiner (selector_miner.py) → mine_posts_with_3x3_crawl()
    ↓ [yields]
CandidatePost objects
    ↓ [pushed to]
Redis candidates queue
    ↓
ExtractorWorker (extractor_worker.py)
    ↓ [creates]
PostTask
    ↓ [pushed to]
Redis storage queue
    ↓
StorageWorker (storage_worker.py)
    ↓ [saves]
PostMetadata to MinIO (s3_bucket_posts)
```

---

## File-by-File Analysis

### 1. `crawler_worker.py` - **CORE CRAWLER LOGIC**

#### How It Works:
- **Purpose**: Fetches HTML pages and mines them for posts using selector mining
- **Main Loop**: `run()` method continuously pops `SiteTask` from Redis, processes sites concurrently
- **Site Processing**: `process_site()` method handles each site:
  1. Calls `selector_miner.mine_posts_with_3x3_crawl()` (async generator)
  2. Receives `(page_url, page_candidates)` tuples as pages are crawled
  3. Enqueues `CandidatePost` objects to Redis candidates queue
  4. Updates site statistics (`pages_crawled`, `posts_found`)

#### Alignment with Post Finding:
✅ **ALREADY ALIGNED** - The worker has been updated to:
- Use `mine_posts_with_3x3_crawl()` instead of image mining
- Track `posts_found` instead of `images_found` in stats
- Handle `CandidatePost` objects (line 50, 57)
- Update stats with `posts_found` (line 146)

#### Issues Found:
⚠️ **MINOR ISSUE** - Line 183 in `_process_site_task()` still updates `images_found`:
```python
await self.redis.update_site_stats_async(
    site_task.site_id,
    {
        'images_found': candidates_count  # ❌ Should be 'posts_found'
    }
)
```
This is redundant since `process_site()` already updates `posts_found` correctly.

---

### 2. `selector_miner.py` - **POST DISCOVERY ENGINE**

#### How It Works:
- **Main Method**: `mine_posts_with_3x3_crawl()` performs multi-page crawling:
  1. Fetches base page
  2. Discovers forum/board category pages using `_discover_category_pages()`
  3. Crawls up to 7 sample pages in parallel
  4. Continues BFS crawl if more pages needed
  5. Yields `(page_url, List[CandidatePost])` for each page

- **Post Extraction**: `mine_posts_for_diabetes()` extracts posts from HTML:
  1. Detects page type (listing vs. detail page)
  2. Uses heuristic patterns to find post containers
  3. Extracts title, content, author, date, URL
  4. Filters for diabetes-related keywords
  5. Returns `List[CandidatePost]`

#### Image Extraction (DEPRECATED):
⚠️ **DISABLED** - Image extraction methods are disabled in post-focused mode:
- `mine_selectors()` - Returns empty list if `nc_enable_image_extraction = False`
- `_extract_noscript_images()` - Returns empty list
- `_extract_jsonld_images()` - Returns empty list
- `_extract_script_images()` - Returns empty list
- `_extract_background_image()` - Returns None
- `_extract_from_srcset()` - Returns None

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - All methods updated for posts:
- `_discover_category_pages()` prioritizes forum/board links
- `mine_posts_for_diabetes()` extracts post metadata
- `_create_post_candidate()` builds `CandidatePost` objects
- `_detect_page_type()` differentiates listing vs. detail pages
- Image extraction methods disabled via `nc_enable_image_extraction` flag

---

### 3. `data_structures.py` - **DATA MODELS**

#### Key Models:
- **`CandidatePost`**: Represents a discovered post with title, content, author, date
- **`PostTask`**: Task for processing a post (bypasses GPU)
- **`PostMetadata`**: Final metadata saved to MinIO

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - All post-related models exist and are properly structured

---

### 4. `extractor_worker.py` - **POST PROCESSING**

#### How It Works:
- **Main Loop**: `run()` pops `CandidatePost` or `CandidateImage` from candidates queue
- **Routing**: `process_candidate()` routes to:
  - `_process_post_candidate()` for posts
  - `_process_image_candidate()` for images (legacy, **SKIPPED** if `nc_enable_image_extraction = False`)

#### Post Processing Flow:
1. Checks URL deduplication
2. Checks site limits (`nc_max_posts_per_site`)
3. Creates content hash
4. Creates `PostTask`
5. **Bypasses GPU** - pushes directly to storage queue

#### Image Processing (DEPRECATED):
⚠️ **DISABLED** - Image candidates are skipped if `nc_enable_image_extraction = False`:
- `CandidateImage` objects return `None` immediately
- No image downloads or temp file creation
- No GPU inbox enqueueing

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - Post processing is complete:
- `_process_post_candidate()` handles posts (line 268)
- Bypasses GPU worker (line 336)
- Pushes to storage queue (line 336)
- Updates stats with `posts_processed` (line 402)
- Image processing skipped when disabled

---

### 5. `storage_worker.py` - **MINIO SAVING**

#### How It Works:
- **Main Loop**: `run()` pops `StorageTask` or `PostTask` from storage queue
- **Routing**: Routes based on task type:
  - `_process_post_task()` for posts
  - `_process_storage_task()` for images (legacy)

#### Post Storage Flow:
1. Calls `storage.save_post_metadata_async()`
2. Saves JSON to `s3_bucket_posts` bucket
3. Updates stats with `posts_saved: 1`
4. Checks site limits and removes remaining items if limit reached

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - Post storage is complete:
- `_process_post_task()` handles posts (line 338)
- Saves to MinIO correctly
- Updates stats correctly

---

### 6. `redis_manager.py` - **QUEUE MANAGEMENT**

#### Key Methods:
- `push_candidate_async()`: Handles both `CandidateImage` and `CandidatePost`
- `pop_candidate()`: Deserializes to correct type
- `push_post_task_async()`: Pushes `PostTask` to storage queue
- `pop_storage_task_async()`: Returns `Union[StorageTask, PostTask]`
- `serialize_post_task()` / `deserialize_post_task()`: Post-specific serialization

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - All queue operations support posts

---

### 7. `storage_manager.py` - **MINIO OPERATIONS**

#### Key Methods:
- `save_post_metadata()`: Saves `PostMetadata` as JSON to `s3_bucket_posts`
- `save_post_metadata_async()`: Async wrapper

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - Post saving implemented

---

### 8. `config.py` - **CONFIGURATION**

#### Post-Related Config:
- `nc_max_posts_per_site: int = 100` - Max posts per site
- `s3_bucket_posts: str = "raw-images"` - MinIO bucket for posts (reuses raw-images bucket)

#### Feature Flags (Post-Focused Mode):
- `nc_enable_image_extraction: bool = False` - **DISABLED** - Controls image extraction in selector_miner
- `nc_enable_gpu_processing: bool = False` - **DISABLED** - Controls GPU processor workers

#### Alignment with Post Finding:
✅ **FULLY ALIGNED** - Post configuration exists with image/GPU features disabled

---

## Summary of Changes Needed

### ✅ Already Complete:
1. ✅ `selector_miner.py` - Post extraction logic
2. ✅ `data_structures.py` - Post models
3. ✅ `extractor_worker.py` - Post processing
4. ✅ `storage_worker.py` - Post storage
5. ✅ `redis_manager.py` - Post queue operations
6. ✅ `storage_manager.py` - Post MinIO saving
7. ✅ `config.py` - Post configuration

### ⚠️ Minor Issues:
1. **`crawler_worker.py` line 183**: Redundant `images_found` update in `_process_site_task()`
   - **Fix**: Remove or change to `posts_found` (though `process_site()` already updates it correctly)

---

## How Crawler Worker Aligns with Post Finding

### Current Flow:
1. **Orchestrator** pushes `SiteTask` to Redis
2. **CrawlerWorker** pops `SiteTask` from Redis
3. **CrawlerWorker.process_site()** calls `selector_miner.mine_posts_with_3x3_crawl()`
4. **SelectorMiner** crawls pages and yields `(page_url, List[CandidatePost])`
5. **CrawlerWorker** enqueues `CandidatePost` objects to Redis candidates queue
6. **ExtractorWorker** pops candidates, creates `PostTask`, pushes to storage queue
7. **StorageWorker** pops `PostTask`, saves `PostMetadata` to MinIO

### Key Alignment Points:
- ✅ Uses `mine_posts_with_3x3_crawl()` instead of image mining
- ✅ Handles `CandidatePost` objects
- ✅ Tracks `posts_found` in statistics
- ✅ Enqueues to correct queue (candidates queue)
- ✅ Supports concurrent site processing

### What's Different from Image Crawling:
1. **No Image Extraction**: Image extraction methods disabled via `nc_enable_image_extraction = False`
2. **No GPU Processing**: GPU processor workers disabled via `nc_enable_gpu_processing = False`
3. **Different Queue Path**: `candidates → storage` (no GPU inbox, no image processing)
4. **Different Statistics**: Tracks `posts_found` instead of `images_found`
5. **Different Storage**: Saves JSON metadata instead of images
6. **Simplified Flow**: Posts go directly from extractor to storage (no GPU worker)

---

## Testing Recommendations

1. **Verify Post Discovery**: Check that `mine_posts_for_diabetes()` finds posts correctly
2. **Verify Queue Flow**: Ensure `CandidatePost` → `PostTask` → `PostMetadata` flow works
3. **Verify Storage**: Confirm posts are saved to `s3_bucket_posts` bucket
4. **Verify Statistics**: Check that `posts_found` and `posts_saved` are tracked correctly
5. **Verify Limits**: Test that `nc_max_posts_per_site` limit is enforced

---

## Post-Focused Mode Implementation

### Image Extraction Disabled
- `mine_selectors()` returns empty list when `nc_enable_image_extraction = False`
- All image extraction helper methods return empty lists/None
- Only `CandidatePost` objects are created, never `CandidateImage`

### GPU Processing Disabled
- `num_gpu_processors` set to 0 when `nc_enable_gpu_processing = False`
- GPU processor workers are not started
- GPU queue monitoring (gpu:inbox, gpu:inflight) is skipped
- Back-pressure monitoring excludes GPU queues

### Post Extraction Flow
1. **Listing Page** → `mine_posts_with_3x3_crawl()` discovers post links
2. **Detail Page** → `mine_posts_for_diabetes()` extracts post content
3. **Extraction** → `_process_post_candidate()` creates `PostTask`
4. **Storage** → `save_post_metadata_async()` saves to MinIO

### MinIO Storage Structure
- **All posts**: `raw-images/posts/{site_id}/{content_hash}.json`
- **All posts HTML**: `raw-images/posts/{site_id}/{content_hash}.html` (if available)
- **Keyword-filtered posts**: `posts/{site_id}/{content_hash}.json` (diabetes keywords only)

## Conclusion

The crawler system is **fully aligned** with the post-finding goal. Image extraction and GPU processing are disabled via configuration flags. The architecture correctly:
- Discovers posts using heuristic patterns
- Processes posts without GPU (directly to storage)
- Saves post metadata to MinIO
- Tracks post-related statistics
- Skips all image extraction and GPU processing

The system is ready for testing and should work correctly for diabetes post crawling in post-focused mode.

