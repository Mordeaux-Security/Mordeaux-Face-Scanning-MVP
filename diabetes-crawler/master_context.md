# Diabetes Crawler Master Context

## Overview

This document provides comprehensive documentation for all Python files in the diabetes crawler system. The crawler is a multi-process, distributed web crawling system designed to extract diabetes-related posts and images from websites. It uses Redis for queue management, MinIO for storage, and includes GPU-accelerated face detection capabilities.

**System Architecture:**

- **Entry Point**: `main.py` - CLI interface and orchestration
- **Orchestration**: `orchestrator.py` - Manages worker processes and system coordination
- **Workers**:
  - `crawler_worker.py` - Fetches HTML and discovers posts/images
  - `extractor_worker.py` - Downloads and processes candidates
  - `gpu_processor_worker.py` - GPU-accelerated face detection
  - `storage_worker.py` - Saves results to MinIO
- **Infrastructure**:
  - `redis_manager.py` - Redis queue operations
  - `storage_manager.py` - MinIO/S3 storage operations
  - `http_utils.py` - HTTP fetching and JavaScript rendering
  - `gpu_interface.py` - GPU worker communication
  - `gpu_scheduler.py` - GPU batch scheduling
- **Utilities**:
  - `selector_miner.py` - Post/image discovery from HTML
  - `cache_manager.py` - Perceptual hash caching
  - `timing_logger.py` - Performance timing instrumentation
  - `extraction_tracer.py` - Extraction attempt tracking
  - `gpu_worker_logger.py` - GPU operation logging
- **Data Models**: `data_structures.py` - Pydantic models for all data structures
- **Configuration**: `config.py` - Centralized configuration management

---

## Files

### 1. `__init__.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/__init__.py`
- **Lines**: 0
- **Scope**: Package initialization
- **Purpose**: Empty package initialization file for Python package structure
- **Dependencies**: None
- **Used By**: Python import system

#### Methods

None (empty file)

---

### 2. `data_structures.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/data_structures.py`
- **Lines**: 349
- **Scope**: Data models and type definitions
- **Purpose**: Defines all Pydantic models used throughout the system for type safety and serialization
- **Dependencies**:
  - `pydantic` (BaseModel, Field, validator, model_serializer, field_validator)
  - `datetime`
  - `enum`
  - `base64`, `json`
- **Used By**: All files that handle data structures (workers, managers, etc.)

#### Classes

##### TaskStatus (Enum)

- **Purpose**: Status enumeration for processing tasks
- **Values**: PENDING, PROCESSING, COMPLETED, FAILED, SKIPPED

##### SiteTask (BaseModel)

- **Purpose**: Task for crawling a site
- **Fields**: url, site_id, priority, max_pages, max_images_per_site, use_3x3_mining, pages_crawled, images_saved, created_at, status
- **Validators**: `validate_url()` - Ensures URL starts with http:// or https://

##### CandidateImage (BaseModel)

- **Purpose**: Candidate image found during crawling
- **Fields**: page_url, img_url, selector_hint, site_id, alt_text, width, height, discovered_at, content_type, estimated_size, has_srcset
- **Validators**: `validate_img_url()` - Ensures image URL starts with http:// or https://

##### CandidatePost (BaseModel)

- **Purpose**: Candidate post found during crawling that may contain diabetes-related content
- **Fields**: page_url, post_url, selector_hint, site_id, title, content, author, date, raw_html, discovered_at
- **Validators**: `validate_post_url()` - Ensures post URL starts with http:// or https://

##### ImageTask (BaseModel)

- **Purpose**: Task for processing an image
- **Fields**: temp_path, phash, candidate, file_size, mime_type, created_at, status

##### PostTask (BaseModel)

- **Purpose**: Task for processing a post for diabetes-related content
- **Fields**: candidate, content_hash, has_keywords, created_at, status

##### PostMetadata (BaseModel)

- **Purpose**: Metadata for a diabetes-related post to be stored in MinIO
- **Fields**: title, content, author, url, date, site_id, content_hash, discovered_at, page_url

##### ImageMetadata (BaseModel)

- **Purpose**: Metadata for image processing (JSON serializable, binary data stored separately)
- **Fields**: phash, candidate, file_size, mime_type, redis_binary_key, created_at

##### FaceDetection (BaseModel)

- **Purpose**: Face detection result
- **Fields**: bbox, landmarks, embedding, quality, age, gender
- **Validators**: `validate_bbox()` - Ensures bounding box has exactly 4 coordinates

##### FaceResult (BaseModel)

- **Purpose**: Result of face processing for an image
- **Fields**: image_task, faces, crop_paths, raw_image_key, thumbnail_keys, processing_time_ms, gpu_used, saved_to_raw, saved_to_thumbs, skip_reason, created_at

##### StorageTask (BaseModel)

- **Purpose**: Task for storage operations - pre-cropped faces ready for I/O
- **Fields**: image_task, face_result, face_crops, batch_start_time, created_at
- **Methods**:
  - `model_dump()` - Override to base64-encode face_crops for JSON serialization
  - `model_dump_json()` - Override to ensure face_crops are base64-encoded
  - `decode_face_crops()` (field_validator) - Decode base64 strings back to bytes if needed

##### ProcessingStats (BaseModel)

- **Purpose**: Statistics for site processing
- **Fields**: site_id, site_url, pages_fetched, pages_crawled, images_found, images_processed, images_saved_raw, images_saved_thumbs, images_skipped_limit, images_cached, faces_detected, errors, start_time, end_time, total_time_seconds, extraction_start_time, extraction_end_time, gpu_processing_start_time, gpu_processing_end_time, storage_start_time, storage_end_time
- **Properties**:
  - `success_rate` - Calculate success rate (images_processed + images_cached) / images_found
  - `images_per_second` - Calculate overall images/second throughput
  - `extraction_images_per_second` - Calculate extraction throughput
  - `gpu_images_per_second` - Calculate GPU processing throughput

##### BatchRequest (BaseModel)

- **Purpose**: Request for GPU worker batch processing
- **Fields**: image_tasks, min_face_quality, require_face, crop_faces, face_margin, batch_id, created_at

##### BatchResponse (BaseModel)

- **Purpose**: Response from GPU worker batch processing
- **Fields**: batch_id, results, processing_time_ms, gpu_used, worker_id, created_at

##### QueueMetrics (BaseModel)

- **Purpose**: Metrics for queue monitoring
- **Fields**: queue_name, depth, max_depth, utilization_percent, last_updated

##### SystemMetrics (BaseModel)

- **Purpose**: Overall system metrics
- **Fields**: active_crawlers, active_extractors, active_gpu_processors, queue_metrics, total_sites_processed, total_images_processed, total_faces_detected, gpu_worker_available, last_updated

##### CrawlResults (BaseModel)

- **Purpose**: Final results of a crawl operation
- **Fields**: sites, system_metrics, total_time_seconds, success_rate, created_at
- **Properties**:
  - `total_images_found` - Total images found across all sites
  - `total_images_processed` - Total images processed across all sites
  - `total_faces_detected` - Total faces detected across all sites
  - `overall_images_per_second` - Calculate overall images/second across all sites

#### Methods

None (data models only)

---

### 3. `config.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/config.py`
- **Lines**: 438
- **Scope**: Configuration management
- **Purpose**: Centralized configuration using Pydantic Settings with environment variable loading, Docker/Windows addressing, and validation
- **Dependencies**:
  - `os`, `logging`, `redis`
  - `pydantic` (BaseModel, Field, field_validator)
  - `pydantic_settings` (BaseSettings)
  - `urllib.parse` (urlparse)
- **Used By**: All files that need configuration (virtually all files)

#### Classes

##### CrawlerConfig (BaseSettings)

- **Purpose**: Configuration for the new crawler system
- **Configuration Categories**:
  - Environment: environment, log_level
  - Redis: redis_url, redis_max_connections, redis_retry_on_timeout
  - Crawling Limits: nc_max_pages_per_site, nc_max_images_per_site, nc_max_posts_per_site, nc_max_post_links_per_page, nc_strict_limits
  - Feature Flags: nc_enable_image_extraction, nc_enable_gpu_processing
  - Debug Logging: nc_debug_logging, nc_gpu_worker_logging, nc_diagnostic_logging, nc_diagnostic_log_interval
  - Queue Configuration: nc_batch_size, nc_max_queue_depth, nc_extractor_concurrency, nc_extractor_batch_pop_size, nc_url_dedup_ttl_hours, nc_cache_ttl_days
  - HTTP Performance: nc_skip_head_check
  - Crawler Performance: nc_max_concurrent_sites_per_worker
  - GPU Performance: nc_max_concurrent_batches_per_worker, nc_batch_flush_timeout, gpu_target_batch, gpu_max_wait_ms, gpu_min_launch_ms, gpu_inbox_key
  - GPU Processor Worker: image_processing_idle_wait
  - MinIO: minio_max_pool_size, minio_pool_timeout
  - HTTP: nc_http_timeout, nc_js_render_timeout, nc_max_redirects, nc_max_retries, nc_retry_base_delay, nc_retry_max_delay, nc_retry_jitter, nc_circuit_breaker_failure_threshold, nc_circuit_breaker_open_timeout_base
  - JavaScript Rendering: nc_js_wait_strategy, nc_js_wait_timeout, nc_js_networkidle_timeout, nc_js_first_visit_compare, nc_js_concurrency, nc_js_browser_pool_size, nc_js_block_resources, nc_js_aggressive_http
  - Image Extraction: nc_extract_background_images, nc_extract_srcset_images, nc_extract_data_attributes
  - Advanced HTTP: nc_same_origin_redirects_only, nc_blocklist_redirect_hosts, nc_realistic_headers
  - Advanced JavaScript: nc_js_wait_selectors, nc_capture_network_images, nc_extract_script_images, nc_extract_noscript_images, nc_extract_jsonld_images
  - Selector Mining: nc_use_3x3_mining, nc_max_selector_patterns
  - Storage: s3_endpoint, s3_region, s3_bucket_raw, s3_bucket_thumbs, s3_bucket_posts, s3_access_key, s3_secret_key, s3_use_ssl
  - Vector Database: vectorization_enabled, default_tenant_id, qdrant_url, vector_index
  - Face Detection: min_face_quality, min_face_size, min_image_file_size_bytes, face_margin, max_face_yaw_deg, max_face_pitch_deg
  - Back-pressure: backpressure_threshold, backpressure_check_interval
  - Monitoring: metrics_interval, health_check_interval
  - Worker Configuration: num_crawlers, num_extractors, num_gpu_processors, num_storage_workers
  - GPU Worker: gpu_worker_enabled, gpu_worker_url, gpu_worker_timeout, gpu_worker_max_retries

#### Methods

##### validate_gpu_worker_url() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_gpu_worker_url(cls, v) -> str`
- **Pseudocode**:
  ```
  INPUT: v (GPU worker URL string)
  MANIPULATION:
    - Check if URL starts with http:// or https://
    - Parse URL using urlparse
    - Validate hostname exists
    - Return validated URL
  OUTPUT: Validated URL string
  ```
- **Assumptions**: URL is a string
- **Called By**: Pydantic validation (automatic)

##### validate_redis_url() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_redis_url(cls, v) -> str`
- **Pseudocode**:
  ```
  INPUT: v (Redis URL string)
  MANIPULATION:
    - Check if URL starts with redis:// or rediss://
    - Return validated URL
  OUTPUT: Validated URL string
  ```
- **Assumptions**: URL is a string
- **Called By**: Pydantic validation (automatic)

##### validate_log_level() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_log_level(cls, v) -> str`
- **Pseudocode**:
  ```
  INPUT: v (log level string)
  MANIPULATION:
    - Check if v.lower() is in allowed list: ['debug', 'info', 'warning', 'error', 'critical']
    - Return lowercase version
  OUTPUT: Validated lowercase log level string
  ```
- **Assumptions**: v is a string
- **Called By**: Pydantic validation (automatic)

##### validate_js_wait_strategy() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_js_wait_strategy(cls, v) -> str`
- **Pseudocode**:
  ```
  INPUT: v (wait strategy string)
  MANIPULATION:
    - Check if v.lower() is in allowed list: ['fixed', 'networkidle', 'both']
    - Return validated strategy
  OUTPUT: Validated strategy string
  ```
- **Assumptions**: v is a string
- **Called By**: Pydantic validation (automatic)

##### validate_js_wait_timeout() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_js_wait_timeout(cls, v) -> float`
- **Pseudocode**:
  ```
  INPUT: v (timeout value)
  MANIPULATION:
    - Check if v is between 1.0 and 30.0 seconds
    - Return validated timeout
  OUTPUT: Validated timeout float
  ```
- **Assumptions**: v is a number
- **Called By**: Pydantic validation (automatic)

##### validate_blocklist_redirect_hosts() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_blocklist_redirect_hosts(cls, v) -> list[str]`
- **Pseudocode**:
  ```
  INPUT: v (list of host strings)
  MANIPULATION:
    - Check if v is a list
    - For each host in v:
      - Check if host is a non-empty string
    - Return validated list
  OUTPUT: Validated list of host strings
  ```
- **Assumptions**: v is a list
- **Called By**: Pydantic validation (automatic)

##### validate_js_wait_selectors() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_js_wait_selectors(cls, v) -> str`
- **Pseudocode**:
  ```
  INPUT: v (CSS selector string)
  MANIPULATION:
    - Check if v is a non-empty string
    - Strip whitespace
    - Return validated selector
  OUTPUT: Validated selector string
  ```
- **Assumptions**: v is a string
- **Called By**: Pydantic validation (automatic)

##### validate_max_pages() (classmethod, field_validator)

- **Signature**: `@classmethod def validate_max_pages(cls, v) -> int`
- **Pseudocode**:
  ```
  INPUT: v (max pages value)
  MANIPULATION:
    - Check if v >= -1 and v != 0
    - Return validated value
  OUTPUT: Validated max pages int
  ```
- **Assumptions**: v is an integer
- **Called By**: Pydantic validation (automatic)

##### is_docker (property)

- **Signature**: `@property def is_docker(self) -> bool`
- **Pseudocode**:
  ```
  INPUT: self (config instance)
  MANIPULATION:
    - Check if /.dockerenv file exists OR DOCKER_CONTAINER env var is 'true'
    - Return True if Docker, False otherwise
  OUTPUT: Boolean indicating Docker environment
  ```
- **Assumptions**: None
- **Called By**: Various files checking environment

##### is_windows (property)

- **Signature**: `@property def is_windows(self) -> bool`
- **Pseudocode**:
  ```
  INPUT: self (config instance)
  MANIPULATION:
    - Check if os.name == 'nt'
    - Return True if Windows, False otherwise
  OUTPUT: Boolean indicating Windows environment
  ```
- **Assumptions**: None
- **Called By**: Various files checking environment

##### gpu_worker_host_resolved (property)

- **Signature**: `@property def gpu_worker_host_resolved(self) -> str`
- **Pseudocode**:
  ```
  INPUT: self (config instance)
  MANIPULATION:
    - Parse gpu_worker_url
    - If already using host.docker.internal, return as-is
    - If in Docker and hostname is localhost/127.0.0.1:
      - Replace with host.docker.internal:port
    - Otherwise return URL as-is
  OUTPUT: Resolved GPU worker URL string
  ```
- **Assumptions**: gpu_worker_url is valid
- **Called By**: gpu_interface.py, main.py

##### queue_names (property)

- **Signature**: `@property def queue_names(self) -> Dict[str, str]`
- **Pseudocode**:
  ```
  INPUT: self (config instance)
  MANIPULATION:
    - Return dictionary mapping queue types to queue names:
      - 'sites': 'nc:sites'
      - 'candidates': 'nc:candidates'
      - 'images': 'nc:images'
      - 'results': 'nc:results'
      - 'storage': 'nc:storage'
      - 'cpu_fallback': 'nc:cpu_fallback'
  OUTPUT: Dictionary of queue names
  ```
- **Assumptions**: None
- **Called By**: redis_manager.py, orchestrator.py

##### cache_keys (property)

- **Signature**: `@property def cache_keys(self) -> Dict[str, str]`
- **Pseudocode**:
  ```
  INPUT: self (config instance)
  MANIPULATION:
    - Return dictionary mapping cache types to key patterns:
      - 'phash': 'nc:cache:phash:{phash}'
      - 'site_stats': 'nc:cache:site_stats:{site_id}'
      - 'processing_stats': 'nc:cache:processing_stats:{site_id}'
  OUTPUT: Dictionary of cache key patterns
  ```
- **Assumptions**: None
- **Called By**: cache_manager.py, redis_manager.py

##### get_queue_name()

- **Signature**: `def get_queue_name(self, queue_type: str) -> str`
- **Pseudocode**:
  ```
  INPUT: queue_type (string), self.queue_names
  MANIPULATION:
    - Check if queue_type exists in queue_names
    - Return full queue name
    - Raise ValueError if unknown queue type
  OUTPUT: Full queue name string
  ```
- **Assumptions**: queue_type is a valid key in queue_names
- **Called By**: redis_manager.py, orchestrator.py, crawler_worker.py, extractor_worker.py, storage_worker.py

##### get_cache_key()

- **Signature**: `def get_cache_key(self, cache_type: str, **kwargs) -> str`
- **Pseudocode**:
  ```
  INPUT: cache_type (string), **kwargs (format parameters), self.cache_keys
  MANIPULATION:
    - Check if cache_type exists in cache_keys
    - Format cache key pattern with kwargs
    - Return formatted key
    - Raise ValueError if unknown cache type
  OUTPUT: Formatted cache key string
  ```
- **Assumptions**: cache_type is a valid key in cache_keys, kwargs match pattern placeholders
- **Called By**: cache_manager.py, redis_manager.py

##### validate_environment()

- **Signature**: `def validate_environment(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self (config instance), SKIP_REDIS_VALIDATION env var
  MANIPULATION:
    - If SKIP_REDIS_VALIDATION=1, skip validation and return
    - Create Redis client from redis_url
    - Ping Redis to test connection
    - Validate GPU worker URL format
    - Log configuration summary
  OUTPUT: None (raises exception on failure)
  ```
- **Assumptions**: Redis is accessible, GPU worker URL is valid
- **Called By**: get_config() (automatic on first access)

##### log_configuration()

- **Signature**: `def log_configuration(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self (config instance)
  MANIPULATION:
    - Log all configuration values (without sensitive data)
    - Include: environment, log level, Docker/Windows status, Redis URL, GPU worker URL, batch sizes, timeouts, etc.
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: Logger is configured
- **Called By**: main.py

#### Functions

##### get_config()

- **Signature**: `def get_config() -> CrawlerConfig`
- **Pseudocode**:
  ```
  INPUT: _config_instance (global singleton)
  MANIPULATION:
    - If _config_instance is None:
      - Create new CrawlerConfig instance
      - Call validate_environment() on instance
      - Store in _config_instance
    - Return _config_instance
  OUTPUT: CrawlerConfig singleton instance
  ```
- **Assumptions**: None
- **Called By**: Virtually all files (main entry point for config access)

##### reload_config()

- **Signature**: `def reload_config() -> CrawlerConfig`
- **Pseudocode**:
  ```
  INPUT: _config_instance (global singleton)
  MANIPULATION:
    - Set _config_instance to None
    - Call get_config() to create new instance
    - Return new instance
  OUTPUT: New CrawlerConfig instance
  ```
- **Assumptions**: None
- **Called By**: Test code, configuration reload scenarios

##### validate_configuration()

- **Signature**: `def validate_configuration() -> bool`
- **Pseudocode**:
  ```
  INPUT: config (from get_config())
  MANIPULATION:
    - Get config instance
    - Check required fields for production:
      - S3 credentials (access_key, secret_key)
      - S3 endpoint
    - Check MinIO configuration for development
    - Check GPU worker configuration
    - Log warnings for missing fields
    - Return True if no warnings, False otherwise
  OUTPUT: Boolean indicating validation success
  ```
- **Assumptions**: Config is accessible
- **Called By**: main.py

---

### 4. `cache_manager.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/cache_manager.py`
- **Lines**: 296
- **Scope**: Caching layer for image deduplication
- **Purpose**: Handles Redis caching with perceptual hash (phash) computation for deduplication. Provides efficient image deduplication and caching of processing results.
- **Dependencies**:
  - `hashlib`, `logging`, `time`
  - `pathlib` (Path)
  - `imagehash` (phash computation)
  - `PIL` (Image)
  - `io`
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.data_structures` (CandidateImage, ImageTask, FaceResult)
- **Used By**: extractor_worker.py, gpu_processor_worker.py, storage_worker.py

#### Classes

##### CacheManager

- **Purpose**: Cache manager for image deduplication and result caching

#### Methods

##### **init**()

- **Signature**: `def __init__(self, config=None, redis_manager=None)`
- **Pseudocode**:
  ```
  INPUT: config (optional), redis_manager (optional)
  MANIPULATION:
    - Set self.config to config or get_config()
    - Set self.redis to redis_manager or get_redis_manager()
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: Config and redis_manager are available if not provided
- **Called By**: get_cache_manager() (singleton creation)

##### compute_phash()

- **Signature**: `def compute_phash(self, image_path: str) -> Optional[str]`
- **Pseudocode**:
  ```
  INPUT: image_path (string path to image file)
  MANIPULATION:
    - Open image file with PIL Image.open()
    - Convert to RGB if necessary
    - Compute perceptual hash using imagehash.phash()
    - Convert to string
    - Return phash string or None on error
  OUTPUT: Perceptual hash string or None
  ```
- **Assumptions**: image_path exists and is a valid image file
- **Called By**: extractor_worker.py::\_process_image_candidate()

##### compute_phash_from_bytes()

- **Signature**: `def compute_phash_from_bytes(self, image_bytes: bytes) -> Optional[str]`
- **Pseudocode**:
  ```
  INPUT: image_bytes (bytes of image data)
  MANIPULATION:
    - Open image from BytesIO(image_bytes)
    - Convert to RGB if necessary
    - Compute perceptual hash using imagehash.phash()
    - Convert to string
    - Return phash string or None on error
  OUTPUT: Perceptual hash string or None
  ```
- **Assumptions**: image_bytes is valid image data
- **Called By**: Internal use, gpu_processor_worker.py

##### compute_file_hash()

- **Signature**: `def compute_file_hash(self, file_path: str) -> Optional[str]`
- **Pseudocode**:
  ```
  INPUT: file_path (string path to file)
  MANIPULATION:
    - Open file in binary mode
    - Read in 4096-byte chunks
    - Update SHA256 hash with each chunk
    - Return hexdigest of hash
    - Return None on error
  OUTPUT: SHA256 hash hexdigest string or None
  ```
- **Assumptions**: file_path exists and is readable
- **Called By**: Internal use

##### get_phash_cache_key()

- **Signature**: `def get_phash_cache_key(self, phash: str) -> str`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string)
  MANIPULATION:
    - Call config.get_cache_key('phash', phash=phash)
    - Return formatted cache key
  OUTPUT: Redis cache key string
  ```
- **Assumptions**: phash is a valid string
- **Called By**: Internal methods (is_image_cached, cache_image_info, etc.)

##### is_image_cached()

- **Signature**: `def is_image_cached(self, phash: str) -> bool`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string)
  MANIPULATION:
    - Get cache key using get_phash_cache_key()
    - Call redis.exists_cache() to check if key exists
    - Return True if cached, False otherwise
  OUTPUT: Boolean indicating if image is cached
  ```
- **Assumptions**: phash is valid, Redis is accessible
- **Called By**: extractor_worker.py::\_process_image_candidate()

##### get_cached_image_info()

- **Signature**: `def get_cached_image_info(self, phash: str) -> Optional[Dict[str, Any]]`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string)
  MANIPULATION:
    - Get cache key using get_phash_cache_key()
    - Call redis.get_cache() to retrieve cached data
    - Return cached info dict or None
  OUTPUT: Cached image info dictionary or None
  ```
- **Assumptions**: phash is valid, Redis is accessible
- **Called By**: should_skip_image(), process_image_task()

##### cache_image_info()

- **Signature**: `def cache_image_info(self, phash: str, image_info: Dict[str, Any]) -> bool`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string), image_info (dictionary)
  MANIPULATION:
    - Get cache key using get_phash_cache_key()
    - Call redis.set_cache() to store image_info
    - Return True on success, False on error
  OUTPUT: Boolean indicating success
  ```
- **Assumptions**: phash is valid, image_info is serializable, Redis is accessible
- **Called By**: Internal use

##### cache_face_result()

- **Signature**: `def cache_face_result(self, phash: str, face_result: FaceResult) -> bool`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string), face_result (FaceResult object)
  MANIPULATION:
    - Extract data from face_result:
      - faces_count, raw_image_key, thumbnail_keys
      - processing_time_ms, gpu_used
    - Create cache_data dict with extracted fields + cached_at timestamp
    - Get cache key using get_phash_cache_key()
    - Call redis.set_cache() to store cache_data
    - Return True on success, False on error
  OUTPUT: Boolean indicating success
  ```
- **Assumptions**: phash is valid, face_result is valid, Redis is accessible
- **Called By**: store_processing_result()

##### get_cached_face_result()

- **Signature**: `def get_cached_face_result(self, phash: str) -> Optional[Dict[str, Any]]`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string)
  MANIPULATION:
    - Get cache key using get_phash_cache_key()
    - Call redis.get_cache() to retrieve cached face result
    - Return cached data dict or None
  OUTPUT: Cached face result dictionary or None
  ```
- **Assumptions**: phash is valid, Redis is accessible
- **Called By**: Internal use

##### should_skip_image()

- **Signature**: `def should_skip_image(self, phash: str) -> Tuple[bool, Optional[Dict[str, Any]]]`
- **Pseudocode**:
  ```
  INPUT: phash (perceptual hash string)
  MANIPULATION:
    - Call get_cached_image_info(phash)
    - If cached_info exists:
      - Log debug message
      - Return (True, cached_info)
    - Otherwise return (False, None)
  OUTPUT: Tuple of (should_skip boolean, cached_info dict or None)
  ```
- **Assumptions**: phash is valid
- **Called By**: process_image_task()

##### process_image_task()

- **Signature**: `def process_image_task(self, image_task: ImageTask) -> Tuple[bool, Optional[Dict[str, Any]]]`
- **Pseudocode**:
  ```
  INPUT: image_task (ImageTask object)
  MANIPULATION:
    - Call should_skip_image(image_task.phash)
    - If should_skip is True:
      - Return (True, cached_info)
    - Otherwise return (False, None)
  OUTPUT: Tuple of (is_cached boolean, cached_info dict or None)
  ```
- **Assumptions**: image_task has valid phash
- **Called By**: extractor_worker.py, gpu_processor_worker.py

##### store_processing_result()

- **Signature**: `def store_processing_result(self, image_task: ImageTask, face_result: FaceResult) -> bool`
- **Pseudocode**:
  ```
  INPUT: image_task (ImageTask), face_result (FaceResult)
  MANIPULATION:
    - Call cache_face_result(image_task.phash, face_result)
    - Log debug message on success
    - Return success boolean
  OUTPUT: Boolean indicating success
  ```
- **Assumptions**: image_task and face_result are valid
- **Called By**: storage_worker.py::\_process_storage_task()

##### get_site_stats_cache_key()

- **Signature**: `def get_site_stats_cache_key(self, site_id: str) -> str`
- **Pseudocode**:
  ```
  INPUT: site_id (string)
  MANIPULATION:
    - Call config.get_cache_key('site_stats', site_id=site_id)
    - Return formatted cache key
  OUTPUT: Redis cache key string
  ```
- **Assumptions**: site_id is valid
- **Called By**: Internal methods

##### cache_site_stats()

- **Signature**: `def cache_site_stats(self, site_id: str, stats: Dict[str, Any]) -> bool`
- **Pseudocode**:
  ```
  INPUT: site_id (string), stats (dictionary)
  MANIPULATION:
    - Get cache key using get_site_stats_cache_key()
    - Call redis.set_cache() to store stats
    - Return True on success, False on error
  OUTPUT: Boolean indicating success
  ```
- **Assumptions**: site_id is valid, stats is serializable
- **Called By**: update_site_stats()

##### get_cached_site_stats()

- **Signature**: `def get_cached_site_stats(self, site_id: str) -> Optional[Dict[str, Any]]`
- **Pseudocode**:
  ```
  INPUT: site_id (string)
  MANIPULATION:
    - Get cache key using get_site_stats_cache_key()
    - Call redis.get_cache() to retrieve cached stats
    - Return cached stats dict or None
  OUTPUT: Cached site stats dictionary or None
  ```
- **Assumptions**: site_id is valid
- **Called By**: update_site_stats()

##### update_site_stats()

- **Signature**: `def update_site_stats(self, site_id: str, updates: Dict[str, Any]) -> bool`
- **Pseudocode**:
  ```
  INPUT: site_id (string), updates (dictionary)
  MANIPULATION:
    - Get existing stats using get_cached_site_stats()
    - Merge updates into existing_stats (or start with empty dict)
    - Add last_updated timestamp
    - Call cache_site_stats() to store updated stats
    - Return True on success, False on error
  OUTPUT: Boolean indicating success
  ```
- **Assumptions**: site_id is valid, updates is a dictionary
- **Called By**: Internal use

##### get_cache_stats()

- **Signature**: `def get_cache_stats(self) -> Dict[str, Any]`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Get Redis client
    - Use client.keys() to find all phash cache keys
    - Use client.keys() to find all site_stats cache keys
    - Count keys
    - Return dict with:
      - phash_cache_count
      - site_stats_cache_count
      - total_cache_keys
      - cache_ttl_days
      - timestamp
  OUTPUT: Dictionary with cache statistics
  ```
- **Assumptions**: Redis is accessible
- **Called By**: health_check(), orchestrator.py

##### clear_cache()

- **Signature**: `def clear_cache(self, pattern: str = None) -> bool`
- **Pseudocode**:
  ```
  INPUT: pattern (optional string pattern)
  MANIPULATION:
    - Get Redis client
    - If pattern provided:
      - Use client.keys(pattern) to find matching keys
      - Delete matching keys
    - Otherwise:
      - Find all phash and site_stats cache keys
      - Delete all found keys
    - Log result
    - Return True
  OUTPUT: Boolean indicating success
  ```
- **Assumptions**: Redis is accessible
- **Called By**: Test code, cleanup operations

##### cleanup_expired_cache()

- **Signature**: `def cleanup_expired_cache(self) -> int`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Call get_cache_stats() to get current cache size
    - Log cleanup completion with current cache size
    - Return total_cache_keys count
    - Note: Redis handles TTL automatically, this is mainly for logging
  OUTPUT: Integer count of cache keys
  ```
- **Assumptions**: Redis is accessible
- **Called By**: Cleanup operations

##### health_check()

- **Signature**: `def health_check(self) -> Dict[str, Any]`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Create test phash and test data
    - Test set operation: set_cache() with TTL
    - Test get operation: get_cache()
    - Test delete operation: delete_cache()
    - Get cache stats
    - Return dict with:
      - status: 'healthy' if all tests pass, 'unhealthy' otherwise
      - set_test, get_test, delete_test: boolean results
      - cache_stats: statistics
      - timestamp
  OUTPUT: Dictionary with health check results
  ```
- **Assumptions**: Redis is accessible
- **Called By**: main.py::health_check()

#### Functions

##### get_cache_manager()

- **Signature**: `def get_cache_manager() -> CacheManager`
- **Pseudocode**:
  ```
  INPUT: _cache_manager (global singleton)
  MANIPULATION:
    - If _cache_manager is None:
      - Create new CacheManager instance
      - Store in _cache_manager
    - Return _cache_manager
  OUTPUT: CacheManager singleton instance
  ```
- **Assumptions**: None
- **Called By**: extractor_worker.py, gpu_processor_worker.py, storage_worker.py

##### close_cache_manager()

- **Signature**: `def close_cache_manager() -> None`
- **Pseudocode**:
  ```
  INPUT: _cache_manager (global singleton)
  MANIPULATION:
    - Set _cache_manager to None
  OUTPUT: None
  ```
- **Assumptions**: None
- **Called By**: Cleanup operations, test code

---

## Documentation Status

**Completed Files (10/20):**

1. ✅ `__init__.py` - Complete
2. ✅ `data_structures.py` - Complete (all classes and models documented)
3. ✅ `config.py` - Complete (all methods and validators documented)
4. ✅ `cache_manager.py` - Complete (all methods documented)
5. ✅ `main.py` - Complete (all methods documented)
6. ✅ `timing_logger.py` - Complete (all methods documented)
7. ✅ `extraction_tracer.py` - Complete (all methods documented)
8. ✅ `gpu_worker_logger.py` - Complete (all methods documented)
9. ✅ `gpu_scheduler.py` - Complete (all methods documented)
10. ✅ `crawler_worker.py` - Complete (all methods documented)

**Remaining Files (16/20):**

- Tier 2: `timing_logger.py`, `extraction_tracer.py`, `gpu_worker_logger.py`
- Tier 3: `http_utils.py`, `redis_manager.py`, `storage_manager.py`, `gpu_interface.py`, `gpu_scheduler.py`
- Tier 4: `crawler_worker.py`, `extractor_worker.py`, `gpu_processor_worker.py`, `storage_worker.py`
- Tier 5: `selector_miner.py`, `orchestrator.py`, `main.py`, `test_suite.py`

**Total Methods to Document:** ~429 methods across 20 files

**Note:** This is a comprehensive documentation effort. The structure above demonstrates the format for all files. Each remaining file should follow the same pattern:

- File metadata (path, lines, scope, purpose, dependencies, used by)
- Classes with descriptions
- Methods with:
  - Signature
  - Pseudocode (inputs, manipulations, outputs)
  - Assumptions
  - Callers (what files/methods call this)

**Key Call Graphs Identified:**

- `cache_manager.compute_phash()` → Called by: `extractor_worker.py::_process_image_candidate()`
- `cache_manager.is_image_cached()` → Called by: `extractor_worker.py::_process_image_candidate()`, `gpu_processor_worker.py`
- `cache_manager.store_processing_result()` → Called by: `storage_worker.py::_process_storage_task()`
- `timing_logger.log_site_start()` → Called by: `crawler_worker.py::process_site()`
- `timing_logger.log_page_end()` → Called by: `crawler_worker.py::process_site()`
- `timing_logger.log_extraction_start()` → Called by: `extractor_worker.py::_process_image_candidate()`, `extractor_worker.py::_process_post_candidate()`
- `selector_miner.mine_posts_with_3x3_crawl()` → Called by: `crawler_worker.py::process_site()`
- `selector_miner.mine_posts_for_diabetes()` → Called by: `selector_miner.py::_fetch_post_page()`, `selector_miner.py::mine_posts_with_3x3_crawl()`
- `selector_miner.mine_selectors()` → Called by: `selector_miner.py::mine_with_3x3_crawl()`, `test_suite.py`

---

## Next Steps for Completion

To complete this documentation:

1. **Continue with Tier 2 files** - Document `timing_logger.py`, `extraction_tracer.py`, `gpu_worker_logger.py` following the same pattern
2. **Document Tier 3 infrastructure files** - These are large files (1000+ lines each) with many methods:
   - `http_utils.py` (1458 lines, ~36 methods)
   - `redis_manager.py` (1636 lines, ~94 methods)
   - `storage_manager.py` (977 lines, ~25 methods)
   - `gpu_interface.py` (1004 lines, ~26 methods)
   - `gpu_scheduler.py` (186 lines, ~7 methods)
3. **Document Tier 4 worker files** - Core worker implementations:
   - `crawler_worker.py` (307 lines, ~8 methods)
   - `extractor_worker.py` (634 lines, ~11 methods)
   - `gpu_processor_worker.py` (1263 lines, ~17 methods)
   - `storage_worker.py` (419 lines, ~9 methods)
4. **Document Tier 5 orchestration files** - System coordination:
   - `selector_miner.py` (3117 lines, ~45 methods) - **LARGEST FILE**
   - `orchestrator.py` (894 lines, ~21 methods)
   - `main.py` (286 lines, ~7 methods)
   - `test_suite.py` (536 lines, ~14 methods)

**Validation Checklist:**

- [ ] All 20 files documented
- [ ] All classes documented
- [ ] All methods documented with pseudocode
- [ ] All callers identified (or marked as unknown)
- [ ] All assumptions documented
- [ ] Cross-references verified
- [ ] Formatting consistent

---

## File Summary Statistics

| File                      | Lines      | Classes | Methods | Status             |
| ------------------------- | ---------- | ------- | ------- | ------------------ |
| `__init__.py`             | 0          | 0       | 0       | ✅ Complete        |
| `data_structures.py`      | 349        | 15      | 0       | ✅ Complete        |
| `config.py`               | 438        | 1       | 20      | ✅ Complete        |
| `cache_manager.py`        | 296        | 1       | 23      | ✅ Complete        |
| `timing_logger.py`        | 135        | 1       | 24      | ⏳ Pending         |
| `extraction_tracer.py`    | 103        | 2       | 4       | ⏳ Pending         |
| `gpu_worker_logger.py`    | 104        | 1       | 23      | ⏳ Pending         |
| `http_utils.py`           | 1458       | 3       | 36      | ⏳ Pending         |
| `redis_manager.py`        | 1636       | 1       | 94      | ⏳ Pending         |
| `storage_manager.py`      | 977        | 1       | 25      | ⏳ Pending         |
| `gpu_interface.py`        | 1004       | 1       | 26      | ⏳ Pending         |
| `gpu_scheduler.py`        | 186        | 1       | 7       | ⏳ Pending         |
| `crawler_worker.py`       | 307        | 1       | 8       | ⏳ Pending         |
| `extractor_worker.py`     | 634        | 1       | 11      | ⏳ Pending         |
| `gpu_processor_worker.py` | 1263       | 1       | 17      | ⏳ Pending         |
| `storage_worker.py`       | 419        | 1       | 9       | ⏳ Pending         |
| `selector_miner.py`       | 3117       | 1       | 45      | ⏳ Pending         |
| `orchestrator.py`         | 894        | 1       | 21      | ⏳ Pending         |
| `main.py`                 | 286        | 0       | 7       | ⏳ Pending         |
| `test_suite.py`           | 536        | 1       | 14      | ⏳ Pending         |
| **TOTAL**                 | **14,643** | **33**  | **429** | **10/20 Complete** |

---

### 19. `main.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/main.py`
- **Lines**: 307
- **Scope**: Entry point and CLI interface
- **Purpose**: Provides command-line interface and main entry point for the new crawler system. Handles argument parsing, logging setup, health checks, and orchestrates the crawl process.
- **Dependencies**:
  - `argparse`, `asyncio`, `logging`, `sys`
  - `pathlib` (Path)
  - `typing` (List)
  - `.config` (get_config, validate_configuration)
  - `.orchestrator` (Orchestrator)
  - `.redis_manager` (get_redis_manager)
  - `.cache_manager` (get_cache_manager)
  - `.storage_manager` (get_storage_manager)
  - `.gpu_interface` (get_gpu_interface)
- **Used By**: Python entry point (`python -m diabetes_crawler.main`)

#### Methods

##### setup_logging()

- **Signature**: `def setup_logging(log_level: str = "info") -> None`
- **Pseudocode**:
  ```
  INPUT: log_level (string, default "info")
  MANIPULATION:
    - Convert log_level to uppercase and get logging level constant
    - Configure logging.basicConfig with:
      - Level from log_level
      - Format string with timestamp, name, level, message
      - Handlers: StreamHandler (stdout) and FileHandler ('diabetes_crawler.log')
  OUTPUT: None (configures logging)
  ```
- **Assumptions**: log_level is valid ('debug', 'info', 'warning', 'error', 'critical')
- **Called By**: main()

##### load_sites_from_file()

- **Signature**: `def load_sites_from_file(file_path: str) -> List[str]`
- **Pseudocode**:
  ```
  INPUT: file_path (string path to file)
  MANIPULATION:
    - Open file in read mode with UTF-8 encoding
    - Read all lines
    - Filter out empty lines and lines starting with '#'
    - Strip whitespace from each line
    - Log number of sites loaded
    - Return list of site URLs
    - Return empty list on error
  OUTPUT: List of site URL strings
  ```
- **Assumptions**: file_path exists and is readable, contains one URL per line
- **Called By**: main()

##### load_sites_from_args()

- **Signature**: `def load_sites_from_args(sites_arg: List[str]) -> List[str]`
- **Pseudocode**:
  ```
  INPUT: sites_arg (list of strings from command line)
  MANIPULATION:
    - For each site in sites_arg:
      - Check if site starts with 'http://' or 'https://'
      - If valid, add to sites list
      - If invalid, log warning and skip
    - Return list of valid site URLs
  OUTPUT: List of valid site URL strings
  ```
- **Assumptions**: sites_arg is a list of strings
- **Called By**: main()

##### health_check()

- **Signature**: `async def health_check() -> bool`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Get config instance
    - Test Redis connection via redis_manager.test_connection()
    - Test cache manager via cache_manager.health_check()
    - Test storage via storage_manager.health_check()
    - Test GPU worker via gpu_interface._check_health()
    - Log health status for each component
    - Return True if all healthy, False otherwise
  OUTPUT: Boolean indicating overall system health
  ```
- **Assumptions**: All managers are initialized
- **Called By**: main(), run_crawl()

##### run_crawl()

- **Signature**: `async def run_crawl(sites: List[str], config_file: str = None) -> CrawlResults`
- **Pseudocode**:
  ```
  INPUT: sites (list of site URLs), config_file (optional string)
  MANIPULATION:
    - Log crawl start
    - Validate configuration via validate_configuration()
    - Perform health check via health_check()
    - Create Orchestrator instance
    - Call orchestrator.crawl_sites(sites) to run crawl
    - Print results via print_results()
    - Return CrawlResults
    - On exception: log error and raise
    - Finally: call orchestrator.stop()
  OUTPUT: CrawlResults object
  ```
- **Assumptions**: sites is non-empty list of valid URLs, system is healthy
- **Called By**: main()

##### print_results()

- **Signature**: `def print_results(results: CrawlResults) -> None`
- **Pseudocode**:
  ```
  INPUT: results (CrawlResults object)
  MANIPULATION:
    - Print summary section:
      - Total time, success rate, sites processed
      - Total images found, processed, faces detected
      - Raw images saved, thumbnails saved
      - Overall throughput (images/second)
    - Print per-site results:
      - For each site in results.sites:
        - Pages crawled, images found, processed, cached
        - Faces detected, raw/thumbs saved
        - Success rate, processing time, throughput
        - Errors (first 3)
    - Print system metrics:
      - Active workers (crawlers, extractors, GPU processors)
      - GPU worker availability
      - Queue metrics (depth, utilization)
  OUTPUT: None (prints to stdout)
  ```
- **Assumptions**: results is a valid CrawlResults object
- **Called By**: run_crawl()

##### main()

- **Signature**: `def main() -> None`
- **Pseudocode**:
  ```
  INPUT: Command line arguments (via argparse)
  MANIPULATION:
    - Parse command line arguments:
      - --sites-file or --sites (mutually exclusive)
      - --config-file, --log-level
      - --health-check, --validate-config
      - --num-crawlers, --num-extractors, --num-gpu-processors, --batch-size
      - --max-pages-per-site, --max-images-per-site
    - Setup logging with log_level
    - Load configuration via get_config()
    - Override config with command line arguments if provided
    - Log configuration
    - Handle operations:
      - If --health-check: run health_check() and exit
      - If --validate-config: run validate_configuration() and exit
      - Otherwise:
        - Load sites from file or args
        - Validate sites list is non-empty
        - Create asyncio event loop
        - Run run_crawl(sites) in event loop
        - Handle KeyboardInterrupt and exceptions
  OUTPUT: None (exits with code 0 on success, 1 on error)
  ```
- **Assumptions**: Command line arguments are valid
- **Called By**: Python entry point (`if __name__ == "__main__"`)

---

### 5. `timing_logger.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/timing_logger.py`
- **Lines**: 142
- **Scope**: Performance timing instrumentation
- **Purpose**: Thread-safe singleton logger for detailed performance timing instrumentation. Writes structured timing data to timings.txt for performance analysis (currently disabled to reduce memory usage).
- **Dependencies**:
  - `threading`, `time`
  - `datetime`
  - `typing` (Optional)
  - `os`
- **Used By**: crawler_worker.py, extractor_worker.py, orchestrator.py, gpu_processor_worker.py, storage_worker.py

#### Classes

##### TimingLogger

- **Purpose**: Thread-safe singleton timing logger
- **Pattern**: Singleton with double-checked locking

#### Methods

##### **new**()

- **Signature**: `def __new__(cls) -> TimingLogger`
- **Pseudocode**:
  ```
  INPUT: cls (class)
  MANIPULATION:
    - If _instance is None:
      - Acquire lock
      - Double-check _instance is still None
      - Create new instance via super().__new__()
      - Set _initialized = False
      - Store in _instance
    - Return _instance
  OUTPUT: TimingLogger singleton instance
  ```
- **Assumptions**: Thread-safe singleton pattern
- **Called By**: get_timing_logger() (automatic on first access)

##### **init**()

- **Signature**: `def __init__(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self (TimingLogger instance)
  MANIPULATION:
    - If already initialized, return early
    - Create threading.Lock for file operations
    - Set _timing_file to "timings04.txt"
    - Set _initialized = True
    - Note: File writing is currently DISABLED to reduce memory usage
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: Called after **new**()
- **Called By**: get_timing_logger() (automatic on first access)

##### \_log()

- **Signature**: `def _log(self, event_type: str, *args) -> None`
- **Pseudocode**:
  ```
  INPUT: event_type (string), *args (variable arguments)
  MANIPULATION:
    - Currently DISABLED (no-op) to reduce memory usage
    - When enabled: would format timestamp, join args, write to file with lock
  OUTPUT: None (no-op currently)
  ```
- **Assumptions**: event_type is a string
- **Called By**: All log\_\* methods (internal)

##### log_system_start()

- **Signature**: `def log_system_start(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Call _log("SYSTEM_START")
  OUTPUT: None
  ```
- **Assumptions**: None
- **Called By**: orchestrator.py::**init**()

##### log_crawl_start()

- **Signature**: `def log_crawl_start(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Call _log("CRAWL_START")
  OUTPUT: None
  ```
- **Assumptions**: None
- **Called By**: orchestrator.py::crawl_sites()

##### log_site_start()

- **Signature**: `def log_site_start(self, site_id: str, url: str) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), url (string)
  MANIPULATION:
    - Call _log("SITE_START", site_id, url)
  OUTPUT: None
  ```
- **Assumptions**: site_id and url are valid strings
- **Called By**: crawler_worker.py::process_site()

##### log_page_start()

- **Signature**: `def log_page_start(self, site_id: str, page_url: str) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), page_url (string)
  MANIPULATION:
    - Call _log("PAGE_START", site_id, page_url)
  OUTPUT: None
  ```
- **Assumptions**: site_id and page_url are valid strings
- **Called By**: crawler_worker.py::process_site()

##### log_page_end()

- **Signature**: `def log_page_end(self, site_id: str, page_url: str, duration_ms: float, candidate_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), page_url (string), duration_ms (float), candidate_count (int)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Format candidate_count as "{candidate_count} candidates"
    - Call _log("PAGE_END", site_id, page_url, formatted_duration, formatted_count)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: crawler_worker.py::process_site()

##### log_site_end()

- **Signature**: `def log_site_end(self, site_id: str, duration_ms: float, total_pages: int, total_candidates: int) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), duration_ms (float), total_pages (int), total_candidates (int)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Format total_pages as "{total_pages} pages"
    - Format total_candidates as "{total_candidates} candidates"
    - Call _log("SITE_END", site_id, formatted_duration, formatted_pages, formatted_candidates)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: crawler_worker.py::process_site()

##### log_extraction_start()

- **Signature**: `def log_extraction_start(self, site_id: str, image_url: str) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), image_url (string)
  MANIPULATION:
    - Call _log("EXTRACT_START", site_id, image_url)
  OUTPUT: None
  ```
- **Assumptions**: site_id and image_url are valid strings
- **Called By**: extractor_worker.py::\_process_image_candidate(), extractor_worker.py::\_process_post_candidate()

##### log_extraction_end()

- **Signature**: `def log_extraction_end(self, site_id: str, image_url: str, duration_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), image_url (string), duration_ms (float)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Call _log("EXTRACT_END", site_id, image_url, formatted_duration)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: extractor_worker.py::\_process_image_candidate(), extractor_worker.py::\_process_post_candidate()

##### log_gpu_batch_start()

- **Signature**: `def log_gpu_batch_start(self, batch_id: str, image_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), image_count (int)
  MANIPULATION:
    - Format image_count as "{image_count} images"
    - Call _log("GPU_BATCH_START", batch_id, formatted_count)
  OUTPUT: None
  ```
- **Assumptions**: batch_id is valid string, image_count is non-negative
- **Called By**: gpu_processor_worker.py::process_batch()

##### log_gpu_recognition_start()

- **Signature**: `def log_gpu_recognition_start(self, batch_id: str) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string)
  MANIPULATION:
    - Call _log("GPU_RECOGNITION_START", batch_id)
  OUTPUT: None
  ```
- **Assumptions**: batch_id is valid string
- **Called By**: gpu_interface.py, gpu_processor_worker.py

##### log_gpu_recognition_end()

- **Signature**: `def log_gpu_recognition_end(self, batch_id: str, duration_ms: float, face_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), duration_ms (float), face_count (int)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Format face_count as "{face_count} faces"
    - Call _log("GPU_RECOGNITION_END", batch_id, formatted_duration, formatted_count)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_interface.py, gpu_processor_worker.py

##### log_gpu_crop_start()

- **Signature**: `def log_gpu_crop_start(self, image_id: str, face_index: int) -> None`
- **Pseudocode**:
  ```
  INPUT: image_id (string), face_index (int)
  MANIPULATION:
    - Format face_index as "face_{face_index}"
    - Call _log("GPU_CROP_START", image_id, formatted_index)
  OUTPUT: None
  ```
- **Assumptions**: image_id is valid string, face_index is non-negative
- **Called By**: gpu_processor_worker.py

##### log_gpu_crop_end()

- **Signature**: `def log_gpu_crop_end(self, image_id: str, face_index: int, duration_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: image_id (string), face_index (int), duration_ms (float)
  MANIPULATION:
    - Format face_index as "face_{face_index}"
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Call _log("GPU_CROP_END", image_id, formatted_index, formatted_duration)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_processor_worker.py

##### log_gpu_storage_start()

- **Signature**: `def log_gpu_storage_start(self, image_id: str, storage_type: str) -> None`
- **Pseudocode**:
  ```
  INPUT: image_id (string), storage_type (string, e.g., "raw" or "thumb")
  MANIPULATION:
    - Call _log("GPU_STORAGE_START", image_id, storage_type)
  OUTPUT: None
  ```
- **Assumptions**: image_id and storage_type are valid strings
- **Called By**: gpu_processor_worker.py

##### log_gpu_storage_end()

- **Signature**: `def log_gpu_storage_end(self, image_id: str, storage_type: str, duration_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: image_id (string), storage_type (string), duration_ms (float)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Call _log("GPU_STORAGE_END", image_id, storage_type, formatted_duration)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_processor_worker.py

##### log_gpu_batch_end()

- **Signature**: `def log_gpu_batch_end(self, batch_id: str, duration_ms: float, images_processed: int) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), duration_ms (float), images_processed (int)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Format images_processed as "{images_processed} images"
    - Call _log("GPU_BATCH_END", batch_id, formatted_duration, formatted_count)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_processor_worker.py::process_batch()

##### log_storage_start()

- **Signature**: `def log_storage_start(self, site_id: str, image_id: str) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), image_id (string)
  MANIPULATION:
    - Call _log("STORAGE_START", site_id, image_id)
  OUTPUT: None
  ```
- **Assumptions**: site_id and image_id are valid strings
- **Called By**: storage_worker.py::\_process_storage_task()

##### log_storage_end()

- **Signature**: `def log_storage_end(self, site_id: str, image_id: str, duration_ms: float, faces_count: int, raw_saved: bool, thumbs_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (string), image_id (string), duration_ms (float), faces_count (int), raw_saved (bool), thumbs_count (int)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Format faces_count as "{faces_count} faces"
    - Format raw_saved as "raw={raw_saved}"
    - Format thumbs_count as "{thumbs_count} thumbs"
    - Call _log("STORAGE_END", site_id, image_id, formatted_duration, formatted_faces, formatted_raw, formatted_thumbs)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: storage_worker.py::\_process_storage_task()

##### log_crawl_end()

- **Signature**: `def log_crawl_end(self, duration_ms: float, total_sites: int, total_images: int) -> None`
- **Pseudocode**:
  ```
  INPUT: duration_ms (float), total_sites (int), total_images (int)
  MANIPULATION:
    - Format duration_ms as "{duration_ms:.0f}ms"
    - Format total_sites as "{total_sites} sites"
    - Format total_images as "{total_images} images"
    - Call _log("CRAWL_END", formatted_duration, formatted_sites, formatted_images)
  OUTPUT: None
  ```
- **Assumptions**: All parameters are valid
- **Called By**: orchestrator.py::crawl_sites()

##### log_system_shutdown()

- **Signature**: `def log_system_shutdown(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Call _log("SYSTEM_SHUTDOWN")
  OUTPUT: None
  ```
- **Assumptions**: None
- **Called By**: orchestrator.py::stop()

#### Functions

##### get_timing_logger()

- **Signature**: `def get_timing_logger() -> TimingLogger`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Call TimingLogger() which returns singleton via __new__()
    - Return singleton instance
  OUTPUT: TimingLogger singleton instance
  ```
- **Assumptions**: None
- **Called By**: crawler_worker.py, extractor_worker.py, orchestrator.py, gpu_processor_worker.py, storage_worker.py

---

### 6. `extraction_tracer.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/extraction_tracer.py`
- **Lines**: 115
- **Scope**: Extraction attempt tracking
- **Purpose**: Tracks every extraction attempt with structured logging to enable debugging and analysis of extraction failures and strategy effectiveness. Currently file writing is disabled to reduce memory usage.
- **Dependencies**:
  - `json`, `logging`, `time`
  - `datetime`
  - `pathlib` (Path)
  - `typing` (Dict, Any, Optional, List)
  - `dataclasses` (dataclass, asdict)
- **Used By**: selector_miner.py

#### Classes

##### ExtractionAttempt (dataclass)

- **Purpose**: Structured data for a single extraction attempt
- **Fields**: url, page_type, strategy_used, success, failure_reason, content_length, title_found, author_found, date_found, timestamp, html_sample, strategy_order, strategy_results

##### ExtractionTracer

- **Purpose**: Tracks extraction attempts and writes structured logs

#### Methods

##### **init**()

- **Signature**: `def __init__(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Initialize self.attempts as empty list
    - Create Path("crawl_output/debug")
    - Create directory if it doesn't exist (mkdir with parents=True, exist_ok=True)
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: File system is writable
- **Called By**: get_extraction_tracer() (singleton creation)

##### log_attempt()

- **Signature**: `def log_attempt(self, url: str, page_type: str, strategy_used: Optional[str] = None, success: bool = False, failure_reason: Optional[str] = None, content_length: int = 0, title_found: bool = False, author_found: bool = False, date_found: bool = False, html_sample: Optional[str] = None, strategy_order: Optional[List[str]] = None, strategy_results: Optional[Dict[str, Any]] = None) -> None`
- **Pseudocode**:
  ```
  INPUT: url, page_type, strategy_used, success, failure_reason, content_length, title_found, author_found, date_found, html_sample, strategy_order, strategy_results
  MANIPULATION:
    - Limit html_sample to first 10000 characters if provided
    - Use empty list for strategy_order if None
    - Use empty dict for strategy_results if None
    - Get current timestamp via time.time()
    - Create ExtractionAttempt dataclass instance with all parameters
    - Append attempt to self.attempts list
  OUTPUT: None (stores attempt in memory)
  ```
- **Assumptions**: url and page_type are valid strings
- **Called By**: selector_miner.py (various extraction methods)

##### flush()

- **Signature**: `def flush(self, site_id: Optional[str] = None) -> None`
- **Pseudocode**:
  ```
  INPUT: site_id (optional string)
  MANIPULATION:
    - Currently DISABLED (no-op) to reduce memory usage
    - When enabled: would write all attempts to JSON file with timestamp, site_id, statistics
    - Clear self.attempts list
  OUTPUT: None (clears attempts list)
  ```
- **Assumptions**: None
- **Called By**: selector_miner.py (after site processing)

#### Functions

##### get_extraction_tracer()

- **Signature**: `def get_extraction_tracer() -> ExtractionTracer`
- **Pseudocode**:
  ```
  INPUT: _tracer_instance (global singleton)
  MANIPULATION:
    - If _tracer_instance is None:
      - Create new ExtractionTracer instance
      - Store in _tracer_instance
    - Return _tracer_instance
  OUTPUT: ExtractionTracer singleton instance
  ```
- **Assumptions**: None
- **Called By**: selector_miner.py

---

### 7. `gpu_worker_logger.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/gpu_worker_logger.py`
- **Lines**: 118
- **Scope**: GPU operation logging
- **Purpose**: Dedicated logger for GPU operations with consistent formatting and comprehensive tracking. Provides structured logging for all GPU worker operations.
- **Dependencies**:
  - `logging`
  - `typing` (Optional)
- **Used By**: gpu_interface.py

#### Classes

##### GPUWorkerLogger

- **Purpose**: Dedicated logger for GPU worker operations

#### Methods

##### **init**()

- **Signature**: `def __init__(self, worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Get logger instance with name f"gpu_worker_{worker_id}"
    - Store logger in self.logger
    - Store worker_id in self.worker_id
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: worker_id is a valid integer
- **Called By**: gpu_interface.py::**init**()

##### log_batch_start()

- **Signature**: `def log_batch_start(self, batch_id: str, image_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), image_count (integer)
  MANIPULATION:
    - Log INFO message: "[GPU-WORKER-{worker_id}] BATCH-START: {batch_id}, images={image_count}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: batch_id is valid string, image_count is non-negative
- **Called By**: gpu_interface.py

##### log_health_check()

- **Signature**: `def log_health_check(self, available: bool, error: Optional[str] = None) -> None`
- **Pseudocode**:
  ```
  INPUT: available (boolean), error (optional string)
  MANIPULATION:
    - If available is True:
      - Log INFO: "[GPU-WORKER-{worker_id}] HEALTH-CHECK: OK"
    - Else:
      - Log WARNING: "[GPU-WORKER-{worker_id}] HEALTH-CHECK: FAILED - {error}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: available is boolean, error is string if provided
- **Called By**: gpu_interface.py::\_check_health()

##### log_request_start()

- **Signature**: `def log_request_start(self, image_count: int, url: str) -> None`
- **Pseudocode**:
  ```
  INPUT: image_count (integer), url (string)
  MANIPULATION:
    - Log DEBUG: "[GPU-WORKER-{worker_id}] REQUEST-START: {image_count} images to {url}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: image_count is non-negative, url is valid string
- **Called By**: gpu_interface.py::\_gpu_worker_request()

##### log_request_complete()

- **Signature**: `def log_request_complete(self, image_count: int, faces_found: int, time_ms: float, gpu_used: bool) -> None`
- **Pseudocode**:
  ```
  INPUT: image_count (integer), faces_found (integer), time_ms (float), gpu_used (boolean)
  MANIPULATION:
    - Determine mode: "GPU" if gpu_used else "CPU"
    - Log INFO: "[GPU-WORKER-{worker_id}] REQUEST-COMPLETE: {image_count} images, {faces_found} faces, {time_ms:.1f}ms, mode={mode}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_interface.py::\_gpu_worker_request()

##### log_request_failed()

- **Signature**: `def log_request_failed(self, error: str) -> None`
- **Pseudocode**:
  ```
  INPUT: error (string)
  MANIPULATION:
    - Log ERROR: "[GPU-WORKER-{worker_id}] REQUEST-FAILED: {error}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: error is valid string
- **Called By**: gpu_interface.py::\_gpu_worker_request()

##### log_fallback_start()

- **Signature**: `def log_fallback_start(self, image_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: image_count (integer)
  MANIPULATION:
    - Log WARNING: "[GPU-WORKER-{worker_id}] FALLBACK-START: CPU processing {image_count} images"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: image_count is non-negative
- **Called By**: gpu_interface.py (CPU fallback path)

##### log_fallback_complete()

- **Signature**: `def log_fallback_complete(self, image_count: int, faces_found: int, time_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: image_count (integer), faces_found (integer), time_ms (float)
  MANIPULATION:
    - Log INFO: "[GPU-WORKER-{worker_id}] FALLBACK-COMPLETE: {image_count} images, {faces_found} faces, {time_ms:.1f}ms"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_interface.py (CPU fallback path)

##### log_batch_result_ok()

- **Signature**: `def log_batch_result_ok(self, batch_id: str, images: int, faces: int, duration_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), images (integer), faces (integer), duration_ms (float)
  MANIPULATION:
    - Log INFO: "[GPU-WORKER-{worker_id}] BATCH-RESULT-OK: batch_id={batch_id}, images={images}, faces={faces}, duration_ms={duration_ms:.1f}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_interface.py::process_batch()

##### log_batch_result_error()

- **Signature**: `def log_batch_result_error(self, batch_id: str, err_type: str, fallback: str, duration_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), err_type (string), fallback (string), duration_ms (float)
  MANIPULATION:
    - Log ERROR: "[GPU-WORKER-{worker_id}] BATCH-RESULT-ERROR: batch_id={batch_id}, err_type={err_type}, fallback={fallback}, duration_ms={duration_ms:.1f}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: All parameters are valid strings/numbers
- **Called By**: gpu_interface.py::process_batch()

##### log_fallback_done()

- **Signature**: `def log_fallback_done(self, batch_id: str, images: int, faces: int, duration_ms: float) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), images (integer), faces (integer), duration_ms (float)
  MANIPULATION:
    - Log INFO: "[GPU-WORKER-{worker_id}] FALLBACK-DONE: batch_id={batch_id}, images={images}, faces={faces}, duration_ms={duration_ms:.1f}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: All parameters are valid
- **Called By**: gpu_interface.py (CPU fallback completion)

##### log_batch_encoding_start()

- **Signature**: `def log_batch_encoding_start(self, image_count: int) -> None`
- **Pseudocode**:
  ```
  INPUT: image_count (integer)
  MANIPULATION:
    - Log DEBUG: "[GPU-WORKER-{worker_id}] BATCH-ENCODING-START: encoding {image_count} images"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: image_count is non-negative
- **Called By**: gpu_interface.py (encoding operations)

##### log_batch_encoding_complete()

- **Signature**: `def log_batch_encoding_complete(self, image_count: int, encoded_size: int) -> None`
- **Pseudocode**:
  ```
  INPUT: image_count (integer), encoded_size (integer)
  MANIPULATION:
    - Log DEBUG: "[GPU-WORKER-{worker_id}] BATCH-ENCODING-COMPLETE: {image_count} images, {encoded_size} bytes"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: Both parameters are non-negative integers
- **Called By**: gpu_interface.py (encoding operations)

##### log_result_decoding_start()

- **Signature**: `def log_result_decoding_start(self, response_size: int) -> None`
- **Pseudocode**:
  ```
  INPUT: response_size (integer)
  MANIPULATION:
    - Log DEBUG: "[GPU-WORKER-{worker_id}] RESULT-DECODING-START: {response_size} bytes"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: response_size is non-negative
- **Called By**: gpu_interface.py::\_gpu_worker_request()

##### log_result_decoding_complete()

- **Signature**: `def log_result_decoding_complete(self, faces_found: int) -> None`
- **Pseudocode**:
  ```
  INPUT: faces_found (integer)
  MANIPULATION:
    - Log DEBUG: "[GPU-WORKER-{worker_id}] RESULT-DECODING-COMPLETE: {faces_found} faces"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: faces_found is non-negative
- **Called By**: gpu_interface.py::\_gpu_worker_request()

##### log_circuit_breaker_open()

- **Signature**: `def log_circuit_breaker_open(self, reason: str) -> None`
- **Pseudocode**:
  ```
  INPUT: reason (string)
  MANIPULATION:
    - Log WARNING: "[GPU-WORKER-{worker_id}] CIRCUIT-BREAKER-OPEN: {reason}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: reason is valid string
- **Called By**: gpu_interface.py::\_check_health()

##### log_circuit_breaker_close()

- **Signature**: `def log_circuit_breaker_close(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Log INFO: "[GPU-WORKER-{worker_id}] CIRCUIT-BREAKER-CLOSE: attempting GPU again"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: None
- **Called By**: gpu_interface.py::\_check_health()

##### log_retry_attempt()

- **Signature**: `def log_retry_attempt(self, attempt: int, max_retries: int, error: str) -> None`
- **Pseudocode**:
  ```
  INPUT: attempt (integer), max_retries (integer), error (string)
  MANIPULATION:
    - Log WARNING: "[GPU-WORKER-{worker_id}] RETRY-ATTEMPT: {attempt}/{max_retries}, error: {error}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: attempt <= max_retries, error is valid string
- **Called By**: gpu_interface.py (retry logic)

##### log_timeout()

- **Signature**: `def log_timeout(self, timeout_seconds: float) -> None`
- **Pseudocode**:
  ```
  INPUT: timeout_seconds (float)
  MANIPULATION:
    - Log WARNING: "[GPU-WORKER-{worker_id}] TIMEOUT: {timeout_seconds}s exceeded"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: timeout_seconds is positive
- **Called By**: gpu_interface.py (timeout handling)

##### log_connection_error()

- **Signature**: `def log_connection_error(self, error: str) -> None`
- **Pseudocode**:
  ```
  INPUT: error (string)
  MANIPULATION:
    - Log ERROR: "[GPU-WORKER-{worker_id}] CONNECTION-ERROR: {error}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: error is valid string
- **Called By**: gpu_interface.py (connection error handling)

##### log_http_error()

- **Signature**: `def log_http_error(self, status_code: int, error: str) -> None`
- **Pseudocode**:
  ```
  INPUT: status_code (integer), error (string)
  MANIPULATION:
    - Log ERROR: "[GPU-WORKER-{worker_id}] HTTP-ERROR: {status_code}, {error}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: status_code is valid HTTP status code, error is valid string
- **Called By**: gpu_interface.py (HTTP error handling)

##### log_processing_error()

- **Signature**: `def log_processing_error(self, error: str) -> None`
- **Pseudocode**:
  ```
  INPUT: error (string)
  MANIPULATION:
    - Log ERROR: "[GPU-WORKER-{worker_id}] PROCESSING-ERROR: {error}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: error is valid string
- **Called By**: gpu_interface.py (processing error handling)

##### log_metrics()

- **Signature**: `def log_metrics(self, metrics: dict) -> None`
- **Pseudocode**:
  ```
  INPUT: metrics (dictionary)
  MANIPULATION:
    - Log INFO: "[GPU-WORKER-{worker_id}] METRICS: {metrics}"
  OUTPUT: None (logs to logger)
  ```
- **Assumptions**: metrics is a dictionary
- **Called By**: gpu_interface.py (metrics reporting)

---

### 8. `gpu_scheduler.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/gpu_scheduler.py`
- **Lines**: 194
- **Scope**: GPU batch scheduling
- **Purpose**: Centralized batching and pacing control for GPU processing. Eliminates sawtooth utilization patterns by controlling batch timing and size. Manages staging area, launch timing, and maximum inflight batches (max 2).
- **Dependencies**:
  - `time`, `logging`
  - `collections` (deque)
  - `typing` (List, Optional)
  - `.data_structures` (ImageTask)
- **Used By**: gpu_processor_worker.py

#### Classes

##### GPUScheduler

- **Purpose**: Centralized GPU batch scheduler that manages batching and pacing

#### Methods

##### **init**()

- **Signature**: `def __init__(self, redis_mgr, deserializer, inbox_key: str, metadata_deserializer=None, target_batch: int = 32, max_wait_ms: int = 12, min_launch_ms: int = 200, config=None)`
- **Pseudocode**:
  ```
  INPUT: redis_mgr, deserializer, inbox_key, metadata_deserializer, target_batch, max_wait_ms, min_launch_ms, config
  MANIPULATION:
    - Store redis_mgr, deserializer (or metadata_deserializer if provided), inbox_key, config
    - Convert target_batch, max_wait_ms, min_launch_ms to int
    - Initialize _staging as empty list
    - Initialize _last_launch_ms to 0.0
    - Initialize _inflight as deque with maxlen=2 (tracks up to 2 inflight batch IDs)
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: redis_mgr has blpop_many and get_queue_length_by_key methods, deserializer converts bytes to ImageTask
- **Called By**: gpu_processor_worker.py::**init**()

##### \_now_ms() (staticmethod)

- **Signature**: `@staticmethod def _now_ms() -> float`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Get current time using time.perf_counter() (high-resolution timer)
    - Multiply by 1000.0 to convert to milliseconds
    - Return milliseconds as float
  OUTPUT: Current time in milliseconds (float)
  ```
- **Assumptions**: None
- **Called By**: \_can_launch(), next_batch(), mark_launched()

##### \_can_launch()

- **Signature**: `def _can_launch(self) -> bool`
- **Pseudocode**:
  ```
  INPUT: self._inflight (deque), self._last_launch_ms (float)
  MANIPULATION:
    - If len(_inflight) >= 2:
      - Return False (max 2 inflight batches)
    - Calculate time since last launch: _now_ms() - _last_launch_ms
    - Return True if time_since_launch >= MIN_LAUNCH_MS, False otherwise
  OUTPUT: Boolean indicating if new batch can be launched
  ```
- **Assumptions**: \_inflight and \_last_launch_ms are initialized
- **Called By**: next_batch()

##### feed()

- **Signature**: `def feed(self) -> int`
- **Pseudocode**:
  ```
  INPUT: self._staging (list), self.TARGET (int), self.inbox_key (string), self.redis, self.deserialize
  MANIPULATION:
    - Calculate need = max(0, TARGET - len(_staging))
    - If need == 0, return 0
    - Get queue depth via redis.get_queue_length_by_key(inbox_key)
    - Call redis.blpop_many(inbox_key, max_n=need, timeout=0.5) to get raw items
    - Calculate wait_time_ms
    - If no items returned:
      - Log INFO diagnostic message
      - Return 0
    - For each raw item:
      - Try to deserialize using self.deserialize()
      - Append to _staging list
      - Increment added counter
      - On exception: log debug and skip
    - Get queue_depth_after
    - Log INFO diagnostic with added count, queue depths, staging size, wait time
    - Return added count
  OUTPUT: Integer count of items added to staging
  ```
- **Assumptions**: Redis is accessible, deserializer is valid function
- **Called By**: gpu_processor_worker.py::run() (main loop)

##### next_batch()

- **Signature**: `def next_batch(self, force_flush: bool = False) -> Optional[List[ImageTask]]`
- **Pseudocode**:
  ```
  INPUT: force_flush (boolean), self._staging (list), self.TARGET (int), self._can_launch(), self._last_launch_ms
  MANIPULATION:
    - If _staging is empty, return None
    - If force_flush is True:
      - Copy all items from _staging to batch
      - Clear _staging
      - Log force flush
      - Return batch
    - If len(_staging) >= TARGET and _can_launch():
      - Take first TARGET items from _staging
      - Remove those items from _staging
      - Log target reached
      - Return batch
    - Calculate waited = _now_ms() - _last_launch_ms
    - If _can_launch() and waited >= MAX_WAIT_MS:
      - Calculate floor = max(8, TARGET // 4) (minimum batch size)
      - If len(_staging) >= floor:
        - Take min(len(_staging), TARGET) items
        - Remove from _staging
        - Log early launch
        - Return batch
    - Log diagnostic if not ready
    - Return None
  OUTPUT: List of ImageTasks ready for processing, or None if not ready
  ```
- **Assumptions**: \_staging contains valid ImageTask objects
- **Called By**: gpu_processor_worker.py::run() (main loop)

##### mark_launched()

- **Signature**: `def mark_launched(self, batch_id: str) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), self._last_launch_ms, self._inflight (deque)
  MANIPULATION:
    - Update _last_launch_ms to current time via _now_ms()
    - Append batch_id to _inflight deque
    - Note: deque automatically maintains maxlen=2
  OUTPUT: None (updates internal state)
  ```
- **Assumptions**: batch_id is valid string, \_inflight has space (maxlen=2)
- **Called By**: gpu_processor_worker.py::process_batch()

##### mark_completed()

- **Signature**: `def mark_completed(self, batch_id: str) -> None`
- **Pseudocode**:
  ```
  INPUT: batch_id (string), self._inflight (deque)
  MANIPULATION:
    - Try to remove batch_id from _inflight deque
    - If ValueError (not found), ignore (handle gracefully)
  OUTPUT: None (updates internal state)
  ```
- **Assumptions**: batch_id is valid string
- **Called By**: gpu_processor_worker.py::process_batch()

---

### 13. `crawler_worker.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/crawler_worker.py`
- **Lines**: 332
- **Scope**: HTML fetching and selector mining worker
- **Purpose**: HTML fetching and selector mining worker process. Handles site crawling with 3x3 mining and pushes candidates to Redis queue. Supports concurrent site processing.
- **Dependencies**:
  - `asyncio`, `json`, `logging`, `multiprocessing`, `os`, `signal`, `sys`, `time`
  - `pathlib` (Path)
  - `typing` (List, Optional)
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.selector_miner` (get_selector_miner)
  - `.http_utils` (get_http_utils)
  - `.data_structures` (SiteTask, CandidateImage, CandidatePost, TaskStatus)
  - `.timing_logger` (get_timing_logger)
- **Used By**: orchestrator.py (spawns worker processes)

#### Classes

##### CrawlerWorker

- **Purpose**: Crawler worker for HTML fetching and selector mining

#### Methods

##### **init**()

- **Signature**: `def __init__(self, worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Store worker_id
    - Get config, redis_manager, selector_miner, http_utils, timing_logger via singletons
    - Initialize running = False
    - Initialize processed_sites = 0, total_candidates = 0
    - Create asyncio.Event for shutdown
    - Store references to redis.incr_active_tasks and redis.decr_active_tasks
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: All singleton managers are available
- **Called By**: crawler_worker_process()

##### \_enqueue_candidates()

- **Signature**: `async def _enqueue_candidates(self, candidates) -> int`
- **Pseudocode**:
  ```
  INPUT: candidates (list of CandidateImage or CandidatePost), self.redis, self.config, self.worker_id
  MANIPULATION:
    - Determine candidate_type from first candidate if list not empty
    - Filter valid candidates:
      - For each candidate:
        - Check if site limit reached via redis.is_site_limit_reached_async()
        - If strict_limits enabled, check pages limit from site stats
        - Skip if limits reached
        - Otherwise add to valid_candidates
    - If no valid candidates, return 0
    - Serialize all valid candidates
    - Batch push to candidates queue via redis.push_many()
    - If diagnostic logging enabled:
      - Get queue depth
      - Track candidates/second throughput
      - Log diagnostic info
    - Return count of pushed candidates
  OUTPUT: Integer count of candidates enqueued
  ```
- **Assumptions**: candidates is a list, Redis is accessible
- **Called By**: process_site() (as background task)

##### process_site()

- **Signature**: `async def process_site(self, site_task: SiteTask) -> int`
- **Pseudocode**:
  ```
  INPUT: site_task (SiteTask object), self.selector_miner, self.timing_logger, self.redis, self.config
  MANIPULATION:
    - Record site_start_time
    - Log site start via timing_logger
    - Increment active tasks counter
    - Initialize enqueue_tasks list and total_pages = 0
    - Async iterate over selector_miner.mine_posts_with_3x3_crawl():
      - For each (page_url, page_candidates):
        - Log page start
        - Create async task to enqueue candidates (background)
        - Log page end with duration and candidate count
        - Increment total_pages
    - Wait for all enqueue tasks to complete
    - Sum total_enqueued from results
    - Update Redis site stats with pages_crawled and posts_found
    - Check pages limit if strict_limits enabled
    - Decrement active tasks counter
    - Log site end with duration, pages, candidates
    - Return total_enqueued
    - On exception: log error and return 0
  OUTPUT: Integer count of candidates found and enqueued
  ```
- **Assumptions**: site_task is valid, selector_miner is initialized
- **Called By**: \_process_site_task()

##### \_process_site_task()

- **Signature**: `async def _process_site_task(self, site_task: SiteTask) -> None`
- **Pseudocode**:
  ```
  INPUT: site_task (SiteTask object)
  MANIPULATION:
    - Call process_site(site_task) to get candidates_count
    - Update local statistics:
      - Increment processed_sites
      - Add candidates_count to total_candidates
    - Log completion with stats
    - On exception: log error
  OUTPUT: None (updates statistics)
  ```
- **Assumptions**: site_task is valid
- **Called By**: run() (as async task)

##### run()

- **Signature**: `async def run(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self.running, self._shutdown_event, self.config, self.redis
  MANIPULATION:
    - Set running = True
    - Initialize active_site_tasks as empty set
    - Get max_concurrent_sites from config
    - While running and not shutdown:
      - Clean up completed tasks from active_site_tasks
      - If capacity available (len < max_concurrent_sites):
        - Pop site task from Redis queue (timeout 2.0s)
        - If site_task received:
          - Create async task for _process_site_task()
          - Add to active_site_tasks
          - Log site start
      - Sleep 0.1s before next iteration
    - Wait for all active tasks to complete
    - Close http_utils
    - Log worker stopped
  OUTPUT: None (runs until stopped)
  ```
- **Assumptions**: Redis is accessible, config is valid
- **Called By**: crawler_worker_process() (via event loop)

##### stop()

- **Signature**: `def stop(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Set self.running = False
    - Set self._shutdown_event
  OUTPUT: None (signals shutdown)
  ```
- **Assumptions**: None
- **Called By**: Orchestrator (on shutdown)

##### get_stats()

- **Signature**: `def get_stats(self) -> dict`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Return dictionary with:
      - worker_id
      - processed_sites
      - total_candidates
      - running status
  OUTPUT: Dictionary with worker statistics
  ```
- **Assumptions**: None
- **Called By**: Orchestrator (monitoring)

#### Functions

##### crawler_worker_process()

- **Signature**: `def crawler_worker_process(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Reset signal handlers to default (SIGINT, SIGTERM)
    - Configure logging for multiprocessing
    - Create CrawlerWorker instance
    - Create new asyncio event loop
    - Run worker.run() in event loop
    - Close event loop
    - On exception: log fatal error
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: orchestrator.py::\_run_crawler_worker() (spawned as Process)

---

### 14. `extractor_worker.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/extractor_worker.py`
- **Lines**: 634
- **Scope**: Image download and batch preparation worker
- **Purpose**: Image download and batch preparation worker process. Downloads images, performs HEAD/GET validation, computes phash, and creates batches. Handles both CandidateImage and CandidatePost processing.
- **Dependencies**:
  - `asyncio`, `logging`, `multiprocessing`, `os`, `signal`, `sys`, `tempfile`, `time`
  - `datetime`
  - `typing` (List, Optional, Tuple, Union)
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.cache_manager` (get_cache_manager)
  - `.http_utils` (get_http_utils)
  - `.data_structures` (CandidateImage, CandidatePost, ImageTask, PostTask, BatchRequest, TaskStatus)
  - `.timing_logger` (get_timing_logger)
- **Used By**: orchestrator.py (spawns worker processes)

#### Classes

##### ExtractorWorker

- **Purpose**: Extractor worker for image download and batch preparation

#### Methods

##### **init**()

- **Signature**: `def __init__(self, worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Store worker_id
    - Get config, redis_manager, cache_manager, http_utils, timing_logger via singletons
    - Initialize running = False
    - Initialize processed_candidates, downloaded_images, cached_images = 0
    - Calculate concurrency_per_worker = total_concurrency // num_extractors
    - Create asyncio.Semaphore with concurrency_per_worker
    - Create temp directory with prefix "extractor_{worker_id}_"
    - Initialize _temp_files dict to track temp files
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: All singleton managers are available
- **Called By**: extractor_worker_process()

##### process_candidate()

- **Signature**: `async def process_candidate(self, candidate) -> Optional[Union[ImageTask, PostTask]]`
- **Pseudocode**:
  ```
  INPUT: candidate (CandidateImage or CandidatePost), extraction_start_time (float)
  MANIPULATION:
    - Record extraction_start_time
    - If candidate is CandidatePost:
      - Call _process_post_candidate()
    - Else if candidate is CandidateImage:
      - If image extraction disabled, return None
      - Call _process_image_candidate()
    - Else: log warning and return None
  OUTPUT: ImageTask, PostTask, or None
  ```
- **Assumptions**: candidate is valid CandidateImage or CandidatePost
- **Called By**: \_process_candidate_with_stats()

##### \_process_image_candidate()

- **Signature**: `async def _process_image_candidate(self, candidate: CandidateImage, extraction_start_time: float) -> Optional[ImageTask]`
- **Pseudocode**:
  ```
  INPUT: candidate (CandidateImage), extraction_start_time (float), self._semaphore, self.config, self.redis, self.cache, self.http_utils
  MANIPULATION:
    - Acquire semaphore (limit concurrent downloads)
    - Check URL deduplication via redis.url_seen_async()
    - If seen, return None
    - Check strict_limits: site image limit and site limit flag
    - Log extraction start
    - If HTML metadata available (content_type, width, height):
      - Skip HEAD check, use metadata
    - Else if not skip_head_check:
      - Perform HEAD check via http_utils.head_check()
      - If invalid, return None
    - Download image to temp file via http_utils.download_to_temp()
    - Track temp file in _temp_files dict
    - Compute phash in thread pool via cache.compute_phash()
    - If phash failed, delete temp file and return None
    - Check cache via cache.is_image_cached()
    - If cached, delete temp file and return None
    - Mark URL as seen via redis.mark_url_seen_async()
    - Check minimum file size (filter tiny avatars)
    - If too small, delete temp file and return None
    - Create ImageTask with temp_path, phash, candidate, file_size, mime_type
    - Serialize and push to GPU inbox queue via redis.push_many()
    - Log extraction end
    - Return ImageTask
    - On exception: log error and return None
  OUTPUT: ImageTask or None
  ```
- **Assumptions**: candidate has valid img_url, temp directory is writable
- **Called By**: process_candidate()

##### \_process_post_candidate()

- **Signature**: `async def _process_post_candidate(self, candidate: CandidatePost, extraction_start_time: float) -> Optional[PostTask]`
- **Pseudocode**:
  ```
  INPUT: candidate (CandidatePost), extraction_start_time (float), self._semaphore, self.config, self.redis
  MANIPULATION:
    - Acquire semaphore (limit concurrent processing)
    - Check URL deduplication via redis.url_seen_async()
    - If seen, return None
    - Check strict_limits: site post limit and site limit flag
    - Log extraction start
    - Create content hash from post content
    - Check if post contains diabetes keywords
    - Mark URL as seen
    - Create PostTask with candidate, content_hash, has_keywords
    - Push to storage queue via redis.push_post_task_async()
    - Log extraction end
    - Return PostTask
    - On exception: log error and return None
  OUTPUT: PostTask or None
  ```
- **Assumptions**: candidate has valid post_url and content
- **Called By**: process_candidate()

##### \_process_candidate_with_stats()

- **Signature**: `async def _process_candidate_with_stats(self, candidate: Union[CandidateImage, CandidatePost], site_extraction_times: dict) -> None`
- **Pseudocode**:
  ```
  INPUT: candidate, site_extraction_times (dict)
  MANIPULATION:
    - Record start time
    - Call process_candidate()
    - Calculate duration
    - Update site_extraction_times statistics
    - Update worker stats (processed_candidates, downloaded_images, cached_images)
  OUTPUT: None (updates statistics)
  ```
- **Assumptions**: candidate is valid
- **Called By**: run() (main loop)

##### run()

- **Signature**: `async def run(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self.running, self.config, self.redis
  MANIPULATION:
    - Set running = True
    - Initialize site_extraction_times dict
    - While running:
      - Pop batch of candidates from queue (batch_pop_size from config)
      - If candidates received:
        - Process each candidate concurrently via _process_candidate_with_stats()
        - Wait for all to complete
      - Else:
        - Sleep briefly
    - On exception: log error and sleep
  OUTPUT: None (runs until stopped)
  ```
- **Assumptions**: Redis is accessible
- **Called By**: extractor_worker_process() (via event loop)

##### \_cleanup_temp_dir()

- **Signature**: `def _cleanup_temp_dir(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self.temp_dir, self._temp_files (dict)
  MANIPULATION:
    - For each temp file in _temp_files:
      - Check if file exists and age
      - Delete if old enough
    - Remove temp directory if empty
  OUTPUT: None (cleans up temp files)
  ```
- **Assumptions**: temp_dir exists
- **Called By**: cleanup()

##### cleanup()

- **Signature**: `def cleanup(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Call _cleanup_temp_dir()
  OUTPUT: None
  ```
- **Assumptions**: None
- **Called By**: extractor_worker_process() (on shutdown)

##### stop()

- **Signature**: `def stop(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Set self.running = False
  OUTPUT: None (signals shutdown)
  ```
- **Assumptions**: None
- **Called By**: Orchestrator (on shutdown)

##### get_stats()

- **Signature**: `def get_stats(self) -> dict`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Return dictionary with:
      - worker_id
      - processed_candidates
      - downloaded_images
      - cached_images
      - running status
  OUTPUT: Dictionary with worker statistics
  ```
- **Assumptions**: None
- **Called By**: Orchestrator (monitoring)

#### Functions

##### extractor_worker_process()

- **Signature**: `def extractor_worker_process(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Reset signal handlers to default
    - Configure logging for multiprocessing
    - Create ExtractorWorker instance
    - Create new asyncio event loop
    - Run worker.run() in event loop
    - Call worker.cleanup() on shutdown
    - Close event loop
    - On exception: log fatal error
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: orchestrator.py::\_run_extractor_worker() (spawned as Process)

---

### 16. `storage_worker.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/storage_worker.py`
- **Lines**: 433
- **Scope**: Storage operations worker
- **Purpose**: Consumes storage tasks from Redis queue and saves them to MinIO. Handles I/O operations only - all compute (cropping) happens in GPU processor. After saving to MinIO, upserts face embeddings to Qdrant vector database.
- **Dependencies**:
  - `asyncio`, `logging`, `multiprocessing`, `os`, `signal`, `sys`, `time`, `uuid`
  - `typing` (Optional, List, Dict, Any)
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.storage_manager` (get_storage_manager)
  - `.cache_manager` (get_cache_manager)
  - `.timing_logger` (get_timing_logger)
  - `.data_structures` (StorageTask, FaceResult, ImageTask, FaceDetection, PostTask)
  - `qdrant_client` (optional, for vector operations)
- **Used By**: orchestrator.py (spawns worker processes)

#### Classes

##### StorageWorker

- **Purpose**: Storage worker that consumes storage queue and saves to MinIO

#### Functions

##### \_get_vector_client()

- **Signature**: `def _get_vector_client() -> QdrantClient`
- **Pseudocode**:
  ```
  INPUT: _vector_client (global singleton), config
  MANIPULATION:
    - If _vector_client is None:
      - Import QdrantClient and models
      - Get config
      - Create QdrantClient with qdrant_url
      - Try to create collection (ignore if exists)
      - Store in _vector_client
    - Return _vector_client
  OUTPUT: QdrantClient singleton instance
  ```
- **Assumptions**: Qdrant is accessible, vectorization_enabled in config
- **Called By**: \_upsert_embeddings_to_vector_db()

#### Methods

##### **init**()

- **Signature**: `def __init__(self, worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Store worker_id
    - Get config, redis_manager, storage_manager, cache_manager, timing_logger via singletons
    - Initialize running = False
    - Initialize processed_tasks = 0, failed_tasks = 0
    - Initialize vectorized_faces = 0, vectorization_errors = 0
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: All singleton managers are available
- **Called By**: storage_worker_process()

##### run()

- **Signature**: `async def run(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self.running, self.config, self.redis
  MANIPULATION:
    - Set running = True
    - Initialize throughput tracking (start_time, last_log_time)
    - While running:
      - Pop task from storage queue via redis.pop_storage_task_async() (timeout 2.0s)
      - If task received:
        - Get queue depth
        - If task is PostTask:
          - Call _process_post_task()
        - Else if task is StorageTask:
          - Call _process_storage_task()
        - Else: log warning and increment failed_tasks
        - Increment processed_tasks
        - Log periodic throughput (every 10 seconds)
      - Else:
        - Log diagnostic (every 10 seconds) and sleep 0.1s
    - Log final stats on shutdown
  OUTPUT: None (runs until stopped)
  ```
- **Assumptions**: Redis is accessible
- **Called By**: storage_worker_process() (via event loop)

##### \_process_storage_task()

- **Signature**: `async def _process_storage_task(self, storage_task: StorageTask) -> None`
- **Pseudocode**:
  ```
  INPUT: storage_task (StorageTask object), self.storage, self.cache, self.redis, self.config
  MANIPULATION:
    - Record storage_start_time
    - Validate storage task structure (faces_count vs crops_count)
    - Log storage start
    - Save to storage via storage.save_storage_task_async()
    - Get save_counts (saved_thumbs)
    - If faces have embeddings:
      - Call _upsert_embeddings_to_vector_db() to vectorize
    - Cache result via cache.store_processing_result()
    - Update Redis site stats:
      - faces_detected
      - images_saved_raw (if saved)
      - images_saved_thumbs
    - Check if site image limit reached (strict_limits):
      - If reached, set site limit flag
      - Remove remaining items from gpu:inbox queue
    - Push face result to results queue
    - Clean up temp file via storage.cleanup_temp_file()
    - Calculate duration
    - Log storage end
    - On exception: increment failed_tasks and log error
  OUTPUT: None (saves to MinIO and updates stats)
  ```
- **Assumptions**: storage_task has valid image_task and face_result
- **Called By**: run() (main loop)

##### \_upsert_embeddings_to_vector_db()

- **Signature**: `async def _upsert_embeddings_to_vector_db(self, face_result: FaceResult, image_task: ImageTask) -> int`
- **Pseudocode**:
  ```
  INPUT: face_result (FaceResult), image_task (ImageTask), self.config
  MANIPULATION:
    - If vectorization disabled, return 0
    - If no faces, return 0
    - Filter faces that have embeddings
    - If no faces with embeddings, return 0
    - Get Qdrant client via _get_vector_client()
    - Build points list:
      - For each face with embedding:
        - Generate unique face_id (UUID)
        - Get thumbnail key if available
        - Build payload with metadata (tenant_id, raw_key, thumb_key, source_url, page_url, site_id, phash, bbox, quality, age, gender, indexed_at)
        - Create PointStruct with id, vector (embedding), payload
    - Upsert points to Qdrant collection via client.upsert()
    - Increment vectorized_faces counter
    - Return count of upserted faces
    - On exception: increment vectorization_errors and return 0
  OUTPUT: Integer count of faces successfully upserted
  ```
- **Assumptions**: Qdrant is accessible, vectorization_enabled is True, faces have embeddings
- **Called By**: \_process_storage_task()

##### \_process_post_task()

- **Signature**: `async def _process_post_task(self, post_task: PostTask) -> None`
- **Pseudocode**:
  ```
  INPUT: post_task (PostTask object), self.storage, self.redis, self.config
  MANIPULATION:
    - Record storage_start_time
    - Save post metadata to MinIO via storage.save_post_metadata_async()
    - If save failed, raise exception
    - Update Redis site stats with posts_saved = 1
    - Check if site post limit reached (strict_limits):
      - If reached, set site limit flag
      - Remove remaining post items from storage queue
    - Calculate duration
    - Log completion
    - On exception: increment failed_tasks and log error
  OUTPUT: None (saves post to MinIO)
  ```
- **Assumptions**: post_task has valid candidate and content_hash
- **Called By**: run() (main loop)

##### stop()

- **Signature**: `def stop(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Set self.running = False
  OUTPUT: None (signals shutdown)
  ```
- **Assumptions**: None
- **Called By**: Orchestrator (on shutdown), signal_handler()

#### Functions

##### storage_worker_process()

- **Signature**: `def storage_worker_process(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Define signal_handler function to stop worker gracefully
    - Configure logging for multiprocessing
    - Create StorageWorker instance
    - Register signal handlers (SIGTERM, SIGINT)
    - Run worker.run() via asyncio.run()
    - On KeyboardInterrupt: stop worker
    - On exception: log fatal error and stop worker
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: orchestrator.py::\_run_storage_worker() (spawned as Process)

---

### 9. `http_utils.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/http_utils.py`
- **Lines**: 1483
- **Scope**: HTTP fetching and JavaScript rendering utilities
- **Purpose**: Provides utilities for HTTP fetching, including retry logic, realistic headers, and JavaScript rendering fallback using Playwright. Manages per-domain connection pools, circuit breakers, and a JS cache.
- **Dependencies**:
  - `asyncio`, `logging`, `time`, `random`, `tempfile`, `os`
  - `typing` (Dict, List, Optional, Tuple, Any)
  - `httpx`
  - `beautifulsoup4` (BeautifulSoup)
  - `playwright` (optional, for JS rendering)
  - `urlparse`, `urljoin` (from urllib.parse)
  - `.config` (get_config)
- **Used By**: crawler_worker.py, selector_miner.py, extractor_worker.py, test_suite.py

#### Classes

##### DomainConnectionPool

- **Purpose**: Manages per-domain HTTP connection pools with circuit breaker pattern

##### HTTPUtils

- **Purpose**: Main HTTP utilities class for fetching HTML and downloading images

##### BrowserPool

- **Purpose**: Manages Playwright browser instances for JavaScript rendering

#### Key Methods

##### HTTPUtils.fetch_html()

- **Signature**: `async def fetch_html(self, url: str, use_js_fallback: bool = True, force_compare_first_visit: bool = False) -> Tuple[Optional[str], str, Optional[Dict[str, int]]]`
- **Pseudocode**:
  ```
  INPUT: url (string), use_js_fallback (boolean), force_compare_first_visit (boolean)
  MANIPULATION:
    - If force_compare_first_visit and JS available:
      - Fetch both HTTP and JS concurrently
      - Compare image counts from both
      - Return better HTML based on heuristic (aggressive HTTP-first strategy)
      - Return comparison_stats dict
    - Try standard HTTP fetch via _fetch_with_redirects()
    - If HTML received:
      - Estimate image candidates count
      - If count < 10, try JS rendering
      - If _needs_js_rendering() returns False, return HTTP HTML
    - If use_js_fallback and JS available:
      - Try JS rendering via _fetch_with_js()
      - Return JS HTML if successful
    - Return HTTP HTML or error
  OUTPUT: Tuple of (html string or None, status message, comparison_stats dict or None)
  ```
- **Assumptions**: url is valid, config is initialized
- **Called By**: crawler_worker.py, selector_miner.py

##### HTTPUtils.head_check()

- **Signature**: `async def head_check(self, url: str) -> Tuple[bool, Dict[str, Any]]`
- **Pseudocode**:
  ```
  INPUT: url (string)
  MANIPULATION:
    - Get HTTP client for URL
    - Perform HEAD request with headers
    - Extract response info (status_code, content_type, content_length, last_modified, etag)
    - Record request metrics
    - Check if content_type is image (exclude SVG)
    - Check size bounds (min_bytes, max_bytes)
    - Return (is_valid_image, info_dict)
  OUTPUT: Tuple of (boolean success, info dictionary)
  ```
- **Assumptions**: url is valid
- **Called By**: extractor_worker.py::\_process_image_candidate()

##### HTTPUtils.download_to_temp()

- **Signature**: `async def download_to_temp(self, url: str, temp_dir: str = None) -> Tuple[Optional[str], Dict[str, Any]]`
- **Pseudocode**:
  ```
  INPUT: url (string), temp_dir (optional string)
  MANIPULATION:
    - For each retry attempt (up to max_retries):
      - Get HTTP client for URL
      - Follow redirects manually (respect same-origin policy, blocklist)
      - Stream response to temp file (avoid loading into memory)
      - Validate content_type is image
      - Check content_length < 10MB
      - Return (temp_file_path, metadata_dict)
      - On failure: exponential backoff with jitter
    - Return (None, error_dict) if all retries fail
  OUTPUT: Tuple of (temp file path or None, metadata dictionary)
  ```
- **Assumptions**: url is valid, temp_dir is writable
- **Called By**: extractor_worker.py::\_process_image_candidate()

##### HTTPUtils.\_fetch_with_redirects()

- **Signature**: `async def _fetch_with_redirects(self, url: str) -> Tuple[Optional[str], str]`
- **Pseudocode**:
  ```
  INPUT: url (string)
  MANIPULATION:
    - Get HTTP client for URL
    - Follow redirects manually (up to max_redirects):
      - Check same-origin policy if enabled
      - Check blocklist hosts
      - Resolve relative redirects
    - Get response content
    - Record request metrics
    - Return (html, status_message)
  OUTPUT: Tuple of (html string or None, status message)
  ```
- **Assumptions**: url is valid
- **Called By**: fetch_html()

##### HTTPUtils.\_fetch_with_js()

- **Signature**: `async def _fetch_with_js(self, url: str) -> Tuple[Optional[str], str]`
- **Pseudocode**:
  ```
  INPUT: url (string)
  MANIPULATION:
    - Get browser from BrowserPool
    - Create new page
    - Navigate to URL with timeout
    - Wait for page load (based on config strategy)
    - Extract HTML content
    - Return (html, status_message)
    - On exception: return (None, error_message)
  OUTPUT: Tuple of (html string or None, status message)
  ```
- **Assumptions**: Playwright is available, url is valid
- **Called By**: fetch_html()

#### Functions

##### get_http_utils()

- **Signature**: `def get_http_utils() -> HTTPUtils`
- **Pseudocode**:
  ```
  INPUT: _http_utils_instance (global singleton)
  MANIPULATION:
    - If _http_utils_instance is None:
      - Create new HTTPUtils instance
      - Store in _http_utils_instance
    - Return _http_utils_instance
  OUTPUT: HTTPUtils singleton instance
  ```
- **Assumptions**: None
- **Called By**: crawler_worker.py, selector_miner.py, extractor_worker.py

##### close_http_utils()

- **Signature**: `async def close_http_utils() -> None`
- **Pseudocode**:
  ```
  INPUT: _http_utils_instance (global singleton)
  MANIPULATION:
    - If _http_utils_instance exists:
      - Call close() to cleanup connections and browsers
      - Set _http_utils_instance to None
  OUTPUT: None (cleans up resources)
  ```
- **Assumptions**: None
- **Called By**: crawler_worker.py (on shutdown)

---

### 10. `redis_manager.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/redis_manager.py`
- **Lines**: 1689
- **Scope**: Redis queue and cache management
- **Purpose**: Manages all interactions with Redis, including connection pooling, serialization/deserialization of Pydantic models, queue operations (push/pop for sites, candidates, images, results, storage, GPU inbox), URL deduplication, site statistics, and active task tracking.
- **Dependencies**:
  - `redis`, `aioredis`
  - `json`, `logging`, `time`
  - `typing` (List, Optional, Dict, Any, Union)
  - `.config` (get_config)
  - `.data_structures` (all Pydantic models)
- **Used By**: cache_manager.py, crawler_worker.py, extractor_worker.py, gpu_processor_worker.py, orchestrator.py, storage_worker.py, main.py, selector_miner.py, test_suite.py

#### Classes

##### RedisManager

- **Purpose**: Manages all Redis operations including queues, caching, and deduplication

#### Key Methods

##### RedisManager.push_site()

- **Signature**: `def push_site(self, site_task: SiteTask, timeout: float = 5.0) -> bool`
- **Pseudocode**:
  ```
  INPUT: site_task (SiteTask), timeout (float)
  MANIPULATION:
    - Serialize site_task to bytes
    - Push to sites queue via Redis LPUSH
    - Return True on success, False on failure
  OUTPUT: Boolean success
  ```
- **Assumptions**: site_task is valid SiteTask
- **Called By**: orchestrator.py::push_sites()

##### RedisManager.pop_site()

- **Signature**: `def pop_site(self, timeout: float = 5.0) -> Optional[SiteTask]`
- **Pseudocode**:
  ```
  INPUT: timeout (float)
  MANIPULATION:
    - Pop from sites queue via Redis BLPOP (blocking)
    - Deserialize bytes to SiteTask
    - Return SiteTask or None
  OUTPUT: SiteTask or None
  ```
- **Assumptions**: None
- **Called By**: crawler_worker.py::run()

##### RedisManager.push_many()

- **Signature**: `def push_many(self, key: str, payloads: list[bytes]) -> int`
- **Pseudocode**:
  ```
  INPUT: key (string), payloads (list of bytes)
  MANIPULATION:
    - Use Redis pipeline for batch LPUSH
    - Push all payloads to queue
    - Return count of pushed items
  OUTPUT: Integer count
  ```
- **Assumptions**: key is valid queue name, payloads is list of bytes
- **Called By**: crawler_worker.py::\_enqueue_candidates(), extractor_worker.py

##### RedisManager.blpop_many()

- **Signature**: `def blpop_many(self, key: str, max_n: int, timeout: float = 0.5) -> list[bytes]`
- **Pseudocode**:
  ```
  INPUT: key (string), max_n (integer), timeout (float)
  MANIPULATION:
    - Use Redis pipeline for batch BLPOP
    - Pop up to max_n items from queue
    - Return list of bytes (may be empty)
  OUTPUT: List of bytes
  ```
- **Assumptions**: key is valid queue name
- **Called By**: extractor_worker.py::run(), gpu_scheduler.py::feed()

##### RedisManager.get_queue_length_by_key()

- **Signature**: `def get_queue_length_by_key(self, key: str) -> int`
- **Pseudocode**:
  ```
  INPUT: key (string)
  MANIPULATION:
    - Call Redis LLEN on key
    - Return queue length
  OUTPUT: Integer queue length
  ```
- **Assumptions**: key is valid queue name
- **Called By**: orchestrator.py, gpu_scheduler.py, many workers

##### RedisManager.mark_url_seen_async()

- **Signature**: `async def mark_url_seen_async(self, url: str, ttl_seconds: Optional[int] = None) -> bool`
- **Pseudocode**:
  ```
  INPUT: url (string), ttl_seconds (optional integer)
  MANIPULATION:
    - Create cache key for URL
    - Set in Redis with TTL
    - Return True on success
  OUTPUT: Boolean success
  ```
- **Assumptions**: url is valid string
- **Called By**: extractor_worker.py, crawler_worker.py

##### RedisManager.url_seen_async()

- **Signature**: `async def url_seen_async(self, url: str) -> bool`
- **Pseudocode**:
  ```
  INPUT: url (string)
  MANIPULATION:
    - Create cache key for URL
    - Check if exists in Redis
    - Return True if seen, False otherwise
  OUTPUT: Boolean indicating if URL was seen
  ```
- **Assumptions**: url is valid string
- **Called By**: extractor_worker.py, crawler_worker.py

##### RedisManager.update_site_stats_async()

- **Signature**: `async def update_site_stats_async(self, site_id: str, stats: Dict[str, Any]) -> bool`
- **Pseudocode**:
  ```
  INPUT: site_id (string), stats (dictionary)
  MANIPULATION:
    - Get existing site stats from Redis
    - Merge new stats with existing
    - Save back to Redis
    - Return True on success
  OUTPUT: Boolean success
  ```
- **Assumptions**: site_id is valid string, stats is dictionary
- **Called By**: crawler_worker.py, storage_worker.py, extractor_worker.py

##### RedisManager.get_site_stats()

- **Signature**: `def get_site_stats(self, site_id: str) -> Optional[Dict[str, Any]]`
- **Pseudocode**:
  ```
  INPUT: site_id (string)
  MANIPULATION:
    - Get site stats from Redis
    - Deserialize JSON
    - Return stats dictionary or None
  OUTPUT: Dictionary or None
  ```
- **Assumptions**: site_id is valid string
- **Called By**: orchestrator.py, extractor_worker.py, many workers

##### RedisManager.set_site_limit_reached_async()

- **Signature**: `async def set_site_limit_reached_async(self, site_id: str) -> bool`
- **Pseudocode**:
  ```
  INPUT: site_id (string)
  MANIPULATION:
    - Set Redis key indicating site limit reached
    - Return True on success
  OUTPUT: Boolean success
  ```
- **Assumptions**: site_id is valid string
- **Called By**: storage_worker.py, extractor_worker.py

##### RedisManager.is_site_limit_reached_async()

- **Signature**: `async def is_site_limit_reached_async(self, site_id: str) -> bool`
- **Pseudocode**:
  ```
  INPUT: site_id (string)
  MANIPULATION:
    - Check Redis key for site limit flag
    - Return True if limit reached, False otherwise
  OUTPUT: Boolean indicating if limit reached
  ```
- **Assumptions**: site_id is valid string
- **Called By**: crawler_worker.py, extractor_worker.py

#### Functions

##### get_redis_manager()

- **Signature**: `def get_redis_manager() -> RedisManager`
- **Pseudocode**:
  ```
  INPUT: _redis_manager_instance (global singleton)
  MANIPULATION:
    - If _redis_manager_instance is None:
      - Create new RedisManager instance
      - Store in _redis_manager_instance
    - Return _redis_manager_instance
  OUTPUT: RedisManager singleton instance
  ```
- **Assumptions**: None
- **Called By**: All workers and managers

##### close_redis_manager()

- **Signature**: `def close_redis_manager() -> None`
- **Pseudocode**:
  ```
  INPUT: _redis_manager_instance (global singleton)
  MANIPULATION:
    - If _redis_manager_instance exists:
      - Close connection pools
      - Set _redis_manager_instance to None
  OUTPUT: None (cleans up resources)
  ```
- **Assumptions**: None
- **Called By**: main.py (on shutdown)

---

### 11. `storage_manager.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/storage_manager.py`
- **Lines**: 993
- **Scope**: MinIO/S3 storage operations
- **Purpose**: Handles saving data to MinIO/S3, including raw images, face thumbnails, and post metadata. Computes content hashes and manages object keys.
- **Dependencies**:
  - `minio`, `io`, `json`, `logging`, `os`, `time`, `tempfile`
  - `asyncio`
  - `typing` (Dict, List, Optional, Tuple, Any)
  - `PIL` (Pillow, for image operations)
  - `.config` (get_config)
  - `.data_structures` (ImageTask, FaceResult, FaceDetection, PostTask)
- **Used By**: storage_worker.py, main.py, test_suite.py

#### Classes

##### StorageManager

- **Purpose**: Manages all MinIO/S3 storage operations

#### Key Methods

##### StorageManager.save_raw_image()

- **Signature**: `def save_raw_image(self, image_task: ImageTask) -> Tuple[Optional[str], Optional[str]]`
- **Pseudocode**:
  ```
  INPUT: image_task (ImageTask)
  MANIPULATION:
    - Validate temp_path exists
    - Read image bytes from temp file
    - Generate key: "{phash}.jpg"
    - Save to MinIO raw-images bucket with retry (3 attempts)
    - Return (key, s3_url)
  OUTPUT: Tuple of (object key or None, S3 URL or None)
  ```
- **Assumptions**: image_task has valid temp_path and phash
- **Called By**: save_storage_task_async(), save_face_result_async()

##### StorageManager.save_face_thumbnail()

- **Signature**: `def save_face_thumbnail(self, face_crop_data: bytes, face_detection: FaceDetection, image_task: ImageTask) -> Tuple[Optional[str], Optional[str]]`
- **Pseudocode**:
  ```
  INPUT: face_crop_data (bytes), face_detection (FaceDetection), image_task (ImageTask)
  MANIPULATION:
    - Compute face hash from detection
    - Generate key: "{face_hash}.jpg"
    - Save to MinIO thumbnails bucket
    - Return (key, s3_url)
  OUTPUT: Tuple of (object key or None, S3 URL or None)
  ```
- **Assumptions**: face_crop_data is valid image bytes
- **Called By**: save_storage_task_async(), save_face_result_async()

##### StorageManager.save_post_metadata()

- **Signature**: `def save_post_metadata(self, post_task: PostTask) -> Tuple[Optional[str], Optional[str]]`
- **Pseudocode**:
  ```
  INPUT: post_task (PostTask)
  MANIPULATION:
    - Create metadata dict from post_task
    - Serialize to JSON
    - Generate key: "posts/{content_hash}.json"
    - Save to MinIO raw-images bucket
    - Return (key, s3_url)
  OUTPUT: Tuple of (object key or None, S3 URL or None)
  ```
- **Assumptions**: post_task has valid candidate and content_hash
- **Called By**: save_post_metadata_async()

##### StorageManager.save_storage_task_async()

- **Signature**: `async def save_storage_task_async(self, storage_task: StorageTask) -> Tuple[FaceResult, Dict[str, int]]`
- **Pseudocode**:
  ```
  INPUT: storage_task (StorageTask)
  MANIPULATION:
    - Save raw image if needed
    - For each face_crop in storage_task.face_crops:
      - Save thumbnail to MinIO
    - Save metadata JSON
    - Return (updated_face_result, save_counts_dict)
  OUTPUT: Tuple of (FaceResult, save counts dictionary)
  ```
- **Assumptions**: storage_task has valid image_task, face_result, and face_crops
- **Called By**: storage_worker.py::\_process_storage_task()

##### StorageManager.health_check()

- **Signature**: `def health_check(self) -> Dict[str, Any]`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Test MinIO connection
    - List buckets
    - Return health status dict
  OUTPUT: Dictionary with health status
  ```
- **Assumptions**: MinIO is accessible
- **Called By**: main.py::health_check()

#### Functions

##### get_storage_manager()

- **Signature**: `def get_storage_manager() -> StorageManager`
- **Pseudocode**:
  ```
  INPUT: _storage_manager_instance (global singleton)
  MANIPULATION:
    - If _storage_manager_instance is None:
      - Create new StorageManager instance
      - Store in _storage_manager_instance
    - Return _storage_manager_instance
  OUTPUT: StorageManager singleton instance
  ```
- **Assumptions**: None
- **Called By**: storage_worker.py, main.py

##### close_storage_manager()

- **Signature**: `def close_storage_manager() -> None`
- **Pseudocode**:
  ```
  INPUT: _storage_manager_instance (global singleton)
  MANIPULATION:
    - If _storage_manager_instance exists:
      - Close MinIO client
      - Set _storage_manager_instance to None
  OUTPUT: None (cleans up resources)
  ```
- **Assumptions**: None
- **Called By**: main.py (on shutdown)

---

### 12. `gpu_interface.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/gpu_interface.py`
- **Lines**: 987
- **Scope**: GPU worker communication interface
- **Purpose**: Provides interface for communicating with external GPU worker service. Handles HTTP requests, image encoding, result decoding, and implements circuit breaker pattern for resilience. Includes CPU fallback for face detection.
- **Dependencies**:
  - `asyncio`, `httpx`, `json`, `logging`, `os`, `time`, `traceback`
  - `typing` (Dict, List, Optional, Any)
  - `insightface` (optional, for CPU fallback)
  - `.config` (get_config)
  - `.data_structures` (ImageTask, FaceDetection)
  - `.gpu_worker_logger` (GPUWorkerLogger)
- **Used By**: gpu_processor_worker.py, main.py, test_suite.py

#### Classes

##### GPUInterface

- **Purpose**: Interface for GPU worker communication with circuit breaker and CPU fallback

#### Key Methods

##### GPUInterface.process_batch()

- **Signature**: `async def process_batch(self, image_tasks: List[ImageTask], batch_id: Optional[str] = None) -> Optional[Dict[str, List[FaceDetection]]]`
- **Pseudocode**:
  ```
  INPUT: image_tasks (list of ImageTask), batch_id (optional string)
  MANIPULATION:
    - Check circuit breaker state
    - If circuit open:
      - Try CPU fallback
      - Return CPU results
    - Try GPU worker request via _gpu_worker_request()
    - If GPU fails:
      - Update circuit breaker
      - Try CPU fallback
      - Return CPU results
    - Return GPU results
  OUTPUT: Dictionary mapping phash to list of FaceDetection, or None
  ```
- **Assumptions**: image_tasks have valid temp_path
- **Called By**: gpu_processor_worker.py::process_batch()

##### GPUInterface.\_gpu_worker_request()

- **Signature**: `async def _gpu_worker_request(self, image_tasks: List[ImageTask]) -> Optional[Dict[str, List[FaceDetection]]]`
- **Pseudocode**:
  ```
  INPUT: image_tasks (list of ImageTask)
  MANIPULATION:
    - Load images from temp paths
    - Build multipart form data (no base64 encoding)
    - POST to GPU worker /detect_faces_batch_multipart endpoint
    - Parse JSON response
    - Decode face detections
    - Filter faces by pose
    - Update metrics
    - Return results dict
  OUTPUT: Dictionary mapping phash to list of FaceDetection, or None
  ```
- **Assumptions**: GPU worker is accessible, images exist at temp_path
- **Called By**: process_batch()

##### GPUInterface.\_cpu_fallback()

- **Signature**: `async def _cpu_fallback(self, image_tasks: List[ImageTask], batch_id: Optional[str] = None) -> Dict[str, List[FaceDetection]]`
- **Pseudocode**:
  ```
  INPUT: image_tasks (list of ImageTask), batch_id (optional string)
  MANIPULATION:
    - Get CPU face detection app (lazy load)
    - Preload all images in parallel
    - Process each image with CPU model
    - Filter faces by pose
    - Return results dict
  OUTPUT: Dictionary mapping phash to list of FaceDetection
  ```
- **Assumptions**: insightface is available, images exist at temp_path
- **Called By**: process_batch() (when GPU unavailable)

##### GPUInterface.\_check_health()

- **Signature**: `async def _check_health(self) -> bool`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Make GET request to GPU worker /health endpoint
    - If successful:
      - Close circuit breaker if open
      - Return True
    - Else:
      - Open circuit breaker
      - Return False
  OUTPUT: Boolean indicating GPU worker health
  ```
- **Assumptions**: GPU worker has /health endpoint
- **Called By**: main.py::health_check()

#### Functions

##### get_gpu_interface()

- **Signature**: `def get_gpu_interface() -> GPUInterface`
- **Pseudocode**:
  ```
  INPUT: _gpu_interface_instance (global singleton)
  MANIPULATION:
    - If _gpu_interface_instance is None:
      - Create new GPUInterface instance
      - Store in _gpu_interface_instance
    - Return _gpu_interface_instance
  OUTPUT: GPUInterface singleton instance
  ```
- **Assumptions**: None
- **Called By**: gpu_processor_worker.py, main.py

##### close_gpu_interface()

- **Signature**: `async def close_gpu_interface() -> None`
- **Pseudocode**:
  ```
  INPUT: _gpu_interface_instance (global singleton)
  MANIPULATION:
    - If _gpu_interface_instance exists:
      - Call close() to cleanup HTTP client
      - Set _gpu_interface_instance to None
  OUTPUT: None (cleans up resources)
  ```
- **Assumptions**: None
- **Called By**: main.py (on shutdown)

---

### 15. `gpu_processor_worker.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/gpu_processor_worker.py`
- **Lines**: 1288
- **Scope**: GPU batch processing worker
- **Purpose**: Processes batches of ImageTask objects. Copies temp files, validates images, sends them to GPU worker via gpu_interface, and then pushes the results as StorageTask to the storage queue. Handles CPU fallback if GPU worker is unavailable.
- **Dependencies**:
  - `asyncio`, `logging`, `multiprocessing`, `os`, `signal`, `sys`, `tempfile`, `time`, `shutil`
  - `typing` (List, Optional, Dict, Any)
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.gpu_interface` (get_gpu_interface)
  - `.gpu_scheduler` (GPUScheduler)
  - `.cache_manager` (get_cache_manager)
  - `.timing_logger` (get_timing_logger)
  - `.data_structures` (ImageTask, BatchRequest, StorageTask, FaceResult, FaceDetection)
- **Used By**: orchestrator.py (spawns worker processes)

#### Classes

##### GPUProcessorWorker

- **Purpose**: GPU processor worker for batch face detection

#### Key Methods

##### GPUProcessorWorker.process_batch()

- **Signature**: `async def process_batch(self, batch_request: BatchRequest) -> int`
- **Pseudocode**:
  ```
  INPUT: batch_request (BatchRequest)
  MANIPULATION:
    - Copy temp files to GPU processor ownership (prevent race conditions)
    - Validate image tasks
    - Fill batch to target size if needed
    - Send batch to GPU worker via gpu_interface.process_batch()
    - Crop faces from images
    - Create StorageTask with face_crops
    - Push to storage queue
    - Clean up GPU temp files
    - Return count of processed images
  OUTPUT: Integer count of processed images
  ```
- **Assumptions**: batch_request has valid image_tasks with temp_path
- **Called By**: run() (main loop)

##### GPUProcessorWorker.run()

- **Signature**: `async def run(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self.running, self.scheduler, self.config
  MANIPULATION:
    - Set running = True
    - While running:
      - Feed scheduler with items from GPU inbox queue
      - Get next batch from scheduler
      - If batch ready:
        - Process batch via process_batch()
        - Mark batch completed in scheduler
      - If queues empty for duration, force flush staging
      - Sleep briefly
    - Cleanup on shutdown
  OUTPUT: None (runs until stopped)
  ```
- **Assumptions**: Redis is accessible, scheduler is initialized
- **Called By**: gpu_processor_worker_process() (via event loop)

#### Functions

##### gpu_processor_worker_process()

- **Signature**: `def gpu_processor_worker_process(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Reset signal handlers to default
    - Configure logging for multiprocessing
    - Create GPUProcessorWorker instance
    - Create new asyncio event loop
    - Run worker.run() in event loop
    - Close event loop
    - On exception: log fatal error
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: orchestrator.py::\_run_gpu_processor_worker() (spawned as Process)

---

### 17. `selector_miner.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/selector_miner.py`
- **Lines**: 3192
- **Scope**: Post discovery and extraction engine
- **Purpose**: The "post discovery engine" that performs multi-page crawling (3x3 approach) to find diabetes-related posts. Extracts post metadata (title, content, author, date) and filters for keywords. Image extraction methods are largely disabled in post-focused mode.
- **Dependencies**:
  - `asyncio`, `logging`, `re`, `time`
  - `typing` (List, Optional, Tuple, Dict, Any, AsyncIterator)
  - `beautifulsoup4` (BeautifulSoup)
  - `datetime`
  - `.config` (get_config)
  - `.http_utils` (get_http_utils)
  - `.redis_manager` (get_redis_manager)
  - `.extraction_tracer` (get_extraction_tracer)
  - `.data_structures` (CandidateImage, CandidatePost)
- **Used By**: crawler_worker.py, test_suite.py

#### Classes

##### SelectorMiner

- **Purpose**: Post discovery and extraction engine with 3x3 crawling strategy

#### Key Methods

##### SelectorMiner.mine_posts_with_3x3_crawl()

- **Signature**: `async def mine_posts_with_3x3_crawl(self, base_url: str, site_id: str, max_pages: int = 5) -> AsyncIterator[Tuple[str, List[CandidatePost]]]`
- **Pseudocode**:
  ```
  INPUT: base_url (string), site_id (string), max_pages (integer)
  MANIPULATION:
    - PHASE 1: Sample crawl
      - Fetch base page
      - Mine posts from base page
      - Discover category pages
      - Fetch up to 6 more sample pages in parallel
      - Yield (page_url, candidates) for each page
    - PHASE 2: BFS crawl (if more pages needed)
      - Continue BFS expansion from discovered URLs
      - Fetch and mine each page
      - Yield (page_url, candidates) for each page
    - Track pages_crawled and checked_urls
  OUTPUT: AsyncIterator yielding (page_url, list of CandidatePost)
  ```
- **Assumptions**: base_url is valid, http_utils is initialized
- **Called By**: crawler_worker.py::process_site()

##### SelectorMiner.mine_posts_for_diabetes()

- **Signature**: `async def mine_posts_for_diabetes(self, html: str, base_url: str, site_id: str) -> List[CandidatePost]`
- **Pseudocode**:
  ```
  INPUT: html (string), base_url (string), site_id (string)
  MANIPULATION:
    - Parse HTML with BeautifulSoup
    - Detect page type (listing vs detail)
    - If listing:
      - Find post listing items
      - Extract post links
      - Navigate to each post and extract full content
    - If detail:
      - Extract full post and replies
    - Filter for diabetes keywords
    - Create CandidatePost objects
    - Return list of candidates
  OUTPUT: List of CandidatePost objects
  ```
- **Assumptions**: html is valid HTML string
- **Called By**: mine_posts_with_3x3_crawl(), \_fetch_post_page()

##### SelectorMiner.\_detect_page_type()

- **Signature**: `def _detect_page_type(self, soup: BeautifulSoup, url: Optional[str] = None) -> str`
- **Pseudocode**:
  ```
  INPUT: soup (BeautifulSoup), url (optional string)
  MANIPULATION:
    - Check URL patterns (detail vs listing indicators)
    - Check HTML structure (post containers, thread lists)
    - Return 'listing' or 'detail'
  OUTPUT: String 'listing' or 'detail'
  ```
- **Assumptions**: soup is valid BeautifulSoup object
- **Called By**: mine_posts_for_diabetes()

##### SelectorMiner.\_extract_full_post_and_replies()

- **Signature**: `async def _extract_full_post_and_replies(self, soup: BeautifulSoup, post_url: str, site_id: str) -> Optional[dict]`
- **Pseudocode**:
  ```
  INPUT: soup (BeautifulSoup), post_url (string), site_id (string)
  MANIPULATION:
    - Find main post content element
    - Extract title, author, date, content
    - Find reply elements
    - Combine post and replies into single content
    - Return post data dict
  OUTPUT: Dictionary with post data or None
  ```
- **Assumptions**: soup is valid BeautifulSoup object
- **Called By**: mine_posts_for_diabetes()

#### Functions

##### get_selector_miner()

- **Signature**: `def get_selector_miner() -> SelectorMiner`
- **Pseudocode**:
  ```
  INPUT: _selector_miner_instance (global singleton)
  MANIPULATION:
    - If _selector_miner_instance is None:
      - Create new SelectorMiner instance
      - Store in _selector_miner_instance
    - Return _selector_miner_instance
  OUTPUT: SelectorMiner singleton instance
  ```
- **Assumptions**: None
- **Called By**: crawler_worker.py

##### close_selector_miner()

- **Signature**: `def close_selector_miner() -> None`
- **Pseudocode**:
  ```
  INPUT: _selector_miner_instance (global singleton)
  MANIPULATION:
    - If _selector_miner_instance exists:
      - Close http_utils
      - Set _selector_miner_instance to None
  OUTPUT: None (cleans up resources)
  ```
- **Assumptions**: None
- **Called By**: main.py (on shutdown)

---

### 18. `orchestrator.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/orchestrator.py`
- **Lines**: 903
- **Scope**: System orchestration and coordination
- **Purpose**: The main coordinator of the crawler system. Starts and stops all worker processes, monitors queue depths for back-pressure, consumes final results, and aggregates system metrics.
- **Dependencies**:
  - `asyncio`, `logging`, `multiprocessing`, `signal`, `time`
  - `typing` (List, Dict, Any, Optional)
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.timing_logger` (get_timing_logger)
  - `.data_structures` (SiteTask, CrawlResults, SystemMetrics, ProcessingStats)
- **Used By**: main.py

#### Classes

##### Orchestrator

- **Purpose**: Main system orchestrator that coordinates all workers

#### Key Methods

##### Orchestrator.**init**()

- **Signature**: `def __init__(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Get config, redis_manager, timing_logger
    - Initialize worker process lists (crawlers, extractors, gpu_processors, storage)
    - Initialize site_stats dict
    - Set max_site_stats limit
    - Register signal handlers
    - Log system start
  OUTPUT: None (initializes instance)
  ```
- **Assumptions**: All managers are available
- **Called By**: main.py::run_crawl()

##### Orchestrator.start_workers()

- **Signature**: `def start_workers(self) -> None`
- **Pseudocode**:
  ```
  INPUT: self.config (worker counts)
  MANIPULATION:
    - Start crawler worker processes (nc_num_crawlers)
    - Start extractor worker processes (nc_num_extractors)
    - Start GPU processor worker processes (nc_num_gpu_processors)
    - Start storage worker processes (nc_num_storage_workers)
    - Store all processes in lists
    - Log worker startup
  OUTPUT: None (starts all worker processes)
  ```
- **Assumptions**: Config has valid worker counts
- **Called By**: crawl_sites()

##### Orchestrator.crawl_sites()

- **Signature**: `async def crawl_sites(self, sites: List[str]) -> CrawlResults`
- **Pseudocode**:
  ```
  INPUT: sites (list of site URLs)
  MANIPULATION:
    - Log crawl start
    - Push sites to queue via push_sites()
    - Start all workers via start_workers()
    - Start monitor_loop() and _consume_results() as background tasks
    - Wait for crawl completion (all queues empty, all sites processed)
    - Aggregate results from site_stats
    - Create CrawlResults object
    - Log crawl end
    - Return CrawlResults
  OUTPUT: CrawlResults object
  ```
- **Assumptions**: sites is non-empty list of valid URLs
- **Called By**: main.py::run_crawl()

##### Orchestrator.monitor_loop()

- **Signature**: `async def monitor_loop(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - While crawl not complete:
      - Monitor queue depths
      - Check back-pressure
      - Log metrics periodically
      - Sleep 5 seconds
    - Log monitoring complete
  OUTPUT: None (runs until crawl complete)
  ```
- **Assumptions**: None
- **Called By**: crawl_sites() (as background task)

##### Orchestrator.\_consume_results()

- **Signature**: `async def _consume_results(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - While crawl not complete:
      - Pop face results from results queue
      - Update site_stats with results
      - Sleep briefly
  OUTPUT: None (consumes results until crawl complete)
  ```
- **Assumptions**: None
- **Called By**: crawl_sites() (as background task)

##### Orchestrator.stop()

- **Signature**: `def stop(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Stop all worker processes (terminate)
    - Wait for processes to join
    - Log system shutdown
  OUTPUT: None (stops all workers)
  ```
- **Assumptions**: None
- **Called By**: main.py::run_crawl() (on completion or error)

##### Orchestrator.get_system_metrics()

- **Signature**: `def get_system_metrics(self) -> SystemMetrics`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Get queue metrics from Redis
    - Get site stats
    - Aggregate worker health
    - Create SystemMetrics object
    - Return metrics
  OUTPUT: SystemMetrics object
  ```
- **Assumptions**: Redis is accessible
- **Called By**: monitor_loop()

#### Functions

##### Orchestrator.\_run_crawler_worker()

- **Signature**: `@staticmethod def _run_crawler_worker(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Import crawler_worker module
    - Call crawler_worker_process(worker_id)
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: start_workers() (spawned as Process)

##### Orchestrator.\_run_extractor_worker()

- **Signature**: `@staticmethod def _run_extractor_worker(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Import extractor_worker module
    - Call extractor_worker_process(worker_id)
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: start_workers() (spawned as Process)

##### Orchestrator.\_run_gpu_processor_worker()

- **Signature**: `@staticmethod def _run_gpu_processor_worker(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Import gpu_processor_worker module
    - Call gpu_processor_worker_process(worker_id)
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: start_workers() (spawned as Process)

##### Orchestrator.\_run_storage_worker()

- **Signature**: `@staticmethod def _run_storage_worker(worker_id: int) -> None`
- **Pseudocode**:
  ```
  INPUT: worker_id (integer)
  MANIPULATION:
    - Import storage_worker module
    - Call storage_worker_process(worker_id)
  OUTPUT: None (process entry point)
  ```
- **Assumptions**: worker_id is valid integer
- **Called By**: start_workers() (spawned as Process)

---

### 19. `test_suite.py`

- **Path**: `diabetes-crawler/src/diabetes_crawler/test_suite.py`
- **Lines**: ~500 (estimated)
- **Scope**: Test suite for crawler components
- **Purpose**: Contains a test suite for various components of the crawler, including Redis, HTTP utilities, selector mining, and the crawler worker.
- **Dependencies**:
  - `asyncio`, `logging`
  - `.config` (get_config)
  - `.redis_manager` (get_redis_manager)
  - `.http_utils` (get_http_utils)
  - `.selector_miner` (get_selector_miner)
  - `.crawler_worker` (CrawlerWorker)
  - Various data structures
- **Used By**: Manual testing, development

#### Classes

##### TestSuite

- **Purpose**: Test suite for crawler components

#### Key Methods

##### TestSuite.run_tests()

- **Signature**: `async def run_tests(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Run test_crawler_worker_only() or full test suite
    - Test Redis operations
    - Test HTTP utilities
    - Test selector mining
    - Test crawler worker
    - Log test results
  OUTPUT: None (runs tests and logs results)
  ```
- **Assumptions**: All managers are initialized
- **Called By**: main() (if test mode)

##### TestSuite.test_crawler_worker_only()

- **Signature**: `async def test_crawler_worker_only(self) -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Create test site task
    - Create crawler worker
    - Process site
    - Verify candidates found
    - Log results
  OUTPUT: None (tests crawler worker)
  ```
- **Assumptions**: Test site is accessible
- **Called By**: run_tests()

#### Functions

##### main()

- **Signature**: `def main() -> None`
- **Pseudocode**:
  ```
  INPUT: None
  MANIPULATION:
    - Setup logging
    - Create TestSuite instance
    - Run tests via asyncio.run()
  OUTPUT: None (runs test suite)
  ```
- **Assumptions**: None
- **Called By**: Python interpreter (`if __name__ == "__main__":`)

---

## Summary

This `master_context.md` file provides comprehensive documentation for all 20 Python files in the `diabetes-crawler` project. Each file is documented with:

1. **File Path and Metadata**: Location, line count, scope, and purpose
2. **Dependencies**: All imports and external dependencies
3. **Used By**: Files that import or use this module
4. **Classes**: All classes with their purpose
5. **Methods**: Every method with:
   - Signature (function signature)
   - Pseudocode (inputs, manipulations, outputs)
   - Assumptions (data expectations)
   - Called By (callers of the method)

The documentation follows a tiered approach:

- **Tier 1**: Core data structures and configuration
- **Tier 2**: Utility and logging modules
- **Tier 3**: Infrastructure (HTTP, Redis, Storage, GPU interface)
- **Tier 4**: Worker processes (Crawler, Extractor, GPU Processor, Storage)
- **Tier 5**: High-level coordination (Selector Miner, Orchestrator, Main, Test Suite)

This comprehensive documentation enables developers to understand the entire system architecture, data flow, and inter-component dependencies.
