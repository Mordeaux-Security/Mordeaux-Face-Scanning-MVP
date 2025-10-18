"""
Crawler Settings and Constants

This module contains all configurable constants for the image crawler,
replacing magic numbers throughout the codebase with named constants.
"""

# ============================================================================
# Crawler knobs (IO/CPU)
# ============================================================================

# HTTP Concurrency Settings
HEAD_CONCURRENCY = 16          # Number of concurrent HEAD requests
GET_CONCURRENCY = 8            # Number of concurrent GET requests  
PER_HOST_LIMIT = 6             # Maximum concurrent requests per host
BATCH_SIZE = 64                # Default batch size for processing

# Image Size Limits
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB - maximum image file size
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB - maximum content length for downloads

# HTTP Timeouts
CONNECT_TIMEOUT_S = 5          # Connection timeout in seconds
READ_TIMEOUT_S = 10            # Read timeout in seconds
DEFAULT_TIMEOUT = 30           # Default HTTP timeout

# ============================================================================
# Face Detection Settings
# ============================================================================

# Face Detection Threading
FACE_THREADS = 4               # Number of threads for face detection (increased from 2)
FACE_MAX_WORKERS = 12          # Maximum workers for face detection thread pool (increased from 8)

# Face Size and Quality
FACE_MIN_SIZE_PX = 30          # Minimum face size in pixels
FACE_MIN_QUALITY = 0.5         # Minimum face detection quality score
FACE_MARGIN = 0.2              # Margin around face crops (20%)

# Face Deduplication
FACE_DUP_DIST_IMAGE = 0.35     # Dedupe threshold within a single image
FACE_DUP_DIST_ALBUM = 0.38     # Dedupe threshold across images in one album
FACE_STRONG_DETECTION_THRESHOLD = 0.8  # Threshold for early exit on strong detection

# Face Detection Scales
FACE_DETECTION_SCALES = [1.0, 2.0]  # Multi-scale detection scales (reduced from [1.0, 2.0, 4.0] for speed)

# ============================================================================
# Memory and Performance Settings
# ============================================================================

# Memory Management
MEMORY_PRESSURE_THRESHOLD = 75  # Memory pressure threshold percentage
MEMORY_CRITICAL_THRESHOLD = 85  # Critical memory threshold percentage
MEMORY_LOW_THRESHOLD = 60       # Low memory threshold percentage
MEMORY_MODERATE_THRESHOLD = 75  # Moderate memory threshold percentage
MEMORY_HIGH_THRESHOLD = 85      # High memory threshold percentage

# Garbage Collection
GC_FREQUENCY = 10              # Force GC every N operations
CPU_SAMPLE_FREQUENCY = 20      # Sample CPU every N operations

# CPU Performance Factors
CPU_HIGH_THRESHOLD = 80        # High CPU usage threshold
CPU_LOW_THRESHOLD = 30         # Low CPU usage threshold
CPU_HIGH_FACTOR = 0.5          # CPU factor when usage is high
CPU_LOW_FACTOR = 1.5           # CPU factor when usage is low
CPU_DEFAULT_FACTOR = 1.0       # Default CPU factor

# ============================================================================
# Concurrency and Limits
# ============================================================================

# Default Concurrency Settings
DEFAULT_MAX_CONCURRENT_IMAGES = 20  # Maximum concurrent image processing
DEFAULT_MAX_CONCURRENT_DOWNLOADS = 50  # Maximum concurrent downloads
DEFAULT_PER_HOST_CONCURRENCY = 3   # Default per-host concurrency
DEFAULT_MAX_CONCURRENCY_CAP = 30   # Maximum concurrency cap

# Queue Sizes
DOWNLOAD_QUEUE_SIZE = 20       # Download queue max size
PROCESSING_QUEUE_SIZE = 10     # Processing queue max size
STORAGE_QUEUE_SIZE = 10        # Storage queue max size

# Batch Processing
DEFAULT_BATCH_SIZE = 50        # Default batch size
SMALL_BATCH_SIZE = 5           # Small batch size for memory efficiency

# ============================================================================
# Crawl Policy Configuration
# ============================================================================

# Crawl Limits
DEFAULT_MAX_DEPTH = 1          # Default maximum crawl depth
DEFAULT_MAX_PAGES = 20         # Default maximum pages to crawl
DEFAULT_MAX_TOTAL_IMAGES = 50  # Default maximum total images
DEFAULT_MAX_REDIRECTS = 3      # Default maximum redirects

# Jitter and Timing
DEFAULT_JITTER_RANGE = (100, 400)  # Jitter range in milliseconds
STORAGE_TIMEOUT = 0.5          # Storage operation timeout

# ============================================================================
# Image Processing Settings
# ============================================================================

# Image Enhancement
IMAGE_ENHANCEMENT_LOW_RES_WIDTH = 500   # Low resolution width threshold
IMAGE_ENHANCEMENT_LOW_RES_HEIGHT = 400  # Low resolution height threshold
IMAGE_ENHANCEMENT_CONTRAST = 1.15       # Contrast enhancement factor
IMAGE_ENHANCEMENT_SHARPNESS = 1.1       # Sharpness enhancement factor
IMAGE_JPEG_QUALITY = 95                 # JPEG quality for saved images

# Thumbnail Settings
THUMBNAIL_SIZE = (150, 150)    # Default thumbnail size
THUMBNAIL_QUALITY = 95         # Thumbnail JPEG quality

# Image Safety
MAX_IMAGE_PIXELS = 50_000_000  # Maximum image pixels for PIL safety

# ============================================================================
# HTTP Client Settings
# ============================================================================

# HTTP Client Limits
HTTP_MAX_KEEPALIVE_CONNECTIONS = 200  # Maximum keepalive connections
HTTP_MAX_CONNECTIONS = 500            # Maximum total connections
HTTP_KEEPALIVE_EXPIRY = 30.0          # Keepalive expiry in seconds

# HTTP Retry Settings
HTTP_MAX_RETRIES = 3           # Maximum HTTP retries
HTTP_RETRY_DELAY = 1.0         # HTTP retry delay in seconds

# ============================================================================
# Logging and Debug Settings
# ============================================================================

# Logging
LOG_TRUNCATE_MAX_LENGTH = 120  # Maximum length for log string truncation
LOG_HASH_SUFFIX_LENGTH = 8     # Length of hash suffix in truncated logs

# ============================================================================
# Face Detection Model Settings
# ============================================================================

# InsightFace Model Settings
FACE_MODEL_NAME = "buffalo_l"  # InsightFace model name
FACE_DETECTION_SIZE = (640, 640)  # Face detection input size

# ============================================================================
# Image Selection and Filtering
# ============================================================================

# Image Selection
PREFERRED_IMAGE_WIDTH = 640    # Preferred width for srcset selection
SIMILARITY_THRESHOLD = 5       # Image similarity threshold
MAX_PREVIEW_IMAGES = 3         # Maximum images to preview

# Content Type Filtering
ALLOWED_CONTENT_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 
    'image/gif', 'image/webp', 'image/bmp'
}
BLOCKED_CONTENT_TYPES = {'image/svg+xml'}  # SVG's often are logos, not images

# ============================================================================
# Album/Gallery Processing Settings
# ============================================================================

# Album Detection
ALBUM_DETECTION_ENABLED = True          # Enable album-specific processing
ALBUM_MIN_IMAGES = 3                    # Minimum images to consider an album
ALBUM_MAX_IMAGES = 1000                 # Maximum images to process in an album

# Album Face Processing
ALBUM_FACE_QUALITY_THRESHOLD = 0.6      # Minimum face quality for album processing
ALBUM_SAVE_ALL_QUALITY_FACES = True     # Save all faces above threshold, not just best
ALBUM_FACE_DEDUPLICATION = True         # Enable face deduplication within albums

# Video Thumbnail Processing
VIDEO_THUMBNAIL_FACE_EXTRACTION = True  # Extract faces from video thumbnails
VIDEO_FACE_QUALITY_THRESHOLD = 0.5      # Minimum face quality for video thumbnails

# Album Metadata
ALBUM_METADATA_TRACKING = True          # Track album metadata and relationships
ALBUM_PERSON_DEDUPLICATION = True       # Deduplicate faces of same person across album

# ============================================================================
# JavaScript Rendering Settings
# ============================================================================

# JavaScript Rendering Configuration
JS_RENDERING_ENABLED = True              # Enable JavaScript rendering capabilities
JS_RENDERING_TIMEOUT = 15.0              # Timeout for JavaScript rendering in seconds (reduced from 30)
JS_RENDERING_WAIT_TIME = 0.5             # Wait time for page to stabilize after load (reduced from 2.0)
JS_RENDERING_MAX_CONCURRENT = 5          # Maximum concurrent JavaScript rendering sessions (increased from 3)
JS_RENDERING_HEADLESS = True             # Run browser in headless mode
JS_RENDERING_VIEWPORT_WIDTH = 1280       # Browser viewport width (reduced from 1920)
JS_RENDERING_VIEWPORT_HEIGHT = 720       # Browser viewport height (reduced from 1080)

# JavaScript Detection Settings
JS_DETECTION_ENABLED = True              # Enable automatic JavaScript detection
JS_DETECTION_KEYWORDS = [                # Keywords that suggest JavaScript-heavy content
    'react', 'vue', 'angular', 'spa', 'single-page',
    'lazy-load', 'infinite-scroll', 'dynamic-content'
]
JS_DETECTION_SCRIPT_THRESHOLD = 5        # Minimum number of script tags to trigger JS rendering
JS_DETECTION_FALLBACK_ENABLED = True     # Fallback to static HTML if JS rendering fails

# JavaScript Performance Settings
JS_RENDERING_MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB memory limit for browser instances
JS_RENDERING_CPU_LIMIT = 80              # CPU usage threshold to limit JS rendering
JS_RENDERING_CACHE_TTL = 300             # Cache rendered content for 5 minutes

# List Crawling Settings
LIST_CRAWL_DEFAULT_SITES_FILE = "sites.txt"     # Default file containing list of sites to crawl
LIST_CRAWL_MAX_PAGES_PER_SITE = 5               # Maximum pages to crawl per site in list mode
LIST_CRAWL_MAX_IMAGES_PER_SITE = 20             # Maximum images to save per site in list mode
LIST_CRAWL_AUTO_SELECTOR_MINING = True          # Automatically run selector miner for new sites
LIST_CRAWL_SKIP_EXISTING_RECIPES = True         # Skip selector mining if site recipe already exists

# ============================================================================
# Constants Not Yet Implemented
# ============================================================================

# The following constants are defined but not yet implemented in the codebase:
# - FACE_DUP_DIST_ALBUM: Face deduplication across images in one album (0.38)
# - FACE_DUP_DIST_IMAGE: Face deduplication within a single image (0.35) - partially implemented
# - HEAD_CONCURRENCY: Number of concurrent HEAD requests (16)
# - GET_CONCURRENCY: Number of concurrent GET requests (8)  
# - PER_HOST_LIMIT: Maximum concurrent requests per host (6)
# - BATCH_SIZE: Default batch size for processing (64)
# - CONNECT_TIMEOUT_S: Connection timeout in seconds (5)
# - READ_TIMEOUT_S: Read timeout in seconds (10)
# - FACE_MIN_SIZE_PX: Minimum face size in pixels (80)

# Note: The following constants are now implemented:
# - FACE_STRONG_DETECTION_THRESHOLD: Early exit threshold for strong face detection (0.8) ✓
# - FACE_DETECTION_SCALES: Multi-scale detection scales [1.0, 2.0, 4.0] ✓
# - HTTP_MAX_RETRIES: HTTP retry configuration (3) ✓
# - HTTP_RETRY_DELAY: HTTP retry delay (1.0) ✓
# - MAX_PREVIEW_IMAGES: Maximum images to preview in results (3) ✓
