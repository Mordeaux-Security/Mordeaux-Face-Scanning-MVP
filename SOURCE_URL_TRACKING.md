# Source URL Tracking Implementation

This document describes the implementation of source URL tracking for thumbnails in the crawler system.

## Overview

The crawler now tracks the source URL of each image when saving thumbnails using **S3/MinIO object tags**. This allows you to see where each image originally came from, which is useful for:

- **Object Tags**: Source URL appears in MinIO/S3 object tags (visible in web UI)
- **File Downloads**: When you download images, the source URL is preserved in the object tags
- **Debugging crawl issues**: Trace images back to their sources
- **Tracking image sources for compliance**: Know exactly where images came from

## Implementation Details

### 1. Storage Service Enhancements

The storage service has been updated to add source URL as S3/MinIO object tags:

#### `put_object()` function
```python
def put_object(bucket: str, key: str, data: bytes, content_type: str, tags: Optional[Dict[str, str]] = None) -> None:
    """Upload bytes to object storage with optional tags."""
    # ... existing code ...
    cli.put_object(
        bucket_name=bucket,
        object_name=key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
        tags=tags,  # <-- Source URL stored as tag
    )
```

#### Tag Implementation
When saving images, the source URL is added as a simple tag:
```python
tags = {"source_url": source_url} if source_url else None
put_object(bucket, key, image_bytes, "image/jpeg", tags)
```

### 2. Cache Service Enhancements

The cache service has been updated to store source URL information:

#### `store_crawled_image()` function
```python
async def store_crawled_image(self, url: str, image_bytes: bytes, raw_key: str, 
                            thumbnail_key: Optional[str] = None, tenant_id: str = "default", source_url: Optional[str] = None) -> bool:
    """Store crawled image metadata (not image bytes) in both Redis and PostgreSQL."""
    # ... existing code ...
    
    # Extract metadata only (no image bytes)
    metadata = {
        'hash': content_hash,
        'length': len(image_bytes),
        'mime': 'image/jpeg',
        'raw_key': raw_key,
        'thumb_key': thumbnail_key
    }
    
    # Add source URL if provided
    if source_url:
        metadata['source_url'] = source_url
```

### 3. Crawler Updates

The crawler now passes the source URL when saving images:

```python
# In _process_images_batch() and _store_single_item()
raw_key, raw_url, thumbnail_key, thumb_url, metadata = await save_raw_and_thumb_content_addressed_async(
    image_bytes, 
    thumbnail_bytes,
    self.tenant_id,
    image_info.url  # Pass source URL for tracking
)

# And when storing in cache
await self.cache_service.store_crawled_image(
    image_info.url,
    image_bytes,
    raw_key,
    thumbnail_key,
    self.tenant_id,
    image_info.url  # Pass source URL for tracking
)
```

## Object Tags Storage

The source URL is stored as a simple S3/MinIO object tag:

**Tag Key**: `source_url`  
**Tag Value**: The full source URL (e.g., `https://example.com/image.jpg`)

### Viewing Tags

**In MinIO Web UI:**
1. Go to your MinIO bucket
2. Click on any image file
3. Look at the "Tags" section - it will show:
   ```
   source_url: https://example.com/image.jpg
   ```

**When Downloading:**
- The tags are preserved when you download files from MinIO/S3
- You can view them using S3-compatible tools or the MinIO web interface

## Database Storage

The source URL is also stored in the existing `crawl_cache` table in the `metadata` JSONB column:

```sql
-- Query database for source URLs
SELECT 
    url_hash,
    raw_image_key,
    thumbnail_key,
    metadata->>'source_url' as source_url,
    processed_at
FROM crawl_cache 
WHERE metadata->>'source_url' IS NOT NULL;
```

## Usage Examples

### Querying Images by Source URL

```sql
-- Find all images from a specific domain
SELECT 
    raw_image_key,
    thumbnail_key,
    metadata->>'source_url' as source_url,
    processed_at
FROM crawl_cache 
WHERE metadata->>'source_url' LIKE '%example.com%';
```

### Getting Source URL for a Specific Image

```sql
-- Get source URL for a specific thumbnail
SELECT 
    metadata->>'source_url' as source_url,
    processed_at
FROM crawl_cache 
WHERE thumbnail_key = 'tenant/ab/abc123_thumb.jpg';
```

### Redis Cache Queries

The source URL is also stored in Redis cache entries:

```python
# Redis key format: crawl:{tenant_id}:{url_hash}
# Cache data includes metadata with source_url
{
    'should_skip': True,
    'raw_key': 'tenant/ab/abc123.jpg',
    'thumbnail_key': 'tenant/ab/abc123_thumb.jpg',
    'metadata': {
        'hash': 'abc123...',
        'length': 45678,
        'mime': 'image/jpeg',
        'raw_key': 'tenant/ab/abc123.jpg',
        'thumb_key': 'tenant/ab/abc123_thumb.jpg',
        'source_url': 'https://example.com/image.jpg'  # <-- New field
    },
    'cached_at': 1234567890.123
}
```

## Backward Compatibility

- All `source_url` parameters are optional with `None` as default
- Existing code will continue to work without modification
- Images crawled before this update will not have source URL information
- The metadata structure is extended, not changed

## Testing

To test the implementation:

1. **Manual Testing**: Run a crawl and check the database for source URL metadata
2. **Database Verification**: Query the `crawl_cache` table to see source URLs in metadata
3. **Cache Verification**: Check Redis cache entries for source URL information

### Example Test Query

```sql
-- Check that source URLs are being stored
SELECT 
    COUNT(*) as total_images,
    COUNT(metadata->>'source_url') as images_with_source_url,
    ROUND(
        COUNT(metadata->>'source_url')::numeric / COUNT(*)::numeric * 100, 
        2
    ) as percentage_with_source_url
FROM crawl_cache 
WHERE processed_at > NOW() - INTERVAL '1 hour';
```

## Benefits

1. **Traceability**: Know exactly where each image came from
2. **Debugging**: Easily identify problematic source websites
3. **Compliance**: Track image sources for legal/compliance purposes
4. **Analytics**: Analyze which websites are most productive for image discovery
5. **Quality Control**: Identify and potentially blacklist low-quality image sources

## Future Enhancements

Potential future improvements could include:

1. **Source URL Analytics**: Dashboard showing most productive image sources
2. **Source URL Filtering**: Ability to crawl only specific domains
3. **Source URL Blacklisting**: Automatically skip known problematic domains
4. **Source URL Reporting**: Generate reports of image sources by tenant
5. **Source URL Validation**: Validate that source URLs are still accessible

## Migration Notes

No database migration is required as the existing `metadata` JSONB column in the `crawl_cache` table already supports storing the source URL information.
