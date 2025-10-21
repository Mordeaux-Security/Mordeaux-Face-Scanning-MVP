"""
Storage Operations Module

Handles all storage operations including image storage, metadata management,
and site recipe storage. Provides unified interface for MinIO/S3 operations.
"""

import asyncio
import json
import logging
import os
import time
import uuid
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image

from .config import CrawlerConfig

logger = logging.getLogger(__name__)


def _create_video_url_sha256(url: str) -> str:
    """Create SHA256 hash of video URL for MinIO header."""
    if not url:
        return ""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def _create_video_url_head(url: str, max_length: int = 50) -> str:
    """Create truncated video URL head for MinIO header."""
    if not url:
        return ""
    if len(url) <= max_length:
        return url
    return url[:max_length-3] + "..."


def _create_minio_headers(doc_id: str, video_url: str) -> Dict[str, str]:
    """Create small MinIO user metadata headers."""
    return {
        'doc-id': doc_id,
        'video-url-sha256': _create_video_url_sha256(video_url),
        'video-url-head': _create_video_url_head(video_url)
    }


@dataclass
class StorageMetadata:
    """Metadata for stored images."""
    image_id: str
    original_url: str
    source_url: str
    file_size: int
    dimensions: Tuple[int, int]
    perceptual_hash: str
    faces_detected: int
    enhancement_applied: bool
    storage_timestamp: str
    storage_bucket: str
    storage_key: str
    thumbnail_key: Optional[str] = None
    face_embeddings: Optional[List[List[float]]] = None
    face_bboxes: Optional[List[List[float]]] = None
    face_scores: Optional[List[float]] = None


@dataclass
class StorageResult:
    """Result of storage operation."""
    success: bool
    image_key: str
    thumbnail_keys: List[str] = field(default_factory=list)
    metadata_key: str = ""
    image_url: str = ""
    thumbnail_urls: List[str] = field(default_factory=list)
    file_size: int = 0
    thumbnail_count: int = 0
    error: Optional[str] = None


class StorageManager:
    """
    Manages storage operations for images, metadata, and site recipes.
    
    Provides unified interface for storage operations with support for
    MinIO/S3 backends and comprehensive metadata management.
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.client = None
        self.bucket_name = config.storage_bucket
        self.region = config.storage_region
        
        # Storage statistics
        self.stats = {
            'images_stored': 0,
            'thumbnails_stored': 0,
            'metadata_stored': 0,
            'recipes_stored': 0,
            'storage_time': 0.0,
            'errors': 0
        }
        
        # Recipe cache
        self._recipe_cache: Dict[str, Dict[str, Any]] = {}
        self._recipe_cache_file = "site_recipes.yaml"
    
    async def initialize(self) -> None:
        """Initialize storage client and ensure bucket exists."""
        try:
            # Initialize MinIO client
            from minio import Minio
            from minio.error import S3Error
            
            # Get credentials from environment
            endpoint = os.getenv('S3_ENDPOINT', os.getenv('MINIO_ENDPOINT', 'localhost:9000'))
            # Remove protocol and path from endpoint for MinIO client
            if endpoint.startswith('http://'):
                endpoint = endpoint[7:]
            elif endpoint.startswith('https://'):
                endpoint = endpoint[8:]
            # Remove any path components
            endpoint = endpoint.split('/')[0]
            
            access_key = os.getenv('S3_ACCESS_KEY', os.getenv('MINIO_ACCESS_KEY', 'minioadmin'))
            secret_key = os.getenv('S3_SECRET_KEY', os.getenv('MINIO_SECRET_KEY', 'minioadmin'))
            secure = os.getenv('S3_USE_SSL', os.getenv('MINIO_SECURE', 'false')).lower() == 'true'
            
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name, location=self.region)
                logger.info(f"Created storage bucket: {self.bucket_name}")
            
            logger.info(f"Storage manager initialized with bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage manager: {e}")
            raise
    
    async def store_image_with_multiple_thumbnails(
        self, 
        image_data: bytes, 
        thumbnail_data_list: List[bytes], 
        metadata: Dict[str, Any],
        site: str = "",
        page_url: str = "",
        source_video_url: str = "",
        source_image_url: str = "",
        selector_source: str = ""
    ) -> StorageResult:
        """Store image with multiple thumbnails (one per face)."""
        start_time = time.time()
        
        try:
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            
            # Generate storage keys using hash-based folders
            hash_prefix = image_id[:2]  # First 2 characters of hash
            image_key = f"images/{hash_prefix}/{image_id}.jpg"
            metadata_key = f"metadata/{hash_prefix}/{image_id}.json"
            
            # Create small MinIO headers for image
            minio_headers = _create_minio_headers(image_id, source_video_url)
            
            # Store main image with MinIO headers
            await self._store_object(image_key, image_data, 'image/jpeg', minio_headers)
            
            # Store multiple thumbnails with MinIO headers
            thumbnail_keys = []
            for i, thumbnail_data in enumerate(thumbnail_data_list):
                if thumbnail_data:
                    thumbnail_key = f"thumbnails/{hash_prefix}/{image_id}_face_{i}.jpg"
                    await self._store_object(thumbnail_key, thumbnail_data, 'image/jpeg', minio_headers)
                    thumbnail_keys.append(thumbnail_key)
            
            # Store comprehensive metadata in sidecar JSON (full URLs included)
            enhanced_metadata = {
                **metadata,  # Include existing metadata
                'thumbnail_keys': thumbnail_keys,
                'thumbnail_count': len(thumbnail_keys),
                'site': site,
                'page_url': page_url,
                'source_video_url': source_video_url,  # Full URL in JSON
                'source_image_url': source_image_url,  # Full URL in JSON
                'selector_source': selector_source,
                'image_id': image_id,
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'storage_bucket': self.bucket_name,
                # Add MinIO header values for reference
                'minio_headers': {
                    'doc_id': image_id,
                    'video_url_sha256': minio_headers['video-url-sha256'],
                    'video_url_head': minio_headers['video-url-head']
                }
            }
            await self._store_object(metadata_key, json.dumps(enhanced_metadata, indent=2).encode('utf-8'), 'application/json')
            
            # Generate URLs
            base_url = str(self.client._base_url)
            if base_url.startswith('http://'):
                base_url = base_url[7:]  # Remove http://
            elif base_url.startswith('https://'):
                base_url = base_url[8:]  # Remove https://

            image_url = f"http://{base_url}/{self.bucket_name}/{image_key}"
            thumbnail_urls = [f"http://{base_url}/{self.bucket_name}/{key}" for key in thumbnail_keys]
            
            return StorageResult(
                success=True,
                image_key=image_key,
                thumbnail_keys=thumbnail_keys,
                metadata_key=metadata_key,
                image_url=image_url,
                thumbnail_urls=thumbnail_urls,
                file_size=len(image_data),
                thumbnail_count=len(thumbnail_keys)
            )
            
        except Exception as e:
            logger.error(f"Error storing image with multiple thumbnails: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                file_size=len(image_data),
                thumbnail_count=0
            )
    
    async def store_image(
        self, 
        image_data: bytes, 
        thumbnail_data: Optional[bytes],
        metadata: Dict[str, Any]
    ) -> StorageResult:
        """
        Store image and thumbnail with metadata.
        
        Args:
            image_data: Image data bytes
            thumbnail_data: Thumbnail data bytes (optional)
            metadata: Image metadata
            
        Returns:
            StorageResult with storage information
        """
        start_time = time.time()
        
        try:
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            
            # Generate storage keys using hash-based folders
            hash_prefix = image_id[:2]  # First 2 characters of hash
            image_key = f"images/{hash_prefix}/{image_id}.jpg"
            thumbnail_key = f"thumbnails/{hash_prefix}/{image_id}.jpg" if thumbnail_data else None
            metadata_key = f"metadata/{hash_prefix}/{image_id}.json"
            
            # Store image
            await self._store_object(image_key, image_data, 'image/jpeg')
            
            # Store thumbnail if provided
            if thumbnail_data:
                await self._store_object(thumbnail_key, thumbnail_data, 'image/jpeg')
            
            # Create storage metadata
            storage_metadata = StorageMetadata(
                image_id=image_id,
                original_url=metadata.get('original_url', ''),
                source_url=metadata.get('source_url', ''),
                file_size=len(image_data),
                dimensions=metadata.get('dimensions', (0, 0)),
                perceptual_hash=metadata.get('perceptual_hash', ''),
                faces_detected=metadata.get('faces_detected', 0),
                enhancement_applied=metadata.get('enhancement_applied', False),
                storage_timestamp=datetime.now(timezone.utc).isoformat(),
                storage_bucket=self.bucket_name,
                storage_key=image_key,
                thumbnail_key=thumbnail_key,
                face_embeddings=metadata.get('face_embeddings'),
                face_bboxes=metadata.get('face_bboxes'),
                face_scores=metadata.get('face_scores')
            )
            
            # Store metadata
            metadata_json = json.dumps(asdict(storage_metadata), indent=2)
            await self._store_object(metadata_key, metadata_json.encode('utf-8'), 'application/json')
            
            # Generate URLs
            base_url = str(self.client._base_url)
            if base_url.startswith('http://'):
                base_url = base_url[7:]  # Remove http://
            elif base_url.startswith('https://'):
                base_url = base_url[8:]  # Remove https://
            
            image_url = f"http://{base_url}/{self.bucket_name}/{image_key}"
            thumbnail_url = f"http://{base_url}/{self.bucket_name}/{thumbnail_key}" if thumbnail_key else None
            
            # Update statistics
            storage_time = time.time() - start_time
            self.stats['images_stored'] += 1
            if thumbnail_data:
                self.stats['thumbnails_stored'] += 1
            self.stats['metadata_stored'] += 1
            self.stats['storage_time'] += storage_time
            
            logger.debug(f"Stored image {image_id} in {storage_time:.3f}s")
            
            return StorageResult(
                success=True,
                image_id=image_id,
                storage_key=image_key,
                storage_url=image_url,
                thumbnail_key=thumbnail_key,
                thumbnail_url=thumbnail_url,
                metadata=storage_metadata
            )
            
        except Exception as e:
            logger.error(f"Error storing image: {e}")
            self.stats['errors'] += 1
            return StorageResult(
                success=False,
                image_id="",
                storage_key="",
                storage_url="",
                error=str(e)
            )
    
    async def _store_object(self, key: str, data: bytes, content_type: str, metadata: Optional[Dict[str, str]] = None) -> None:
        """Store object in storage backend with optional MinIO user metadata."""
        try:
            from io import BytesIO
            
            data_stream = BytesIO(data)
            
            # Convert metadata to MinIO format (x-amz-meta-* headers)
            minio_metadata = {}
            if metadata:
                for k, v in metadata.items():
                    minio_metadata[f'x-amz-meta-{k}'] = v
            
            self.client.put_object(
                self.bucket_name,
                key,
                data_stream,
                len(data),
                content_type=content_type,
                metadata=minio_metadata
            )
            
        except Exception as e:
            logger.error(f"Error storing object {key}: {e}")
            raise
    
    async def get_image(self, image_id: str) -> Optional[bytes]:
        """Retrieve image data by ID."""
        try:
            # Find image key from metadata
            metadata = await self.get_image_metadata(image_id)
            if not metadata:
                return None
            
            # Retrieve image data
            response = self.client.get_object(self.bucket_name, metadata.storage_key)
            return response.read()
            
        except Exception as e:
            logger.error(f"Error retrieving image {image_id}: {e}")
            return None
    
    async def get_image_metadata(self, image_id: str) -> Optional[StorageMetadata]:
        """Retrieve image metadata by ID."""
        try:
            # Search for metadata file
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=f"metadata/",
                recursive=True
            )
            
            for obj in objects:
                if obj.object_name.endswith(f"{image_id}.json"):
                    # Retrieve metadata
                    response = self.client.get_object(self.bucket_name, obj.object_name)
                    metadata_json = response.read().decode('utf-8')
                    metadata_dict = json.loads(metadata_json)
                    
                    return StorageMetadata(**metadata_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving metadata for {image_id}: {e}")
            return None
    
    async def save_recipe(self, url: str, recipe: Dict[str, Any]) -> bool:
        """Save site recipe to storage."""
        try:
            from urllib.parse import urlparse
            
            # Extract domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Load existing recipes
            recipes = await self._load_recipes()
            
            # Update recipes
            if 'sites' not in recipes:
                recipes['sites'] = {}
            
            recipes['sites'][domain] = recipe
            
            # Save recipes
            await self._save_recipes(recipes)
            
            # Update cache
            self._recipe_cache[domain] = recipe
            
            self.stats['recipes_stored'] += 1
            logger.info(f"Saved recipe for domain: {domain}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving recipe for {url}: {e}")
            return False
    
    def get_recipe_for_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get existing recipe for URL."""
        try:
            from urllib.parse import urlparse
            
            # Extract domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Check cache first
            if domain in self._recipe_cache:
                return self._recipe_cache[domain]
            
            # Load from file
            recipes = self._load_recipes_sync()
            if recipes and 'sites' in recipes:
                recipe = recipes['sites'].get(domain)
                if recipe:
                    self._recipe_cache[domain] = recipe
                    return recipe
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recipe for {url}: {e}")
            return None
    
    async def _load_recipes(self) -> Dict[str, Any]:
        """Load recipes from storage."""
        try:
            # Try to load from file first
            if os.path.exists(self._recipe_cache_file):
                with open(self._recipe_cache_file, 'r', encoding='utf-8') as f:
                    import yaml
                    return yaml.safe_load(f) or {}
            
            # Return default structure
            return {
                'schema_version': 2,
                'defaults': self._get_default_recipe(),
                'sites': {}
            }
            
        except Exception as e:
            logger.error(f"Error loading recipes: {e}")
            return {
                'schema_version': 2,
                'defaults': self._get_default_recipe(),
                'sites': {}
            }
    
    def _load_recipes_sync(self) -> Dict[str, Any]:
        """Load recipes synchronously."""
        try:
            if os.path.exists(self._recipe_cache_file):
                with open(self._recipe_cache_file, 'r', encoding='utf-8') as f:
                    import yaml
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            logger.error(f"Error loading recipes sync: {e}")
            return {}
    
    async def _save_recipes(self, recipes: Dict[str, Any]) -> None:
        """Save recipes to storage with canonical formatting."""
        try:
            import yaml
            from collections import OrderedDict
            
            # Create canonical YAML with fixed key order
            canonical_recipes = self._create_canonical_recipes(recipes)
            
            # Save to file with canonical formatting (convert OrderedDict to regular dict for clean YAML)
            clean_recipes = self._convert_ordered_dict_to_dict(canonical_recipes)
            
            with open(self._recipe_cache_file, 'w', encoding='utf-8') as f:
                yaml.dump(clean_recipes, f, 
                         default_flow_style=False,
                         sort_keys=False,  # Maintain our custom order
                         allow_unicode=True,
                         width=120,
                         indent=2)
            
            # Also save to storage backend if available
            if self.client:
                recipes_json = json.dumps(canonical_recipes, indent=2)
                await self._store_object("recipes/site_recipes.json", recipes_json.encode('utf-8'), 'application/json')
            
        except Exception as e:
            logger.error(f"Error saving recipes: {e}")
            raise
    
    def _create_canonical_recipes(self, recipes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create canonical recipes with fixed key order and normalized attributes.
        
        Args:
            recipes: Original recipes dictionary
            
        Returns:
            Canonical recipes dictionary with consistent formatting
        """
        from collections import OrderedDict
        
        # Canonical key order for site recipes
        YAML_KEY_ORDER = ["selectors", "attributes_priority", "extra_sources", "method", "confidence"]
        CANONICAL_ATTRIBUTES = ["data-src", "data-srcset", "srcset", "src"]
        EXTRA_SOURCES_PATTERNS = [
            "meta[property='og:image']::attr(content)",
            "link[rel='image_src']::attr(href)",
            "video::attr(poster)",
            "script[type='application/ld+json']",
            "[style*='background-image']"
        ]
        
        # Create ordered dictionary to maintain key order
        canonical_recipes = OrderedDict()
        
        # Add schema version first
        canonical_recipes['schema_version'] = recipes.get('schema_version', 2)
        
        # Add defaults with canonical order
        if 'defaults' in recipes:
            canonical_defaults = OrderedDict()
            defaults = recipes['defaults']
            
            # Add defaults keys in canonical order
            for key in YAML_KEY_ORDER:
                if key in defaults:
                    if key == "attributes_priority":
                        # Normalize attributes to canonical order
                        canonical_defaults[key] = CANONICAL_ATTRIBUTES
                    elif key == "extra_sources":
                        # Use canonical extra sources patterns
                        canonical_defaults[key] = EXTRA_SOURCES_PATTERNS
                    else:
                        canonical_defaults[key] = defaults[key]
            
            canonical_recipes['defaults'] = canonical_defaults
        
        # Add sites with sorted domain names for consistency
        if 'sites' in recipes:
            canonical_sites = OrderedDict()
            for domain in sorted(recipes['sites'].keys()):
                site_recipe = recipes['sites'][domain]
                canonical_site = OrderedDict()
                
                # Add site keys in canonical order
                for key in YAML_KEY_ORDER:
                    if key in site_recipe:
                        if key == "attributes_priority":
                            # Normalize attributes to canonical order
                            canonical_site[key] = CANONICAL_ATTRIBUTES
                        elif key == "extra_sources":
                            # Use canonical extra sources patterns
                            canonical_site[key] = EXTRA_SOURCES_PATTERNS
                        elif key == "selectors":
                            # Sort selectors by score for consistency
                            selectors = site_recipe[key]
                            if isinstance(selectors, list):
                                # Sort by score if available, otherwise by CSS selector
                                sorted_selectors = sorted(selectors, 
                                                        key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, 
                                                        reverse=True)
                                canonical_site[key] = sorted_selectors
                            else:
                                canonical_site[key] = selectors
                        else:
                            canonical_site[key] = site_recipe[key]
                
                canonical_sites[domain] = canonical_site
            
            canonical_recipes['sites'] = canonical_sites
        
        return canonical_recipes
    
    def _convert_ordered_dict_to_dict(self, obj):
        """Convert OrderedDict objects to regular dicts for clean YAML output."""
        if isinstance(obj, dict):
            return {key: self._convert_ordered_dict_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_ordered_dict_to_dict(item) for item in obj]
        else:
            return obj
    
    def _get_default_recipe(self) -> Dict[str, Any]:
        """Get default recipe configuration with canonical formatting."""
        return {
            "selectors": [
                {"kind": "video_grid", "css": "img"},
                {"kind": "video_grid", "css": ".thumbnail img"},
                {"kind": "video_grid", "css": ".thumb img"}
            ],
            "attributes_priority": ["data-src", "data-srcset", "srcset", "src"],
            "extra_sources": [
                "meta[property='og:image']::attr(content)",
                "link[rel='image_src']::attr(href)",
                "video::attr(poster)",
                "script[type='application/ld+json']",
                "[style*='background-image']"
            ],
            "method": "smart"
        }
    
    async def cleanup(self) -> None:
        """Cleanup storage resources."""
        try:
            if self.client:
                # Close any open connections
                pass
            
            logger.info("Storage manager cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during storage cleanup: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset storage statistics."""
        self.stats = {
            'images_stored': 0,
            'thumbnails_stored': 0,
            'metadata_stored': 0,
            'recipes_stored': 0,
            'storage_time': 0.0,
            'errors': 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on storage system."""
        try:
            start_time = time.time()
            
            # Test bucket access
            bucket_exists = self.client.bucket_exists(self.bucket_name)
            
            # Test object listing
            objects = list(self.client.list_objects(self.bucket_name, max_keys=1))
            
            response_time = time.time() - start_time
            
            return {
                'healthy': True,
                'bucket_exists': bucket_exists,
                'can_list_objects': True,
                'response_time': response_time,
                'bucket_name': self.bucket_name,
                'region': self.region
            }
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'bucket_name': self.bucket_name,
                'region': self.region
            }
