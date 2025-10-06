from typing import Dict, Any, Optional
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class MordeauxError(Exception):
    """Base exception class for Mordeaux application errors."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(MordeauxError):
    """Raised when input validation fails."""
    pass

class AuthenticationError(MordeauxError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(MordeauxError):
    """Raised when authorization fails."""
    pass

class ResourceNotFoundError(MordeauxError):
    """Raised when a requested resource is not found."""
    pass

class RateLimitError(MordeauxError):
    """Raised when rate limit is exceeded."""
    pass

class StorageError(MordeauxError):
    """Raised when storage operations fail."""
    pass

class VectorDBError(MordeauxError):
    """Raised when vector database operations fail."""
    pass

class FaceProcessingError(MordeauxError):
    """Raised when face processing operations fail."""
    pass

class BatchProcessingError(MordeauxError):
    """Raised when batch processing operations fail."""
    pass

class CacheError(MordeauxError):
    """Raised when cache operations fail."""
    pass

# Error code definitions
ERROR_CODES = {
    # Validation errors (1000-1999)
    "INVALID_IMAGE_FORMAT": {
        "code": "INVALID_IMAGE_FORMAT",
        "message": "Invalid image format. Please upload a JPG or PNG image.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "IMAGE_TOO_LARGE": {
        "code": "IMAGE_TOO_LARGE",
        "message": "Image size exceeds the maximum allowed size of 10MB.",
        "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        "category": "validation"
    },
    "EMPTY_FILE": {
        "code": "EMPTY_FILE",
        "message": "No file provided or file is empty.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "MISSING_TENANT_ID": {
        "code": "MISSING_TENANT_ID",
        "message": "X-Tenant-ID header is required.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "INVALID_TENANT_ID": {
        "code": "INVALID_TENANT_ID",
        "message": "X-Tenant-ID must be at least 3 characters long.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "INVALID_BATCH_SIZE": {
        "code": "INVALID_BATCH_SIZE",
        "message": "Batch size cannot exceed 100 images.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "NO_IMAGE_URLS": {
        "code": "NO_IMAGE_URLS",
        "message": "No image URLs provided for batch processing.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "INVALID_WEBHOOK_EVENTS": {
        "code": "INVALID_WEBHOOK_EVENTS",
        "message": "Invalid webhook events provided.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "validation"
    },
    "WEBHOOK_NOT_FOUND": {
        "code": "WEBHOOK_NOT_FOUND",
        "message": "Webhook endpoint not found.",
        "http_status": status.HTTP_404_NOT_FOUND,
        "category": "not_found"
    },
    
    # Authentication/Authorization errors (2000-2999)
    "TENANT_ACCESS_DENIED": {
        "code": "TENANT_ACCESS_DENIED",
        "message": "Access denied to this resource.",
        "http_status": status.HTTP_403_FORBIDDEN,
        "category": "authorization"
    },
    "BATCH_ACCESS_DENIED": {
        "code": "BATCH_ACCESS_DENIED",
        "message": "Access denied to this batch job.",
        "http_status": status.HTTP_403_FORBIDDEN,
        "category": "authorization"
    },
    
    # Rate limiting errors (3000-3999)
    "RATE_LIMIT_EXCEEDED": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded. Please try again later.",
        "http_status": status.HTTP_429_TOO_MANY_REQUESTS,
        "category": "rate_limit"
    },
    
    # Resource not found errors (4000-4999)
    "BATCH_NOT_FOUND": {
        "code": "BATCH_NOT_FOUND",
        "message": "Batch job not found.",
        "http_status": status.HTTP_404_NOT_FOUND,
        "category": "not_found"
    },
    "IMAGE_NOT_FOUND": {
        "code": "IMAGE_NOT_FOUND",
        "message": "Image not found in storage.",
        "http_status": status.HTTP_404_NOT_FOUND,
        "category": "not_found"
    },
    
    # Storage errors (5000-5999)
    "STORAGE_UPLOAD_FAILED": {
        "code": "STORAGE_UPLOAD_FAILED",
        "message": "Failed to upload image to storage.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "storage"
    },
    "STORAGE_DOWNLOAD_FAILED": {
        "code": "STORAGE_DOWNLOAD_FAILED",
        "message": "Failed to download image from URL.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "storage"
    },
    "STORAGE_CONNECTION_FAILED": {
        "code": "STORAGE_CONNECTION_FAILED",
        "message": "Failed to connect to storage service.",
        "http_status": status.HTTP_503_SERVICE_UNAVAILABLE,
        "category": "storage"
    },
    
    # Vector database errors (6000-6999)
    "VECTOR_DB_CONNECTION_FAILED": {
        "code": "VECTOR_DB_CONNECTION_FAILED",
        "message": "Failed to connect to vector database.",
        "http_status": status.HTTP_503_SERVICE_UNAVAILABLE,
        "category": "vector_db"
    },
    "VECTOR_DB_UPSERT_FAILED": {
        "code": "VECTOR_DB_UPSERT_FAILED",
        "message": "Failed to store face embeddings in vector database.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "vector_db"
    },
    "VECTOR_DB_SEARCH_FAILED": {
        "code": "VECTOR_DB_SEARCH_FAILED",
        "message": "Failed to search for similar faces.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "vector_db"
    },
    
    # Face processing errors (7000-7999)
    "FACE_DETECTION_FAILED": {
        "code": "FACE_DETECTION_FAILED",
        "message": "Failed to detect faces in the image.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "face_processing"
    },
    "NO_FACES_DETECTED": {
        "code": "NO_FACES_DETECTED",
        "message": "No faces detected in the uploaded image.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "face_processing"
    },
    "FACE_EMBEDDING_FAILED": {
        "code": "FACE_EMBEDDING_FAILED",
        "message": "Failed to generate face embeddings.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "face_processing"
    },
    "PHASH_COMPUTATION_FAILED": {
        "code": "PHASH_COMPUTATION_FAILED",
        "message": "Failed to compute perceptual hash.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "face_processing"
    },
    
    # Batch processing errors (8000-8999)
    "BATCH_CREATION_FAILED": {
        "code": "BATCH_CREATION_FAILED",
        "message": "Failed to create batch processing job.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "batch_processing"
    },
    "BATCH_PROCESSING_FAILED": {
        "code": "BATCH_PROCESSING_FAILED",
        "message": "Batch processing failed.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "batch_processing"
    },
    "BATCH_CANCELLATION_FAILED": {
        "code": "BATCH_CANCELLATION_FAILED",
        "message": "Cannot cancel completed or failed batch job.",
        "http_status": status.HTTP_400_BAD_REQUEST,
        "category": "batch_processing"
    },
    
    # Cache errors (9000-9999)
    "CACHE_OPERATION_FAILED": {
        "code": "CACHE_OPERATION_FAILED",
        "message": "Cache operation failed.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "cache"
    },
    
    # System errors (10000+)
    "INTERNAL_SERVER_ERROR": {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An internal server error occurred.",
        "http_status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "category": "system"
    },
    "SERVICE_UNAVAILABLE": {
        "code": "SERVICE_UNAVAILABLE",
        "message": "Service is temporarily unavailable.",
        "http_status": status.HTTP_503_SERVICE_UNAVAILABLE,
        "category": "system"
    }
}

def create_http_exception(error_code: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
    """Create an HTTPException from an error code."""
    if error_code not in ERROR_CODES:
        error_code = "INTERNAL_SERVER_ERROR"
    
    error_info = ERROR_CODES[error_code]
    
    response_detail = {
        "error_code": error_info["code"],
        "message": error_info["message"],
        "category": error_info["category"]
    }
    
    if details:
        response_detail["details"] = details
    
    return HTTPException(
        status_code=error_info["http_status"],
        detail=response_detail
    )

def handle_mordeaux_error(error: MordeauxError) -> HTTPException:
    """Convert a MordeauxError to an HTTPException."""
    response_detail = {
        "error_code": error.error_code,
        "message": error.message,
        "details": error.details
    }
    
    # Get HTTP status from error code if available
    if error.error_code in ERROR_CODES:
        http_status = ERROR_CODES[error.error_code]["http_status"]
        response_detail["category"] = ERROR_CODES[error.error_code]["category"]
    else:
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        response_detail["category"] = "system"
    
    return HTTPException(
        status_code=http_status,
        detail=response_detail
    )

def handle_generic_error(error: Exception) -> HTTPException:
    """Handle generic exceptions and convert to HTTPException."""
    logger.error(f"Unhandled error: {error}", exc_info=True)
    
    return create_http_exception(
        "INTERNAL_SERVER_ERROR",
        {"original_error": str(error)}
    )

# Error response model for API documentation
class ErrorResponse:
    """Standard error response model."""
    
    def __init__(self, error_code: str, message: str, category: str, details: Optional[Dict[str, Any]] = None):
        self.error_code = error_code
        self.message = message
        self.category = category
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category,
            "details": self.details
        }
