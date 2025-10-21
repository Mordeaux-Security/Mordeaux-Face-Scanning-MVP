# API v0.1 Contract Summary

## 🎯 **STABLE API CONTRACT - FROZEN FOR INTEGRATION**

**Date**: October 20, 2025  
**Version**: v0.1  
**Status**: ✅ **FROZEN AND STABLE**  

This document summarizes the stable API v0.1 contract that is now frozen for safe integration by teams A and B.

---

## 📋 **API Endpoints (FROZEN)**

### 1. **POST** `/api/v1/search`
**Purpose**: Search for similar faces by image or vector  
**Status**: ✅ **STABLE**

**Request Options**:
- **Image Upload**: `multipart/form-data` with image file
- **Vector Search**: `application/json` with embedding vector

**Response Format**:
```json
{
  "query": {
    "tenant_id": "tenant-123",
    "search_mode": "image",
    "top_k": 10,
    "threshold": 0.75
  },
  "hits": [
    {
      "face_id": "face-uuid-123",
      "score": 0.95,
      "payload": {
        "site": "example.com",
        "url": "https://example.com/image.jpg",
        "ts": "2024-01-01T12:00:00Z",
        "bbox": [100, 150, 200, 250],
        "p_hash": "b2c3d4e5f6g7h8i9",
        "quality": 0.92
      },
      "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=..."
    }
  ],
  "count": 1
}
```

### 2. **GET** `/api/v1/faces/{face_id}`
**Purpose**: Retrieve face details by ID  
**Status**: ✅ **STABLE**

**Response Format**:
```json
{
  "face_id": "face-uuid-123",
  "payload": {
    "site": "example.com",
    "url": "https://example.com/image.jpg",
    "ts": "2024-01-01T12:00:00Z",
    "bbox": [100, 150, 200, 250],
    "p_hash": "b2c3d4e5f6g7h8i9",
    "quality": 0.92
  },
  "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=..."
}
```

### 3. **GET** `/api/v1/stats`
**Purpose**: Get pipeline statistics  
**Status**: ✅ **STABLE**

**Response Format**:
```json
{
  "processed": 1250,
  "rejected": 45,
  "dup_skipped": 12
}
```

### 4. **GET** `/api/v1/health`
**Purpose**: API health check  
**Status**: ✅ **STABLE**

**Response Format**:
```json
{
  "status": "healthy",
  "service": "face-pipeline-search-api",
  "version": "0.1.0-dev2",
  "api_version": "v0.1",
  "note": "API v0.1 contract frozen - all endpoints stable"
}
```

---

## 🔒 **Security Features**

### **Presigned URLs**
- **TTL**: 10 minutes maximum
- **Format**: HTTPS URLs with authentication parameters
- **Security**: No raw object URLs exposed
- **Thumbnail Size**: 256px longest side

### **Metadata Filtering**
Only the following fields are exposed in API responses:
- ✅ `site` - Source website domain
- ✅ `url` - Original image URL
- ✅ `ts` - Timestamp when processed
- ✅ `bbox` - Face bounding box coordinates
- ✅ `p_hash` - Perceptual hash
- ✅ `quality` - Image quality score

**Forbidden Fields** (never exposed):
- ❌ `raw_url` - Raw image URLs
- ❌ `raw_key` - Internal storage keys
- ❌ `det_score` - Face detection scores
- ❌ `embedding` - Face embedding vectors

### **Multi-tenant Isolation**
- **Header Required**: `X-Tenant-ID` for all requests
- **Data Isolation**: All operations scoped by tenant
- **Storage Isolation**: Object keys prefixed with tenant ID
- **Vector Isolation**: Search results filtered by tenant

---

## 📚 **Documentation**

### **OpenAPI Specification**
- **File**: `api/openapi.yaml`
- **Format**: OpenAPI 3.0.3
- **Coverage**: Complete specification for all endpoints
- **Examples**: Comprehensive mock JSON examples

### **HTML Documentation**
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI JSON**: `/openapi.json`

### **Version Information**
- **Header**: `X-API-Version: v0.1` in all responses
- **Contract**: v0.1 is frozen and stable
- **Compatibility**: No breaking changes will be made

---

## 🧪 **Testing Examples**

### **Search by Image**
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "X-Tenant-ID: tenant-123" \
  -F "image=@test-image.jpg" \
  -F "top_k=10" \
  -F "threshold=0.75"
```

### **Search by Vector**
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "X-Tenant-ID: tenant-123" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "tenant_id": "tenant-123",
    "top_k": 10,
    "threshold": 0.75
  }'
```

### **Get Face Details**
```bash
curl -X GET "http://localhost:8000/api/v1/faces/face-uuid-123" \
  -H "X-Tenant-ID: tenant-123"
```

### **Get Statistics**
```bash
curl -X GET "http://localhost:8000/api/v1/stats" \
  -H "X-Tenant-ID: tenant-123"
```

### **Health Check**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

---

## 📊 **Response Headers**

All API responses include:
- **X-API-Version**: `v0.1` (API contract version)
- **Content-Type**: `application/json`
- **X-Request-ID**: Request correlation ID (if provided)

---

## 🎯 **Integration Guidelines**

### **For Teams A & B**

1. **Use Stable Endpoints**: All `/api/v1/*` endpoints are frozen
2. **Include Tenant ID**: Always include `X-Tenant-ID` header
3. **Handle Presigned URLs**: Thumbnail URLs expire after 10 minutes
4. **Check Version Header**: Verify `X-API-Version: v0.1` in responses
5. **Follow Error Format**: Standardized error responses with proper HTTP codes

### **Error Handling**
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid request parameters",
  "timestamp": "2024-01-01T12:00:00Z",
  "details": {
    "field": "threshold",
    "issue": "must be between 0.0 and 1.0"
  }
}
```

---

## ✅ **Contract Guarantees**

### **Stability**
- ✅ No breaking changes to v0.1 endpoints
- ✅ Response format will not change
- ✅ Field names and types are frozen
- ✅ HTTP status codes are standardized

### **Security**
- ✅ Presigned URLs with TTL enforcement
- ✅ Metadata filtering prevents data exposure
- ✅ Tenant isolation is guaranteed
- ✅ No internal system details exposed

### **Documentation**
- ✅ Complete OpenAPI specification
- ✅ Comprehensive mock examples
- ✅ Version information in all responses
- ✅ HTML documentation auto-generated

---

## 🎉 **Ready for Integration**

The API v0.1 contract is now **production-ready** and **integration-safe**:

- **✅ Stable Contract**: No breaking changes will be made
- **✅ Comprehensive Docs**: Full OpenAPI specification with examples
- **✅ Security Compliant**: Implements presigned URL policy
- **✅ Multi-tenant Ready**: Complete tenant isolation
- **✅ Version Controlled**: Proper versioning with headers
- **✅ Error Handling**: Standardized error responses

**Teams A and B can now safely integrate using the frozen v0.1 API contract!** 🚀
