# Quick Deployment Guide

**Simplified deployment guide for verification-first flow**

---

## 🚀 Quick Start

### 1. Pre-Deployment Checks

```bash
# Verify environment variables are set
cd face-pipeline
cat .env | grep -E "(IDENTITY_COLLECTION|VERIFY_HI_THRESHOLD|QDRANT)"

# Should show:
# IDENTITY_COLLECTION=identities_v1
# VERIFY_HI_THRESHOLD=0.78
# QDRANT_URL=http://qdrant:6333
# QDRANT_COLLECTION=faces_v1
```

### 2. Deploy Code

**Option A: Docker**
```bash
docker-compose build face-pipeline
docker-compose up -d face-pipeline
```

**Option B: Kubernetes**
```bash
kubectl set image deployment/face-pipeline face-pipeline=registry/face-pipeline:latest
kubectl rollout status deployment/face-pipeline
```

**Option C: Direct**
```bash
systemctl restart face-pipeline
```

### 3. Verify Deployment

**Using Python script:**
```bash
python scripts/deploy.py dev        # For dev
python scripts/deploy.py staging    # For staging
python scripts/deploy.py prod        # For production
```

**Or manually:**
```bash
# Check health
curl http://localhost:8001/api/v1/health

# Check collections
curl http://qdrant:6333/collections | jq '.result.collections[]?.name'

# Should show: faces_v1 and identities_v1
```

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] Health check passes: `curl http://localhost:8001/api/v1/health`
- [ ] Collections exist: `identities_v1` and `faces_v1`
- [ ] Enrollment works: `POST /api/v1/enroll_identity` (returns 200 or 422)
- [ ] Verification works: `POST /api/v1/verify` (returns 200, 404, or 422)
- [ ] Search still works: `POST /api/v1/search` (backward compatibility)
- [ ] No errors in logs: `docker logs face-pipeline | grep -i error`

---

## 🧪 Quick Test

```bash
# Test enrollment (will fail validation with dummy data, but endpoint should respond)
curl -X POST http://localhost:8001/api/v1/enroll_identity \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test",
    "identity_id": "user-1",
    "images_b64": ["data:image/jpeg;base64,/9j/4AAQ=="]
  }'

# Expected: HTTP 200 (success) or 422 (validation error)

# Test verification (will return 404 for non-existent identity, but endpoint should respond)
curl -X POST http://localhost:8001/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test",
    "identity_id": "user-1",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQ=="
  }'

# Expected: HTTP 404 (not enrolled) or 422 (validation error)
```

---

## 📋 Environment Variables

Required environment variables:

```bash
# Collections
IDENTITY_COLLECTION=identities_v1
QDRANT_COLLECTION=faces_v1

# Verification defaults
VERIFY_HI_THRESHOLD=0.78
VERIFY_TOP_K=50
VERIFY_HNSW_EF=128

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=your-key-here
VECTOR_DIM=512
```

---

## 🔄 Rollback

If issues occur:

```bash
# Option 1: Revert code
git revert HEAD
# Redeploy

# Option 2: Kubernetes rollback
kubectl rollout undo deployment/face-pipeline

# Option 3: Scale down
kubectl scale deployment/face-pipeline --replicas=0
```

**Note**: No data migration was required, so rolling back is safe.

---

## 📊 Monitoring

After deployment, monitor:

1. **Enrollment success rate** (target: >95%)
2. **Verification pass rate** (target: 85-92%)
3. **API latency** (target: < 500ms P95)
4. **Error rate** (target: < 1%)

---

## 📚 Full Documentation

For detailed deployment steps, see:
- **Deployment Plan**: `DEPLOYMENT_PLAN.md` (comprehensive guide)
- **Operating Guide**: `VERIFICATION_FLOW_GUIDE.md` (production recommendations)
- **Quick Start**: `VERIFICATION_QUICKSTART.md` (examples and usage)

---

**Ready to deploy?** Run: `python scripts/deploy.py [dev|staging|prod]`

