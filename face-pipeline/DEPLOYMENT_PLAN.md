# Verification-First Flow: Deployment Plan

**Target Date**: [TO BE DETERMINED]  
**Estimated Duration**: 2-4 hours  
**Rollback Window**: Immediate (code changes only, no data migration)

---

## 📋 Pre-Deployment Checklist

### Code Review & Testing

- [ ] All code reviewed and approved
- [ ] Unit tests pass (`pytest face-pipeline/tests/`)
- [ ] Integration tests pass
- [ ] Linting passes (`flake8`, `mypy`, etc.)
- [ ] No linting errors in modified files
- [ ] Code changes match implementation summary

### Environment Configuration

- [ ] Review `env.example` → `.env` for all environments (dev, staging, prod)
- [ ] Set `IDENTITY_COLLECTION=identities_v1` in all environments
- [ ] Set `VERIFY_HI_THRESHOLD=0.78` (production default)
- [ ] Verify `QDRANT_URL` and `QDRANT_API_KEY` are correct
- [ ] Verify `QDRANT_COLLECTION=faces_v1` is correct
- [ ] Ensure `VECTOR_DIM=512` is set consistently

### Database/Infrastructure

- [ ] Qdrant is accessible from all environments
- [ ] Qdrant API key has write permissions
- [ ] Qdrant has sufficient disk space (estimate: ~10KB per identity)
- [ ] Backup Qdrant data (recommended before any schema changes)
- [ ] Verify network connectivity (Qdrant, MinIO, Redis)

### Documentation

- [ ] Team briefed on new endpoints (`/enroll_identity`, `/verify`)
- [ ] API documentation updated (OpenAPI/Swagger)
- [ ] Frontend team notified of new endpoints
- [ ] Support team briefed on error messages
- [ ] Runbook created for common issues

---

## 🚀 Deployment Steps

### Phase 1: Preparation (30 minutes)

#### Step 1.1: Backup Current State

```bash
# Backup Qdrant collections (if possible)
# Option A: Qdrant snapshot (if enabled)
curl -X POST http://qdrant:6333/collections/faces_v1/snapshots

# Option B: Export collection metadata
python scripts/backup_qdrant.py --collection faces_v1

# Option C: Document current state
curl http://qdrant:6333/collections/faces_v1 | jq > backups/faces_v1_before.json
```

#### Step 1.2: Verify Code is Ready

```bash
cd face-pipeline

# Pull latest code
git pull origin main  # or your branch

# Verify all changes are present
git log --oneline -10
git diff main --name-only | grep -E "(ensure.py|main.py|processor.py|indexer.py)"

# Run tests
pytest tests/ -v
```

#### Step 1.3: Update Environment Variables

**For each environment (dev, staging, prod):**

```bash
# Update .env file (or environment config)
IDENTITY_COLLECTION=identities_v1
VERIFY_HI_THRESHOLD=0.78
VERIFY_TOP_K=50
VERIFY_HNSW_EF=128

# Verify existing settings
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=faces_v1
VECTOR_DIM=512
```

### Phase 2: Code Deployment (15 minutes)

#### Step 2.1: Deploy to Dev Environment

```bash
# Deploy code (adjust based on your deployment method)
# Option A: Docker
docker-compose -f docker-compose.yml build face-pipeline
docker-compose -f docker-compose.yml up -d face-pipeline

# Option B: Kubernetes
kubectl set image deployment/face-pipeline face-pipeline=your-registry/face-pipeline:latest
kubectl rollout status deployment/face-pipeline -n face-pipeline

# Option C: Direct deployment
systemctl restart face-pipeline
```

#### Step 2.2: Verify Collections Created

```bash
# Check logs for ensure_all() success
docker logs face-pipeline | grep -i "identities"
# OR
kubectl logs deployment/face-pipeline | grep -i "identities"

# Verify collection exists
curl http://qdrant:6333/collections | jq '.result.collections[] | .name'

# Expected output should include:
# - faces_v1
# - identities_v1  # <-- New collection
```

#### Step 2.3: Verify Indexes Created

```bash
# Check identities_v1 collection info
curl http://qdrant:6333/collections/identities_v1 | jq '.result.config.payload'

# Should show indexes for:
# - tenant_id (KEYWORD)
# - identity_id (KEYWORD)

# Check faces_v1 collection info (verify identity_id index added)
curl http://qdrant:6333/collections/faces_v1 | jq '.result.config.payload'

# Should show indexes for:
# - tenant_id (KEYWORD)
# - p_hash_prefix (KEYWORD)
# - site (KEYWORD)
# - identity_id (KEYWORD)  # <-- New index
```

### Phase 3: Smoke Tests (30 minutes)

#### Step 3.1: Health Checks

```bash
# Basic health check
curl http://localhost:8001/api/v1/health

# Expected: {"status": "healthy", "service": "face-pipeline-search-api"}

# Verify all endpoints respond
curl http://localhost:8001/docs  # Swagger UI should load
```

#### Step 3.2: Test Enrollment Endpoint

```bash
# Prepare test images (3 photos: frontal, left, right)
# Convert to base64
FRONTAL_B64=$(cat test_images/frontal.jpg | base64 -w 0)
LEFT_B64=$(cat test_images/left.jpg | base64 -w 0)
RIGHT_B64=$(cat test_images/right.jpg | base64 -w 0)

# Test enrollment
curl -X POST http://localhost:8001/api/v1/enroll_identity \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"test-tenant\",
    \"identity_id\": \"test-user-1\",
    \"images_b64\": [
      \"data:image/jpeg;base64,${FRONTAL_B64}\",
      \"data:image/jpeg;base64,${LEFT_B64}\",
      \"data:image/jpeg;base64,${RIGHT_B64}\"
    ]
  }"

# Expected response:
# {"ok": true, "identity": {"tenant_id": "test-tenant", "identity_id": "test-user-1"}, "vector_dim": 512}
```

#### Step 3.3: Test Verification Endpoint

```bash
# Test with same person (should pass)
PROBE_B64=$(cat test_images/probe_same_person.jpg | base64 -w 0)

curl -X POST http://localhost:8001/api/v1/verify \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"test-tenant\",
    \"identity_id\": \"test-user-1\",
    \"image_b64\": \"data:image/jpeg;base64,${PROBE_B64}\",
    \"hi_threshold\": 0.78
  }"

# Expected: {"verified": true, "similarity": 0.80-0.95, "results": [...], "count": 0}

# Test with different person (should fail)
DIFFERENT_B64=$(cat test_images/different_person.jpg | base64 -w 0)

curl -X POST http://localhost:8001/api/v1/verify \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"test-tenant\",
    \"identity_id\": \"test-user-1\",
    \"image_b64\": \"data:image/jpeg;base64,${DIFFERENT_B64}\",
    \"hi_threshold\": 0.78
  }"

# Expected: {"verified": false, "similarity": 0.50-0.70, "results": [], "count": 0}

# Test with non-existent identity (should return 404)
curl -X POST http://localhost:8001/api/v1/verify \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"test-tenant\",
    \"identity_id\": \"non-existent\",
    \"image_b64\": \"data:image/jpeg;base64,${PROBE_B64}\"
  }"

# Expected: 404 {"detail": "identity_not_enrolled"}
```

#### Step 3.4: Test Search Endpoint (Legacy)

```bash
# Verify legacy search still works
curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"test-tenant\",
    \"image_b64\": \"data:image/jpeg;base64,${PROBE_B64}\",
    \"top_k\": 10
  }"

# Expected: {"query": {...}, "hits": [...], "count": N}
```

### Phase 4: Staging Deployment (45 minutes)

#### Step 4.1: Deploy to Staging

```bash
# Same steps as Phase 2, but for staging environment
# Update environment variables for staging
# Deploy code
# Verify collections created
# Verify indexes created
```

#### Step 4.2: Integration Tests

```bash
# Run full integration test suite
pytest tests/integration/ -v

# Test enrollment → verification flow end-to-end
python tests/integration/test_verification_flow.py

# Test face tagging during ingestion
python tests/integration/test_identity_tagging.py
```

#### Step 4.3: Load Testing (Optional)

```bash
# Test with concurrent enrollments
python scripts/load_test_enrollment.py --concurrent 10 --iterations 100

# Test with concurrent verifications
python scripts/load_test_verification.py --concurrent 50 --iterations 500

# Monitor:
# - Response times (should be < 500ms for enrollment, < 300ms for verification)
# - Error rates (should be < 1%)
# - Qdrant latency
```

### Phase 5: Production Deployment (60 minutes)

#### Step 5.1: Pre-Production Checklist

- [ ] All smoke tests pass in staging
- [ ] Integration tests pass
- [ ] Load tests acceptable (if run)
- [ ] Monitoring dashboards ready
- [ ] Rollback plan reviewed
- [ ] Support team on standby
- [ ] Maintenance window scheduled (if required)

#### Step 5.2: Deploy to Production

```bash
# Switch to production context
export ENV=production

# Final code review
git log --oneline -5

# Deploy (choose your method)
# Option A: Blue-Green Deployment (recommended)
# - Deploy to green environment
# - Run smoke tests
# - Switch traffic from blue → green
# - Monitor for 15 minutes
# - Keep blue available for rollback

# Option B: Rolling Update
kubectl set image deployment/face-pipeline face-pipeline=registry/face-pipeline:v1.0.0
kubectl rollout status deployment/face-pipeline

# Option C: Canary Deployment
# - Deploy to 10% of pods
# - Monitor metrics for 10 minutes
# - Gradually increase to 100%
```

#### Step 5.3: Verify Collections Created

```bash
# Check production Qdrant
curl https://qdrant-prod.example.com/collections | jq '.result.collections[] | .name'

# Should show:
# - faces_v1
# - identities_v1  # <-- New collection

# Verify indexes
curl https://qdrant-prod.example.com/collections/identities_v1 | jq '.result.config.payload'
```

#### Step 5.4: Production Smoke Tests

```bash
# Run smoke tests against production
# (Use test tenant, not real user data)

# Test enrollment
curl -X POST https://api-prod.example.com/api/v1/enroll_identity \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: test-tenant" \
  -d '{...}'

# Test verification
curl -X POST https://api-prod.example.com/api/v1/verify \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: test-tenant" \
  -d '{...}'
```

---

## 📊 Post-Deployment Verification

### Immediate Checks (First 15 minutes)

- [ ] Health checks passing (`/api/v1/health`)
- [ ] Collections created (`identities_v1` exists)
- [ ] Indexes created (`identity_id` indexed in both collections)
- [ ] Enrollment endpoint responds (200 OK)
- [ ] Verification endpoint responds (200 OK)
- [ ] Search endpoint still works (backward compatibility)
- [ ] No increase in error rates
- [ ] No increase in response latency

### Monitoring Dashboard Setup

#### Key Metrics to Monitor

1. **Enrollment Metrics**
   - Success rate (target: >95%)
   - Latency (P50, P95, P99)
   - Error rate by type

2. **Verification Metrics**
   - Pass rate (target: 85-92%)
   - Similarity score distribution
   - Latency (P50, P95, P99)
   - Error rate by type

3. **System Metrics**
   - Qdrant latency
   - Collection sizes (identities_v1 count)
   - API error rates
   - Request rate

#### Example Prometheus Queries

```promql
# Enrollment success rate
rate(enrollment_success_total[5m]) / rate(enrollment_attempts_total[5m])

# Verification pass rate
rate(verification_verified_total[5m]) / rate(verification_attempts_total[5m])

# Average similarity score
avg(verification_similarity_score)

# P95 latency (enrollment)
histogram_quantile(0.95, rate(enrollment_latency_seconds_bucket[5m]))

# P95 latency (verification)
histogram_quantile(0.95, rate(verification_latency_seconds_bucket[5m]))
```

#### Example Grafana Dashboard

Create dashboard with panels:
1. Enrollment success rate (gauge, green if >95%)
2. Verification pass rate (gauge, green if 85-92%)
3. Enrollment latency (line chart, P50/P95/P99)
4. Verification latency (line chart, P50/P95/P99)
5. Similarity score distribution (histogram)
6. Error rate by type (pie chart)
7. Collection sizes (gauge, identities_v1 count)

### Extended Monitoring (First 24 hours)

- [ ] Monitor error logs for new patterns
- [ ] Track verification pass rates (should stabilize)
- [ ] Monitor user complaints (false rejects/accepts)
- [ ] Check Qdrant performance (no degradation)
- [ ] Verify no increase in support tickets

---

## 🔄 Rollback Plan

### Immediate Rollback (< 5 minutes)

If critical issues detected immediately after deployment:

```bash
# Option A: Revert code deployment
git revert HEAD
# Redeploy previous version

# Option B: Switch back to previous environment (blue-green)
# Switch traffic back to blue environment

# Option C: Kubernetes rollback
kubectl rollout undo deployment/face-pipeline

# Option D: Scale down new version
kubectl scale deployment/face-pipeline --replicas=0
```

**Rollback Checklist:**
- [ ] Revert code changes
- [ ] Redeploy previous version
- [ ] Verify health checks pass
- [ ] Verify legacy endpoints work
- [ ] Monitor for stability (15 minutes)
- [ ] Document rollback reason

### Partial Rollback (Disable New Features)

If enrollment/verification have issues but legacy search is fine:

```bash
# Option A: Feature flag (if implemented)
export ENABLE_VERIFICATION_FLOW=false

# Option B: Return 503 for new endpoints
# (Add middleware to return 503 for /enroll_identity and /verify)

# Option C: Route traffic away from new endpoints
# (Configure load balancer to reject requests to new endpoints)
```

### Data Rollback

**Note**: No data migration was required, so no data rollback needed.

If needed to remove test data:
```bash
# Remove test identities (if created)
# This is safe - no production data should exist yet
python scripts/cleanup_test_identities.py --tenant test-tenant
```

---

## 🚨 Troubleshooting Guide

### Issue: Collection Not Created

**Symptoms**: `identities_v1` collection missing after deployment

**Debugging**:
```bash
# Check logs for ensure_all() errors
docker logs face-pipeline | grep -i "identities"

# Verify Qdrant connectivity
curl http://qdrant:6333/health

# Manually create collection
python scripts/create_identities_collection.py
```

**Solution**: Restart service or manually run `ensure_identities()`

### Issue: Indexes Not Created

**Symptoms**: Verification queries are slow

**Debugging**:
```bash
# Check collection info
curl http://qdrant:6333/collections/identities_v1 | jq '.result.config.payload'

# Manually create indexes
python scripts/create_identity_indexes.py
```

**Solution**: Indexes are created lazily; restart service or wait for next upsert

### Issue: Enrollment Fails

**Symptoms**: 422 "no_face_detected" or 500 errors

**Debugging**:
```bash
# Check logs
docker logs face-pipeline | grep -i "enroll"

# Verify InsightFace model loaded
curl http://localhost:8001/api/v1/health

# Test with single image
curl -X POST ... --data '{"images_b64": ["..."]}'
```

**Solution**: Check image quality, ensure model files are present

### Issue: Verification Always Fails

**Symptoms**: All verifications return `verified: false` with low similarity

**Debugging**:
```bash
# Check if identity exists
curl http://qdrant:6333/collections/identities_v1/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "identity_id", "match": {"value": "test-user-1"}}]}}'

# Test similarity calculation
python scripts/test_similarity.py --identity test-user-1 --probe probe.jpg
```

**Solution**: 
- Verify enrollment succeeded
- Check threshold (may be too high, try 0.76)
- Ensure same tenant_id used for enrollment and verification

### Issue: High Latency

**Symptoms**: Enrollment/verification takes > 1 second

**Debugging**:
```bash
# Check Qdrant latency
curl -w "@curl-format.txt" http://qdrant:6333/collections

# Profile API endpoint
python scripts/profile_endpoint.py --endpoint /api/v1/verify

# Check CPU/memory usage
docker stats face-pipeline
```

**Solution**: 
- Reduce `top_k` or `hnsw_ef` if too high
- Scale up Qdrant if needed
- Enable GPU for InsightFace (if available)

### Issue: Memory Leak

**Symptoms**: Memory usage increasing over time

**Debugging**:
```bash
# Monitor memory
docker stats face-pipeline --no-stream | grep face-pipeline

# Check for unclosed clients
python scripts/check_resource_leaks.py
```

**Solution**: 
- Restart service periodically (if needed)
- Check Qdrant client connection pooling
- Verify InsightFace model isn't reloading

---

## 📝 Deployment Sign-Off

### Pre-Deployment Sign-Off

- [ ] Code reviewed and approved by: _________________
- [ ] Infrastructure checked by: _________________
- [ ] Rollback plan reviewed by: _________________
- [ ] Support team briefed by: _________________

### Post-Deployment Sign-Off

- [ ] Smoke tests passed by: _________________
- [ ] Monitoring confirmed by: _________________
- [ ] Production verified by: _________________
- [ ] Deployment completed at: _________________ (date/time)

---

## 🎯 Success Criteria

### Deployment Successful If:

1. ✅ All health checks pass
2. ✅ Collections created (`identities_v1` exists)
3. ✅ Indexes created (`identity_id` indexed)
4. ✅ Enrollment endpoint works (test enroll succeeds)
5. ✅ Verification endpoint works (test verify succeeds)
6. ✅ Legacy search still works (backward compatibility)
7. ✅ No increase in error rates (< 1% error rate)
8. ✅ No increase in latency (< 500ms P95)
9. ✅ Monitoring dashboards show expected metrics
10. ✅ No user-facing issues reported

### Deployment Failed If:

- ❌ Health checks fail
- ❌ Collections not created
- ❌ Error rate > 5%
- ❌ Latency > 1 second (P95)
- ❌ Critical bugs reported
- ❌ Data corruption detected

**Action**: Rollback immediately if any critical issues detected.

---

## 📅 Timeline

### Estimated Duration

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Preparation | 30 min | 30 min |
| Code Deployment | 15 min | 45 min |
| Smoke Tests | 30 min | 1h 15min |
| Staging Deployment | 45 min | 2h 0min |
| Production Deployment | 60 min | 3h 0min |
| Post-Deployment Verification | 30 min | 3h 30min |
| **Total** | **~3-4 hours** | |

### Recommended Schedule

- **Dev Deployment**: Monday, 10:00 AM (low traffic)
- **Staging Deployment**: Tuesday, 2:00 PM (after dev verified)
- **Production Deployment**: Wednesday, 2:00 AM (maintenance window) or during low-traffic period

---

## 🔗 Related Documentation

- **Operating Guide**: `VERIFICATION_FLOW_GUIDE.md`
- **Quick Start**: `VERIFICATION_QUICKSTART.md`
- **Implementation Summary**: `VERIFICATION_IMPLEMENTATION_SUMMARY.md`
- **API Documentation**: `/docs` (Swagger UI)
- **Runbook**: `TROUBLESHOOTING.md` (create if needed)

---

## 📞 Emergency Contacts

- **On-Call Engineer**: _________________
- **DevOps Lead**: _________________
- **Engineering Manager**: _________________
- **Support Lead**: _________________

---

**Last Updated**: [Date]  
**Next Review**: [Date + 1 month]  
**Version**: 1.0

