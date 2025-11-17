# Deployment Checklist

Use this checklist during deployment to ensure all steps are completed.

## Pre-Deployment

### Code & Testing
- [ ] All code reviewed and approved
- [ ] Unit tests pass (`pytest tests/`)
- [ ] Integration tests pass
- [ ] Linting passes (no errors)
- [ ] Code changes match implementation summary

### Environment Configuration
- [ ] `.env` updated with `IDENTITY_COLLECTION=identities_v1`
- [ ] `.env` updated with `VERIFY_HI_THRESHOLD=0.78`
- [ ] `.env` updated with `VERIFY_TOP_K=50`
- [ ] `.env` updated with `VERIFY_HNSW_EF=128`
- [ ] All environment variables verified (dev, staging, prod)

### Infrastructure
- [ ] Qdrant accessible from all environments
- [ ] Qdrant API key has write permissions
- [ ] Qdrant has sufficient disk space
- [ ] Network connectivity verified (Qdrant, MinIO, Redis)
- [ ] Backup created (if applicable)

### Documentation
- [ ] Team briefed on new endpoints
- [ ] API documentation updated
- [ ] Frontend team notified
- [ ] Support team briefed
- [ ] Runbook created (if needed)

---

## Deployment

### Dev Environment
- [ ] Code deployed to dev
- [ ] Service started successfully
- [ ] Logs checked (no errors)
- [ ] Health check passes (`/api/v1/health`)
- [ ] Collections verified (`faces_v1`, `identities_v1`)
- [ ] Indexes verified (`identity_id` indexed)
- [ ] Smoke tests pass
- [ ] Integration tests pass

### Staging Environment
- [ ] Code deployed to staging
- [ ] Service started successfully
- [ ] Logs checked (no errors)
- [ ] Health check passes (`/api/v1/health`)
- [ ] Collections verified (`faces_v1`, `identities_v1`)
- [ ] Indexes verified (`identity_id` indexed)
- [ ] Smoke tests pass
- [ ] Integration tests pass
- [ ] Load tests pass (if run)

### Production Environment
- [ ] Pre-deployment checklist complete
- [ ] Rollback plan reviewed
- [ ] Support team on standby
- [ ] Maintenance window scheduled (if required)
- [ ] Code deployed to production
- [ ] Service started successfully
- [ ] Logs checked (no errors)
- [ ] Health check passes (`/api/v1/health`)
- [ ] Collections verified (`faces_v1`, `identities_v1`)
- [ ] Indexes verified (`identity_id` indexed)
- [ ] Smoke tests pass
- [ ] Monitoring dashboards configured

---

## Post-Deployment Verification

### Immediate Checks (First 15 minutes)
- [ ] Health checks passing (`/api/v1/health`)
- [ ] Enrollment endpoint works (`POST /api/v1/enroll_identity`)
- [ ] Verification endpoint works (`POST /api/v1/verify`)
- [ ] Search endpoint works (`POST /api/v1/search`)
- [ ] No increase in error rates
- [ ] No increase in response latency
- [ ] Logs show no critical errors

### Monitoring Setup
- [ ] Enrollment metrics dashboard created
- [ ] Verification metrics dashboard created
- [ ] System metrics dashboard updated
- [ ] Alerts configured (pass rate, error rate, latency)
- [ ] Dashboards accessible to team

### Extended Monitoring (First 24 hours)
- [ ] Enrollment success rate monitored (>95% target)
- [ ] Verification pass rate monitored (85-92% target)
- [ ] Average similarity score tracked
- [ ] Error logs reviewed (no new patterns)
- [ ] User complaints tracked (none expected initially)
- [ ] Qdrant performance monitored (no degradation)

---

## Testing

### Smoke Tests
- [ ] Test enrollment with 3 images (200 OK)
- [ ] Test enrollment with 1 image (422 error expected)
- [ ] Test verification with enrolled identity (verified=true)
- [ ] Test verification with different person (verified=false)
- [ ] Test verification with non-existent identity (404 error)
- [ ] Test search endpoint (200 OK)
- [ ] Test health endpoint (200 OK)

### Integration Tests
- [ ] End-to-end enrollment → verification flow
- [ ] Face tagging during ingestion
- [ ] Multiple identities per tenant
- [ ] Overwrite enrollment
- [ ] Tenant isolation

### Load Tests (Optional)
- [ ] Concurrent enrollments (10 users)
- [ ] Concurrent verifications (50 requests)
- [ ] Response times acceptable (< 500ms enrollment, < 300ms verification)
- [ ] Error rates acceptable (< 1%)

---

## Rollback Plan (If Needed)

### Immediate Rollback (< 5 minutes)
- [ ] Issue detected (describe: _________________)
- [ ] Rollback initiated
- [ ] Code reverted to previous version
- [ ] Service restarted
- [ ] Health checks passing
- [ ] Legacy endpoints working
- [ ] Issue resolved

### Partial Rollback (Disable New Features)
- [ ] New endpoints disabled (503 responses)
- [ ] Legacy endpoints working
- [ ] Monitoring shows stability
- [ ] Issue investigated

---

## Sign-Off

### Pre-Deployment Sign-Off
- [ ] Code reviewed by: _________________
- [ ] Infrastructure checked by: _________________
- [ ] Rollback plan reviewed by: _________________
- [ ] Support team briefed by: _________________

### Post-Deployment Sign-Off
- [ ] Smoke tests passed by: _________________
- [ ] Monitoring confirmed by: _________________
- [ ] Production verified by: _________________
- [ ] Deployment completed at: _________________ (date/time)

---

## Notes

### Deployment Details
- **Environment**: [ ] Dev [ ] Staging [ ] Prod
- **Version**: _________________
- **Deployed by**: _________________
- **Deployment time**: _________________
- **Duration**: _________________

### Issues Encountered
1. _________________
2. _________________
3. _________________

### Follow-Up Actions
1. _________________
2. _________________
3. _________________

---

**Last Updated**: [Date]  
**Next Review**: [Date + 1 month]

