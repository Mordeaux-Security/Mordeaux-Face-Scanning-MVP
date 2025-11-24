/**
 * VerifySearchPage - Identity-Safe Verification & Search
 * =======================================================
 * 
 * Uses the new /api/v1/verify endpoint to verify an identity
 * and return only that identity's faces (prevents cross-recognition).
 */

import { useState, useCallback } from 'react';
import './VerifySearchPage.css';
import QueryImage from '../components/QueryImage';
import ResultCard, { SearchHit } from '../components/ResultCard';
import ResultListItem from '../components/ResultListItem';
import LoadingState from '../components/LoadingState';
import ErrorState from '../components/ErrorState';
import EmptyState from '../components/EmptyState';

export default function VerifySearchPage() {
  const [tenantId, setTenantId] = useState('demo-tenant');
  const [identityId, setIdentityId] = useState('');
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string>('');
  const [results, setResults] = useState<SearchHit[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [error, setError] = useState<string>('');
  const [verificationResult, setVerificationResult] = useState<{
    verified: boolean;
    similarity: number;
    threshold: number;
    count: number;
  } | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;

    setQueryImage(file);
    const reader = new FileReader();
    reader.onload = (e) => setQueryPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  }, []);

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleVerify = useCallback(async () => {
    if (!queryImage) {
      setError('Please select a query image');
      return;
    }

    if (!identityId.trim()) {
      setError('Please enter an identity ID');
      return;
    }

    setStatus('loading');
    setError('');
    setResults([]);
    setVerificationResult(null);

    try {
      const imageB64 = await convertToBase64(queryImage);

      const response = await fetch('http://localhost:8001/api/v1/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          identity_id: identityId,
          image_b64: imageB64,
          hi_threshold: 0.78,
          top_k: 50,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.error || data.detail || 'Verification failed');
      }

      // Check if verified
      if (!data.verified) {
        setStatus('error');
        setError(`❌ Verification failed. Similarity: ${(data.similarity * 100).toFixed(1)}% (threshold: ${(data.threshold * 100).toFixed(1)}%)`);
        setVerificationResult({
          verified: false,
          similarity: data.similarity,
          threshold: data.threshold,
          count: 0,
        });
        return;
      }

      // Convert results to SearchHit format
      const searchHits: SearchHit[] = (data.results || []).map((r: any) => ({
        face_id: r.id,
        score: r.score,
        payload: r.payload || {},
        thumb_url: r.payload?.thumb_url || '',
      }));

      setResults(searchHits);
      setVerificationResult({
        verified: true,
        similarity: data.similarity,
        threshold: data.threshold,
        count: data.count || searchHits.length,
      });
      setStatus('success');
    } catch (err: any) {
      setStatus('error');
      setError(err.message || 'Failed to verify identity');
    }
  }, [queryImage, identityId, tenantId]);

  return (
    <div className="verify-page">
      <nav className="page-nav">
        <a href="/enroll" className="nav-link">Enroll</a>
        <a href="/verify" className="nav-link active">Verify & Search</a>
        <a href="/dev/search" className="nav-link">Legacy Search</a>
      </nav>
      <header className="verify-header">
        <h1>🔍 Identity-Safe Search</h1>
        <p>Verify an identity and search only their faces (prevents cross-recognition)</p>
      </header>

      <main className="verify-main">
        <div className="verify-form">
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="tenant-id">Tenant ID:</label>
              <input
                id="tenant-id"
                type="text"
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
                placeholder="demo-tenant"
              />
            </div>

            <div className="form-group">
              <label htmlFor="identity-id">Identity ID: *</label>
              <input
                id="identity-id"
                type="text"
                value={identityId}
                onChange={(e) => setIdentityId(e.target.value)}
                placeholder="user-alice"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="query-file">Query Image: *</label>
            <div className="file-upload-area">
              <input
                id="query-file"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="query-file" className="file-upload-label">
                📁 Click to select query image
              </label>
              {queryPreview && (
                <div className="query-preview">
                  <img src={queryPreview} alt="Query preview" />
                </div>
              )}
            </div>
          </div>

          <button
            type="button"
            className="verify-button"
            onClick={handleVerify}
            disabled={status === 'loading' || !queryImage || !identityId.trim()}
          >
            {status === 'loading' ? '⏳ Verifying...' : '🔍 Verify & Search'}
          </button>

          {error && (
            <div className="message error">
              {error}
            </div>
          )}

          {verificationResult && (
            <div className={`verification-status ${verificationResult.verified ? 'success' : 'failed'}`}>
              <h3>
                {verificationResult.verified ? '✅ Verified' : '❌ Not Verified'}
              </h3>
              <p>
                Similarity: {(verificationResult.similarity * 100).toFixed(1)}% 
                (Threshold: {(verificationResult.threshold * 100).toFixed(1)}%)
              </p>
              {verificationResult.verified && (
                <p>Found {verificationResult.count} face(s) for this identity</p>
              )}
            </div>
          )}
        </div>

        {status === 'loading' && <LoadingState />}

        {status === 'error' && !verificationResult && <ErrorState message={error} />}

        {status === 'success' && results.length === 0 && (
          <EmptyState message="No faces found for this identity" />
        )}

        {status === 'success' && results.length > 0 && (
          <div className="results-section">
            <div className="results-header">
              <h2>Results ({results.length})</h2>
              <div className="view-toggle">
                <button
                  type="button"
                  className={viewMode === 'grid' ? 'active' : ''}
                  onClick={() => setViewMode('grid')}
                >
                  ⬜ Grid
                </button>
                <button
                  type="button"
                  className={viewMode === 'list' ? 'active' : ''}
                  onClick={() => setViewMode('list')}
                >
                  ☰ List
                </button>
              </div>
            </div>

            {viewMode === 'grid' ? (
              <div className="match-grid" role="list">
                {results.map((hit) => (
                  <ResultCard
                    key={hit.face_id}
                    hit={hit}
                    showDistance={false}
                    onCopyId={(id) => navigator.clipboard.writeText(id)}
                  />
                ))}
              </div>
            ) : (
              <div className="match-list" role="list">
                {results.map((hit) => (
                  <ResultListItem
                    key={hit.face_id}
                    hit={hit}
                    showDistance={true}
                    onCopyId={(id) => navigator.clipboard.writeText(id)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

