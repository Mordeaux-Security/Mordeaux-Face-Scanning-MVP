/**
 * UploadTestPage - Simple Image Upload for Testing E2E Integration
 * =================================================================
 * 
 * Quick test page to:
 * 1. Upload an image to the backend
 * 2. Verify it gets stored in MinIO with tenant_id
 * 3. Trigger face pipeline processing
 * 4. Search for the uploaded face
 * 5. Display results with full metadata
 */

import { useState } from 'react';
import './SearchDevPage.css';

const BACKEND_URL = 'http://localhost/api';
const TENANT_ID = 'demo-tenant';

interface UploadResult {
  success: boolean;
  message: string;
  data?: any;
}

interface SearchResult {
  count: number;
  hits: Array<{
    face_id: string;
    score: number;
    payload: {
      tenant_id?: string;
      site?: string;
      url?: string;
      ts?: string;
      bbox?: number[];
      quality?: number;
      quality_is_usable?: boolean;
    };
    thumb_url?: string;
  }>;
}

export default function UploadTestPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setUploadResult(null);
      setSearchResults(null);
    }
  };

  const handleUploadAndSearch = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadResult(null);
    setSearchResults(null);

    try {
      // Convert image to base64
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onload = () => {
          const base64 = reader.result as string;
          // Remove data:image/...;base64, prefix
          const base64Data = base64.split(',')[1];
          resolve(base64Data);
        };
        reader.onerror = reject;
      });

      reader.readAsDataURL(selectedFile);
      const imageB64 = await base64Promise;

      console.log('📤 Sending search request with image...');

      // Send search request
      const response = await fetch(`${BACKEND_URL}/v1/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tenant_id: TENANT_ID,
          image_b64: imageB64,
          top_k: 50,
          threshold: 0.70,
        }),
      });

      const data = await response.json();
      console.log('✅ Search response:', data);

      // Check for error response
      if (!response.ok || data.detail) {
        const errorMsg = data.detail 
          ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
          : `HTTP ${response.status}: ${response.statusText}`;
        throw new Error(errorMsg);
      }

      // Ensure we have the expected response structure
      if (!data.hits) {
        console.warn('Unexpected response format:', data);
        data.hits = [];
        data.count = 0;
      }

      setUploadResult({
        success: true,
        message: `Search completed! Found ${data.count || 0} matches.`,
        data,
      });

      setSearchResults(data);
    } catch (error: any) {
      console.error('❌ Search failed:', error);
      setUploadResult({
        success: false,
        message: `Error: ${error.message}`,
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="search-dev-page">
      <header className="page-header">
        <div className="container">
          <h1 className="header-title">
            🧪 E2E Integration Test - Upload & Search
          </h1>
          <p style={{ color: '#666', marginTop: '0.5rem' }}>
            Test the complete flow: Upload image → Process → Search → Display results
          </p>
        </div>
      </header>

      <main className="page-main">
        <div className="container">
          {/* Upload Section */}
          <section className="query-panel" style={{ marginBottom: '2rem' }}>
            <h2 style={{ marginBottom: '1rem' }}>1. Select Image</h2>
            
            <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{
                    padding: '0.5rem',
                    border: '2px dashed #ccc',
                    borderRadius: '4px',
                    width: '100%',
                    cursor: 'pointer',
                  }}
                />
                
                {selectedFile && (
                  <div style={{ marginTop: '1rem' }}>
                    <p><strong>Selected:</strong> {selectedFile.name}</p>
                    <p><strong>Size:</strong> {(selectedFile.size / 1024).toFixed(2)} KB</p>
                    <p><strong>Type:</strong> {selectedFile.type}</p>
                  </div>
                )}
              </div>

              {previewUrl && (
                <div style={{ flex: '0 0 200px' }}>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{
                      width: '100%',
                      height: 'auto',
                      border: '2px solid #ddd',
                      borderRadius: '8px',
                    }}
                  />
                </div>
              )}
            </div>

            <button
              onClick={handleUploadAndSearch}
              disabled={!selectedFile || uploading}
              style={{
                marginTop: '1rem',
                padding: '0.75rem 2rem',
                fontSize: '1rem',
                fontWeight: 'bold',
                backgroundColor: selectedFile && !uploading ? '#007bff' : '#ccc',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: selectedFile && !uploading ? 'pointer' : 'not-allowed',
              }}
            >
              {uploading ? '⏳ Searching...' : '🔍 Search for Similar Faces'}
            </button>
          </section>

          {/* Upload Result */}
          {uploadResult && (
            <section className="query-panel" style={{ marginBottom: '2rem' }}>
              <h2 style={{ marginBottom: '1rem' }}>2. Upload Result</h2>
              <div
                style={{
                  padding: '1rem',
                  backgroundColor: uploadResult.success ? '#d4edda' : '#f8d7da',
                  border: `1px solid ${uploadResult.success ? '#c3e6cb' : '#f5c6cb'}`,
                  borderRadius: '4px',
                  color: uploadResult.success ? '#155724' : '#721c24',
                }}
              >
                <p style={{ margin: 0 }}>
                  {uploadResult.success ? '✅' : '❌'} {uploadResult.message}
                </p>
              </div>
            </section>
          )}

          {/* Search Results */}
          {searchResults && (
            <section className="results-section">
              <h2 style={{ marginBottom: '1rem' }}>
                3. Search Results ({searchResults.count} matches found)
              </h2>

              {(!searchResults.hits || searchResults.hits.length === 0) ? (
                <div
                  style={{
                    padding: '2rem',
                    textAlign: 'center',
                    backgroundColor: '#f8f9fa',
                    borderRadius: '8px',
                  }}
                >
                  <p style={{ fontSize: '1.2rem', color: '#666' }}>
                    No matches found in the database.
                  </p>
                  <p style={{ color: '#999', marginTop: '0.5rem' }}>
                    This is expected if the database is empty. Try uploading more images or
                    running the crawler to populate the database.
                  </p>
                </div>
              ) : (
                <div className="match-grid">
                  {searchResults.hits.map((hit, index) => (
                    <div
                      key={hit.face_id}
                      style={{
                        border: '1px solid #ddd',
                        borderRadius: '8px',
                        padding: '1rem',
                        backgroundColor: 'white',
                      }}
                    >
                      {/* Thumbnail */}
                      {hit.thumb_url && (
                        <img
                          src={hit.thumb_url}
                          alt={`Match ${index + 1}`}
                          style={{
                            width: '100%',
                            height: 'auto',
                            borderRadius: '4px',
                            marginBottom: '0.5rem',
                          }}
                          onError={(e) => {
                            (e.target as HTMLImageElement).src =
                              'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23ddd" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999"%3ENo Image%3C/text%3E%3C/svg%3E';
                          }}
                        />
                      )}

                      {/* Score */}
                      <div
                        style={{
                          fontSize: '1.2rem',
                          fontWeight: 'bold',
                          color: hit.score >= 0.9 ? '#28a745' : hit.score >= 0.8 ? '#ffc107' : '#6c757d',
                          marginBottom: '0.5rem',
                        }}
                      >
                        Score: {(hit.score * 100).toFixed(1)}%
                      </div>

                      {/* Metadata */}
                      <div style={{ fontSize: '0.875rem', color: '#666' }}>
                        <p><strong>Face ID:</strong> {hit.face_id.substring(0, 16)}...</p>
                        <p><strong>Tenant:</strong> {hit.payload.tenant_id || 'N/A'}</p>
                        <p><strong>Site:</strong> {hit.payload.site || 'N/A'}</p>
                        <p><strong>Quality:</strong> {hit.payload.quality ? (hit.payload.quality * 100).toFixed(0) + '%' : 'N/A'}</p>
                        {hit.payload.url && (
                          <p>
                            <strong>URL:</strong>{' '}
                            <a
                              href={hit.payload.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{ color: '#007bff', textDecoration: 'none' }}
                            >
                              {hit.payload.url.substring(0, 40)}...
                            </a>
                          </p>
                        )}
                        {hit.payload.ts && (
                          <p><strong>Timestamp:</strong> {new Date(hit.payload.ts).toLocaleString()}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Debug Info */}
          {searchResults && (
            <section className="debug-panel" style={{ marginTop: '2rem' }}>
              <details>
                <summary style={{ cursor: 'pointer', fontWeight: 'bold', padding: '0.5rem' }}>
                  🔍 View Raw JSON Response
                </summary>
                <pre
                  style={{
                    backgroundColor: '#f8f9fa',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    fontSize: '0.875rem',
                  }}
                >
                  {JSON.stringify(searchResults, null, 2)}
                </pre>
              </details>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

