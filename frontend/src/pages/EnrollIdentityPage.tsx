/**
 * EnrollIdentityPage - Identity Enrollment UI
 * ============================================
 * 
 * Allows users to enroll an identity by uploading 3-5 face photos.
 * Uses the new /api/v1/enroll_identity endpoint.
 */

import { useState, useCallback } from 'react';
import './EnrollIdentityPage.css';
import SafeImage from '../components/SafeImage';

interface EnrollState {
  tenantId: string;
  identityId: string;
  images: File[];
  previews: string[];
  status: 'idle' | 'uploading' | 'success' | 'error';
  message: string;
}

export default function EnrollIdentityPage() {
  const [state, setState] = useState<EnrollState>({
    tenantId: 'demo-tenant',
    identityId: '',
    images: [],
    previews: [],
    status: 'idle',
    message: '',
  });

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    
    if (imageFiles.length === 0) return;
    
    // Limit to 5 images max
    const limited = imageFiles.slice(0, 5);
    
    // Create previews
    const previewPromises = limited.map(file => {
      return new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.readAsDataURL(file);
      });
    });
    
    Promise.all(previewPromises).then(previews => {
      setState(prev => ({
        ...prev,
        images: limited,
        previews,
      }));
    });
  }, []);

  const removeImage = useCallback((index: number) => {
    setState(prev => ({
      ...prev,
      images: prev.images.filter((_, i) => i !== index),
      previews: prev.previews.filter((_, i) => i !== index),
    }));
  }, []);

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleEnroll = useCallback(async () => {
    if (state.images.length < 3) {
      setState(prev => ({
        ...prev,
        status: 'error',
        message: `Please upload at least 3 photos. Currently have ${prev.images.length}.`,
      }));
      return;
    }

    if (!state.identityId.trim()) {
      setState(prev => ({
        ...prev,
        status: 'error',
        message: 'Please enter an identity ID.',
      }));
      return;
    }

    setState(prev => ({ ...prev, status: 'uploading', message: 'Uploading and processing images...' }));

    try {
      // Convert all images to base64
      const imagesB64 = await Promise.all(state.images.map(convertToBase64));

      // Call enrollment endpoint
      const response = await fetch('http://localhost:8001/api/v1/enroll_identity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: state.tenantId,
          identity_id: state.identityId,
          images_b64: imagesB64,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.error || data.detail || 'Enrollment failed');
      }

      setState(prev => ({
        ...prev,
        status: 'success',
        message: `✅ Successfully enrolled "${state.identityId}" with ${data.num_images_used || state.images.length} photos!`,
      }));

      // Clear form after 3 seconds
      setTimeout(() => {
        setState({
          tenantId: 'demo-tenant',
          identityId: '',
          images: [],
          previews: [],
          status: 'idle',
          message: '',
        });
      }, 3000);
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        status: 'error',
        message: `❌ Error: ${error.message || 'Failed to enroll identity'}`,
      }));
    }
  }, [state]);

  return (
    <div className="enroll-page">
      <nav className="page-nav">
        <a href="/enroll" className="nav-link active">Enroll</a>
        <a href="/verify" className="nav-link">Verify & Search</a>
        <a href="/dev/search" className="nav-link">Legacy Search</a>
      </nav>
      <header className="enroll-header">
        <h1>🔐 Enroll Identity</h1>
        <p>Upload 3-5 face photos to create an identity profile</p>
      </header>

      <main className="enroll-main">
        <div className="enroll-form">
          <div className="form-group">
            <label htmlFor="tenant-id">Tenant ID:</label>
            <input
              id="tenant-id"
              type="text"
              value={state.tenantId}
              onChange={(e) => setState(prev => ({ ...prev, tenantId: e.target.value }))}
              placeholder="demo-tenant"
            />
          </div>

          <div className="form-group">
            <label htmlFor="identity-id">Identity ID: *</label>
            <input
              id="identity-id"
              type="text"
              value={state.identityId}
              onChange={(e) => setState(prev => ({ ...prev, identityId: e.target.value }))}
              placeholder="user-alice"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="file-input">Face Photos (3-5 required): *</label>
            <div className="file-upload-area">
              <input
                id="file-input"
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input" className="file-upload-label">
                📁 Click to select photos (or drag & drop)
              </label>
              <p className="file-hint">
                {state.images.length === 0 && 'Select 3-5 face photos'}
                {state.images.length > 0 && state.images.length < 3 && 
                  `⚠️ Need ${3 - state.images.length} more photo(s) (minimum 3 required)`}
                {state.images.length >= 3 && state.images.length < 5 && 
                  `✅ ${state.images.length} photos selected (can add up to 5)`}
                {state.images.length === 5 && '✅ Maximum 5 photos selected'}
              </p>
            </div>
          </div>

          {state.previews.length > 0 && (
            <div className="preview-grid">
              {state.previews.map((preview, index) => (
                <div key={index} className="preview-item">
                  <SafeImage
                    src={preview}
                    alt={`Preview ${index + 1}`}
                    className="preview-image"
                  />
                  <button
                    type="button"
                    className="remove-button"
                    onClick={() => removeImage(index)}
                    aria-label={`Remove image ${index + 1}`}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            type="button"
            className="enroll-button"
            onClick={handleEnroll}
            disabled={state.status === 'uploading' || state.images.length < 3 || !state.identityId.trim()}
          >
            {state.status === 'uploading' ? '⏳ Enrolling...' : '✅ Enroll Identity'}
          </button>

          {state.message && (
            <div className={`message ${state.status}`}>
              {state.message}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

