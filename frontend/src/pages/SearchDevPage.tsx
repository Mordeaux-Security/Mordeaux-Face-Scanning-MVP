/**
 * SearchDevPage - Phase 7 Filters, Pagination & URL Sync + Phase 9 Performance
 * ==============================================================================
 * 
 * Main search results visualization page (dev-only).
 * Layout matches Phase 1 wireframes.
 * 
 * Phase 7 Features:
 * - Min score filter
 * - Pagination
 * - URL state synchronization (deep-linking)
 * - State persists across page reloads
 * 
 * Phase 9 Features (NEW):
 * - Virtualized rendering for large result sets
 * - Performance monitoring
 * - Feature flag to toggle virtualization
 */

import { useState, useMemo, useCallback } from 'react';
import './SearchDevPage.css';
import './SearchDevPage_Phase7.css';
import './SearchDevPage_Phase9.css';
import '../styles/accessibility.css'; // Phase 12
import LoadingState from '../components/LoadingState';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import QueryImage from '../components/QueryImage';
import SkipLink from '../components/SkipLink'; // Phase 12
import ResultCard, { SearchHit } from '../components/ResultCard';
import ResultListItem from '../components/ResultListItem';
// Temporarily disabled - causing import errors
// import VirtualizedResultGrid from '../components/VirtualizedResultGrid';
// import VirtualizedResultList from '../components/VirtualizedResultList';
import MinScoreSlider from '../components/MinScoreSlider';
import Pagination from '../components/Pagination';
import { useUrlState, urlParsers, copyCurrentUrl } from '../hooks/useUrlState';
import { usePerformanceMonitor } from '../hooks/usePerformanceMonitor';

type PageState = 'loading' | 'results' | 'empty' | 'error';
type ViewMode = 'grid' | 'list';

// Mock data for Phase 7 testing - Extended to 100 results
const mockResults: SearchHit[] = Array.from({ length: 100 }, (_, index) => ({
  face_id: `face-${Math.random().toString(36).substr(2, 16)}`,
  score: 0.95 - (index * 0.008), // Decreasing scores (0.95 to 0.15)
  payload: {
    site: ['example.com', 'demo-site.org', 'test-faces.net'][index % 3],
    url: `https://example.com/images/photo-${1000 + index}.jpg`,
    ts: new Date(Date.now() - index * 3600000).toISOString(),
    bbox: [
      Math.floor(Math.random() * 400), // x
      Math.floor(Math.random() * 400), // y
      Math.floor(Math.random() * 200 + 100), // width
      Math.floor(Math.random() * 200 + 100), // height
    ] as [number, number, number, number],
    p_hash: Math.random().toString(36).substr(2, 16),
    quality: 0.75 + Math.random() * 0.25,
  },
  thumb_url: `https://minio.example.com/thumbnails/demo-tenant/face-${index}_thumb.jpg?X-Amz-Signature=mock`,
}));

export default function SearchDevPage() {
  console.log('[SearchDevPage] Component rendering...');
  
  // Demo state toggle (no real data)
  const [pageState, setPageState] = useState<PageState>('results');
  
  // Phase 9: Performance monitoring
  const { mark, measure } = usePerformanceMonitor('SearchDevPage', true);
  
  // Phase 9: Feature flag for virtualization (toggle for testing)
  const [useVirtualization, setUseVirtualization] = useState(false);
  
  // URL-synced state (Phase 7)
  const [urlState, setUrlState, resetUrlState] = useUrlState({
    view: { 
      default: 'grid' as ViewMode, 
      parse: (v) => (v === 'list' ? 'list' : 'grid') 
    },
    minScore: { 
      default: 0, 
      parse: urlParsers.number 
    },
    page: { 
      default: 1, 
      parse: urlParsers.int 
    },
    pageSize: { 
      default: 25, 
      parse: urlParsers.int 
    },
    site: { 
      default: '', 
      parse: urlParsers.string 
    },
  });
  
  // Filter and paginate results (Phase 9: added performance mark)
  const { filteredResults, paginatedResults, totalPages } = useMemo(() => {
    mark('filter-start');
    // Filter by minimum score
    const filtered = mockResults.filter(hit => hit.score >= urlState.minScore);
    
    // Filter by site if selected
    const siteFiltered = urlState.site 
      ? filtered.filter(hit => hit.payload.site === urlState.site)
      : filtered;
    
    // Calculate pagination
    const total = Math.ceil(siteFiltered.length / urlState.pageSize);
    const startIndex = (urlState.page - 1) * urlState.pageSize;
    const endIndex = startIndex + urlState.pageSize;
    const paginated = siteFiltered.slice(startIndex, endIndex);
    
    const result = {
      filteredResults: siteFiltered,
      paginatedResults: paginated,
      totalPages: total,
    };
    
    mark('filter-end');
    measure('filter-duration', 'filter-start', 'filter-end');
    
    return result;
  }, [urlState.minScore, urlState.site, urlState.page, urlState.pageSize, mark, measure]);
  
  // Phase 9: Memoized callback to prevent re-renders
  const handleCopyId = useCallback((faceId: string) => {
    console.log('Copied face ID:', faceId);
    // In real implementation, show toast notification
  }, []);
  
  const handleCopyUrl = async () => {
    const success = await copyCurrentUrl();
    if (success) {
      console.log('URL copied to clipboard!');
      // In real implementation, show toast notification
    }
  };
  
  const handleResetFilters = () => {
    resetUrlState();
  };
  
  return (
    <div className="search-dev-page">
      {/* Skip link for accessibility - Phase 12 */}
      <SkipLink targetId="main-content" />
      
      {/* Header */}
      <header className="page-header" role="banner">
        <div className="container">
          <div className="header-content">
            <h1 className="header-title">
              <span className="header-icon" aria-hidden="true">🔍</span>
              Mordeaux Search Results
            </h1>
            
            <div className="header-meta">
              <span className="search-id">Search ID: abc-123</span>
              <button 
                className="copy-url-button" 
                type="button"
                onClick={handleCopyUrl}
                title="Copy current URL (with filters)"
              >
                📋 Copy URL
              </button>
              <button className="upload-new-button" type="button">
                Upload New
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main id="main-content" className="page-main">
        <div className="container">
          
          {/* Query Panel */}
          <section className="query-panel" aria-labelledby="query-panel-title">
            <h2 id="query-panel-title" className="visually-hidden">Query Information</h2>
            
            <div className="query-content">
              {/* Query Image - Now with SafeImage */}
              <div className="query-image-section">
                <QueryImage
                  thumbnailUrl="https://minio.example.com/thumbnails/demo-tenant/query-abc123_thumb.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=mock"
                  fullResolutionUrl="https://minio.example.com/images/demo-tenant/query-abc123.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=mock"
                  alt="Query face image"
                  size={150}
                  metadata={{
                    fileName: 'query-image.jpg',
                    fileSize: 245678,
                    dimensions: { width: 1024, height: 1024 },
                  }}
                />
              </div>
              
              {/* Query Metadata */}
              <div className="query-metadata">
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="metadata-label">Uploaded:</span>
                    <span className="metadata-value">2024-01-15 14:32:05</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Tenant:</span>
                    <span className="metadata-value">demo-tenant</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Top K:</span>
                    <span className="metadata-value">50</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Threshold:</span>
                    <span className="metadata-value">0.75</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Search Mode:</span>
                    <span className="metadata-value">Image</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Backend:</span>
                    <span className="metadata-value">Qdrant</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
          
          {/* Filters Panel - Phase 7 */}
          <section className="filters-panel" aria-labelledby="filters-title">
            <h2 id="filters-title" className="section-title">Filters</h2>
            
            <div className="filters-content">
              {/* Min Score Slider */}
              <div className="filter-item">
                <MinScoreSlider
                  value={urlState.minScore}
                  onChange={(value) => setUrlState({ minScore: value, page: 1 })} // Reset to page 1 on filter change
                  showLabel={true}
                />
              </div>
              
              {/* Site Filter */}
              <div className="filter-item">
                <label htmlFor="site-filter" className="filter-label">
                  Filter by Site:
                </label>
                <select 
                  id="site-filter" 
                  className="filter-select"
                  value={urlState.site}
                  onChange={(e) => setUrlState({ site: e.target.value, page: 1 })}
                >
                  <option value="">All Sites</option>
                  <option value="example.com">example.com</option>
                  <option value="demo-site.org">demo-site.org</option>
                  <option value="test-faces.net">test-faces.net</option>
                </select>
              </div>
              
              {/* Filter Summary */}
              <div className="filter-summary">
                <span className="filter-results-count">
                  {filteredResults.length} of {mockResults.length} results
                </span>
                <button 
                  className="reset-filters-button" 
                  type="button"
                  onClick={handleResetFilters}
                  disabled={urlState.minScore === 0 && urlState.site === ''}
                >
                  Reset Filters
                </button>
              </div>
            </div>
          </section>
          
          {/* Controls Bar */}
          <section className="controls-bar" aria-labelledby="controls-title">
            <h2 id="controls-title" className="visually-hidden">Display Controls</h2>
            
            <div className="controls-content">
              {/* View Toggle */}
              <div className="view-toggle-group">
                <h3 className="controls-label">View:</h3>
                <div className="view-toggle" role="group" aria-label="View mode">
                  <button 
                    className={`view-toggle-button ${urlState.view === 'grid' ? 'active' : ''}`}
                    type="button"
                    onClick={() => setUrlState({ view: 'grid' })}
                    aria-pressed={urlState.view === 'grid'}
                    aria-label="Grid view"
                  >
                    Grid
                  </button>
                  <button 
                    className={`view-toggle-button ${urlState.view === 'list' ? 'active' : ''}`}
                    type="button"
                    onClick={() => setUrlState({ view: 'list' })}
                    aria-pressed={urlState.view === 'list'}
                    aria-label="List view"
                  >
                    List
                  </button>
                </div>
              </div>
              
              {/* Phase 9: Virtualization Toggle */}
              <div className="virtualization-toggle-group">
                <h3 className="controls-label">Performance:</h3>
                <label className="virtualization-toggle-label">
                  <input
                    type="checkbox"
                    checked={useVirtualization}
                    onChange={(e) => setUseVirtualization(e.target.checked)}
                  />
                  <span>Use Virtualization</span>
                  <span className="toggle-hint">(for 2k+ results)</span>
                </label>
              </div>
            </div>
          </section>
          
          {/* Results Section */}
          <section className="results-section" aria-labelledby="results-title">
            <h2 id="results-title" className="results-count">
              Showing 1-25 of 156 results
            </h2>
            
            {/* Demo State Toggle Buttons (for Phase 4 demo only) */}
            <div className="demo-controls">
              <button onClick={() => setPageState('loading')} className="demo-button">
                Show Loading
              </button>
              <button onClick={() => setPageState('results')} className="demo-button">
                Show Results
              </button>
              <button onClick={() => setPageState('empty')} className="demo-button">
                Show Empty
              </button>
              <button onClick={() => setPageState('error')} className="demo-button">
                Show Error
              </button>
            </div>
            
            {/* Conditional Rendering based on state */}
            {pageState === 'loading' && <LoadingState />}
            
            {pageState === 'results' && (
              <>
                {paginatedResults.length > 0 ? (
                  <>
                    {/* Phase 9: Conditional rendering - virtualized or standard */}
                    {useVirtualization ? (
                      // Virtualized rendering (Phase 9)
                      <>
                        {urlState.view === 'grid' ? (
                          <VirtualizedResultGrid
                            results={filteredResults}
                            onCopyId={handleCopyId}
                            showDistance={false}
                          />
                        ) : (
                          <VirtualizedResultList
                            results={filteredResults}
                            onCopyId={handleCopyId}
                            showDistance={true}
                          />
                        )}
                        <div className="virtualization-note">
                          ⚡ Virtualized mode: rendering {filteredResults.length} results efficiently
                        </div>
                      </>
                    ) : (
                      // Standard rendering (Phase 1-7)
                      <>
                        {urlState.view === 'grid' ? (
                          <div className="match-grid" role="list">
                            {paginatedResults.map((hit) => (
                              <ResultCard
                                key={hit.face_id}
                                hit={hit}
                                showDistance={false}
                                onCopyId={handleCopyId}
                              />
                            ))}
                          </div>
                        ) : (
                          <div className="match-list" role="list">
                            {paginatedResults.map((hit) => (
                              <ResultListItem
                                key={hit.face_id}
                                hit={hit}
                                showDistance={true}
                                onCopyId={handleCopyId}
                              />
                            ))}
                          </div>
                        )}
                        
                        {/* Pagination - Phase 7 (only in standard mode) */}
                        <Pagination
                          currentPage={urlState.page}
                          totalPages={totalPages}
                          totalItems={filteredResults.length}
                          itemsPerPage={urlState.pageSize}
                          onPageChange={(page) => setUrlState({ page })}
                          onPageSizeChange={(pageSize) => setUrlState({ pageSize, page: 1 })}
                          pageSizeOptions={[10, 25, 50, 100]}
                        />
                      </>
                    )}
                  </>
                ) : (
                  <EmptyState />
                )}
              </>
            )}
            
            {pageState === 'empty' && <EmptyState />}
            
            {pageState === 'error' && <ErrorState />}
          </section>
          
          {/* Debug Panel */}
          <section className="debug-panel" aria-labelledby="debug-panel-title">
            <button 
              className="debug-toggle" 
              type="button"
              aria-expanded="false"
              aria-controls="debug-content"
            >
              <span id="debug-panel-title">▼ Show Debug Info</span>
            </button>
            
            <div id="debug-content" className="debug-content" hidden>
              <div className="debug-section">
                <h3 className="debug-section-title">Timing Metrics:</h3>
                <ul className="debug-list">
                  <li>• API Call Duration: 245ms</li>
                  <li>• Time to First Byte (TTFB): 120ms</li>
                  <li>• Image Load Time (avg): 85ms</li>
                </ul>
              </div>
              
              <div className="debug-section">
                <h3 className="debug-section-title">Query Parameters:</h3>
                <ul className="debug-list">
                  <li>• Search ID: abc-123-def-456</li>
                  <li>• Tenant ID: demo-tenant</li>
                  <li>• Top K: 50</li>
                  <li>• Threshold: 0.75</li>
                  <li>• Vector Backend: Qdrant</li>
                </ul>
              </div>
              
              <div className="debug-section">
                <h3 className="debug-section-title">API Response (JSON):</h3>
                <pre className="debug-json">
{`{
  "query": {
    "tenant_id": "demo-tenant",
    "search_mode": "image",
    "top_k": 50,
    "threshold": 0.75
  },
  "hits": [...],
  "count": 156
}`}
                </pre>
                <div className="debug-actions">
                  <button className="debug-action-button" type="button">
                    📋 Copy JSON
                  </button>
                  <button className="debug-action-button" type="button">
                    💾 Download JSON
                  </button>
                </div>
              </div>
            </div>
          </section>
          
        </div>
      </main>
      
      {/* Footer */}
      <footer className="page-footer" role="contentinfo">
        <div className="container">
          <p className="footer-text">
            Mordeaux Face Scanning MVP • Dev Search Page • API v0.1
          </p>
        </div>
      </footer>
    </div>
  );
}

