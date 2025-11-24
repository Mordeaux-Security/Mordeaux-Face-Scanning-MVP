/**
 * App.tsx - Phase 4 Non-Functional Shell
 * ========================================
 * 
 * Main application with routing.
 * No data calls yet - just routing and layout structure.
 */

import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './tokens.css';
import './App.css';
import TestPage from './TestPage';
import EnrollIdentityPage from './pages/EnrollIdentityPage';
import VerifySearchPage from './pages/VerifySearchPage';

// Lazy load SearchDevPage to avoid loading broken components on initial load
const SearchDevPage = lazy(() => import('./pages/SearchDevPage'));

function App() {
  console.log('[App] Rendering App component...');
  
  return (
    <BrowserRouter>
      <Routes>
        {/* Diagnostics route (optional) */}
        <Route path="/test" element={<TestPage />} />
        
        {/* New Identity-Safe Flow */}
        <Route path="/enroll" element={<EnrollIdentityPage />} />
        <Route path="/verify" element={<VerifySearchPage />} />
        
        {/* Legacy dev search route - Lazy loaded */}
        <Route 
          path="/dev/search" 
          element={
            <Suspense fallback={<div style={{padding: '40px'}}>Loading Dev Interface...</div>}>
              <SearchDevPage />
            </Suspense>
          }
        />
        
        {/* Redirect root to enroll page (new default) */}
        <Route path="/" element={<Navigate to="/enroll" replace />} />
        
        {/* 404 - redirect to enroll */}
        <Route path="*" element={<Navigate to="/enroll" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

