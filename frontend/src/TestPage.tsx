/**
 * Simple test page to verify React is working
 */

import * as React from 'react';

export default function TestPage() {
  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif' }}>
      <h1 style={{ color: 'green' }}>✅ React is Working!</h1>
      <p>If you see this, React is rendering correctly.</p>
      <div style={{ background: '#f0f0f0', padding: '20px', marginTop: '20px', borderRadius: '8px' }}>
        <h2>Debug Info:</h2>
        <ul>
          <li>React: {React.version}</li>
          <li>Time: {new Date().toLocaleString()}</li>
          <li>URL: {window.location.href}</li>
          <li>Dev Mode: {import.meta.env.DEV ? 'true' : 'false'}</li>
        </ul>
      </div>
      <div style={{ marginTop: '20px' }}>
        <a href="/dev/search" style={{ color: 'blue', textDecoration: 'underline' }}>
          Go to Search Dev Page →
        </a>
      </div>
    </div>
  );
}


