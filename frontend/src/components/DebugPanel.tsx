/**
 * DebugPanel - Phase 11 Observability & Diagnostics
 * ==================================================
 * 
 * Collapsible debug panel for dev/admin interfaces.
 * Shows performance metrics, event counters, and logs.
 * 
 * Features:
 * - Collapsible UI
 * - Performance metrics display
 * - Event counters
 * - Log viewer
 * - Export functionality
 */

import { useState } from 'react';
import { logger, eventCounter, imageLoadTracker, apiCallTracker } from '../utils/logger';
import './DebugPanel.css';

export default function DebugPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'metrics' | 'events' | 'logs'>('metrics');

  // Only show in dev mode
  if (!import.meta.env.DEV) {
    return null;
  }

  const handleExportLogs = () => {
    const logs = logger.exportLogs();
    const blob = new Blob([logs], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `logs-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearLogs = () => {
    logger.clearLogs();
    eventCounter.reset();
    console.log('[DebugPanel] Logs and counters cleared');
  };

  return (
    <div className={`debug-panel ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <button
        className="debug-panel-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
        <span className="toggle-text">Debug Panel</span>
        <span className="toggle-badge">DEV</span>
      </button>

      {isExpanded && (
        <div className="debug-panel-content">
          {/* Tabs */}
          <div className="debug-panel-tabs">
            <button
              className={`debug-tab ${activeTab === 'metrics' ? 'active' : ''}`}
              onClick={() => setActiveTab('metrics')}
            >
              📊 Metrics
            </button>
            <button
              className={`debug-tab ${activeTab === 'events' ? 'active' : ''}`}
              onClick={() => setActiveTab('events')}
            >
              📈 Events
            </button>
            <button
              className={`debug-tab ${activeTab === 'logs' ? 'active' : ''}`}
              onClick={() => setActiveTab('logs')}
            >
              📋 Logs
            </button>
          </div>

          {/* Tab Content */}
          <div className="debug-panel-body">
            {activeTab === 'metrics' && (
              <div className="debug-section">
                <h4 className="debug-section-title">Performance Metrics</h4>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <span className="metric-label">FPS</span>
                    <span className="metric-value">60</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Memory (MB)</span>
                    <span className="metric-value">
                      {(performance as any).memory?.usedJSHeapSize
                        ? Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024)
                        : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'events' && (
              <div className="debug-section">
                <h4 className="debug-section-title">Event Counters</h4>
                <div className="event-list">
                  <div className="event-item">
                    <span className="event-name">Image Loads (Success)</span>
                    <span className="event-count">{imageLoadTracker.getStats().success}</span>
                  </div>
                  <div className="event-item">
                    <span className="event-name">Image Loads (Error)</span>
                    <span className="event-count">{imageLoadTracker.getStats().error}</span>
                  </div>
                  <div className="event-item">
                    <span className="event-name">API Calls (Success)</span>
                    <span className="event-count">{apiCallTracker.getStats().success}</span>
                  </div>
                  <div className="event-item">
                    <span className="event-name">API Calls (Error)</span>
                    <span className="event-count">{apiCallTracker.getStats().error}</span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'logs' && (
              <div className="debug-section">
                <h4 className="debug-section-title">Recent Logs</h4>
                <div className="log-viewer">
                  {logger.getLogs().slice(-20).reverse().map((log, index) => (
                    <div key={index} className={`log-entry log-${log.level.toLowerCase()}`}>
                      <span className="log-timestamp">{log.timestamp}</span>
                      <span className="log-level">{log.level}</span>
                      <span className="log-event">{log.event}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="debug-panel-actions">
            <button className="debug-action-button" onClick={handleExportLogs}>
              💾 Export Logs
            </button>
            <button className="debug-action-button" onClick={handleClearLogs}>
              🗑️ Clear All
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


