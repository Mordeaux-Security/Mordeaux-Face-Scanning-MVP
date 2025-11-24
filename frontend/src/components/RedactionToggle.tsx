/**
 * RedactionToggle - Phase 10 Security/Privacy
 * ============================================
 * 
 * UI component for toggling data redaction in dev/admin interfaces.
 * Only visible and functional in dev mode.
 */

import { useRedaction } from '../utils/dataRedaction';
import './RedactionToggle.css';

export default function RedactionToggle() {
  const { reveal, toggleReveal } = useRedaction();

  // Only show in dev mode
  if (!import.meta.env.DEV) {
    return null;
  }

  return (
    <div className="redaction-toggle">
      <label className="redaction-toggle-label">
        <input
          type="checkbox"
          checked={reveal}
          onChange={toggleReveal}
          className="redaction-toggle-checkbox"
        />
        <span className="redaction-toggle-text">
          🔓 Reveal Sensitive Data
        </span>
        <span className="redaction-toggle-hint">(Dev Only)</span>
      </label>
    </div>
  );
}


