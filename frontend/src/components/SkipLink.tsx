/**
 * SkipLink - Phase 12 Accessibility
 * ==================================
 * 
 * Skip navigation link for keyboard users.
 * Allows jumping directly to main content.
 */

import './SkipLink.css';

interface SkipLinkProps {
  targetId: string;
  label?: string;
}

export default function SkipLink({ targetId, label = 'Skip to main content' }: SkipLinkProps) {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    const target = document.getElementById(targetId);
    if (target) {
      target.focus();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <a
      href={`#${targetId}`}
      className="skip-link"
      onClick={handleClick}
    >
      {label}
    </a>
  );
}


