import { ReactNode, MouseEvent } from 'react';
import './SafeLink.css';
import { validateExternalLink, logLinkAudit } from '../utils/linkAudit';

interface SafeLinkProps {
  href: string;
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
  onUnsafeClick?: () => void;
}

export default function SafeLink({
  href,
  children,
  className = '',
  ariaLabel,
  onUnsafeClick,
}: SafeLinkProps) {
  const validation = validateExternalLink(href);

  const handleUnsafeClick = (event: MouseEvent<HTMLSpanElement>) => {
    event.preventDefault();
    onUnsafeClick?.();
  };

  if (!validation.isSafe) {
    logLinkAudit({ type: 'LINK_BLOCKED', url: href, reason: validation.reason });
    return (
      <span
        className={`safe-link safe-link--disabled ${className}`}
        role="text"
        title={validation.reason}
        onClick={handleUnsafeClick}
      >
        {children}
      </span>
    );
  }

  logLinkAudit({ type: 'LINK_ALLOWED', url: href });

  return (
    <a
      href={href}
      className={`safe-link ${className}`}
      target="_blank"
      rel="noreferrer noopener nofollow"
      aria-label={ariaLabel}
    >
      {children}
    </a>
  );
}


