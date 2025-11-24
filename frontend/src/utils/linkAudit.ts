/**
 * linkAudit.ts - Phase 8 Source/Storage Actions
 * ------------------------------------------------
 * Shared utilities for validating and auditing external links.
 */

export type LinkSafetyResult = {
  isSafe: boolean;
  reason?: string;
  sanitizedUrl?: string;
};

const ALLOWED_PROTOCOLS = ['https:', 'http:'];
const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1']);

function sanitizeUrl(rawUrl: string): string {
  try {
    const url = new URL(rawUrl);
    url.searchParams.delete('token');
    url.searchParams.delete('signature');
    url.searchParams.delete('X-Amz-Signature');
    url.searchParams.delete('X-Amz-Credential');
    return url.toString();
  } catch {
    return '[invalid-url]';
  }
}

export function validateExternalLink(href: string): LinkSafetyResult {
  if (!href || typeof href !== 'string') {
    return { isSafe: false, reason: 'Missing URL' };
  }

  // Block obvious javascript/data URIs before parsing
  const lowered = href.trim().toLowerCase();
  if (lowered.startsWith('javascript:') || lowered.startsWith('data:')) {
    return { isSafe: false, reason: 'Protocol not allowed' };
  }

  try {
    const url = new URL(href);

    if (!ALLOWED_PROTOCOLS.includes(url.protocol)) {
      return { isSafe: false, reason: 'Protocol not allowed' };
    }

    if (url.protocol === 'http:' && !LOCAL_HOSTNAMES.has(url.hostname)) {
      return { isSafe: false, reason: 'Insecure HTTP blocked' };
    }

    if (!url.hostname) {
      return { isSafe: false, reason: 'Missing hostname' };
    }

    return { isSafe: true, sanitizedUrl: sanitizeUrl(href) };
  } catch {
    return { isSafe: false, reason: 'Invalid URL' };
  }
}

export function logLinkAudit(event: {
  type: 'LINK_ALLOWED' | 'LINK_BLOCKED';
  url: string;
  reason?: string;
}) {
  const payload = {
    timestamp: new Date().toISOString(),
    type: event.type,
    url: sanitizeUrl(event.url),
    reason: event.reason,
  };

  if (event.type === 'LINK_BLOCKED') {
    console.warn('[SafeLink]', payload);
  } else if (import.meta.env.DEV) {
    console.info('[SafeLink]', payload);
  }
}

export type StorageProvider = 'minio' | 's3' | 'external';

export interface StorageInfo {
  provider: StorageProvider;
  hostname?: string;
  bucket?: string;
  rawUrl?: string;
}

export function detectStorageProvider(url?: string): StorageInfo | undefined {
  if (!url) return undefined;

  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();

    if (host.includes('minio')) {
      return { provider: 'minio', hostname: host, rawUrl: url };
    }

    if (host.includes('amazonaws') || host.includes('s3')) {
      const bucket = host.split('.')[0];
      return { provider: 's3', hostname: host, bucket, rawUrl: url };
    }

    return { provider: 'external', hostname: host, rawUrl: url };
  } catch {
    return undefined;
  }
}


