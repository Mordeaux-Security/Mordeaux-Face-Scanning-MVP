/**
 * dataRedaction - Phase 10 Security/Privacy
 * ==========================================
 * 
 * Utilities for redacting sensitive data in dev/admin interfaces.
 * Protects PII and internal system details.
 * 
 * Features:
 * - Multiple redaction strategies
 * - Toggle redaction on/off (dev only)
 * - Configurable rules per field type
 * - Type-safe redaction
 */

import { useState } from 'react';

export type RedactionStrategy = 'masked' | 'hidden' | 'sanitized' | 'partial' | 'none';

export interface RedactionConfig {
  fields: {
    [key: string]: RedactionStrategy;
  };
  revealForDev: boolean;
}

// Default redaction configuration
export const DEFAULT_REDACTION_CONFIG: RedactionConfig = {
  fields: {
    // Network
    ip_address: 'masked',
    user_agent: 'partial',
    
    // IDs
    internal_id: 'hidden',
    session_id: 'partial',
    request_id: 'partial',
    
    // URLs
    presigned_url: 'sanitized',
    callback_url: 'sanitized',
    
    // Auth
    api_key: 'hidden',
    access_token: 'hidden',
    refresh_token: 'hidden',
    signature: 'hidden',
    
    // PII
    email: 'partial',
    phone: 'masked',
    name: 'partial',
    address: 'masked',
    
    // System
    error_stack: 'partial',
    debug_info: 'hidden',
  },
  revealForDev: true, // Allow devs to toggle reveal
};

/**
 * Mask an IP address
 * Example: 192.168.1.123 → 192.168.x.x
 */
export function maskIpAddress(ip: string): string {
  const parts = ip.split('.');
  if (parts.length === 4) {
    return `${parts[0]}.${parts[1]}.x.x`;
  }
  // IPv6 or invalid
  return 'x.x.x.x';
}

/**
 * Mask a phone number
 * Example: +1-555-123-4567 → +1-555-xxx-xxxx
 */
export function maskPhoneNumber(phone: string): string {
  // Keep country code and area code, mask the rest
  return phone.replace(/(\d{3})[\d-]+(\d{4})$/, '$1-xxx-$2');
}

/**
 * Partial redaction (show first and last few chars)
 * Example: user@example.com → u***@example.com
 *          abc-123-def-456 → abc***456
 */
export function partialRedact(value: string, showChars: number = 3): string {
  if (value.length <= showChars * 2) {
    return '*'.repeat(value.length);
  }

  // Email special case
  if (value.includes('@')) {
    const [local, domain] = value.split('@');
    return `${local.charAt(0)}***@${domain}`;
  }

  // General case
  const start = value.slice(0, showChars);
  const end = value.slice(-showChars);
  return `${start}***${end}`;
}

/**
 * Sanitize URL (remove sensitive query params)
 * Example: https://example.com?token=abc123 → https://example.com?token=[REDACTED]
 */
export function sanitizeUrl(url: string): string {
  try {
    const urlObj = new URL(url);
    
    // Sensitive param keys to redact
    const sensitiveParams = [
      'token',
      'api_key',
      'apikey',
      'access_token',
      'refresh_token',
      'signature',
      'sig',
      'X-Amz-Signature',
      'X-Amz-Security-Token',
      'auth',
      'authorization',
    ];

    sensitiveParams.forEach((param) => {
      if (urlObj.searchParams.has(param)) {
        urlObj.searchParams.set(param, '[REDACTED]');
      }
    });

    return urlObj.toString();
  } catch {
    return '[INVALID-URL]';
  }
}

/**
 * Apply redaction strategy to a value
 */
export function redactValue(
  value: any,
  strategy: RedactionStrategy
): string {
  if (value === null || value === undefined) {
    return '';
  }

  const strValue = String(value);

  switch (strategy) {
    case 'hidden':
      return '[HIDDEN]';

    case 'masked':
      // For IPs, phone numbers, etc.
      if (strValue.match(/^\d+\.\d+\.\d+\.\d+$/)) {
        return maskIpAddress(strValue);
      }
      if (strValue.match(/[\d-+()]/)) {
        return maskPhoneNumber(strValue);
      }
      return '*'.repeat(Math.min(strValue.length, 10));

    case 'sanitized':
      // For URLs
      if (strValue.startsWith('http')) {
        return sanitizeUrl(strValue);
      }
      return strValue;

    case 'partial':
      return partialRedact(strValue);

    case 'none':
    default:
      return strValue;
  }
}

/**
 * Redact an object based on configuration
 */
export function redactObject<T extends Record<string, any>>(
  obj: T,
  config: RedactionConfig = DEFAULT_REDACTION_CONFIG,
  reveal: boolean = false
): Record<string, any> {
  // If reveal is enabled and allowed, return original
  if (reveal && config.revealForDev && import.meta.env.DEV) {
    return obj;
  }

  const redacted: Record<string, any> = {};

  Object.keys(obj).forEach((key) => {
    const value = obj[key];
    const strategy = config.fields[key] || 'none';

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      // Recursive redaction for nested objects
      redacted[key] = redactObject(value, config, reveal);
    } else if (Array.isArray(value)) {
      // Redact array items
      redacted[key] = value.map((item) =>
        typeof item === 'object' ? redactObject(item, config, reveal) : redactValue(item, strategy)
      );
    } else {
      // Redact primitive value
      redacted[key] = redactValue(value, strategy);
    }
  });

  return redacted;
}

/**
 * React hook for managing redaction state
 */
export function useRedaction() {
  const [reveal, setReveal] = useState(false);

  const toggleReveal = () => {
    if (DEFAULT_REDACTION_CONFIG.revealForDev && import.meta.env.DEV) {
      setReveal(!reveal);
      console.log(`[Redaction] Reveal mode: ${!reveal ? 'ON' : 'OFF'}`);
    } else {
      console.warn('[Redaction] Reveal not allowed in this environment');
    }
  };

  return {
    reveal,
    toggleReveal,
    redact: <T extends Record<string, any>>(obj: T) =>
      redactObject(obj, DEFAULT_REDACTION_CONFIG, reveal),
  };
}

// Type guard to check if value needs redaction
export function needsRedaction(key: string, config: RedactionConfig = DEFAULT_REDACTION_CONFIG): boolean {
  return key in config.fields && config.fields[key] !== 'none';
}

