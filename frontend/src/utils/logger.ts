/**
 * logger - Phase 11 Observability & Diagnostics
 * ==============================================
 * 
 * Structured logging utility for dev/admin interfaces.
 * Provides consistent, searchable logs with timestamps and context.
 * 
 * Features:
 * - Multiple log levels (INFO, WARN, ERROR, DEBUG)
 * - Structured payloads
 * - Timestamp-based logging
 * - Dev-only controls
 * - Performance tracking integration
 */

export type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  event: string;
  payload?: any;
  context?: string;
}

/**
 * Configuration for logger
 */
export interface LoggerConfig {
  enabled: boolean;
  minLevel: LogLevel;
  includeTimestamp: boolean;
  includeContext: boolean;
  maxPayloadSize: number; // Max characters for payload
}

const DEFAULT_CONFIG: LoggerConfig = {
  enabled: import.meta.env.DEV,
  minLevel: 'DEBUG',
  includeTimestamp: true,
  includeContext: true,
  maxPayloadSize: 1000,
};

// Log level hierarchy for filtering
const LOG_LEVELS: Record<LogLevel, number> = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
};

/**
 * Truncate large payloads
 */
function truncatePayload(payload: any, maxSize: number): any {
  const str = JSON.stringify(payload);
  if (str.length <= maxSize) {
    return payload;
  }

  return {
    _truncated: true,
    _originalSize: str.length,
    _preview: str.slice(0, maxSize) + '...',
  };
}

/**
 * Format log entry for console
 */
function formatLogEntry(entry: LogEntry): string {
  const parts: string[] = [];

  if (entry.timestamp) {
    parts.push(`[${entry.timestamp}]`);
  }

  parts.push(`[${entry.level}]`);

  if (entry.context) {
    parts.push(`[${entry.context}]`);
  }

  parts.push(entry.event);

  return parts.join(' ');
}

/**
 * Main logger class
 */
class Logger {
  private config: LoggerConfig;
  private context?: string;
  private logs: LogEntry[] = [];

  constructor(config: Partial<LoggerConfig> = {}, context?: string) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.context = context;
  }

  /**
   * Check if log should be output based on level
   */
  private shouldLog(level: LogLevel): boolean {
    if (!this.config.enabled) {
      return false;
    }

    return LOG_LEVELS[level] >= LOG_LEVELS[this.config.minLevel];
  }

  /**
   * Create log entry
   */
  private createEntry(level: LogLevel, event: string, payload?: any): LogEntry {
    return {
      timestamp: this.config.includeTimestamp ? new Date().toISOString() : '',
      level,
      event,
      payload: payload ? truncatePayload(payload, this.config.maxPayloadSize) : undefined,
      context: this.config.includeContext ? this.context : undefined,
    };
  }

  /**
   * Output log to console
   */
  private output(entry: LogEntry) {
    if (!this.shouldLog(entry.level)) {
      return;
    }

    // Store log
    this.logs.push(entry);

    // Output to console
    const message = formatLogEntry(entry);

    switch (entry.level) {
      case 'DEBUG':
        console.debug(message, entry.payload || '');
        break;
      case 'INFO':
        console.info(message, entry.payload || '');
        break;
      case 'WARN':
        console.warn(message, entry.payload || '');
        break;
      case 'ERROR':
        console.error(message, entry.payload || '');
        break;
    }
  }

  /**
   * Log methods
   */
  debug(event: string, payload?: any) {
    this.output(this.createEntry('DEBUG', event, payload));
  }

  info(event: string, payload?: any) {
    this.output(this.createEntry('INFO', event, payload));
  }

  warn(event: string, payload?: any) {
    this.output(this.createEntry('WARN', event, payload));
  }

  error(event: string, payload?: any) {
    this.output(this.createEntry('ERROR', event, payload));
  }

  /**
   * Get all logs
   */
  getLogs(): LogEntry[] {
    return [...this.logs];
  }

  /**
   * Clear logs
   */
  clearLogs() {
    this.logs = [];
  }

  /**
   * Export logs as JSON
   */
  exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }

  /**
   * Create child logger with context
   */
  child(context: string): Logger {
    const childContext = this.context ? `${this.context}:${context}` : context;
    return new Logger(this.config, childContext);
  }
}

/**
 * Default logger instance
 */
export const logger = new Logger();

/**
 * Create contextual logger
 */
export function createLogger(context: string, config?: Partial<LoggerConfig>): Logger {
  return new Logger(config, context);
}

/**
 * Event counters for diagnostics
 */
class EventCounter {
  private counts: Map<string, number> = new Map();

  increment(event: string) {
    const current = this.counts.get(event) || 0;
    this.counts.set(event, current + 1);
  }

  get(event: string): number {
    return this.counts.get(event) || 0;
  }

  getAll(): Record<string, number> {
    return Object.fromEntries(this.counts);
  }

  reset(event?: string) {
    if (event) {
      this.counts.delete(event);
    } else {
      this.counts.clear();
    }
  }
}

export const eventCounter = new EventCounter();

/**
 * Helper for image load tracking
 */
export const imageLoadTracker = {
  success: () => {
    eventCounter.increment('image_load_success');
    logger.debug('IMAGE_LOAD_SUCCESS', { total: eventCounter.get('image_load_success') });
  },
  error: (url: string, error: Error) => {
    eventCounter.increment('image_load_error');
    logger.warn('IMAGE_LOAD_ERROR', { url, error: error.message, total: eventCounter.get('image_load_error') });
  },
  getStats: () => ({
    success: eventCounter.get('image_load_success'),
    error: eventCounter.get('image_load_error'),
    total: eventCounter.get('image_load_success') + eventCounter.get('image_load_error'),
    successRate: eventCounter.get('image_load_success') / (eventCounter.get('image_load_success') + eventCounter.get('image_load_error')) || 0,
  }),
};

/**
 * Helper for API call tracking
 */
export const apiCallTracker = {
  start: (endpoint: string) => {
    eventCounter.increment('api_call_started');
    logger.info('API_CALL_START', { endpoint });
  },
  success: (endpoint: string, duration: number) => {
    eventCounter.increment('api_call_success');
    logger.info('API_CALL_SUCCESS', { endpoint, duration });
  },
  error: (endpoint: string, error: Error) => {
    eventCounter.increment('api_call_error');
    logger.error('API_CALL_ERROR', { endpoint, error: error.message });
  },
  getStats: () => ({
    started: eventCounter.get('api_call_started'),
    success: eventCounter.get('api_call_success'),
    error: eventCounter.get('api_call_error'),
    successRate: eventCounter.get('api_call_success') / eventCounter.get('api_call_started') || 0,
  }),
};


