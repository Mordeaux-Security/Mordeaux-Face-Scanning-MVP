import winston from 'winston';
import { v4 as uuidv4 } from 'uuid';

export interface LogContext {
  requestId?: string;
  service?: string;
  userId?: string;
  tenantId?: string;
  [key: string]: any;
}

class Logger {
  private logger: winston.Logger;
  private serviceName: string;

  constructor(serviceName: string) {
    this.serviceName = serviceName;
    this.logger = winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      defaultMeta: { service: serviceName },
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          )
        })
      ]
    });
  }

  private createContext(context?: LogContext): LogContext {
    return {
      requestId: context?.requestId || uuidv4(),
      service: this.serviceName,
      ...context
    };
  }

  info(message: string, context?: LogContext): void {
    this.logger.info(message, this.createContext(context));
  }

  error(message: string, error?: Error, context?: LogContext): void {
    this.logger.error(message, {
      ...this.createContext(context),
      error: error ? {
        message: error.message,
        stack: error.stack,
        name: error.name
      } : undefined
    });
  }

  warn(message: string, context?: LogContext): void {
    this.logger.warn(message, this.createContext(context));
  }

  debug(message: string, context?: LogContext): void {
    this.logger.debug(message, this.createContext(context));
  }

  // Request logging helper
  logRequest(method: string, url: string, statusCode: number, duration: number, context?: LogContext): void {
    this.info('HTTP Request', {
      ...context,
      method,
      url,
      statusCode,
      duration
    });
  }
}

export function createLogger(serviceName: string): Logger {
  return new Logger(serviceName);
}

export { Logger };
