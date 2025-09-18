// Re-export all schemas for convenience
export * from './events';
export * from './dtos';

// Additional utility schemas
import { z } from 'zod';

// Common schemas
export const UuidSchema = z.string().uuid();
export const TimestampSchema = z.string().datetime();
export const TenantIdSchema = z.string().uuid();

// Health check response
export const HealthCheckSchema = z.object({
  status: z.enum(['healthy', 'unhealthy']),
  timestamp: z.string().datetime(),
  service: z.string(),
  version: z.string(),
  uptime: z.number()
});

export type HealthCheck = z.infer<typeof HealthCheckSchema>;

// Ready check response
export const ReadyCheckSchema = z.object({
  status: z.enum(['ready', 'not_ready']),
  timestamp: z.string().datetime(),
  service: z.string(),
  dependencies: z.record(z.enum(['up', 'down']))
});

export type ReadyCheck = z.infer<typeof ReadyCheckSchema>;
