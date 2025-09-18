import { z } from 'zod';

// DTO: SearchByVectorRequest
export const SearchByVectorRequestSchema = z.object({
  vector: z.array(z.number()).length(512),
  filters: z.object({
    tenant_id: z.string().uuid(),
    site: z.string().optional(),
    ts_from: z.string().datetime().optional(),
    ts_to: z.string().datetime().optional()
  }).optional()
});

export type SearchByVectorRequest = z.infer<typeof SearchByVectorRequestSchema>;

// DTO: SearchResult
export const SearchResultSchema = z.object({
  content_id: z.string().uuid(),
  face_id: z.string().uuid(),
  score: z.number().min(0).max(1),
  thumb_s3_key: z.string(),
  cluster_id: z.string().uuid().optional()
});

export type SearchResult = z.infer<typeof SearchResultSchema>;

// DTO: SearchByImageRequest
export const SearchByImageRequestSchema = z.object({
  image: z.string(), // base64 encoded image
  filters: z.object({
    tenant_id: z.string().uuid(),
    site: z.string().optional(),
    ts_from: z.string().datetime().optional(),
    ts_to: z.string().datetime().optional()
  }).optional()
});

export type SearchByImageRequest = z.infer<typeof SearchByImageRequestSchema>;

// DTO: Policy
export const PolicySchema = z.object({
  tenant_id: z.string().uuid(),
  name: z.string(),
  rules: z.array(z.object({
    type: z.enum(['block', 'alert', 'allow']),
    conditions: z.record(z.any()),
    actions: z.array(z.string())
  })),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
});

export type Policy = z.infer<typeof PolicySchema>;
