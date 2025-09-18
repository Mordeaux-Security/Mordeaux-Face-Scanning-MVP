import { z } from 'zod';

// Event: NEW_CONTENT
export const NewContentEventSchema = z.object({
  content_id: z.string().uuid(),
  tenant_id: z.string().uuid(),
  source_id: z.string().uuid(),
  s3_key_raw: z.string(),
  url: z.string().url(),
  fetch_ts: z.string().datetime()
});

export type NewContentEvent = z.infer<typeof NewContentEventSchema>;

// Event: FACES_EXTRACTED
export const FaceExtractedSchema = z.object({
  face_id: z.string().uuid(),
  bbox: z.object({
    x: z.number(),
    y: z.number(),
    width: z.number(),
    height: z.number()
  }),
  quality: z.number().min(0).max(1),
  aligned_s3_key: z.string()
});

export const FacesExtractedEventSchema = z.object({
  content_id: z.string().uuid(),
  faces: z.array(FaceExtractedSchema)
});

export type FaceExtracted = z.infer<typeof FaceExtractedSchema>;
export type FacesExtractedEvent = z.infer<typeof FacesExtractedEventSchema>;

// Event: INDEXED
export const IndexedEventSchema = z.object({
  content_id: z.string().uuid(),
  embedding_ids: z.array(z.string().uuid()),
  index_ns: z.string(),
  ts: z.string().datetime()
});

export type IndexedEvent = z.infer<typeof IndexedEventSchema>;

// Union of all events
export const EventSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('NEW_CONTENT'), data: NewContentEventSchema }),
  z.object({ type: z.literal('FACES_EXTRACTED'), data: FacesExtractedEventSchema }),
  z.object({ type: z.literal('INDEXED'), data: IndexedEventSchema })
]);

export type Event = z.infer<typeof EventSchema>;
