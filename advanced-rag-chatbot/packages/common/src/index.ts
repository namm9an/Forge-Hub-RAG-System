import { z } from 'zod';

// Shared schemas and types for the monorepo
export const ChatMessageSchema = z.object({
  id: z.string().uuid(),
  role: z.enum(['system', 'user', 'assistant']),
  content: z.string().min(1),
  createdAt: z.string().optional()
});

export type ChatMessage = z.infer<typeof ChatMessageSchema>;

export const AskRequestSchema = z.object({
  query: z.string().min(1),
  topK: z.number().int().min(1).max(50).default(5)
});

export type AskRequest = z.infer<typeof AskRequestSchema>;

export const AskResponseSchema = z.object({
  answer: z.string(),
  sources: z.array(
    z.object({
      id: z.string(),
      title: z.string().optional(),
      url: z.string().url().optional(),
      score: z.number().optional()
    })
  )
});

export type AskResponse = z.infer<typeof AskResponseSchema>;

export const version = '0.1.0';

