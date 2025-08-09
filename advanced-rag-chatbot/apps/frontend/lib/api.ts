export type Source = { id: string; score?: number };
export type ChatMessage = { role: 'user' | 'assistant'; content: string; sources?: Source[] };
export type DocumentInfo = { id: string; filename: string; pages?: number; chunks?: number; file_size?: number };

const BASE_URL = (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {})
    }
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Request failed ${res.status}: ${txt}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listDocuments: (sessionId?: string) =>
    req<{ documents: DocumentInfo[] }>(`/api/documents${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`),

  deleteDocument: (id: string) =>
    req<{ status: string }>(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  uploadDocuments: async (files: File[], sessionId?: string) => {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    const res = await fetch(`${BASE_URL}/api/documents/upload${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Upload failed ${res.status}: ${txt}`);
    }
    return res.json();
  },

  generateEmbeddings: () => req<{ status: string; generated: number }>(`/api/embeddings/generate`, { method: 'POST' }),

  chat: (query: string, sessionId?: string, topK?: number, threshold?: number) =>
    req<{ answer: string; sources: Source[]; processing_time_ms: number }>(`/api/chat`, {
      method: 'POST',
      body: JSON.stringify({ query, session_id: sessionId, top_k: topK, threshold })
    }),

  chatStream: async (
    query: string,
    sessionId: string | undefined,
    onChunk: (text: string) => void,
    opts?: { topK?: number; threshold?: number }
  ): Promise<{ fullText: string }> => {
    const res = await fetch(`${BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: sessionId, top_k: opts?.topK, threshold: opts?.threshold })
    });
    if (!res.ok || !res.body) {
      const txt = await res.text().catch(() => '');
      throw new Error(`Stream failed ${res.status}: ${txt}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      full += chunk;
      onChunk(chunk);
    }
    return { fullText: full };
  },

  history: (sessionId: string, limit = 50) =>
    req<{ messages: Array<{ id: string; user_message: string; assistant_message: string; source_documents?: Source[]; created_at: string }> }>(
      `/api/chat/history?session_id=${encodeURIComponent(sessionId)}&limit=${limit}`
    ),

  stats: () => req<{ total_documents: number; total_chunks: number; total_embeddings: number; total_sessions: number; total_messages: number }>(`/api/stats`)
};
