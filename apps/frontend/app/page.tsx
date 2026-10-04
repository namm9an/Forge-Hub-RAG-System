"use client";

import { useEffect, useMemo, useState } from 'react';
import { api, ChatMessage, DocumentInfo } from '../lib/api';
import { ChatInterface } from '../components/ChatInterface';
import { FileUpload } from '../components/FileUpload';
import { DocumentList } from '../components/DocumentList';

function useSessionId() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  useEffect(() => {
    const existing = localStorage.getItem('chat_session_id');
    if (existing) setSessionId(existing);
    else {
      const id = crypto.randomUUID();
      localStorage.setItem('chat_session_id', id);
      setSessionId(id);
    }
  }, []);
  return sessionId;
}

export default function HomePage() {
  const sessionId = useSessionId();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);

  useEffect(() => {
    if (!sessionId) return;
    (async () => {
      try {
        const h = await api.history(sessionId, 50);
        const merged: ChatMessage[] = [];
        h.messages
          .slice()
          .reverse()
          .forEach((m) => {
            merged.push({ role: 'user', content: m.user_message });
            merged.push({ role: 'assistant', content: m.assistant_message, sources: m.source_documents });
          });
        setMessages(merged);
        const d = await api.listDocuments(sessionId);
        setDocs(d.documents);
      } catch (e) {
        console.error(e);
      }
    })();
  }, [sessionId]);

  async function handleSend(text: string) {
    if (!sessionId) return;
    const q = text;
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setBusy(true);
    try {
      // streaming mode
      let acc = '';
      setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);
      const idx = messages.length + 1; // last pushed assistant index
      await api.chatStream(q, sessionId, (chunk) => {
        acc += chunk;
        setMessages((prev) => {
          const copy = [...prev];
          copy[idx] = { role: 'assistant', content: acc };
          return copy;
        });
      });
      // sources shown after next history load; optional fetch now
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(files: FileList) {
    if (!sessionId) return;
    await api.uploadDocuments(Array.from(files), sessionId);
    api.generateEmbeddings().catch(() => {});
    const d = await api.listDocuments(sessionId);
    setDocs(d.documents);
  }

  async function handleDeleteDoc(id: string) {
    await api.deleteDocument(id);
    const d = await api.listDocuments(sessionId || undefined);
    setDocs(d.documents);
  }

  return (
    <main className="min-h-screen bg-white text-gray-900">
      <div className="mx-auto max-w-6xl p-4 md:p-8">
        <header className="mb-6 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Advanced RAG Chatbot</h1>
          <FileUpload onUpload={handleUpload} />
        </header>

        <div className="grid gap-6 md:grid-cols-3">
          <div className="md:col-span-2">
            <ChatInterface messages={messages} onSend={handleSend} busy={busy} />
          </div>
          <DocumentList docs={docs} onDelete={handleDeleteDoc} />
        </div>
      </div>
    </main>
  );
}

