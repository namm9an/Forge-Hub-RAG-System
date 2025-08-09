"use client";
import { useRef, useState } from 'react';
import type { ChatMessage, Source } from '../lib/api';

function SourceChips({ sources }: { sources?: Source[] }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-500">
      {sources.map((s, i) => (
        <span key={i} className="rounded-full border px-2 py-0.5">#{s.id.slice(0, 6)} {s.score !== undefined ? `(${s.score.toFixed(2)})` : ''}</span>
      ))}
    </div>
  );
}

export function ChatInterface({
  messages,
  onSend,
  busy
}: {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  busy: boolean;
}) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <section className="rounded-lg border p-4">
      <div className="mb-4 h-[60vh] overflow-y-auto pr-2">
        {messages.length === 0 ? (
          <div className="text-sm text-gray-500">No messages yet. Upload PDFs and ask a question.</div>
        ) : (
          <ul className="space-y-4">
            {messages.map((m, i) => (
              <li key={i} className={m.role === 'user' ? 'text-right' : ''}>
                <div
                  className={`inline-block max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
                    m.role === 'user' ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  {m.content}
                </div>
                {m.role === 'assistant' && <SourceChips sources={m.sources} />}
              </li>
            ))}
            {busy && (
              <li className="text-left">
                <div className="inline-block rounded-2xl bg-gray-100 px-4 py-2 text-sm text-gray-500">Assistant is typing…</div>
              </li>
            )}
          </ul>
        )}
      </div>

      <div className="flex items-center gap-2">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && input.trim()) {
              onSend(input.trim());
              setInput('');
            }
          }}
          placeholder="Ask something about your PDFs…"
          className="flex-1 rounded-md border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-gray-300"
        />
        <button
          onClick={() => {
            if (input.trim()) {
              onSend(input.trim());
              setInput('');
              inputRef.current?.focus();
            }
          }}
          disabled={busy || !input.trim()}
          className="rounded-md bg-gray-900 px-4 py-2 text-sm text-white hover:bg-black disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </section>
  );
}
