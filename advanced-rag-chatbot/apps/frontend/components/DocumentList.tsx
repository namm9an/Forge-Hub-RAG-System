"use client";
import type { DocumentInfo } from '../lib/api';

export function DocumentList({ docs, onDelete }: { docs: DocumentInfo[]; onDelete: (id: string) => void }) {
  return (
    <aside className="rounded-lg border p-4">
      <h2 className="mb-3 text-sm font-medium text-gray-700">Documents</h2>
      {docs.length === 0 ? (
        <div className="text-sm text-gray-500">No documents uploaded.</div>
      ) : (
        <ul className="space-y-2 text-sm">
          {docs.map((d) => (
            <li key={d.id} className="flex items-center justify-between gap-2 rounded-md border px-2 py-1.5">
              <div>
                <div className="font-medium">{d.filename}</div>
                <div className="text-xs text-gray-500">{d.pages ?? 0} pages · {d.chunks ?? 0} chunks</div>
              </div>
              <button onClick={() => onDelete(d.id)} className="rounded-md border px-2 py-1 text-xs hover:bg-gray-50">
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
