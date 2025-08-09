"use client";
import { useRef, useState } from 'react';

export function FileUpload({ onUpload }: { onUpload: (files: FileList) => Promise<void> }) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);

  async function choose() {
    const el = inputRef.current;
    if (!el) return;
    el.click();
  }

  async function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files) return;
    setBusy(true);
    try {
      await onUpload(e.target.files);
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  }

  return (
    <div className="flex items-center gap-2">
      <input ref={inputRef} type="file" className="hidden" accept="application/pdf" multiple onChange={onChange} />
      <button onClick={choose} disabled={busy} className="rounded-md border px-3 py-1.5 text-sm hover:bg-gray-50 disabled:opacity-50">
        {busy ? 'Uploading…' : 'Upload PDFs'}
      </button>
    </div>
  );
}
