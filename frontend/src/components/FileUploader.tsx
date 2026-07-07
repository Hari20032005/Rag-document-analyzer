import { type DragEvent, useRef, useState } from 'react';
import { uploadPdf } from '../services/api';

interface Props {
  onUploaded: (jobId: string) => void;
  onError: (message: string) => void;
}

export default function FileUploader({ onUploaded, onError }: Props): JSX.Element {
  const [title, setTitle] = useState('');
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  async function handleFile(file: File): Promise<void> {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      onError('Please upload a PDF file.');
      return;
    }

    setUploading(true);
    setProgress(0);
    try {
      const response = await uploadPdf(file, title, setProgress);
      onUploaded(response.job_id);
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Upload failed';
      onError(detail);
    } finally {
      setUploading(false);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      void handleFile(dropped);
    }
  }

  return (
    <div className="card space-y-5">
      <div>
        <div className="flex items-center gap-2">
          <span className="chip">Step 1</span>
          <h2 className="text-xl font-bold tracking-tight text-ink-900">Upload an academic PDF</h2>
        </div>
        <p className="mt-1.5 text-sm text-ink-600">
          Drop a paper below. It gets parsed, chunked, embedded and structured into a section-tree before you can ask
          questions.
        </p>
      </div>

      <div>
        <label className="label" htmlFor="title">
          Document title <span className="font-normal text-ink-400">(optional)</span>
        </label>
        <input
          id="title"
          className="field"
          placeholder="e.g., Attention Is All You Need"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
      </div>

      <div
        className={`rounded-2xl border-2 border-dashed px-4 py-12 text-center transition ${
          dragging ? 'border-brand-400 bg-brand-50/70' : 'border-ink-200 bg-ink-50/60 hover:border-brand-300'
        }`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-gradient text-2xl text-white shadow-glow">
          ⬆
        </div>
        <p className="mb-1 text-sm font-semibold text-ink-800">Drag &amp; drop your PDF here</p>
        <p className="mb-4 text-xs text-ink-500">or</p>
        <button className="btn-ghost" type="button" onClick={() => inputRef.current?.click()} disabled={uploading}>
          Browse files
        </button>
        <input
          ref={inputRef}
          className="hidden"
          type="file"
          accept="application/pdf"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              void handleFile(file);
            }
          }}
        />
      </div>

      {uploading ? (
        <div>
          <div className="mb-1.5 flex justify-between text-xs font-medium text-ink-600">
            <span>Uploading…</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-ink-100">
            <div className="h-2 rounded-full bg-brand-gradient transition-all duration-200" style={{ width: `${progress}%` }} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
