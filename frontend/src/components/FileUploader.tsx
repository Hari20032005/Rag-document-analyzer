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
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      void handleFile(dropped);
    }
  }

  return (
    <div className="card space-y-4">
      <div>
        <h2 className="page-title">Upload Academic PDF</h2>
        <p className="text-sm text-slate-600">Drag and drop a paper or select a file to start indexing.</p>
      </div>

      <label className="block text-sm font-medium text-slate-700" htmlFor="title">
        Document title (optional)
      </label>
      <input
        id="title"
        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
        placeholder="e.g., Attention Is All You Need"
        value={title}
        onChange={(event) => setTitle(event.target.value)}
      />

      <div
        className="rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-10 text-center"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        <p className="mb-3 text-sm text-slate-600">Drop your PDF here</p>
        <button
          className="rounded-lg border border-slate-400 px-4 py-2 text-sm font-medium hover:bg-slate-100"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
        >
          Select PDF
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
          <div className="mb-1 flex justify-between text-xs text-slate-600">
            <span>Upload progress</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200">
            <div className="h-2 rounded-full bg-slatebrand-900" style={{ width: `${progress}%` }} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
