import { Link } from 'react-router-dom';
import type { DocumentInfo } from '../services/api';

interface Props {
  docs: DocumentInfo[];
  onDelete: (docId: string) => void;
  onOpenChat: (docId: string) => void;
}

export default function DocList({ docs, onDelete, onOpenChat }: Props): JSX.Element {
  if (docs.length === 0) {
    return (
      <div className="card flex flex-col items-center gap-3 py-12 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 text-2xl">📄</div>
        <div>
          <p className="font-semibold text-ink-800">No documents yet</p>
          <p className="mt-1 text-sm text-ink-500">Upload a PDF to start asking structure-aware questions.</p>
        </div>
        <Link to="/upload" className="btn-primary mt-1">
          Upload your first PDF
        </Link>
      </div>
    );
  }

  return (
    <ul className="space-y-3">
      {docs.map((doc) => (
        <li
          key={doc.id}
          className="group flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-ink-200/70 bg-white/80 p-4 shadow-sm transition hover:border-brand-300 hover:shadow-card"
        >
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-brand-gradient text-lg text-white">
              📘
            </div>
            <div className="min-w-0">
              <h3 className="truncate font-semibold text-ink-900">{doc.title}</h3>
              <p className="text-xs text-ink-500">
                {doc.page_count} pages · {new Date(doc.upload_date).toLocaleDateString(undefined, { dateStyle: 'medium' })}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn-primary px-3 py-2 text-xs" onClick={() => onOpenChat(doc.id)}>
              Open chat
            </button>
            <button type="button" className="btn-danger px-3 py-2 text-xs" onClick={() => onDelete(doc.id)}>
              Delete
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
