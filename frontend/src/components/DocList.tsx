import type { DocumentInfo } from '../services/api';

interface Props {
  docs: DocumentInfo[];
  onDelete: (docId: string) => void;
  onOpenChat: (docId: string) => void;
}

export default function DocList({ docs, onDelete, onOpenChat }: Props): JSX.Element {
  if (docs.length === 0) {
    return <p className="text-sm text-slate-600">No documents uploaded yet.</p>;
  }

  return (
    <ul className="space-y-3">
      {docs.map((doc) => (
        <li key={doc.id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="font-semibold text-slate-900">{doc.title}</h3>
              <p className="text-xs text-slate-500">
                {new Date(doc.upload_date).toLocaleString()} | {doc.page_count} pages
              </p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                className="rounded-lg border border-slate-300 px-3 py-1 text-sm hover:bg-slate-100"
                onClick={() => onOpenChat(doc.id)}
              >
                Open Chat
              </button>
              <button
                type="button"
                className="rounded-lg border border-red-300 px-3 py-1 text-sm text-red-700 hover:bg-red-50"
                onClick={() => onDelete(doc.id)}
              >
                Delete
              </button>
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}
