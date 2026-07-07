import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import DocList from '../components/DocList';
import Loader from '../components/Loader';
import { deleteDocument, listDocuments, type DocumentInfo } from '../services/api';

export default function DocsPage(): JSX.Element {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function fetchDocs(): Promise<void> {
    try {
      setLoading(true);
      setError(null);
      const response = await listDocuments();
      setDocs(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load documents');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void fetchDocs();
  }, []);

  return (
    <section className="mx-auto max-w-3xl space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow">Your library</p>
          <h2 className="mt-1 text-3xl font-bold tracking-tight text-ink-900">Documents</h2>
          <p className="mt-1 text-sm text-ink-600">Indexed papers you can chat with.</p>
        </div>
        <Link to="/upload" className="btn-primary">
          + Upload new
        </Link>
      </div>

      {loading ? (
        <div className="card">
          <Loader label="Loading documents…" />
        </div>
      ) : null}

      {error ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">{error}</div>
      ) : null}

      {!loading && !error ? (
        <DocList
          docs={docs}
          onDelete={(docId) => {
            void (async () => {
              await deleteDocument(docId);
              await fetchDocs();
            })();
          }}
          onOpenChat={(docId) => navigate(`/chat/${docId}`)}
        />
      ) : null}
    </section>
  );
}
