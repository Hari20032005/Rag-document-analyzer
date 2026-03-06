import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
    <section className="card space-y-4">
      <h2 className="page-title">Uploaded Documents</h2>
      {loading ? <Loader label="Loading documents..." /> : null}
      {error ? <p className="text-sm text-red-700">{error}</p> : null}
      {!loading ? (
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
