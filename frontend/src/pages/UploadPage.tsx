import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUploader from '../components/FileUploader';
import Loader from '../components/Loader';
import { getJobStatus, type JobStatusResponse } from '../services/api';

export default function UploadPage(): JSX.Element {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!jobId) {
      return;
    }

    const poll = setInterval(async () => {
      try {
        const payload = await getJobStatus(jobId);
        setStatus(payload);
        if (payload.status === 'completed' || payload.status === 'failed') {
          clearInterval(poll);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to fetch status');
        clearInterval(poll);
      }
    }, 1000);

    return () => clearInterval(poll);
  }, [jobId]);

  return (
    <div className="space-y-4">
      <FileUploader
        onUploaded={(id) => {
          setError(null);
          setStatus(null);
          setJobId(id);
        }}
        onError={(message) => setError(message)}
      />

      {jobId ? (
        <div className="card space-y-2 text-sm text-slate-700">
          <p>
            Job ID: <span className="font-mono text-xs">{jobId}</span>
          </p>
          {status ? (
            <>
              <p>Status: {status.status}</p>
              <p>
                Progress: {status.progress}% | Pages: {status.pages_parsed}/{status.total_pages} | Chunks:{' '}
                {status.chunks_created}
              </p>
              {status.status === 'completed' && status.doc_id ? (
                <button
                  type="button"
                  className="rounded-lg bg-slatebrand-900 px-3 py-2 text-white"
                  onClick={() => navigate(`/chat/${status.doc_id}`)}
                >
                  Open Chat for Document
                </button>
              ) : null}
              {status.status === 'failed' ? <p className="text-red-700">{status.error}</p> : null}
            </>
          ) : (
            <Loader label="Processing document..." />
          )}
        </div>
      ) : null}

      {error ? <p className="text-sm text-red-700">{error}</p> : null}
    </div>
  );
}
