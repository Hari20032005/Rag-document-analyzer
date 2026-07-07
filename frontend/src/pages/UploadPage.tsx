import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUploader from '../components/FileUploader';
import { getJobStatus, type JobStatusResponse } from '../services/api';

function Stat({ label, value }: { label: string; value: string | number }): JSX.Element {
  return (
    <div className="rounded-xl border border-ink-200 bg-white/70 px-3 py-2.5 text-center">
      <div className="text-lg font-bold text-ink-900">{value}</div>
      <div className="text-[11px] font-medium uppercase tracking-wide text-ink-500">{label}</div>
    </div>
  );
}

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

  const done = status?.status === 'completed';
  const failed = status?.status === 'failed';

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <FileUploader
        onUploaded={(id) => {
          setError(null);
          setStatus(null);
          setJobId(id);
        }}
        onError={(message) => setError(message)}
      />

      {jobId ? (
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="chip">Step 2</span>
              <h3 className="font-bold text-ink-900">Indexing document</h3>
            </div>
            <span
              className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                done
                  ? 'bg-emerald-100 text-emerald-700'
                  : failed
                    ? 'bg-rose-100 text-rose-700'
                    : 'bg-brand-100 text-brand-700'
              }`}
            >
              {done ? 'Completed' : failed ? 'Failed' : 'Processing…'}
            </span>
          </div>

          <div>
            <div className="mb-1.5 flex justify-between text-xs font-medium text-ink-600">
              <span>Progress</span>
              <span>{Math.round(status?.progress ?? 0)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-ink-100">
              <div
                className={`h-2.5 rounded-full transition-all duration-300 ${failed ? 'bg-rose-400' : 'bg-brand-gradient'}`}
                style={{ width: `${status?.progress ?? 4}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <Stat label="Pages" value={status ? `${status.pages_parsed}/${status.total_pages}` : '—'} />
            <Stat label="Chunks" value={status?.chunks_created ?? '—'} />
            <Stat label="Status" value={status?.status ?? 'queued'} />
          </div>

          <p className="text-[11px] text-ink-400">
            First upload after the server has been idle can take ~30–60s while the embedding model warms up.
          </p>

          {done && status?.doc_id ? (
            <button type="button" className="btn-primary w-full py-3 text-base" onClick={() => navigate(`/chat/${status.doc_id}`)}>
              Open chat for this document →
            </button>
          ) : null}
          {failed ? <p className="text-sm font-medium text-rose-600">{status?.error}</p> : null}
        </div>
      ) : null}

      {error ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">{error}</div>
      ) : null}
    </div>
  );
}
