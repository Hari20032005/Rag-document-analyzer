import { useState } from 'react';
import type { SourceChunk } from '../services/api';

interface Props {
  source: SourceChunk;
}

export default function SourceHighlight({ source }: Props): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const snippet = expanded ? source.snippet : `${source.snippet.slice(0, 160)}${source.snippet.length > 160 ? '...' : ''}`;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="font-semibold">Page {source.page}</span>
        <span className="rounded bg-slate-200 px-2 py-0.5">{Math.round(source.score * 100)}% match</span>
      </div>
      <p>{snippet}</p>
      {source.snippet.length > 160 ? (
        <button
          className="mt-2 text-xs font-medium text-blue-700 hover:underline"
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      ) : null}
    </div>
  );
}
