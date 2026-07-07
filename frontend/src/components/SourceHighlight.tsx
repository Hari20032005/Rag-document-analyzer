import { useState } from 'react';
import type { SourceChunk } from '../services/api';

interface Props {
  source: SourceChunk;
}

export default function SourceHighlight({ source }: Props): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const snippet = expanded ? source.snippet : `${source.snippet.slice(0, 160)}${source.snippet.length > 160 ? '…' : ''}`;
  const pct = Math.round(source.score * 100);

  return (
    <div className="rounded-xl border border-ink-200/70 bg-ink-50/70 p-3 text-xs text-ink-600">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1 rounded-md bg-white px-2 py-0.5 font-semibold text-brand-700 shadow-sm">
          📄 Page {source.page}
        </span>
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-ink-200">
            <div className="h-1.5 rounded-full bg-brand-gradient" style={{ width: `${Math.max(pct, 4)}%` }} />
          </div>
          <span className="font-mono font-semibold text-ink-500">{pct}%</span>
        </div>
      </div>
      <p className="leading-relaxed">{snippet}</p>
      {source.snippet.length > 160 ? (
        <button
          className="mt-2 text-[11px] font-semibold text-brand-600 transition hover:text-brand-700"
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      ) : null}
    </div>
  );
}
