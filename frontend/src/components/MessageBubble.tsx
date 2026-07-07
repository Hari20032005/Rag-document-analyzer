import type { SourceChunk } from '../services/api';
import SourceHighlight from './SourceHighlight';

interface Props {
  role: 'user' | 'assistant';
  text: string;
  sources?: SourceChunk[];
}

export default function MessageBubble({ role, text, sources = [] }: Props): JSX.Element {
  const isAssistant = role === 'assistant';

  if (!isAssistant) {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl rounded-2xl rounded-br-md bg-brand-gradient px-4 py-2.5 text-sm text-white shadow-sm">
          <p className="whitespace-pre-wrap leading-relaxed">{text}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start gap-2.5">
      <div className="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-brand-gradient text-sm text-white shadow-sm">
        ✨
      </div>
      <div className="max-w-2xl rounded-2xl rounded-tl-md border border-ink-200/70 bg-white px-4 py-3 text-sm text-ink-800 shadow-sm">
        <p className="whitespace-pre-wrap leading-relaxed">{text}</p>
        {sources.length > 0 ? (
          <div className="mt-3 border-t border-ink-100 pt-3">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-400">
              Sources · {sources.length}
            </p>
            <div className="space-y-2">
              {sources.slice(0, 3).map((source) => (
                <SourceHighlight key={source.chunk_id} source={source} />
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
