import type { SourceChunk } from '../services/api';
import SourceHighlight from './SourceHighlight';

interface Props {
  role: 'user' | 'assistant';
  text: string;
  sources?: SourceChunk[];
}

export default function MessageBubble({ role, text, sources = [] }: Props): JSX.Element {
  const isAssistant = role === 'assistant';

  return (
    <div className={`flex ${isAssistant ? 'justify-start' : 'justify-end'}`}>
      <div
        className={`max-w-3xl rounded-2xl px-4 py-3 text-sm shadow-sm ${
          isAssistant ? 'bg-white text-slate-800' : 'bg-slatebrand-900 text-white'
        }`}
      >
        <p className="whitespace-pre-wrap leading-relaxed">{text}</p>
        {isAssistant && sources.length > 0 ? (
          <div className="mt-3 space-y-2 border-t border-slate-200 pt-3">
            {sources.slice(0, 3).map((source) => (
              <SourceHighlight key={source.chunk_id} source={source} />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
