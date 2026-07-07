import { useState } from 'react';
import type { AskResponse } from '../services/api';
import Loader from './Loader';
import MessageBubble from './MessageBubble';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources?: AskResponse['sources'];
}

interface Props {
  messages: ChatMessage[];
  loading: boolean;
  topK: number;
  mode: 'explain' | 'summary' | 'qa';
  temperature: number;
  onTopKChange: (value: number) => void;
  onModeChange: (value: 'explain' | 'summary' | 'qa') => void;
  onTemperatureChange: (value: number) => void;
  onSend: (question: string) => void;
}

const modes: { value: Props['mode']; label: string }[] = [
  { value: 'qa', label: 'Q&A' },
  { value: 'explain', label: 'Explain' },
  { value: 'summary', label: 'Summary' },
];

const suggestions = [
  'What problem does this paper solve?',
  'Summarise the methodology.',
  'What are the key results?',
  'What are the limitations?',
];

export default function ChatBox({
  messages,
  loading,
  topK,
  mode,
  temperature,
  onTopKChange,
  onModeChange,
  onTemperatureChange,
  onSend,
}: Props): JSX.Element {
  const [input, setInput] = useState('');

  function submit(text: string): void {
    if (!text.trim() || loading) {
      return;
    }
    onSend(text.trim());
    setInput('');
  }

  return (
    <div className="card flex min-h-[68vh] flex-col gap-4 p-4 sm:p-5">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-3 border-b border-ink-100 pb-4">
        <div className="inline-flex rounded-xl border border-ink-200 bg-ink-50 p-1">
          {modes.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => onModeChange(m.value)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                mode === m.value ? 'bg-white text-brand-700 shadow-sm' : 'text-ink-500 hover:text-ink-700'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-2 text-xs font-medium text-ink-600">
          Top-K
          <select
            className="rounded-lg border border-ink-200 bg-white px-2 py-1.5 text-xs font-semibold text-ink-800 focus:border-brand-400 focus:outline-none"
            value={topK}
            onChange={(event) => onTopKChange(Number(event.target.value))}
          >
            {[3, 5, 7, 10].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-1 items-center gap-2 text-xs font-medium text-ink-600">
          Temp
          <input
            className="flex-1 accent-brand-600"
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={temperature}
            onChange={(event) => onTemperatureChange(Number(event.target.value))}
          />
          <span className="w-8 font-mono text-ink-800">{temperature.toFixed(1)}</span>
        </label>
      </div>

      {/* messages */}
      <div className="scroll-slim flex-1 space-y-4 overflow-y-auto rounded-2xl bg-gradient-to-b from-ink-50/60 to-white p-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-5 py-8 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-gradient text-2xl text-white shadow-glow">
              💬
            </div>
            <div>
              <p className="font-semibold text-ink-800">Ask anything about this document</p>
              <p className="mt-1 text-sm text-ink-500">Every answer is grounded in retrieved passages with citations.</p>
            </div>
            <div className="flex max-w-md flex-wrap justify-center gap-2">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => submit(s)}
                  className="rounded-full border border-ink-200 bg-white px-3 py-1.5 text-xs font-medium text-ink-600 transition hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} role={message.role} text={message.text} sources={message.sources} />
          ))
        )}
        {loading ? (
          <div className="flex items-center gap-2 rounded-2xl bg-white px-4 py-3 shadow-sm">
            <Loader label="Retrieving & generating…" />
          </div>
        ) : null}
      </div>

      {/* composer */}
      <form
        className="flex items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit(input);
        }}
      >
        <input
          className="field flex-1"
          placeholder="Ask about methods, results, assumptions…"
          value={input}
          onChange={(event) => setInput(event.target.value)}
        />
        <button className="btn-primary px-5" type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
