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

  return (
    <div className="card flex min-h-[65vh] flex-col gap-4">
      <div className="grid gap-3 md:grid-cols-3">
        <label className="text-sm text-slate-700">
          Top K
          <select
            className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-1 text-sm"
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

        <label className="text-sm text-slate-700">
          Mode
          <select
            className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-1 text-sm"
            value={mode}
            onChange={(event) => onModeChange(event.target.value as 'explain' | 'summary' | 'qa')}
          >
            <option value="qa">Q&A</option>
            <option value="explain">Explain</option>
            <option value="summary">Summary</option>
          </select>
        </label>

        <label className="text-sm text-slate-700">
          Temperature ({temperature.toFixed(1)})
          <input
            className="mt-2 w-full"
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={temperature}
            onChange={(event) => onTemperatureChange(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto rounded-xl border border-slate-200 bg-slate-50 p-3">
        {messages.length === 0 ? <p className="text-sm text-slate-500">Start by asking a question.</p> : null}
        {messages.map((message) => (
          <MessageBubble key={message.id} role={message.role} text={message.text} sources={message.sources} />
        ))}
      </div>

      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!input.trim() || loading) {
            return;
          }
          onSend(input.trim());
          setInput('');
        }}
      >
        <input
          className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"
          placeholder="Ask about methods, results, assumptions..."
          value={input}
          onChange={(event) => setInput(event.target.value)}
        />
        <button
          className="rounded-lg bg-slatebrand-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
          type="submit"
          disabled={loading}
        >
          Send
        </button>
      </form>

      {loading ? <Loader label="Generating answer..." /> : null}
    </div>
  );
}
