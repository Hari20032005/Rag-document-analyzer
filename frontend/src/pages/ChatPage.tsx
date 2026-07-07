import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import ChatBox, { type ChatMessage } from '../components/ChatBox';
import Loader from '../components/Loader';
import { askQuestion, listDocuments, type DocumentInfo } from '../services/api';

const confidenceStyles: Record<string, string> = {
  High: 'bg-emerald-100 text-emerald-700',
  Medium: 'bg-amber-100 text-amber-700',
  Low: 'bg-rose-100 text-rose-700',
};

export default function ChatPage(): JSX.Element {
  const { docId } = useParams();
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<'explain' | 'summary' | 'qa'>('qa');
  const [temperature, setTemperature] = useState(0.2);

  useEffect(() => {
    void (async () => {
      const fetchedDocs = await listDocuments();
      setDocs(fetchedDocs);
    })();
  }, []);

  const activeDoc = useMemo(() => docs.find((doc) => doc.id === docId), [docs, docId]);

  async function handleSend(question: string): Promise<void> {
    if (!docId) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text: question,
    };
    setMessages((prev) => [...prev, userMessage]);
    setRecentQueries((prev) => [question, ...prev.slice(0, 4)]);

    setLoading(true);
    try {
      const response = await askQuestion({ doc_id: docId, question, top_k: topK, mode, temperature });
      const assistantMessage: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        text: response.answer,
        sources: response.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const fallbackMessage: ChatMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        text: err instanceof Error ? err.message : 'Failed to get answer',
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setLoading(false);
    }
  }

  const confidence = useMemo(() => {
    const latest = [...messages].reverse().find((message) => message.role === 'assistant' && message.sources?.length);
    if (!latest?.sources?.length) {
      return null;
    }
    const avg = latest.sources.reduce((acc, source) => acc + source.score, 0) / latest.sources.length;
    if (avg > 0.75) return 'High';
    if (avg > 0.45) return 'Medium';
    return 'Low';
  }, [messages]);

  return (
    <div className="space-y-4">
      <Link to="/docs" className="inline-flex items-center gap-1 text-sm font-medium text-ink-500 transition hover:text-brand-700">
        ← Back to documents
      </Link>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <aside className="card h-fit space-y-5">
          <div>
            <p className="eyebrow">Document</p>
            {!activeDoc ? (
              <div className="mt-2">
                <Loader label="Loading…" />
              </div>
            ) : (
              <div className="mt-2 space-y-2">
                <p className="font-bold leading-snug text-ink-900">{activeDoc.title}</p>
                <div className="flex flex-wrap gap-1.5 text-xs">
                  <span className="rounded-md bg-ink-100 px-2 py-1 font-medium text-ink-600">{activeDoc.page_count} pages</span>
                  <span className="rounded-md bg-ink-100 px-2 py-1 font-medium text-ink-600">
                    {new Date(activeDoc.upload_date).toLocaleDateString(undefined, { dateStyle: 'medium' })}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-xl border border-ink-200 bg-white/70 p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-ink-700">Retrieval confidence</span>
              <span
                className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                  confidence ? confidenceStyles[confidence] : 'bg-ink-100 text-ink-500'
                }`}
              >
                {confidence ?? 'N/A'}
              </span>
            </div>
            <p className="mt-1.5 text-[11px] leading-snug text-ink-500">
              Cross-signal agreement between the section-tree and vector retrieval.
            </p>
          </div>

          <div>
            <h3 className="mb-2 text-sm font-semibold text-ink-700">Recent queries</h3>
            {recentQueries.length === 0 ? (
              <p className="text-xs text-ink-400">Your questions will appear here.</p>
            ) : (
              <ul className="space-y-1.5">
                {recentQueries.map((item, index) => (
                  <li
                    key={`${item}-${index}`}
                    className="truncate rounded-lg bg-ink-50 px-2.5 py-1.5 text-xs text-ink-600"
                    title={item}
                  >
                    {item}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        <ChatBox
          messages={messages}
          loading={loading}
          topK={topK}
          mode={mode}
          temperature={temperature}
          onTopKChange={setTopK}
          onModeChange={setMode}
          onTemperatureChange={setTemperature}
          onSend={(question) => {
            void handleSend(question);
          }}
        />
      </div>
    </div>
  );
}
