import { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ChatBox, { type ChatMessage } from '../components/ChatBox';
import Loader from '../components/Loader';
import { askQuestion, listDocuments, type DocumentInfo } from '../services/api';

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
      const response = await askQuestion({
        doc_id: docId,
        question,
        top_k: topK,
        mode,
        temperature,
      });

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
    if (avg > 0.75) {
      return 'High';
    }
    if (avg > 0.45) {
      return 'Medium';
    }
    return 'Low';
  }, [messages]);

  return (
    <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
      <aside className="card space-y-4">
        <h2 className="text-lg font-semibold text-slatebrand-900">Document Context</h2>
        {!activeDoc ? <Loader label="Loading document info..." /> : null}
        {activeDoc ? (
          <div className="space-y-1 text-sm text-slate-700">
            <p className="font-semibold">{activeDoc.title}</p>
            <p>{activeDoc.page_count} pages</p>
            <p>Uploaded: {new Date(activeDoc.upload_date).toLocaleString()}</p>
            <p>Confidence: {confidence ?? 'N/A'}</p>
          </div>
        ) : null}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-800">Recent Queries</h3>
          <ul className="space-y-2 text-xs text-slate-600">
            {recentQueries.length === 0 ? <li>None yet.</li> : null}
            {recentQueries.map((item, index) => (
              <li key={`${item}-${index}`} className="rounded bg-slate-100 px-2 py-1">
                {item}
              </li>
            ))}
          </ul>
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
  );
}
