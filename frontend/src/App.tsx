import { Link, Route, Routes } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import DocsPage from './pages/DocsPage';
import UploadPage from './pages/UploadPage';

export default function App(): JSX.Element {
  return (
    <div className="min-h-screen bg-transparent px-4 py-6 md:px-8">
      <header className="mx-auto mb-6 flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 rounded-2xl border border-slate-200 bg-white/80 px-5 py-4 shadow-soft backdrop-blur-sm">
        <div>
          <h1 className="text-xl font-bold text-slatebrand-900">RAG Academic Explainer</h1>
          <p className="text-sm text-slate-600">Upload papers, ask grounded questions, inspect citations.</p>
        </div>
        <nav className="flex gap-2 text-sm font-medium">
          <Link className="rounded-lg border border-slate-300 px-3 py-1.5 hover:bg-slate-100" to="/">
            Upload
          </Link>
          <Link className="rounded-lg border border-slate-300 px-3 py-1.5 hover:bg-slate-100" to="/docs">
            Documents
          </Link>
        </nav>
      </header>
      <main className="mx-auto w-full max-w-6xl">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/docs" element={<DocsPage />} />
          <Route path="/chat/:docId" element={<ChatPage />} />
        </Routes>
      </main>
    </div>
  );
}
