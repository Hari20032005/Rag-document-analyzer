import { NavLink, Route, Routes } from 'react-router-dom';
import Brand from './components/Brand';
import ChatPage from './pages/ChatPage';
import DocsPage from './pages/DocsPage';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';

const navItems = [
  { to: '/', label: 'Overview', end: true },
  { to: '/upload', label: 'Upload', end: false },
  { to: '/docs', label: 'Documents', end: false },
];

export default function App(): JSX.Element {
  return (
    <div className="flex min-h-screen flex-col overflow-x-hidden">
      <header className="sticky top-0 z-30 border-b border-ink-200/60 bg-white/70 backdrop-blur-md">
        <div className="container-app flex h-16 items-center justify-between gap-3">
          <NavLink to="/" className="flex min-w-0 items-center gap-2.5">
            <Brand className="h-9 w-9 flex-shrink-0" />
            <div className="min-w-0 leading-tight">
              <span className="block text-[15px] font-bold tracking-tight text-ink-900">StructRAG</span>
              <span className="hidden text-[11px] font-medium text-ink-500 sm:block">Structure-Aware Explainer</span>
            </div>
          </NavLink>

          <nav className="flex flex-shrink-0 items-center gap-0.5 rounded-full border border-ink-200/70 bg-white/60 p-1 text-xs sm:gap-1 sm:text-sm">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-full px-2.5 py-1.5 font-medium transition sm:px-3.5 ${
                    isActive
                      ? 'bg-brand-gradient text-white shadow-sm'
                      : 'text-ink-600 hover:bg-brand-50 hover:text-brand-700'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 py-8 sm:py-10">
        <div className="container-app">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/docs" element={<DocsPage />} />
            <Route path="/chat/:docId" element={<ChatPage />} />
          </Routes>
        </div>
      </main>

      <footer className="border-t border-ink-200/60 bg-white/60 py-6 backdrop-blur-sm">
        <div className="container-app flex flex-col items-center justify-between gap-3 text-sm text-ink-500 sm:flex-row">
          <div className="flex items-center gap-2">
            <Brand className="h-5 w-5" />
            <span>StructRAG — dual-index retrieval with ATB-RRF fusion</span>
          </div>
          <a
            href="https://github.com/Hari20032005/Rag-document-analyzer"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-ink-600 transition hover:text-brand-700"
          >
            View source on GitHub →
          </a>
        </div>
      </footer>
    </div>
  );
}
