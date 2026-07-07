import { Link } from 'react-router-dom';
import PipelineDiagram from '../components/PipelineDiagram';

const innovations = [
  {
    tag: 'ATB-RRF',
    title: 'Adaptive Tree-Boosted Rank Fusion',
    body: 'Standard RRF merges rankings with a fixed constant. ATB-RRF computes a per-chunk boost that adapts to section affinity, tree depth, and cross-signal agreement.',
    icon: '⚡',
  },
  {
    tag: 'Ontology',
    title: 'Bidirectional Section Affinity',
    body: 'A 13-type academic section ontology routes queries to the right sections — and actively dampens mismatched ones with a negative-affinity penalty.',
    icon: '🧭',
  },
  {
    tag: 'Depth',
    title: 'Depth-Weighted Navigation',
    body: 'Deeper, more specific sub-sections earn a specificity bonus (1 + 0.15·d), so precise passages outrank generic top-level containers.',
    icon: '🌳',
  },
  {
    tag: 'Confidence',
    title: 'Retrieval Confidence Score',
    body: 'A [0–1] metric measures agreement between structural and vector signals. When they disagree, the structural boost is attenuated to prevent noise.',
    icon: '🎯',
  },
  {
    tag: 'Coherence',
    title: 'Coherent-Narrative Re-ranking',
    body: 'After fusion, chunks from the same or adjacent tree nodes are grouped, producing context windows that read in the document’s natural order.',
    icon: '📖',
  },
  {
    tag: 'Fallback',
    title: 'Graceful Degradation',
    body: 'No tree index or API key? The system transparently falls back to pure vector retrieval — nothing breaks, results just get simpler.',
    icon: '🛟',
  },
];

const stats = [
  { value: 'Dual', label: 'Index architecture' },
  { value: '13', label: 'Section ontology types' },
  { value: '6', label: 'Novel innovations' },
  { value: '100%', label: 'Cited, grounded answers' },
];

export default function HomePage(): JSX.Element {
  return (
    <div className="space-y-16">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-ink-950 px-6 py-14 text-white shadow-soft sm:px-10 sm:py-20">
        <div className="pointer-events-none absolute inset-0 bg-hero-radial" aria-hidden />
        <div className="pointer-events-none absolute inset-0 bg-grid [background-size:34px_34px] opacity-40" aria-hidden />
        <div className="relative mx-auto max-w-3xl text-center">
          <span className="inline-flex animate-fade-up items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3.5 py-1.5 text-xs font-semibold text-brand-100 backdrop-blur">
            <span className="h-1.5 w-1.5 rounded-full bg-fuchsia-400" />
            StructRAG · structure-aware retrieval
          </span>
          <h1 className="mt-6 animate-fade-up text-4xl font-extrabold leading-[1.1] tracking-tight sm:text-6xl">
            Ask academic papers
            <span className="block bg-gradient-to-r from-brand-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent">
              structure-aware questions
            </span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl animate-fade-up text-base text-ink-200 sm:text-lg">
            Upload a PDF and get grounded, cited answers. Under the hood, a semantic section-tree and a vector index are
            fused with <span className="font-semibold text-white">Adaptive Tree-Boosted Reciprocal Rank Fusion</span> —
            retrieval that understands a document’s structure, not just its words.
          </p>
          <div className="mt-8 flex animate-fade-up flex-col items-center justify-center gap-3 sm:flex-row">
            <Link to="/upload" className="btn-primary w-full px-6 py-3 text-base sm:w-auto">
              Upload a paper
            </Link>
            <a href="#how-it-works" className="btn w-full border border-white/20 bg-white/5 px-6 py-3 text-base text-white hover:bg-white/10 sm:w-auto">
              See how it works
            </a>
          </div>
        </div>

        <div className="relative mx-auto mt-12 grid max-w-3xl grid-cols-2 gap-3 sm:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4 text-center backdrop-blur">
              <div className="bg-gradient-to-r from-brand-200 to-fuchsia-200 bg-clip-text text-2xl font-extrabold text-transparent">
                {s.value}
              </div>
              <div className="mt-1 text-[11px] font-medium uppercase tracking-wide text-ink-400">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Vanilla vs StructRAG */}
      <section className="space-y-6">
        <div className="text-center">
          <p className="eyebrow">Why it’s different</p>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-ink-900">Beyond vanilla RAG</h2>
          <p className="mx-auto mt-3 max-w-2xl text-ink-600">
            Most RAG systems treat a document as a flat bag of chunks. StructRAG keeps the document’s hierarchy and uses
            it as a second retrieval signal.
          </p>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <div className="card border-ink-200">
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-100 text-lg">📎</span>
              <h3 className="text-lg font-bold text-ink-800">Vanilla RAG</h3>
            </div>
            <ul className="space-y-2.5 text-sm text-ink-600">
              <li className="flex gap-2"><span className="text-ink-400">—</span> One signal: cosine similarity over chunk embeddings.</li>
              <li className="flex gap-2"><span className="text-ink-400">—</span> Blind to sections; can surface a fragment out of context.</li>
              <li className="flex gap-2"><span className="text-ink-400">—</span> Fixed ranking, no notion of its own confidence.</li>
              <li className="flex gap-2"><span className="text-ink-400">—</span> Chunks arrive in similarity order, not reading order.</li>
            </ul>
          </div>

          <div className="card relative overflow-hidden border-transparent shadow-glow">
            <span className="absolute inset-x-0 top-0 h-1 bg-brand-gradient" aria-hidden />
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-gradient text-lg">🧠</span>
              <h3 className="text-lg font-bold text-ink-900">StructRAG (this project)</h3>
            </div>
            <ul className="space-y-2.5 text-sm text-ink-700">
              <li className="flex gap-2"><span className="text-brand-500">✓</span> Two signals fused: a section-tree <em>and</em> vector search.</li>
              <li className="flex gap-2"><span className="text-brand-500">✓</span> Section ontology routes queries and penalises mismatches.</li>
              <li className="flex gap-2"><span className="text-brand-500">✓</span> Scores cross-signal confidence and self-corrects when unsure.</li>
              <li className="flex gap-2"><span className="text-brand-500">✓</span> Re-ranks into a coherent, narratively ordered context.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section id="how-it-works" className="scroll-mt-20 space-y-8">
        <div className="text-center">
          <p className="eyebrow">The pipeline</p>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-ink-900">How a question gets answered</h2>
          <p className="mx-auto mt-3 max-w-2xl text-ink-600">
            Each PDF is indexed two ways at once. At query time, both indexes are fused — with an adaptive boost — then
            gated by confidence before the LLM ever sees the context.
          </p>
        </div>
        <div className="card border-ink-200/70 bg-white/70 p-6 sm:p-8">
          <PipelineDiagram />
        </div>
      </section>

      {/* Innovations */}
      <section className="space-y-8">
        <div className="text-center">
          <p className="eyebrow">The novelty</p>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-ink-900">Six innovations over prior art</h2>
          <p className="mx-auto mt-3 max-w-2xl text-ink-600">
            StructRAG is designed to be novel over RAPTOR, PageIndex and classic RRF. Here’s what makes it distinct.
          </p>
        </div>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {innovations.map((item) => (
            <article
              key={item.tag}
              className="group card border-ink-200/70 transition hover:-translate-y-1 hover:border-brand-300 hover:shadow-glow"
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand-50 text-2xl transition group-hover:scale-110">
                  {item.icon}
                </span>
                <span className="chip">{item.tag}</span>
              </div>
              <h3 className="text-base font-bold text-ink-900">{item.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-600">{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative overflow-hidden rounded-3xl bg-brand-gradient px-6 py-12 text-center text-white shadow-soft sm:px-10">
        <div className="pointer-events-none absolute inset-0 bg-grid [background-size:30px_30px] opacity-20" aria-hidden />
        <div className="relative mx-auto max-w-xl">
          <h2 className="text-2xl font-bold sm:text-3xl">Try it on your own paper</h2>
          <p className="mx-auto mt-3 max-w-md text-white/85">
            Upload a PDF, watch it get indexed, then ask anything — every answer comes back with its sources.
          </p>
          <Link to="/upload" className="btn mt-6 bg-white px-6 py-3 text-base font-semibold text-brand-700 hover:bg-ink-50">
            Upload a PDF →
          </Link>
        </div>
      </section>
    </div>
  );
}
