/**
 * Visual explanation of the StructRAG retrieval pipeline.
 * Reads left-to-right on desktop, top-to-bottom on mobile.
 */

function Arrow(): JSX.Element {
  return (
    <div className="flex items-center justify-center text-brand-400" aria-hidden>
      {/* right chevron on desktop, down chevron on mobile */}
      <svg className="hidden h-6 w-6 lg:block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
        <path d="M5 12h13M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <svg className="h-6 w-6 lg:hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
        <path d="M12 5v13M6 13l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

interface NodeProps {
  icon: string;
  title: string;
  subtitle: string;
  accent: string; // tailwind ring/border/text accent classes
  highlight?: boolean;
}

function Node({ icon, title, subtitle, accent, highlight = false }: NodeProps): JSX.Element {
  return (
    <div
      className={`relative w-full rounded-2xl border p-4 text-center shadow-sm transition lg:w-44 ${
        highlight ? 'border-transparent bg-brand-gradient shadow-glow' : `bg-white ${accent}`
      }`}
    >
      <div
        className={`mx-auto mb-2 flex h-11 w-11 items-center justify-center rounded-xl text-xl ${
          highlight ? 'bg-white/25' : 'bg-ink-50'
        }`}
      >
        <span>{icon}</span>
      </div>
      <p className={`text-sm font-bold ${highlight ? 'text-white' : 'text-ink-900'}`}>{title}</p>
      <p className={`mt-1 text-[11px] leading-snug ${highlight ? 'text-white/80' : 'text-ink-500'}`}>{subtitle}</p>
    </div>
  );
}

export default function PipelineDiagram(): JSX.Element {
  return (
    <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-center lg:justify-center">
      <Node icon="📄" title="Academic PDF" subtitle="Parsed page-by-page with metadata" accent="border-ink-200" />

      <Arrow />

      {/* Dual index — the core idea */}
      <div className="relative rounded-2xl border-2 border-dashed border-violet-300 bg-violet-50/40 p-3">
        <span className="chip absolute -top-3 left-1/2 -translate-x-1/2 whitespace-nowrap border-violet-300 bg-white text-violet-700">
          Dual Index
        </span>
        <div className="flex flex-col gap-3">
          <Node
            icon="🌳"
            title="Section-Tree"
            subtitle="13-type academic ontology (no vectors needed)"
            accent="border-emerald-200 ring-1 ring-emerald-100"
          />
          <Node
            icon="🔢"
            title="Vector Index"
            subtitle="Dense MiniLM embeddings in ChromaDB"
            accent="border-brand-200 ring-1 ring-brand-100"
          />
        </div>
      </div>

      <Arrow />

      <div className="relative">
        <span className="absolute inset-0 -z-10 animate-pulse-ring rounded-2xl" aria-hidden />
        <Node icon="⚡" title="ATB-RRF Fusion" subtitle="Adaptive tree-boosted rank fusion" highlight accent="" />
      </div>

      <Arrow />

      <Node
        icon="🎯"
        title="Confidence Gate"
        subtitle="Cross-signal agreement damps tree noise"
        accent="border-sky-200 ring-1 ring-sky-100"
      />

      <Arrow />

      <Node icon="✅" title="Cited Answer" subtitle="Grounded, with page + chunk citations" accent="border-emerald-200 ring-1 ring-emerald-100" />
    </div>
  );
}
