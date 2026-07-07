interface Props {
  className?: string;
}

/** StructRAG brand mark — a small hierarchical tree node, echoing the dual-index structure. */
export default function Brand({ className = 'h-8 w-8' }: Props): JSX.Element {
  return (
    <svg viewBox="0 0 64 64" className={className} role="img" aria-label="StructRAG logo">
      <defs>
        <linearGradient id="brandGrad" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#6366f1" />
          <stop offset="0.5" stopColor="#8b5cf6" />
          <stop offset="1" stopColor="#d946ef" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#brandGrad)" />
      <g stroke="#ffffff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <path d="M32 20.5 L20 37.5 M32 20.5 L44 37.5 M23.4 42 H40.6" opacity="0.92" />
      </g>
      <g fill="#ffffff">
        <circle cx="32" cy="16" r="4.4" />
        <circle cx="19" cy="42" r="4.4" />
        <circle cx="45" cy="42" r="4.4" />
      </g>
    </svg>
  );
}
