export default function Loader({ label = 'Loading...' }: { label?: string }): JSX.Element {
  return (
    <div className="flex items-center gap-2.5 text-sm text-ink-600" role="status" aria-live="polite">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-brand-200 border-t-brand-600" />
      <span>{label}</span>
    </div>
  );
}
