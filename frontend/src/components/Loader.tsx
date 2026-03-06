export default function Loader({ label = 'Loading...' }: { label?: string }): JSX.Element {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-600" role="status" aria-live="polite">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700" />
      <span>{label}</span>
    </div>
  );
}
