// Risk status: colour + icon + words, so it never relies on colour alone.

import { LEVEL_TEXT, type Level } from "@/lib/format";

const STYLE: Record<Level, string> = {
  check: "bg-crimson-soft text-crimson",
  elevated: "bg-amber-soft text-amber",
  normal: "bg-pine-soft text-pine",
};

function Icon({ level }: { level: Level }) {
  // Simple geometric icons: triangle (check in), circle-dot (elevated), check mark (usual)
  if (level === "check")
    return (
      <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden>
        <path d="M8 1.5 15 14H1z" fill="currentColor" />
        <path d="M8 6v4M8 11.5v.5" stroke="var(--surface)" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    );
  if (level === "elevated")
    return (
      <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden>
        <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
        <circle cx="8" cy="8" r="2.5" fill="currentColor" />
      </svg>
    );
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden>
      <path d="m3 8.5 3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function StatusBadge({ level }: { level: Level }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-sm font-medium ${STYLE[level]}`}>
      <Icon level={level} />
      {LEVEL_TEXT[level]}
    </span>
  );
}
