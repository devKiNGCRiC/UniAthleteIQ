// The "lane" visual: risk shown as a marker along a horizontal track, with a fixed
// tick at 1× (an average day). Scale is logarithmic so 0.5× and 2× sit symmetrically
// around the average, which matches how relative risk should be read.

import type { Level } from "@/lib/format";

const MIN = 0.25, MAX = 4; // display range of relative risk (values outside are pinned to the ends)

function position(rel: number): number {
  const clamped = Math.min(MAX, Math.max(MIN, rel));
  return ((Math.log(clamped) - Math.log(MIN)) / (Math.log(MAX) - Math.log(MIN))) * 100;
}

const MARKER: Record<Level, string> = {
  check: "bg-crimson",
  elevated: "bg-amber",
  normal: "bg-pine",
};

export function RiskTrack({ relative, level }: { relative: number; level: Level }) {
  const x = position(relative);
  const avg = position(1);
  return (
    <div className="relative h-6 w-full" role="img" aria-label={`${relative.toFixed(1)} times an average day's risk`}>
      {/* the track */}
      <div className="absolute top-1/2 h-1 w-full -translate-y-1/2 rounded-full bg-line" />
      {/* fill from the average line to the athlete's position */}
      <div
        className={`absolute top-1/2 h-1 -translate-y-1/2 rounded-full ${MARKER[level]} opacity-40`}
        style={{ left: `${Math.min(x, avg)}%`, width: `${Math.abs(x - avg)}%` }}
      />
      {/* the 1× average tick */}
      <div className="absolute top-0 h-6 w-px bg-ink-soft" style={{ left: `${avg}%` }} />
      {/* the athlete */}
      <div
        className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface ${MARKER[level]}`}
        style={{ left: `${x}%` }}
      />
    </div>
  );
}

export function TrackScale() {
  // Axis labels shown once above the list
  const ticks = [0.5, 1, 2, 4];
  return (
    <div className="relative h-4 w-full text-xs text-ink-soft tnum" aria-hidden>
      {ticks.map((t) => (
        <span key={t} className="absolute -translate-x-1/2" style={{ left: `${position(t)}%` }}>
          {t === 1 ? "1× avg" : `${t}×`}
        </span>
      ))}
    </div>
  );
}
