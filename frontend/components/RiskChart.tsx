"use client";

// Relative risk over time. Log scale, like the team track, so 0.5× and 2× are
// equally far from the 1× average line. Flagged days are drawn as crimson dots.

import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import type { RiskPoint } from "@/lib/api";
import { relative, shortDate } from "@/lib/format";

type DotProps = { cx?: number; cy?: number; payload?: RiskPoint };

function FlagDot({ cx, cy, payload }: DotProps) {
  if (!payload?.flagged || cx === undefined || cy === undefined) return null;
  return <circle cx={cx} cy={cy} r={4} fill="var(--crimson)" stroke="var(--surface)" strokeWidth={2} />;
}

export function RiskChart({ timeline }: { timeline: RiskPoint[] }) {
  // Clamp to the plotted range so a single extreme day doesn't flatten the rest
  const data = timeline.map((p) => ({ ...p, shown: Math.min(4, Math.max(0.25, p.relative_risk)) }));
  return (
    <div className="h-64 w-full" role="img" aria-label="Risk relative to an average day over time">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={40}
                 tick={{ fill: "var(--ink-soft)", fontSize: 12 }} stroke="var(--line)" />
          <YAxis dataKey="shown" scale="log" domain={[0.25, 4]} ticks={[0.25, 0.5, 1, 2, 4]}
                 tickFormatter={(v: number) => `${v}×`} allowDataOverflow
                 tick={{ fill: "var(--ink-soft)", fontSize: 12 }} stroke="var(--line)" />
          <ReferenceLine y={1} stroke="var(--ink-soft)" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 6 }}
            labelFormatter={(d) => shortDate(String(d))}
            formatter={(_v, _n, item) => {
              const p = item.payload as RiskPoint;
              return [`${relative(p.relative_risk)} avg${p.flagged ? " (check in)" : ""}`, "Risk"];
            }}
          />
          <Line type="monotone" dataKey="shown" stroke="var(--pine)" strokeWidth={2}
                dot={<FlagDot />} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
