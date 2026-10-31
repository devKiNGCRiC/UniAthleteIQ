// Shared wording for risk levels, so every screen describes risk the same way.

import type { RiskPoint } from "./api";

export type Level = "check" | "elevated" | "normal";

/** "check" = above the model's alert threshold; "elevated" = above average but not flagged. */
export function riskLevel(r: RiskPoint): Level {
  if (r.flagged) return "check";
  if (r.relative_risk >= 1.2) return "elevated";
  return "normal";
}

export const LEVEL_TEXT: Record<Level, string> = {
  check: "Check in",
  elevated: "Elevated",
  normal: "Usual",
};

export function relative(r: number): string {
  return `${r.toFixed(1)}×`;
}

export function shortDate(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export function isoDay(d: Date): string {
  // Local calendar day as YYYY-MM-DD (toISOString would shift to UTC)
  const y = d.getFullYear(), m = String(d.getMonth() + 1).padStart(2, "0"), day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** A feature value written for people, e.g. "12.0 km" or "+1.8 SD vs normal". */
export function reasonValue(feature: string, value: number | null): string {
  if (value === null) return "no data";
  if (feature.endsWith("_vs_personal")) return `${value >= 0 ? "+" : ""}${value.toFixed(1)} SD vs their normal`;
  if (feature.includes("km") && !feature.includes("ratio")) return `${value.toFixed(1)} km`;
  if (feature.startsWith("rest_days") || feature.startsWith("sessions")) return value.toFixed(0);
  if (feature.startsWith("exertion") || feature.startsWith("recovery") || feature.startsWith("training_success"))
    return `${(value * 10).toFixed(1)} / 10`;
  return value.toFixed(2);
}
