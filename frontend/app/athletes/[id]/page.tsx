"use client";

// One athlete: today's risk, why, how it has moved, and their recent training.

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { RiskChart } from "@/components/RiskChart";
import { StatusBadge } from "@/components/StatusBadge";
import { api, ApiError, type AthleteRisk, type DailyLog, type Reason } from "@/lib/api";
import { isoDay, reasonValue, relative, riskLevel, shortDate } from "@/lib/format";

const RANGES = [30, 90, 180];

export default function AthletePage() {
  const id = Number(useParams<{ id: string }>().id);
  const [days, setDays] = useState(90);
  const [risk, setRisk] = useState<AthleteRisk | null>(null);
  const [logs, setLogs] = useState<DailyLog[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.risk(id, days).then(setRisk).catch((e: ApiError) => setError(e.message));
  }, [id, days]);

  useEffect(() => {
    const since = new Date();
    since.setDate(since.getDate() - 14);
    api.logs(id, isoDay(since)).then((l) => setLogs(l.reverse())).catch(() => setLogs([]));
  }, [id]);

  if (error) return <p role="alert" className="rounded-md bg-crimson-soft p-4 text-crimson">{error}</p>;
  if (!risk) return <p className="text-ink-soft">Loading…</p>;

  const latest = risk.latest;
  const level = latest ? riskLevel(latest) : null;

  return (
    <div className="space-y-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link href="/" className="text-sm text-ink-soft hover:text-ink">Team</Link>
          <h1 className="font-display text-4xl font-bold tracking-tight">{risk.athlete.name}</h1>
        </div>
        <Link href={`/athletes/${id}/log`} className="rounded-md bg-pine px-4 py-2 font-medium text-surface">
          Log training
        </Link>
      </div>

      {!latest || !level ? (
        <p className="text-ink-soft">No training logged yet. Log a few days to see a risk estimate.</p>
      ) : (
        <>
          <section className="space-y-3">
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
              <p className="font-display text-5xl font-semibold tnum">{relative(latest.relative_risk)}</p>
              <p className="text-lg text-ink-soft">an average day&apos;s injury risk, today</p>
              <StatusBadge level={level} />
            </div>
            <DataQuality latest={latest} />
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between gap-4">
              <h2 className="font-display text-2xl font-semibold">Risk over time</h2>
              <div role="group" aria-label="Time range" className="flex rounded-md border border-line">
                {RANGES.map((r) => (
                  <button key={r} onClick={() => setDays(r)} aria-pressed={days === r}
                          className={`px-3 py-1 text-sm ${days === r ? "bg-pine text-surface" : "text-ink-soft"}`}>
                    {r} days
                  </button>
                ))}
              </div>
            </div>
            <RiskChart timeline={risk.timeline} />
            <p className="text-sm text-ink-soft">Dashed line: an average day. Dots: days above the alert threshold.</p>
          </section>

          <section className="grid gap-8 md:grid-cols-2">
            <ReasonList title="Pushing risk up" reasons={risk.reasons_up} empty="Nothing notable." />
            <ReasonList title="Keeping risk down" reasons={risk.reasons_down} empty="Nothing notable." />
            <p className="text-sm text-ink-soft md:col-span-2">
              These are patterns the model learned, not causes. For example, a very easy day yesterday can raise the
              estimate because runners often ease off when something starts to hurt.
            </p>
          </section>
        </>
      )}

      <section className="space-y-3">
        <h2 className="font-display text-2xl font-semibold">Last two weeks</h2>
        {logs.length === 0 ? (
          <p className="text-ink-soft">No logs in the last 14 days.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[32rem] text-left text-sm tnum">
              <thead className="text-ink-soft">
                <tr className="border-b border-line">
                  <th className="py-2 font-medium">Day</th>
                  <th className="py-2 font-medium">Km</th>
                  <th className="py-2 font-medium">Hard km</th>
                  <th className="py-2 font-medium">Exertion</th>
                  <th className="py-2 font-medium">Recovery</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((l) => (
                  <tr key={l.date} className="border-b border-line">
                    <td className="py-2">{shortDate(l.date)}</td>
                    <td className="py-2">{l.sessions === 0 ? "Rest" : l.total_km.toFixed(1)}</td>
                    <td className="py-2">{(l.km_z5_t1_t2 + l.km_sprinting).toFixed(1)}</td>
                    <td className="py-2">{score10(l.perceived_exertion)}</td>
                    <td className="py-2">{score10(l.perceived_recovery)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function score10(v: number | null) {
  return v === null ? "–" : `${Math.round(v * 10)}/10`;
}

function DataQuality({ latest }: { latest: NonNullable<AthleteRisk["latest"]> }) {
  // Tell the coach when the estimate rests on thin data, rather than hiding it
  const notes: string[] = [];
  if (latest.days_logged_last_21 < 18)
    notes.push(`Only ${latest.days_logged_last_21} of the last 21 days are logged; unlogged days count as rest.`);
  if (!latest.has_full_history) notes.push("Less than 3 weeks of history, so weekly load trends are incomplete.");
  if (!latest.baseline_ready) notes.push("Personal baseline still forming (needs about 4 weeks of logs).");
  if (notes.length === 0) return null;
  return (
    <ul className="space-y-1 text-sm text-amber">
      {notes.map((n) => <li key={n}>{n}</li>)}
    </ul>
  );
}

function ReasonList({ title, reasons, empty }: { title: string; reasons: Reason[]; empty: string }) {
  return (
    <div>
      <h3 className="mb-2 font-display text-xl font-semibold">{title}</h3>
      {reasons.length === 0 ? (
        <p className="text-ink-soft">{empty}</p>
      ) : (
        <ul className="divide-y divide-line border-y border-line">
          {reasons.map((r) => (
            <li key={r.feature} className="flex items-baseline justify-between gap-4 py-2">
              <span>{r.label}</span>
              <span className="shrink-0 text-sm text-ink-soft tnum">{reasonValue(r.feature, r.value)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
