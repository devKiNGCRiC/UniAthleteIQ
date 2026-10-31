"use client";

// Team view: every athlete ranked by today's risk, like a race start list.

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { RiskTrack, TrackScale } from "@/components/RiskTrack";
import { StatusBadge } from "@/components/StatusBadge";
import { api, ApiError, type TeamRow } from "@/lib/api";
import { reasonValue, relative, riskLevel, shortDate } from "@/lib/format";

// Below this many logged days (of the last 21) the risk estimate is shaky
const MIN_LOGGED = 18;

export default function TeamPage() {
  const [rows, setRows] = useState<TeamRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.team().then(setRows).catch((e: ApiError) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const today = new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
  const flagged = rows?.filter((r) => r.risk?.flagged).length ?? 0;

  return (
    <div className="space-y-8">
      <section>
        {/* Rendered on the server at build time and again in the browser; the date text can differ */}
        <h1 className="font-display text-4xl font-bold tracking-tight" suppressHydrationWarning>{today}</h1>
        <p className="mt-2 max-w-2xl text-lg text-ink-soft">
          {rows === null
            ? "Loading today's risk…"
            : flagged === 0
              ? "No one is above the alert threshold today."
              : `${flagged} ${flagged === 1 ? "athlete is" : "athletes are"} worth a quick check-in today.`}
        </p>
      </section>

      {error && <p role="alert" className="rounded-md bg-crimson-soft p-4 text-crimson">{error}</p>}

      {rows && rows.length > 0 && (
        <section aria-label="Athletes ranked by today's risk">
          <div className="hidden grid-cols-[2rem_1fr_minmax(12rem,1.2fr)_7rem] gap-4 px-3 pb-2 md:grid">
            <span />
            <span />
            <TrackScale />
            <span />
          </div>
          <ol className="divide-y divide-line border-y border-line bg-surface">
            {rows.map((row, i) => (
              <TeamRowItem key={row.athlete.id} row={row} rank={i + 1} />
            ))}
          </ol>
        </section>
      )}

      {rows && rows.length === 0 && (
        <p className="text-ink-soft">No athletes yet. Add your first athlete below to start logging training.</p>
      )}

      <AddAthlete onAdded={load} />
    </div>
  );
}

function TeamRowItem({ row, rank }: { row: TeamRow; rank: number }) {
  const { athlete, risk, top_reason } = row;
  const level = risk ? riskLevel(risk) : null;
  return (
    <li className="grid grid-cols-[2rem_1fr] items-center gap-x-4 gap-y-2 px-3 py-4 md:grid-cols-[2rem_1fr_minmax(12rem,1.2fr)_7rem]">
      <span className="font-display text-2xl font-semibold text-ink-soft tnum" aria-label={`Rank ${rank}`}>
        {rank}
      </span>
      <div className="min-w-0">
        <Link href={`/athletes/${athlete.id}`} className="font-medium text-ink underline-offset-4 hover:underline">
          {athlete.name}
        </Link>
        {risk ? (
          <p className="truncate text-sm text-ink-soft">
            {top_reason
              ? `Main factor: ${top_reason.label} (${reasonValue(top_reason.feature, top_reason.value)})`
              : "No single factor stands out"}
            {risk.days_logged_last_21 < MIN_LOGGED && ` · only ${risk.days_logged_last_21}/21 days logged`}
          </p>
        ) : (
          <p className="text-sm text-ink-soft">No training logged yet</p>
        )}
      </div>
      {risk && level && (
        <>
          <div className="col-start-2 md:col-start-auto">
            <RiskTrack relative={risk.relative_risk} level={level} />
          </div>
          <div className="col-start-2 flex items-center gap-3 md:col-start-auto md:flex-col md:items-end md:gap-1">
            <span className="font-display text-xl font-semibold tnum">{relative(risk.relative_risk)}</span>
            <StatusBadge level={level} />
          </div>
        </>
      )}
      {row.last_log_date && (
        <span className="sr-only">Last log {shortDate(row.last_log_date)}</span>
      )}
    </li>
  );
}

function AddAthlete({ onAdded }: { onAdded: () => void }) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.createAthlete(name);
      setName("");
      onAdded();
    } catch (err) {
      setError((err as ApiError).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex max-w-md flex-wrap items-end gap-3">
      <label className="flex flex-1 flex-col gap-1 text-sm">
        <span className="text-ink-soft">Add an athlete</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          maxLength={100}
          placeholder="Full name"
          className="rounded-md border border-line bg-surface px-3 py-2 text-base text-ink"
        />
      </label>
      <button
        disabled={busy || !name.trim()}
        className="rounded-md bg-pine px-4 py-2 font-medium text-surface disabled:opacity-50"
      >
        Add athlete
      </button>
      {error && <p role="alert" className="w-full text-sm text-crimson">{error}</p>}
    </form>
  );
}
