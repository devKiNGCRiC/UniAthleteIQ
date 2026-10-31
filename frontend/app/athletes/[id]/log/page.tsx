"use client";

// Daily log, designed to take ~30 seconds on a phone. Field names follow the
// research dataset the model was trained on; perceived scores are 0-10 here and
// sent to the API on its 0-1 scale.

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { api, ApiError, type DailyLog } from "@/lib/api";
import { isoDay } from "@/lib/format";

type Form = Omit<DailyLog, "date" | "perceived_exertion" | "perceived_training_success" | "perceived_recovery"> & {
  exertion: number;
  success: number;
  recovery: number;
};

const EMPTY: Form = {
  sessions: 1, total_km: 0, km_z3_4: 0, km_z5_t1_t2: 0, km_sprinting: 0,
  strength_training: 0, hours_alternative: 0, exertion: 5, success: 5, recovery: 5,
};

function fromLog(l: DailyLog): Form {
  const s = (v: number | null) => (v === null ? 5 : Math.round(v * 10));
  return { ...l, exertion: s(l.perceived_exertion), success: s(l.perceived_training_success), recovery: s(l.perceived_recovery) };
}

export default function LogPage() {
  const id = Number(useParams<{ id: string }>().id);
  const [date, setDate] = useState(isoDay(new Date()));
  const [form, setForm] = useState<Form>(EMPTY);
  const [showZones, setShowZones] = useState(false);
  const [status, setStatus] = useState<{ kind: "saved" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  // Load the existing entry for the chosen day, so saving edits rather than duplicates
  useEffect(() => {
    api.logs(id, date)
      .then((logs) => {
        const existing = logs.find((l) => l.date === date);
        setForm(existing ? fromLog(existing) : EMPTY);
      })
      .catch(() => setForm(EMPTY));
  }, [id, date]);

  const rest = form.sessions === 0;
  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.saveLog(id, date, {
        sessions: form.sessions,
        total_km: rest ? 0 : form.total_km,
        km_z3_4: rest ? 0 : form.km_z3_4,
        km_z5_t1_t2: rest ? 0 : form.km_z5_t1_t2,
        km_sprinting: rest ? 0 : form.km_sprinting,
        strength_training: form.strength_training,
        hours_alternative: form.hours_alternative,
        // Rest days have no perceived scores (matches the research data)
        perceived_exertion: rest ? null : form.exertion / 10,
        perceived_training_success: rest ? null : form.success / 10,
        perceived_recovery: rest ? null : form.recovery / 10,
      });
      setStatus({ kind: "saved", text: `Saved ${date}.` });
    } catch (err) {
      setStatus({ kind: "error", text: (err as ApiError).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={save} className="mx-auto max-w-md space-y-6">
      <div>
        <Link href={`/athletes/${id}`} className="text-sm text-ink-soft hover:text-ink">Back to athlete</Link>
        <h1 className="font-display text-3xl font-bold tracking-tight">Log training</h1>
      </div>

      <Field label="Day">
        <input type="date" value={date} max={isoDay(new Date())} onChange={(e) => { setDate(e.target.value); setStatus(null); }}
               className="input" required />
      </Field>

      <Field label="Running sessions">
        <div className="flex gap-2" role="group" aria-label="Running sessions">
          {[0, 1, 2, 3].map((n) => (
            <button type="button" key={n} onClick={() => set("sessions", n)} aria-pressed={form.sessions === n}
                    className={`flex-1 rounded-md border py-2 font-medium ${form.sessions === n
                      ? "border-pine bg-pine text-surface" : "border-line bg-surface text-ink"}`}>
              {n === 0 ? "Rest" : n}
            </button>
          ))}
        </div>
      </Field>

      {!rest && (
        <>
          <NumberField label="Total distance (km)" value={form.total_km} step={0.1} max={150}
                       onChange={(v) => set("total_km", v)} />

          <div>
            <button type="button" onClick={() => setShowZones((s) => !s)} aria-expanded={showZones}
                    className="text-sm font-medium text-pine underline underline-offset-4">
              {showZones ? "Hide intensity breakdown" : "Add intensity breakdown"}
            </button>
            {showZones && (
              <div className="mt-3 space-y-4 border-l-2 border-line pl-4">
                <NumberField label="Tempo / threshold km (Z3–4)" value={form.km_z3_4} step={0.1} max={150}
                             onChange={(v) => set("km_z3_4", v)} />
                <NumberField label="Race-pace and interval km (Z5)" value={form.km_z5_t1_t2} step={0.1} max={150}
                             onChange={(v) => set("km_z5_t1_t2", v)} />
                <NumberField label="Sprint km" value={form.km_sprinting} step={0.1} max={50}
                             onChange={(v) => set("km_sprinting", v)} />
              </div>
            )}
          </div>

          <Slider label="How hard did it feel?" low="Very easy" high="Maximal" value={form.exertion}
                  onChange={(v) => set("exertion", v)} />
          <Slider label="How well did training go?" low="Poorly" high="Very well" value={form.success}
                  onChange={(v) => set("success", v)} />
          <Slider label="How recovered do you feel?" low="Not at all" high="Fully" value={form.recovery}
                  onChange={(v) => set("recovery", v)} />
        </>
      )}

      <NumberField label="Strength sessions" value={form.strength_training} step={1} max={5}
                   onChange={(v) => set("strength_training", v)} />
      <NumberField label="Cross-training (hours)" value={form.hours_alternative} step={0.25} max={12}
                   onChange={(v) => set("hours_alternative", v)} />

      <button disabled={busy} className="w-full rounded-md bg-pine py-3 font-medium text-surface disabled:opacity-50">
        {busy ? "Saving…" : "Save log"}
      </button>
      {status && (
        <p role="status" className={status.kind === "saved" ? "text-pine" : "text-crimson"}>
          {status.text}{" "}
          {status.kind === "saved" && (
            <Link href={`/athletes/${id}`} className="underline underline-offset-4">See updated risk</Link>
          )}
        </p>
      )}
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium text-ink-soft">{label}</span>
      {children}
    </label>
  );
}

function NumberField({ label, value, step, max, onChange }:
  { label: string; value: number; step: number; max: number; onChange: (v: number) => void }) {
  return (
    <Field label={label}>
      <input type="number" inputMode="decimal" min={0} max={max} step={step} value={value}
             onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))} className="input" />
    </Field>
  );
}

function Slider({ label, low, high, value, onChange }:
  { label: string; low: string; high: string; value: number; onChange: (v: number) => void }) {
  return (
    <Field label={`${label} ${value}/10`}>
      <input type="range" min={0} max={10} step={1} value={value} onChange={(e) => onChange(Number(e.target.value))}
             className="w-full accent-pine" />
      <span className="flex justify-between text-xs text-ink-soft"><span>{low}</span><span>{high}</span></span>
    </Field>
  );
}
