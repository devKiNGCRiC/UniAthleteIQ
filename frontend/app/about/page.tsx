"use client";

// Plain-language explanation of what the model does and how accurate it is.
// Numbers come from the live model bundle, so this page never goes stale.

import { useEffect, useState } from "react";

import { api, type ModelInfo } from "@/lib/api";

export default function AboutPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  useEffect(() => {
    api.model().then(setInfo).catch(() => setInfo(null));
  }, []);

  return (
    <article className="max-w-2xl space-y-6 text-lg leading-relaxed">
      <h1 className="font-display text-4xl font-bold tracking-tight">How the risk estimate works</h1>

      <p>
        Each day, the model looks at the previous three weeks of an athlete&apos;s training log: how far they ran,
        how much was fast, how many rest days they took, and how hard and how recovered they felt. It compares
        those numbers with the athlete&apos;s own usual pattern and estimates the chance of an injury that day.
      </p>

      <p>
        Risk is shown relative to an average day. <strong>2×</strong> means twice an average day&apos;s risk.
        Because injuries are rare (about {info ? (info.base_rate * 100).toFixed(1) : "1.3"}% of days), even a
        doubled risk is still a small absolute chance.
      </p>

      <h2 className="font-display text-2xl font-semibold">How accurate is it?</h2>
      <p>
        Tested on athletes it had never seen, the model ranks a risky day above a safe one about{" "}
        {info ? Math.round(info.cv_roc_auc * 100) : 63}% of the time (50% would be guessing). Flagging the riskiest
        fifth of days catches roughly a third of injuries. That makes it useful for deciding whom to check in
        with, and not for diagnosing anyone.
      </p>

      <h2 className="font-display text-2xl font-semibold">What it can&apos;t do</h2>
      <ul className="list-disc space-y-2 pl-6">
        <li>It was trained on competitive runners only. Other sports are not supported yet.</li>
        <li>It needs a log for every day; unlogged days are treated as rest.</li>
        <li>Its reasons are patterns, not causes. Use them to start a conversation.</li>
      </ul>

      {info && (
        <p className="text-base text-ink-soft">
          Model {info.version}, trained on {info.n_athletes} athletes, {new Date(info.trained_at).toLocaleDateString()}.
        </p>
      )}
    </article>
  );
}
