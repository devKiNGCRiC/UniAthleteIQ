// Typed client for the FastAPI backend. Shapes mirror api/schemas.py.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Athlete = { id: number; name: string; sport: string };

export type RiskPoint = {
  date: string;
  probability: number;
  relative_risk: number; // probability / an average day's probability
  flagged: boolean;
  days_logged_last_21: number;
  has_full_history: boolean;
  baseline_ready: boolean;
};

export type Reason = { feature: string; label: string; value: number | null; contribution: number };

export type AthleteRisk = {
  athlete: Athlete;
  timeline: RiskPoint[];
  latest: RiskPoint | null;
  reasons_up: Reason[];
  reasons_down: Reason[];
};

export type TeamRow = {
  athlete: Athlete;
  risk: RiskPoint | null;
  top_reason: Reason | null;
  last_log_date: string | null;
};

export type DailyLog = {
  date: string;
  sessions: number;
  total_km: number;
  km_z3_4: number;
  km_z5_t1_t2: number;
  km_sprinting: number;
  strength_training: number;
  hours_alternative: number;
  perceived_exertion: number | null; // 0-1 scale in the API, shown as 0-10 in the UI
  perceived_training_success: number | null;
  perceived_recovery: number | null;
};

export type ModelInfo = {
  version: string;
  trained_at: string;
  n_athletes: number;
  cv_roc_auc: number;
  threshold: number;
  base_rate: number;
};

/** Error carrying the API's own message, so the UI can show what went wrong. */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, `Can't reach the API at ${API_URL}. Start it with: uvicorn api.main:app`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, describeError(body) ?? `Request failed (${res.status})`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

// FastAPI returns {detail: string} or {detail: [{loc, msg}, ...]} for validation errors
function describeError(body: unknown): string | null {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { loc?: string[]; msg?: string }) =>
      `${d.loc?.slice(1).join(".") || "input"}: ${d.msg}`).join("; ");
  }
  if (detail && typeof detail === "object" && "errors" in detail) {
    return (detail as { errors: string[] }).errors.join("; ");
  }
  return null;
}

export const api = {
  team: () => request<TeamRow[]>("/api/team/today"),
  model: () => request<ModelInfo>("/api/model"),
  createAthlete: (name: string) =>
    request<Athlete>("/api/athletes", { method: "POST", body: JSON.stringify({ name }) }),
  risk: (id: number, days = 90) => request<AthleteRisk>(`/api/athletes/${id}/risk?days=${days}`),
  logs: (id: number, start?: string) =>
    request<DailyLog[]>(`/api/athletes/${id}/logs${start ? `?start=${start}` : ""}`),
  saveLog: (id: number, date: string, log: Omit<DailyLog, "date">) =>
    request<DailyLog>(`/api/athletes/${id}/logs/${date}`, { method: "PUT", body: JSON.stringify(log) }),
};
