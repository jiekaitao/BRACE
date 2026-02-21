import { getApiBase } from "./api";
import { getToken } from "./auth";
import type {
  WorkoutSummary,
  WorkoutDetail,
  TrendData,
  InjuryEvent,
  InjuryGuidelines,
} from "./types";

function authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function fetchWorkouts(
  limit = 20,
  offset = 0,
): Promise<{ workouts: WorkoutSummary[]; total: number }> {
  const res = await fetch(
    `${getApiBase()}/api/dashboard/workouts?limit=${limit}&offset=${offset}`,
    { headers: authHeaders() },
  );
  if (!res.ok) throw new Error("Failed to fetch workouts");
  return res.json();
}

export async function fetchWorkout(id: string): Promise<WorkoutDetail> {
  const res = await fetch(`${getApiBase()}/api/dashboard/workouts/${id}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch workout");
  return res.json();
}

export async function fetchTrends(n = 20): Promise<TrendData> {
  const res = await fetch(`${getApiBase()}/api/dashboard/trends?n=${n}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch trends");
  return res.json();
}

export async function fetchInjuryHistory(
  limit = 50,
): Promise<{ events: InjuryEvent[] }> {
  const res = await fetch(
    `${getApiBase()}/api/dashboard/injury-history?limit=${limit}`,
    { headers: authHeaders() },
  );
  if (!res.ok) throw new Error("Failed to fetch injury history");
  return res.json();
}

export async function fetchGuidelines(
  injury_type: string,
  severity: string,
  location: string,
): Promise<InjuryGuidelines> {
  const res = await fetch(`${getApiBase()}/api/dashboard/guidelines`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ injury_type, severity, location }),
  });
  if (!res.ok) throw new Error("Failed to fetch guidelines");
  return res.json();
}
