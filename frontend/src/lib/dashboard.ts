import { getApiBase } from "./api";

// --- Types ---

export interface WorkoutSummary {
  _id: string;
  user_id: string;
  video_name?: string;
  duration_sec: number;
  clusters: Record<string, unknown>;
  injury_risks: unknown[];
  fatigue_score?: number;
  concussion_rating?: number;
  game_id?: string;
  activity_label?: string;
  created_at: string;
}

export interface TrendPoint {
  date: string | null;
  value: number;
  duration_sec?: number;
  video_name?: string;
}

export interface InjuryRiskTrendPoint {
  date: string | null;
  total_risks: number;
  risk_counts: Record<string, number>;
}

export interface GameSummary {
  _id: string;
  session_id: string;
  video_name: string;
  sport?: string;
  status: "pending" | "processing" | "complete" | "error";
  player_count: number;
  total_frames: number;
  progress: number;
  created_at: string;
}

export interface GamePlayer {
  _id: string;
  game_id: string;
  subject_id: number;
  label: string;
  jersey_number?: number | null;
  jersey_color?: string | null;
  risk_status: "GREEN" | "YELLOW" | "RED";
  total_frames: number;
  injury_events: unknown[];
  workload: Record<string, unknown>;
}

export interface Guideline {
  _id: string;
  user_id: string;
  activity: string;
  guidelines: string[];
  injury_context?: string | null;
  created_at: string;
}

// --- API Functions ---

export async function fetchWorkouts(userId: string, limit = 20): Promise<WorkoutSummary[]> {
  const res = await fetch(`${getApiBase()}/api/dashboard/workouts?user_id=${userId}&limit=${limit}`);
  const data = await res.json();
  return data.workouts ?? [];
}

export async function fetchWorkout(workoutId: string): Promise<WorkoutSummary | null> {
  const res = await fetch(`${getApiBase()}/api/dashboard/workouts/${workoutId}`);
  const data = await res.json();
  return data.error ? null : data;
}

export async function fetchTrends(userId: string, metric = "fatigue_score", limit = 30): Promise<TrendPoint[]> {
  const res = await fetch(
    `${getApiBase()}/api/dashboard/trends?user_id=${userId}&metric=${metric}&limit=${limit}`
  );
  const data = await res.json();
  return data.points ?? [];
}

export async function fetchInjuryRiskTrends(userId: string, limit = 30): Promise<InjuryRiskTrendPoint[]> {
  const res = await fetch(
    `${getApiBase()}/api/dashboard/trends/injury-risks?user_id=${userId}&limit=${limit}`
  );
  const data = await res.json();
  return data.points ?? [];
}

export async function fetchGames(userId?: string, limit = 20): Promise<GameSummary[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${getApiBase()}/api/dashboard/games?${params}`);
  const data = await res.json();
  return data.games ?? [];
}

export async function fetchGamePlayers(gameId: string): Promise<GamePlayer[]> {
  const res = await fetch(`${getApiBase()}/api/dashboard/games/${gameId}/players`);
  const data = await res.json();
  return data.players ?? [];
}

export async function fetchGuidelines(userId: string, activity?: string): Promise<Guideline[]> {
  const params = new URLSearchParams({ user_id: userId });
  if (activity) params.set("activity", activity);
  const res = await fetch(`${getApiBase()}/api/dashboard/guidelines?${params}`);
  const data = await res.json();
  return data.guidelines ?? [];
}

export async function submitGame(sessionId: string, sport = "basketball"): Promise<{ game_id: string }> {
  const res = await fetch(
    `${getApiBase()}/api/games?session_id=${sessionId}&sport=${sport}`,
    { method: "POST" }
  );
  return res.json();
}
