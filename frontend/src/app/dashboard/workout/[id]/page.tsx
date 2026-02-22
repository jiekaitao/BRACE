"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { fetchWorkout, type WorkoutSummary } from "@/lib/dashboard";
import BiomechanicsChart from "@/components/dashboard/BiomechanicsChart";

export default function WorkoutDetailPage() {
  const params = useParams();
  const router = useRouter();
  const workoutId = params.id as string;

  const [workout, setWorkout] = useState<WorkoutSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!workoutId) return;
    fetchWorkout(workoutId).then((data) => {
      setWorkout(data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [workoutId]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[#AFAFAF]">
        Loading workout...
      </div>
    );
  }

  if (!workout) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <p className="text-[#777777]">Workout not found</p>
        <button
          onClick={() => router.push("/dashboard")}
          className="text-sm text-[#4FC3F7] hover:underline"
        >
          Back to Dashboard
        </button>
      </div>
    );
  }

  const risks = (workout.injury_risks ?? []) as Array<{
    risk_name?: string;
    joint?: string;
    severity?: string;
    value?: number;
  }>;

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-8">
      <div className="w-full max-w-3xl">
        <button
          onClick={() => router.push("/dashboard")}
          className="text-xs text-[#4FC3F7] hover:underline mb-4"
        >
          &larr; Back to Dashboard
        </button>

        <h1 className="text-xl font-extrabold text-[#3C3C3C] mb-1">
          {workout.video_name ?? "Workout Detail"}
        </h1>
        <p className="text-sm text-[#AFAFAF] mb-6">
          {new Date(workout.created_at).toLocaleString()} &middot;{" "}
          {Math.round(workout.duration_sec)}s
        </p>

        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          <div className="bg-[#F7F7F7] rounded-xl p-3 border border-[#E5E5E5] text-center">
            <p className="text-lg font-bold text-[#3C3C3C]">
              {Object.keys(workout.clusters ?? {}).length}
            </p>
            <p className="text-xs text-[#AFAFAF]">Movements</p>
          </div>
          <div className="bg-[#F7F7F7] rounded-xl p-3 border border-[#E5E5E5] text-center">
            <p className="text-lg font-bold text-[#3C3C3C]">
              {risks.length}
            </p>
            <p className="text-xs text-[#AFAFAF]">Injury Risks</p>
          </div>
          <div className="bg-[#F7F7F7] rounded-xl p-3 border border-[#E5E5E5] text-center">
            <p className="text-lg font-bold text-[#3C3C3C]">
              {workout.fatigue_score != null
                ? `${Math.round(workout.fatigue_score * 100)}%`
                : "N/A"}
            </p>
            <p className="text-xs text-[#AFAFAF]">Fatigue</p>
          </div>
        </div>

        {/* Injury Risks Table */}
        {risks.length > 0 && (
          <div className="mb-6">
            <p className="text-xs font-bold text-[#3C3C3C] mb-2 uppercase tracking-wider">
              Detected Injury Risks
            </p>
            <div className="bg-white rounded-xl border border-[#E5E5E5] overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-[#F7F7F7] text-[#777777]">
                    <th className="text-left p-2">Risk</th>
                    <th className="text-left p-2">Joint</th>
                    <th className="text-left p-2">Severity</th>
                    <th className="text-right p-2">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {risks.map((r, i) => (
                    <tr key={i} className="border-t border-[#E5E5E5]">
                      <td className="p-2 text-[#3C3C3C]">{r.risk_name ?? "unknown"}</td>
                      <td className="p-2 text-[#777777]">{r.joint ?? "-"}</td>
                      <td className="p-2">
                        <span
                          className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
                            r.severity === "high"
                              ? "bg-[#FFEBEE] text-[#C62828]"
                              : "bg-[#FFF8E1] text-[#F57F17]"
                          }`}
                        >
                          {r.severity ?? "medium"}
                        </span>
                      </td>
                      <td className="p-2 text-right text-[#3C3C3C]">
                        {r.value != null ? r.value.toFixed(1) : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
