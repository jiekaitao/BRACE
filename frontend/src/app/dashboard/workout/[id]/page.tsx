"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";
import { fetchWorkout } from "@/lib/dashboard";
import type { WorkoutDetail, FatigueTimeline } from "@/lib/types";
import Card from "@/components/ui/Card";
import BiomechanicsChart from "@/components/dashboard/BiomechanicsChart";

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

const SEVERITY_COLORS: Record<string, string> = {
  low: "#FF9600",
  medium: "#FF9600",
  high: "#EA2B2B",
};

function FatigueChart({
  timeline,
  width = 400,
  height = 100,
}: {
  timeline: FatigueTimeline;
  width?: number;
  height?: number;
}) {
  const { timestamps, fatigue, form_scores } = timeline;
  const n = timestamps.length;
  if (n < 2) return null;

  const pad = 6;
  const tMin = timestamps[0];
  const tMax = timestamps[n - 1];
  const tRange = Math.max(tMax - tMin, 1);

  function x(t: number) {
    return pad + ((t - tMin) / tRange) * (width - 2 * pad);
  }

  const fatiguePath = fatigue
    .map((f, i) => {
      const px = x(timestamps[i]);
      const py = height - pad - f * (height - 2 * pad);
      return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");

  const formPath = form_scores
    .map((f, i) => {
      const px = x(timestamps[i]);
      const py = height - pad - (f / 100) * (height - 2 * pad);
      return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div>
      <svg
        width={width}
        height={height}
        className="w-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
      >
        <path d={fatiguePath} fill="none" stroke="#FF9600" strokeWidth="2" strokeLinecap="round" />
        <path d={formPath} fill="none" stroke="#58CC02" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <div className="flex gap-4 mt-1">
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-[#FF9600] rounded-full" />
          <span className="text-[11px] text-[#777777]">Fatigue</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-[#58CC02] rounded-full" />
          <span className="text-[11px] text-[#777777]">Form Score</span>
        </div>
      </div>
    </div>
  );
}

export default function WorkoutDetailPage() {
  const router = useRouter();
  const params = useParams();
  const { user, loading: authLoading } = useAuth();
  const [workout, setWorkout] = useState<WorkoutDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const id = params.id as string;

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      router.push("/onboarding");
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchWorkout(id);
        if (!cancelled) setWorkout(data);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [id, user, authLoading, router]);

  if (authLoading || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-[#AFAFAF] text-sm">Loading...</div>
      </div>
    );
  }

  if (error || !workout) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <div className="text-sm text-[#EA2B2B]">{error || "Workout not found"}</div>
        <Link href="/dashboard" className="text-sm font-bold text-[#1CB0F6] no-underline">
          &larr; Back to Dashboard
        </Link>
      </div>
    );
  }

  const clusterEntries = Object.entries(workout.clusters || {});
  const primaryActivity =
    workout.activity_labels
      ? Object.values(workout.activity_labels)[0] || "unknown"
      : "unknown";

  return (
    <div className="min-h-screen px-5 py-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link
          href="/dashboard"
          className="text-sm font-bold text-[#1CB0F6] hover:underline no-underline"
        >
          &larr; Dashboard
        </Link>
      </div>

      {/* Workout Header Card */}
      <Card className="mb-4">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <h1 className="text-lg font-extrabold text-[#3C3C3C]">
              {formatDate(workout.created_at)}
            </h1>
            <div className="flex items-center gap-2 text-sm text-[#777777]">
              <span>{formatDuration(workout.duration_sec)}</span>
              <span className="capitalize font-bold text-[#1CB0F6]">{primaryActivity}</span>
            </div>
          </div>
          <div className="flex gap-3 ml-auto">
            {workout.form_score_avg !== null && (
              <div className="text-center">
                <div className="text-2xl font-extrabold text-[#58CC02]">
                  {Math.round(workout.form_score_avg)}
                </div>
                <div className="text-[11px] text-[#AFAFAF] uppercase">Form</div>
              </div>
            )}
            {workout.fatigue_score !== null && (
              <div className="text-center">
                <div className="text-2xl font-extrabold text-[#FF9600]">
                  {workout.fatigue_score.toFixed(2)}
                </div>
                <div className="text-[11px] text-[#AFAFAF] uppercase">Fatigue</div>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Cluster Cards */}
      {clusterEntries.length > 0 && (
        <div className="mb-4">
          <h2 className="text-base font-extrabold text-[#3C3C3C] mb-2">
            Movement Clusters
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {clusterEntries.map(([cid, info]) => {
              const label =
                workout.activity_labels?.[cid] || info.activity_label || `Cluster ${cid}`;
              return (
                <Card key={cid}>
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-sm font-bold text-[#3C3C3C] capitalize">
                      {label}
                    </span>
                    <span className="text-xs text-[#AFAFAF]">
                      {info.count} reps
                    </span>
                  </div>
                  <div className="flex gap-3 text-xs">
                    <div>
                      <span className="text-[#777777]">Score: </span>
                      <span className="font-bold text-[#4B4B4B]">
                        {(info.mean_score * 100).toFixed(0)}
                      </span>
                    </div>
                    {info.composite_fatigue !== undefined && (
                      <div>
                        <span className="text-[#777777]">Fatigue: </span>
                        <span className="font-bold text-[#4B4B4B]">
                          {info.composite_fatigue.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {info.anomaly_count > 0 && (
                      <div>
                        <span className="text-[#777777]">Anomalies: </span>
                        <span className="font-bold text-[#EA2B2B]">
                          {info.anomaly_count}
                        </span>
                      </div>
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {/* Biomechanics Timeline */}
      {workout.biomechanics_timeline && workout.biomechanics_timeline.length > 0 && (
        <div className="mb-4">
          <BiomechanicsChart data={workout.biomechanics_timeline} />
        </div>
      )}

      {/* Fatigue Timeline */}
      {workout.fatigue_timeline && workout.fatigue_timeline.timestamps.length > 1 && (
        <div className="mb-4">
          <Card>
            <h3 className="text-base font-extrabold text-[#3C3C3C] mb-2">
              Fatigue Timeline
            </h3>
            <FatigueChart timeline={workout.fatigue_timeline} />
          </Card>
        </div>
      )}

      {/* Injury Risks */}
      {workout.injury_risks && workout.injury_risks.length > 0 && (
        <div className="mb-4">
          <h2 className="text-base font-extrabold text-[#3C3C3C] mb-2">
            Injury Risk Events
          </h2>
          <div className="space-y-1">
            {workout.injury_risks.map((risk, i) => (
              <Card key={i}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{
                        backgroundColor: SEVERITY_COLORS[risk.severity] || "#AFAFAF",
                      }}
                    />
                    <span className="text-sm font-bold text-[#3C3C3C]">
                      {risk.risk}
                    </span>
                    <span className="text-xs text-[#777777]">{risk.joint}</span>
                  </div>
                  <div className="text-xs text-[#777777]">
                    {risk.value.toFixed(1)} / {risk.threshold.toFixed(1)}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
