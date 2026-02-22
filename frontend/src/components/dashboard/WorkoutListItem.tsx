"use client";

import Link from "next/link";
import type { WorkoutSummary } from "@/lib/dashboard";

interface WorkoutListItemProps {
  workout: WorkoutSummary;
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function WorkoutListItem({ workout }: WorkoutListItemProps) {
  const riskCount = workout.injury_risks?.length ?? 0;
  const clusterCount = Object.keys(workout.clusters ?? {}).length;

  return (
    <Link
      href={`/dashboard/workout/${workout._id}`}
      className="block no-underline"
    >
      <div className="bg-white rounded-xl p-4 border border-[#E5E5E5] hover:border-[#4FC3F7] transition-colors">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-semibold text-[#3C3C3C]">
            {workout.video_name ?? workout.activity_label ?? "Workout"}
          </p>
          <p className="text-xs text-[#AFAFAF]">
            {formatDate(workout.created_at)}
          </p>
        </div>
        <div className="flex gap-4 text-xs text-[#777777]">
          <span>{formatDuration(workout.duration_sec)}</span>
          <span>{clusterCount} movement{clusterCount !== 1 ? "s" : ""}</span>
          {riskCount > 0 && (
            <span className={riskCount > 3 ? "text-[#EF5350]" : "text-[#FFB74D]"}>
              {riskCount} risk{riskCount !== 1 ? "s" : ""}
            </span>
          )}
          {workout.fatigue_score != null && (
            <span>Fatigue: {Math.round(workout.fatigue_score * 100)}%</span>
          )}
        </div>
      </div>
    </Link>
  );
}
