"use client";

import type { WorkoutSummary } from "@/lib/types";
import Card from "@/components/ui/Card";

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

interface Props {
  workout: WorkoutSummary;
  onClick: () => void;
}

export default function WorkoutListItem({ workout, onClick }: Props) {
  const formColor =
    workout.form_score_avg !== null && workout.form_score_avg >= 80
      ? "#58CC02"
      : workout.form_score_avg !== null && workout.form_score_avg >= 60
        ? "#FF9600"
        : "#EA2B2B";

  const fatigueColor =
    workout.fatigue_score !== null && workout.fatigue_score < 0.4
      ? "#58CC02"
      : workout.fatigue_score !== null && workout.fatigue_score < 0.7
        ? "#FF9600"
        : "#EA2B2B";

  return (
    <Card interactive className="cursor-pointer" onClick={onClick}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-sm font-bold text-[#3C3C3C]">
              {formatDate(workout.created_at)}
            </span>
            <span className="text-xs text-[#AFAFAF]">
              {formatDuration(workout.duration_sec)}
            </span>
            <span className="text-xs font-bold text-[#1CB0F6] capitalize">
              {workout.activity}
            </span>
          </div>
          {workout.video_name && (
            <div className="text-[11px] text-[#AFAFAF] truncate">
              {workout.video_name}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {workout.form_score_avg !== null && (
            <span
              className="text-xs font-bold px-2 py-0.5 rounded-full"
              style={{
                color: formColor,
                backgroundColor: `${formColor}18`,
              }}
            >
              Form {Math.round(workout.form_score_avg)}
            </span>
          )}
          {workout.fatigue_score !== null && (
            <span
              className="text-xs font-bold px-2 py-0.5 rounded-full"
              style={{
                color: fatigueColor,
                backgroundColor: `${fatigueColor}18`,
              }}
            >
              F {workout.fatigue_score.toFixed(1)}
            </span>
          )}
          {workout.risk_count > 0 && (
            <span className="text-xs font-bold text-[#EA2B2B] bg-[#EA2B2B18] px-2 py-0.5 rounded-full">
              {workout.risk_count}
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}
