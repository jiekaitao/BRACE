"use client";

import { useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import DuoButton from "@/components/ui/DuoButton";
import CrashCollisionCard from "@/components/CrashCollisionCard";
import { useCrashAnalysis } from "@/hooks/useCrashAnalysis";

const RISK_COLORS: Record<string, string> = {
  LOW: "text-green-600",
  MODERATE: "text-yellow-600",
  HIGH: "text-orange-600",
  CRITICAL: "text-red-600",
};

export default function CrashAnalysisPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    phase,
    progress,
    collisionCount,
    subjectsTracked,
    result,
    error,
    startAnalysis,
    reset,
  } = useCrashAnalysis();

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) startAnalysis(file);
    },
    [startAnalysis]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) startAnalysis(file);
    },
    [startAnalysis]
  );

  return (
    <div className="min-h-screen flex flex-col items-center px-4 py-8">
      <div className="w-full max-w-2xl space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className="text-[#AFAFAF] hover:text-[#3C3C3C] transition"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
          <h1 className="text-xl font-semibold">Crash Analysis</h1>
        </div>

        {/* Upload Phase */}
        {phase === "idle" && (
          <div
            className="border-2 border-dashed border-[#E5E5E5] rounded-2xl p-12 text-center cursor-pointer hover:border-[#AFAFAF] transition"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <svg
              className="mx-auto mb-4 text-[#AFAFAF]"
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="text-[#777777] mb-2">
              Drop a video here or click to browse
            </p>
            <p className="text-[#AFAFAF] text-sm">
              Supports MP4, MOV, AVI formats
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleFileSelect}
            />
          </div>
        )}

        {/* Uploading Phase */}
        {phase === "uploading" && (
          <div className="rounded-2xl border border-[#E5E5E5] bg-white p-8 text-center space-y-4">
            <div className="animate-spin mx-auto w-8 h-8 border-2 border-[#E5E5E5] border-t-[#58CC02] rounded-full" />
            <p className="text-[#777777]">Uploading video...</p>
          </div>
        )}

        {/* Processing Phase */}
        {phase === "processing" && (
          <div className="rounded-2xl border border-[#E5E5E5] bg-white p-8 space-y-4">
            <div className="flex items-center justify-between text-sm text-[#777777]">
              <span>Analyzing collisions...</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-[#E5E5E5] rounded-full h-2 overflow-hidden">
              <div
                className="bg-green-500 h-full rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex gap-6 text-sm text-[#AFAFAF]">
              <span>{subjectsTracked} subjects tracked</span>
              <span>{collisionCount} collisions detected</span>
            </div>
          </div>
        )}

        {/* Error Phase */}
        {phase === "error" && (
          <div className="rounded-2xl border border-red-200 bg-red-50 p-8 text-center space-y-4">
            <p className="text-red-700">{error}</p>
            <DuoButton variant="secondary" onClick={reset}>
              Try Again
            </DuoButton>
          </div>
        )}

        {/* Results Phase */}
        {phase === "complete" && result && (
          <div className="space-y-6">
            {/* Overall Summary */}
            <div className="rounded-2xl border border-[#E5E5E5] bg-white p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-[#3C3C3C]">Overall Assessment</h2>
                <span
                  className={`text-lg font-bold ${RISK_COLORS[result.overall_risk] ?? "text-[#3C3C3C]"}`}
                >
                  {result.overall_risk}
                </span>
              </div>
              <p className="text-[#777777] text-sm">
                {result.overall_recommendation}
              </p>
              <div className="grid grid-cols-3 gap-4 pt-2 text-center text-sm">
                <div>
                  <div className="text-[#AFAFAF] text-xs">Duration</div>
                  <div className="font-mono text-[#3C3C3C]">{(result.duration_sec ?? 0).toFixed(1)}s</div>
                </div>
                <div>
                  <div className="text-[#AFAFAF] text-xs">Subjects</div>
                  <div className="font-mono text-[#3C3C3C]">{result.subjects_tracked ?? 0}</div>
                </div>
                <div>
                  <div className="text-[#AFAFAF] text-xs">Collisions</div>
                  <div className="font-mono text-[#3C3C3C]">
                    {result.collision_events?.length ?? 0}
                  </div>
                </div>
              </div>
            </div>

            {/* Collision Events */}
            {(result.collision_events?.length ?? 0) > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-[#777777] uppercase tracking-wider">
                  Collision Events
                </h3>
                {result.collision_events!.map((event) => (
                  <CrashCollisionCard key={event.event_id} event={event} />
                ))}
              </div>
            )}

            {(result.collision_events?.length ?? 0) === 0 && (
              <div className="rounded-2xl border border-green-200 bg-green-50 p-6 text-center">
                <p className="text-green-700 font-medium">
                  No collisions detected
                </p>
                <p className="text-[#AFAFAF] text-sm mt-1">
                  The video did not contain any detectable person-to-person
                  collisions.
                </p>
              </div>
            )}

            {/* Per-Subject Summaries */}
            {Object.keys(result.subject_summaries ?? {}).length > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-[#777777] uppercase tracking-wider">
                  Subject Summaries
                </h3>
                <div className="grid gap-2">
                  {Object.values(result.subject_summaries ?? {}).map((s) => (
                    <div
                      key={s.subject_id}
                      className="flex items-center justify-between rounded-lg border border-[#E5E5E5] bg-white px-4 py-3"
                    >
                      <span className="font-medium">
                        Subject {s.subject_id}
                      </span>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-[#AFAFAF]">
                          {s.collision_count} collision
                          {s.collision_count !== 1 ? "s" : ""}
                        </span>
                        <span
                          className={
                            RISK_COLORS[s.worst_risk_level] ?? "text-[#AFAFAF]"
                          }
                        >
                          {s.worst_risk_level}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3">
              <DuoButton variant="secondary" fullWidth onClick={reset}>
                Analyze Another Video
              </DuoButton>
              <DuoButton
                variant="green"
                fullWidth
                onClick={() => router.push("/")}
              >
                Back to Home
              </DuoButton>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
