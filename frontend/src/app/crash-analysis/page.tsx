"use client";

import { useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import DuoButton from "@/components/ui/DuoButton";
import CrashCollisionCard from "@/components/CrashCollisionCard";
import { useCrashAnalysis } from "@/hooks/useCrashAnalysis";

const RISK_COLORS: Record<string, string> = {
  LOW: "text-green-400",
  MODERATE: "text-yellow-400",
  HIGH: "text-orange-400",
  CRITICAL: "text-red-400",
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
    <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center px-4 py-8">
      <div className="w-full max-w-2xl space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className="text-white/50 hover:text-white transition"
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
            className="border-2 border-dashed border-white/20 rounded-2xl p-12 text-center cursor-pointer hover:border-white/40 transition"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <svg
              className="mx-auto mb-4 text-white/40"
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
            <p className="text-white/60 mb-2">
              Drop a video here or click to browse
            </p>
            <p className="text-white/30 text-sm">
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
          <div className="rounded-2xl border border-white/10 bg-white/5 p-8 text-center space-y-4">
            <div className="animate-spin mx-auto w-8 h-8 border-2 border-white/20 border-t-white rounded-full" />
            <p className="text-white/60">Uploading video...</p>
          </div>
        )}

        {/* Processing Phase */}
        {phase === "processing" && (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-8 space-y-4">
            <div className="flex items-center justify-between text-sm text-white/60">
              <span>Analyzing collisions...</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
              <div
                className="bg-green-500 h-full rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex gap-6 text-sm text-white/50">
              <span>{subjectsTracked} subjects tracked</span>
              <span>{collisionCount} collisions detected</span>
            </div>
          </div>
        )}

        {/* Error Phase */}
        {phase === "error" && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-8 text-center space-y-4">
            <p className="text-red-400">{error}</p>
            <DuoButton variant="secondary" onClick={reset}>
              Try Again
            </DuoButton>
          </div>
        )}

        {/* Results Phase */}
        {phase === "complete" && result && (
          <div className="space-y-6">
            {/* Overall Summary */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Overall Assessment</h2>
                <span
                  className={`text-lg font-bold ${RISK_COLORS[result.overall_risk] ?? "text-white"}`}
                >
                  {result.overall_risk}
                </span>
              </div>
              <p className="text-white/60 text-sm">
                {result.overall_recommendation}
              </p>
              <div className="grid grid-cols-3 gap-4 pt-2 text-center text-sm">
                <div>
                  <div className="text-white/40 text-xs">Duration</div>
                  <div className="font-mono">{(result.duration_sec ?? 0).toFixed(1)}s</div>
                </div>
                <div>
                  <div className="text-white/40 text-xs">Subjects</div>
                  <div className="font-mono">{result.subjects_tracked ?? 0}</div>
                </div>
                <div>
                  <div className="text-white/40 text-xs">Collisions</div>
                  <div className="font-mono">
                    {result.collision_events?.length ?? 0}
                  </div>
                </div>
              </div>
            </div>

            {/* Collision Events */}
            {(result.collision_events?.length ?? 0) > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider">
                  Collision Events
                </h3>
                {result.collision_events!.map((event) => (
                  <CrashCollisionCard key={event.event_id} event={event} />
                ))}
              </div>
            )}

            {(result.collision_events?.length ?? 0) === 0 && (
              <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-6 text-center">
                <p className="text-green-400 font-medium">
                  No collisions detected
                </p>
                <p className="text-white/40 text-sm mt-1">
                  The video did not contain any detectable person-to-person
                  collisions.
                </p>
              </div>
            )}

            {/* Per-Subject Summaries */}
            {Object.keys(result.subject_summaries ?? {}).length > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider">
                  Subject Summaries
                </h3>
                <div className="grid gap-2">
                  {Object.values(result.subject_summaries ?? {}).map((s) => (
                    <div
                      key={s.subject_id}
                      className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-4 py-3"
                    >
                      <span className="font-medium">
                        Subject {s.subject_id}
                      </span>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-white/50">
                          {s.collision_count} collision
                          {s.collision_count !== 1 ? "s" : ""}
                        </span>
                        <span
                          className={
                            RISK_COLORS[s.worst_risk_level] ?? "text-white/50"
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
