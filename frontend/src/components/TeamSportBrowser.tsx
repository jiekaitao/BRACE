"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { SPORTS } from "@/lib/teamSportsData";
import { getApiBase } from "@/lib/api";

const SPORT_ICONS: Record<string, React.ReactNode> = {
  basketball: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="32" cy="32" r="28" />
      <path d="M32 4v56" />
      <path d="M4 32h56" />
      <path d="M8.5 12C18 20 18 44 8.5 52" />
      <path d="M55.5 12C46 20 46 44 55.5 52" />
      <path d="M12 8.5C20 18 44 18 52 8.5" />
      <path d="M12 55.5C20 46 44 46 52 55.5" />
    </svg>
  ),
  football: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="32" cy="32" rx="28" ry="16" transform="rotate(-30 32 32)" />
      <path d="M18 46L46 18" />
      <path d="M24 28l4 4" />
      <path d="M28 24l4 4" />
      <path d="M32 20l4 4" />
      <path d="M36 16l4 4" />
      <path d="M10 42c-2-6-2-12 0-18" />
      <path d="M54 42c2-6 2-12 0-18" />
    </svg>
  ),
  soccer: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="32" cy="32" r="28" />
      <polygon points="32,14 40,20 38,30 26,30 24,20" />
      <line x1="32" y1="4" x2="32" y2="14" />
      <line x1="40" y1="20" x2="56" y2="14" />
      <line x1="38" y1="30" x2="54" y2="40" />
      <line x1="26" y1="30" x2="10" y2="40" />
      <line x1="24" y1="20" x2="8" y2="14" />
      <line x1="32" y1="60" x2="32" y2="48" />
      <line x1="10" y1="40" x2="16" y2="48" />
      <line x1="54" y1="40" x2="48" y2="48" />
      <line x1="16" y1="48" x2="32" y2="48" />
      <line x1="48" y1="48" x2="32" y2="48" />
    </svg>
  ),
  "large-class": (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      {/* Center person */}
      <circle cx="32" cy="14" r="5" />
      <path d="M32 19v12" />
      <path d="M24 25h16" />
      <path d="M26 40l6-9 6 9" />
      {/* Left person */}
      <circle cx="14" cy="22" r="4" />
      <path d="M14 26v10" />
      <path d="M8 30h12" />
      <path d="M10 44l4-8 4 8" />
      {/* Right person */}
      <circle cx="50" cy="22" r="4" />
      <path d="M50 26v10" />
      <path d="M44 30h12" />
      <path d="M46 44l4-8 4 8" />
      {/* Back-row dots (crowd) */}
      <circle cx="22" cy="50" r="2" fill="white" fillOpacity="0.4" />
      <circle cx="32" cy="50" r="2" fill="white" fillOpacity="0.4" />
      <circle cx="42" cy="50" r="2" fill="white" fillOpacity="0.4" />
      <circle cx="12" cy="54" r="1.5" fill="white" fillOpacity="0.25" />
      <circle cx="52" cy="54" r="1.5" fill="white" fillOpacity="0.25" />
    </svg>
  ),
  "total-body-workout": (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      {/* Person */}
      <circle cx="32" cy="10" r="5" />
      <path d="M32 15v16" />
      <path d="M24 22h16" />
      <path d="M26 42l6-11 6 11" />
      {/* Dumbbell left */}
      <rect x="8" y="20" width="4" height="8" rx="1" fill="white" fillOpacity="0.5" />
      <path d="M12 24h12" strokeWidth="2" />
      {/* Dumbbell right */}
      <rect x="52" y="20" width="4" height="8" rx="1" fill="white" fillOpacity="0.5" />
      <path d="M40 24h12" strokeWidth="2" />
      {/* Energy lines */}
      <path d="M20 48l-4 8" strokeWidth="1" opacity="0.4" />
      <path d="M32 48v8" strokeWidth="1" opacity="0.4" />
      <path d="M44 48l4 8" strokeWidth="1" opacity="0.4" />
    </svg>
  ),
  boxing: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      {/* Left glove */}
      <path d="M10 28c0-6 4-10 10-10h2c4 0 7 3 7 7v14c0 4-3 7-7 7h-2c-6 0-10-4-10-10z" />
      <path d="M22 22v-4c0-2 1.5-3.5 3.5-3.5S29 16 29 18" />
      <path d="M10 30h19" />
      {/* Right glove */}
      <path d="M54 28c0-6-4-10-10-10h-2c-4 0-7 3-7 7v14c0 4 3 7 7 7h2c6 0 10-4 10-10z" />
      <path d="M42 22v-4c0-2-1.5-3.5-3.5-3.5S35 16 35 18" />
      <path d="M35 30h19" />
      {/* Wrist wraps */}
      <path d="M14 42h12v6H14z" />
      <path d="M38 42h12v6H38z" />
      {/* Impact lines */}
      <path d="M30 12l2-5" />
      <path d="M34 12l-2-5" />
      <path d="M32 13v-5" />
    </svg>
  ),
};

const LONG_PRESS_MS = 600;

interface TeamSportBrowserProps {
  onClose: () => void;
  showContent: boolean;
}

/** Circular progress ring for the precompute loading overlay. */
function ProgressRing({ progress, size = 120 }: { progress: number; size?: number }) {
  const stroke = 6;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(progress, 1));

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      {/* Background ring */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={stroke}
      />
      {/* Progress ring */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="#58CC02"
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        style={{ transition: "stroke-dashoffset 0.3s ease" }}
      />
    </svg>
  );
}

export default function TeamSportBrowser({ onClose, showContent }: TeamSportBrowserProps) {
  const router = useRouter();

  // Long-press precompute state
  const [precomputeState, setPrecomputeState] = useState<{
    sportId: string;
    sportName: string;
    demoVideo: string;
    jobId: string | null;
    progress: number;
    status: "starting" | "processing" | "complete" | "error";
  } | null>(null);

  const longPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdProgressRef = useRef(0);
  const holdAnimRef = useRef(0);
  const [holdingId, setHoldingId] = useState<string | null>(null);
  const [holdProgress, setHoldProgress] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (precomputeState) {
          cancelPrecompute();
        } else {
          onClose();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose, precomputeState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (longPressTimerRef.current) clearTimeout(longPressTimerRef.current);
      if (pollRef.current) clearInterval(pollRef.current);
      cancelAnimationFrame(holdAnimRef.current);
    };
  }, []);

  const cancelPrecompute = useCallback(() => {
    if (precomputeState?.jobId) {
      fetch(`${getApiBase()}/api/precompute/${precomputeState.jobId}/cancel`, {
        method: "POST",
      }).catch(() => {});
    }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setPrecomputeState(null);
  }, [precomputeState]);

  const startPrecompute = useCallback(async (sportId: string, sportName: string, demoVideo: string) => {
    setPrecomputeState({
      sportId,
      sportName,
      demoVideo,
      jobId: null,
      progress: 0,
      status: "starting",
    });

    try {
      const res = await fetch(`${getApiBase()}/api/precompute/${encodeURIComponent(demoVideo)}`, {
        method: "POST",
      });
      const data = await res.json();

      if (data.cached) {
        // Already computed — go straight to analyze
        router.push(
          `/analyze?mode=precomputed&video=${encodeURIComponent(demoVideo)}&job_id=${data.job_id}`
        );
        return;
      }

      const jobId = data.job_id;
      setPrecomputeState((prev) => prev ? { ...prev, jobId, status: "processing" } : null);

      // Poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${getApiBase()}/api/precompute/${jobId}/status`);
          const status = await statusRes.json();

          setPrecomputeState((prev) => {
            if (!prev) return null;
            return { ...prev, progress: status.progress || 0, status: status.status };
          });

          if (status.status === "complete") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            router.push(
              `/analyze?mode=precomputed&video=${encodeURIComponent(demoVideo)}&job_id=${jobId}`
            );
          } else if (status.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setPrecomputeState((prev) =>
              prev ? { ...prev, status: "error" } : null
            );
          }
        } catch {
          // Retry on next poll
        }
      }, 500);
    } catch (err) {
      console.error("Failed to start precompute:", err);
      setPrecomputeState((prev) =>
        prev ? { ...prev, status: "error" } : null
      );
    }
  }, [router]);

  const handlePointerDown = useCallback((sportId: string, sportName: string, demoVideo: string) => {
    setHoldingId(sportId);
    holdProgressRef.current = 0;
    setHoldProgress(0);

    const startTime = performance.now();

    function animateHold() {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / LONG_PRESS_MS, 1);
      holdProgressRef.current = progress;
      setHoldProgress(progress);

      if (progress < 1) {
        holdAnimRef.current = requestAnimationFrame(animateHold);
      }
    }
    holdAnimRef.current = requestAnimationFrame(animateHold);

    longPressTimerRef.current = setTimeout(() => {
      setHoldingId(null);
      setHoldProgress(0);
      cancelAnimationFrame(holdAnimRef.current);
      // Long press detected — start precompute
      startPrecompute(sportId, sportName, demoVideo);
    }, LONG_PRESS_MS);
  }, [startPrecompute]);

  const handlePointerUp = useCallback((demoVideo: string) => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
    cancelAnimationFrame(holdAnimRef.current);

    // If hold wasn't long enough, do normal click (instant navigate)
    if (holdProgressRef.current < 1 && !precomputeState) {
      router.push(`/analyze?mode=demo&video=${encodeURIComponent(demoVideo)}`);
    }

    setHoldingId(null);
    setHoldProgress(0);
    holdProgressRef.current = 0;
  }, [router, precomputeState]);

  const handlePointerLeave = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
    cancelAnimationFrame(holdAnimRef.current);
    setHoldingId(null);
    setHoldProgress(0);
    holdProgressRef.current = 0;
  }, []);

  return (
    <motion.div
      animate={{ opacity: showContent ? 1 : 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col h-full"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[#222]">
        <h2 className="text-xl font-extrabold text-white m-0">
          Choose a Sport
        </h2>
        <button
          onClick={onClose}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-[#1A1A1A] border border-[#333] text-white/60 font-bold text-lg cursor-pointer hover:bg-[#2A2A2A] transition-colors"
        >
          &times;
        </button>
      </div>

      {/* Hint */}
      <div className="px-6 pt-3 pb-0">
        <p className="text-[11px] text-white/30 font-medium m-0">
          Tap to stream live &middot; Hold to pre-analyze at max quality
        </p>
      </div>

      {/* Sport grid */}
      <div className="flex-1 px-6 py-4 overflow-y-auto">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2 }}
          className="grid grid-cols-3 gap-4"
        >
          {SPORTS.map((s) => {
            const isHolding = holdingId === s.id;
            return (
              <button
                key={s.id}
                onPointerDown={(e) => {
                  if (s.demoVideo) {
                    e.preventDefault();
                    handlePointerDown(s.id, s.name, s.demoVideo);
                  }
                }}
                onPointerUp={() => {
                  if (s.demoVideo) handlePointerUp(s.demoVideo);
                }}
                onPointerLeave={handlePointerLeave}
                onContextMenu={(e) => e.preventDefault()}
                disabled={!s.demoVideo}
                className={`group relative overflow-hidden rounded-[16px] border bg-[#111] aspect-[4/3] flex flex-col items-center justify-center gap-3 transition-all duration-100 select-none ${
                  s.demoVideo
                    ? "border-[#333] cursor-pointer shadow-[0_4px_0_#222] hover:border-white/40 hover:shadow-[0_4px_0_#444] active:shadow-none active:translate-y-[4px]"
                    : "border-[#222] cursor-default opacity-40"
                }`}
              >
                {/* Hold progress ring overlay */}
                {isHolding && holdProgress > 0 && (
                  <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/40">
                    <ProgressRing progress={holdProgress} size={60} />
                  </div>
                )}

                <div className={`transition-opacity duration-200 ${s.demoVideo ? "opacity-40 group-hover:opacity-80" : "opacity-30"}`}>
                  {SPORT_ICONS[s.id]}
                </div>
                <span className="text-white font-extrabold text-sm">
                  {s.name}
                </span>
                {!s.demoVideo && (
                  <span className="text-white/30 text-[10px] font-bold">Coming soon</span>
                )}
              </button>
            );
          })}
        </motion.div>
      </div>

      {/* Precompute loading overlay */}
      <AnimatePresence>
        {precomputeState && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/90 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-6">
              {/* Progress ring */}
              <div className="relative">
                <ProgressRing
                  progress={precomputeState.progress}
                  size={140}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-white text-2xl font-extrabold">
                    {Math.round(precomputeState.progress * 100)}%
                  </span>
                </div>
              </div>

              {/* Sport name */}
              <div className="text-center">
                <h3 className="text-white text-lg font-extrabold m-0">
                  {precomputeState.sportName}
                </h3>
                <p className="text-white/50 text-sm mt-1 m-0">
                  {precomputeState.status === "starting"
                    ? "Starting analysis..."
                    : precomputeState.status === "error"
                    ? "Analysis failed"
                    : "Analyzing at maximum quality..."}
                </p>
              </div>

              {/* Cancel button */}
              <button
                onClick={cancelPrecompute}
                className="mt-4 px-6 py-2 rounded-full border border-white/20 text-white/60 text-sm font-bold bg-transparent cursor-pointer hover:bg-white/10 hover:text-white transition-colors"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
