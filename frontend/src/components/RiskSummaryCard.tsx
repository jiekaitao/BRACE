"use client";

import { useEffect, useState } from "react";
import type { SubjectState } from "@/lib/types";
import { getTopRiskMoments, type RiskMoment } from "@/lib/riskTimeline";
import Card from "@/components/ui/Card";

const RISK_LABELS: Record<string, string> = {
  acl_valgus: "ACL Risk",
  hip_drop: "Hip Drop",
  trunk_lean: "Trunk Lean",
  asymmetry: "Asymmetry",
  angular_velocity_spike: "Velocity Spike",
};

const SEVERITY_COLORS: Record<string, string> = {
  high: "#EA2B2B",
  medium: "#FF9600",
};

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

interface Props {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  replaying: boolean;
}

export default function RiskSummaryCard({ subjectsRef, selectedSubjectRef, videoRef, replaying }: Props) {
  const [moments, setMoments] = useState<RiskMoment[]>([]);

  // Poll for risk data at 1Hz (only during replay)
  useEffect(() => {
    if (!replaying) return;

    const interval = setInterval(() => {
      const sel = selectedSubjectRef.current;
      if (sel === null) return;
      const subject = subjectsRef.current.get(sel);
      if (!subject?.replayTimeline?.length) return;
      setMoments(getTopRiskMoments(subject.replayTimeline, 5));
    }, 1000);

    return () => clearInterval(interval);
  }, [replaying, subjectsRef, selectedSubjectRef]);

  if (!replaying || moments.length === 0) return null;

  const seekTo = (time: number) => {
    const video = videoRef.current;
    if (video) video.currentTime = time;
  };

  return (
    <Card className="mt-3">
      <h3 className="text-sm font-bold text-[#3C3C3C] mb-2">
        Risk Highlights
      </h3>
      <div className="flex flex-col gap-1.5">
        {moments.map((m, i) => (
          <button
            key={i}
            onClick={() => seekTo(m.startTime)}
            className="flex items-center gap-2 text-left w-full px-2 py-1.5 rounded-lg hover:bg-[#F7F7F7] transition-colors"
          >
            {/* Severity indicator */}
            <div
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: SEVERITY_COLORS[m.severity] }}
            />
            {/* Risk info */}
            <div className="flex-1 min-w-0">
              <div className="text-xs font-bold text-[#3C3C3C] truncate">
                {RISK_LABELS[m.riskType] ?? m.riskType}
                <span className="font-normal text-[#AFAFAF] ml-1">
                  ({m.joint})
                </span>
              </div>
              <div className="text-[10px] text-[#AFAFAF]">
                {m.peakValue.toFixed(1)} peak · {m.duration.toFixed(1)}s
              </div>
            </div>
            {/* Timestamp */}
            <span className="text-[10px] font-mono text-[#1CB0F6] flex-shrink-0">
              {formatTime(m.startTime)}
            </span>
          </button>
        ))}
      </div>
    </Card>
  );
}
