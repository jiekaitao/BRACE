"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { SubjectState, TimelineData, ClusterSegment, RiskSegment } from "@/lib/types";
import { buildTimelineData } from "@/lib/timelineUtils";
import { buildRiskMoments, type RiskMoment } from "@/lib/riskTimeline";
import RiskMarker from "@/components/RiskMarker";

const CLUSTER_PALETTE = [
  "#1CB0F6", // blue
  "#58CC02", // green
  "#CE82FF", // purple
  "#FF9600", // orange
  "#FF4B4B", // red
  "#2B70C9", // dark blue
  "#FFD900", // yellow
  "#00CD9C", // teal
  "#FF86D0", // pink
  "#A1D2FF", // light blue
];

const RISK_COLORS: Record<string, string> = {
  high: "rgba(255, 120, 120, 0.45)",
  medium: "rgba(255, 200, 120, 0.35)",
};

const RISK_LABELS: Record<string, string> = {
  knee_valgus: "Knee Valgus",
  acl_valgus: "Knee Valgus",  // backward compat
  hip_drop: "Hip Drop",
  trunk_lean: "Trunk Lean",
  asymmetry: "Asymmetry",
  angular_velocity_spike: "Velocity Spike",
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

interface TooltipInfo {
  x: number;
  time: number;
  cluster?: ClusterSegment;
  risks: RiskSegment[];
}

export default function WorkoutTimeline({ subjectsRef, selectedSubjectRef, videoRef, replaying }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);
  const [timeline, setTimeline] = useState<TimelineData | null>(null);
  const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);
  const [riskMoments, setRiskMoments] = useState<RiskMoment[]>([]);
  const isDraggingRef = useRef(false);

  // Poll for timeline data at 2Hz
  useEffect(() => {
    const interval = setInterval(() => {
      const sel = selectedSubjectRef.current;
      if (sel === null) return;
      const s = subjectsRef.current.get(sel);
      if (!s) return;
      const data = buildTimelineData(s);
      setTimeline((prev) => {
        // Only update if data changed (different length or new subject)
        if (data === prev) return prev;
        if (!data && !prev) return prev;
        return data;
      });
      // Compute risk moments from replay timeline
      if (s.replayTimeline?.length) {
        setRiskMoments(buildRiskMoments(s.replayTimeline));
      }
    }, 500);
    return () => clearInterval(interval);
  }, [subjectsRef, selectedSubjectRef]);

  // Animate playhead at 60fps via rAF (no React re-renders)
  useEffect(() => {
    if (!timeline) return;
    function tick() {
      const video = videoRef.current;
      const playhead = playheadRef.current;
      const track = trackRef.current;
      if (video && playhead && track && timeline) {
        const frac = video.currentTime / timeline.duration;
        const pct = Math.min(100, Math.max(0, frac * 100));
        playhead.style.left = `${pct}%`;
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [timeline, videoRef]);

  // Seek video to time corresponding to mouse X position
  const seekToX = useCallback(
    (clientX: number) => {
      const track = trackRef.current;
      const video = videoRef.current;
      if (!track || !video || !timeline) return;
      const rect = track.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      video.currentTime = frac * timeline.duration;
    },
    [videoRef, timeline]
  );

  // Mouse/touch handlers for scrubbing (only in replay mode)
  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (!replaying) return; // Can't scrub during first pass
      e.preventDefault();
      isDraggingRef.current = true;
      seekToX(e.clientX);

      const onMove = (ev: PointerEvent) => seekToX(ev.clientX);
      const onUp = () => {
        isDraggingRef.current = false;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [seekToX, replaying]
  );

  // Tooltip on hover
  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isDraggingRef.current) return;
      const track = trackRef.current;
      if (!track || !timeline) return;
      const rect = track.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const time = frac * timeline.duration;

      // Find cluster at this time
      const cluster = timeline.clusterSegments.find(
        (s) => time >= s.startTime && time <= s.endTime
      );
      // Find risks at this time
      const risks = timeline.riskSegments.filter(
        (s) => time >= s.startTime && time <= s.endTime
      );

      setTooltip({ x: e.clientX - rect.left, time, cluster, risks });
    },
    [timeline]
  );

  const onMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  if (!timeline || timeline.duration <= 0) return null;

  const { duration, clusterSegments, riskSegments } = timeline;

  return (
    <div className="relative select-none" style={{ height: 40 }}>
      {/* Track area */}
      <div
        ref={trackRef}
        className={`absolute inset-0 overflow-hidden ${replaying ? "cursor-pointer" : "cursor-default"}`}
        onPointerDown={onPointerDown}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        style={{ touchAction: "none", backgroundColor: "rgba(0,0,0,0.65)" }}
      >
        {/* Cluster segment bars (full height, 70% opacity) */}
        {clusterSegments.map((seg, i) => {
          const left = (seg.startTime / duration) * 100;
          const width = ((seg.endTime - seg.startTime) / duration) * 100;
          const color = CLUSTER_PALETTE[seg.clusterId % CLUSTER_PALETTE.length];
          const showLabel = width > 5;
          return (
            <div
              key={`c-${i}`}
              className="absolute top-0 bottom-0 flex items-center justify-center overflow-hidden"
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 0.3)}%`,
                backgroundColor: color,
                opacity: 0.7,
              }}
            >
              {showLabel && seg.activityLabel && (
                <span className="text-[10px] font-bold text-white/90 truncate px-1 pointer-events-none">
                  {seg.activityLabel}
                </span>
              )}
            </div>
          );
        })}

        {/* Risk bars (top strip, 8px) */}
        {riskSegments.map((seg, i) => {
          const left = (seg.startTime / duration) * 100;
          const width = ((seg.endTime - seg.startTime) / duration) * 100;
          const color = RISK_COLORS[seg.severity] ?? RISK_COLORS.medium;
          return (
            <div
              key={`r-${i}`}
              className="absolute top-0"
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 0.3)}%`,
                height: 8,
                backgroundColor: color,
              }}
            />
          );
        })}

        {/* Risk diamond markers */}
        {riskMoments.slice(0, 10).map((m, i) => {
          const pos = (m.startTime / duration) * 100;
          return (
            <RiskMarker
              key={`rm-${i}`}
              moment={m}
              position={pos}
              onClick={() => {
                const video = videoRef.current;
                if (video && replaying) video.currentTime = m.startTime;
              }}
            />
          );
        })}

        {/* Playhead */}
        <div
          ref={playheadRef}
          className="absolute top-0 bottom-0 pointer-events-none"
          style={{ width: 3, backgroundColor: "#fff", left: "0%", zIndex: 10 }}
        />

      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-black/85 text-white rounded px-2 py-1 text-[11px] whitespace-nowrap"
          style={{
            bottom: "100%",
            left: tooltip.x,
            transform: "translateX(-50%)",
            marginBottom: 4,
            zIndex: 20,
          }}
        >
          <div className="font-mono">{formatTime(tooltip.time)}</div>
          {tooltip.cluster?.activityLabel && (
            <div className="font-semibold" style={{ color: CLUSTER_PALETTE[tooltip.cluster.clusterId % CLUSTER_PALETTE.length] }}>
              {tooltip.cluster.activityLabel}
              {tooltip.cluster.guidelineName && (
                <span className="text-[9px] text-white/60 ml-1 font-normal">
                  ({tooltip.cluster.guidelineName})
                </span>
              )}
            </div>
          )}
          {tooltip.risks.map((r, i) => (
            <div key={i} style={{ color: RISK_COLORS[r.severity] }}>
              {RISK_LABELS[r.riskType] ?? r.riskType} ({r.joint})
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
