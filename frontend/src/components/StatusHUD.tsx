"use client";

import type { ClusterInfo, SubjectState } from "@/lib/types";
import type { ConnectionStatus } from "@/hooks/useAnalysisWebSocket";
import { PHASE_COLORS } from "@/lib/colors";
import Card from "./ui/Card";
import ClusterSummary from "./ClusterSummary";

interface SubjectIdentity {
  label: string;
  identityStatus: "unknown" | "tentative" | "confirmed";
}

interface StatusHUDProps {
  phase: "calibrating" | "normal" | "anomaly";
  clusterId: number | null;
  clusterSummary: Record<string, ClusterInfo>;
  connected: ConnectionStatus;
  activeTrackIds: number[];
  selectedSubjectId: number | null;
  subjectLabels: Record<number, string>;
  subjectIdentities?: Record<number, SubjectIdentity>;
  onSelectSubject: (trackId: number) => void;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  fatigueIndex: number;
  peakVelocity: number;
}

const PHASE_LABELS: Record<string, string> = {
  calibrating: "CALIBRATING",
  normal: "ANALYZING",
  anomaly: "ANOMALY",
};

export default function StatusHUD({
  phase,
  clusterId,
  clusterSummary,
  connected,
  activeTrackIds,
  selectedSubjectId,
  subjectLabels,
  subjectIdentities,
  onSelectSubject,
  highlightedClusterRef,
  subjectsRef,
  selectedSubjectRef,
  fatigueIndex,
  peakVelocity,
}: StatusHUDProps) {
  const phaseColor = PHASE_COLORS[phase];

  return (
    <div className="flex flex-col gap-2 w-full">
      {/* Connection / phase status */}
      <div className="flex items-center gap-2">
        <div
          className="w-2.5 h-2.5 rounded-full"
          style={{ backgroundColor: connected === "connected" ? phaseColor : connected === "connecting" ? "#F5A623" : "#EA2B2B" }}
        />
        <span
          className="text-xs font-bold uppercase tracking-[0.03em]"
          style={{ color: connected === "connected" ? phaseColor : connected === "connecting" ? "#F5A623" : "#EA2B2B" }}
        >
          {connected === "connected" ? PHASE_LABELS[phase] : connected === "connecting" ? "Connecting" : "Disconnected"}
        </span>
      </div>

      {/* Subject picker pills */}
      {activeTrackIds.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {activeTrackIds.map((trackId) => {
            const isSelected = trackId === selectedSubjectId;
            const identity = subjectIdentities?.[trackId];
            const label = identity?.label || subjectLabels[trackId] || `S${trackId}`;
            const status = identity?.identityStatus || "confirmed";
            const color = isSelected ? phaseColor : "#CCCCCC";
            const isDashed = status === "unknown";
            const isConfirmed = status === "confirmed";

            return (
              <button
                key={trackId}
                onClick={() => onSelectSubject(trackId)}
                className="px-3 py-1.5 rounded-full text-xs font-bold transition-all duration-200 flex items-center gap-1"
                style={
                  isSelected
                    ? {
                      backgroundColor: color,
                      color: "#FFFFFF",
                      boxShadow: `0 2px 8px ${color}40`,
                      borderStyle: isDashed ? "dashed" : "solid",
                      borderWidth: "2px",
                      borderColor: "transparent",
                    }
                    : {
                      backgroundColor: "#FFFFFF",
                      color: "#777777",
                      borderStyle: isDashed ? "dashed" : "solid",
                      borderWidth: "2px",
                      borderColor: color,
                    }
                }
              >
                {isConfirmed && (
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                    <path
                      d="M2 6L5 9L10 3"
                      stroke={isSelected ? "#FFFFFF" : color}
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                )}
                {label}
              </button>
            );
          })}
        </div>
      )}

      {/* Concussion indicator */}
      {fatigueIndex >= 0 && ( /* show whenever tracking is on */
        <Card>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em]">
                Concussion Risk
              </span>
              <span
                className="text-xs font-bold px-2 py-0.5 rounded-full text-white"
                style={{
                  backgroundColor:
                    (() => {
                      const sel = selectedSubjectRef.current;
                      if (sel === null) return "#CCCCCC";
                      const s = subjectsRef.current.get(sel);
                      const cr = s?.quality?.concussion_rating ?? 0;
                      if (cr > 70) return "#EA2B2B";
                      if (cr > 40) return "#F5A623";
                      return "#58CC02";
                    })()
                }}
              >
                {(() => {
                  const sel = selectedSubjectRef.current;
                  if (sel === null) return "—";
                  const s = subjectsRef.current.get(sel);
                  return `${Math.round(s?.quality?.concussion_rating ?? 0)}`;
                })()}
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* Fatigue indicator */}
      {peakVelocity > 0 && (
        <Card>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em]">
                Fatigue
              </span>
              <span
                className="text-xs font-bold px-2 py-0.5 rounded-full text-white"
                style={{
                  backgroundColor:
                    (() => {
                      const sel = selectedSubjectRef.current;
                      if (sel === null) return "#CCCCCC";
                      const s = subjectsRef.current.get(sel);
                      const fr = s?.quality?.fatigue_rating ?? (fatigueIndex * 100);
                      if (fr > 60) return "#EA2B2B";
                      if (fr > 30) return "#F5A623";
                      return "#58CC02";
                    })()
                }}
              >
                {(() => {
                  const sel = selectedSubjectRef.current;
                  if (sel === null) return "—";
                  const s = subjectsRef.current.get(sel);
                  const fr = s?.quality?.fatigue_rating ?? (fatigueIndex * 100);
                  return `${Math.round(fr)}%`;
                })()}
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-[#E5E5E5] overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${(() => {
                    const sel = selectedSubjectRef.current;
                    if (sel === null) return 0;
                    const s = subjectsRef.current.get(sel);
                    const fr = s?.quality?.fatigue_rating ?? (fatigueIndex * 100);
                    return Math.min(fr, 100);
                  })()}%`,
                  backgroundColor:
                    (() => {
                      const sel = selectedSubjectRef.current;
                      if (sel === null) return "#CCCCCC";
                      const s = subjectsRef.current.get(sel);
                      const fr = s?.quality?.fatigue_rating ?? (fatigueIndex * 100);
                      if (fr > 60) return "#EA2B2B";
                      if (fr > 30) return "#F5A623";
                      return "#58CC02";
                    })()
                }}
              />
            </div>
            <div className="flex justify-between text-xs text-[#AFAFAF]">
              <span>Peak: {peakVelocity.toFixed(2)}</span>
              <span>
                Current:{" "}
                {(() => {
                  const sel = selectedSubjectRef.current;
                  if (sel === null) return "—";
                  const s = subjectsRef.current.get(sel);
                  if (!s || s.velocity.rolling.length === 0) return "—";
                  const cur = s.velocity.rolling[s.velocity.rolling.length - 1];
                  const drop = peakVelocity > 0 ? ((peakVelocity - cur) / peakVelocity) * 100 : 0;
                  return `${cur.toFixed(2)} (${drop > 0 ? "↓" : ""}${Math.abs(drop).toFixed(0)}%)`;
                })()}
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* Cluster summary */}
      <Card>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em]">
            Cluster Summary
          </h3>
          {clusterId !== null && (
            <span
              className="text-xs font-bold text-white px-2 py-0.5 rounded-full"
              style={{
                backgroundColor:
                  PHASE_COLORS[phase === "anomaly" ? "anomaly" : "normal"],
              }}
            >
              #{clusterId + 1}
            </span>
          )}
        </div>
        <ClusterSummary
          clusters={clusterSummary}
          activeCluster={clusterId}
          highlightedClusterRef={highlightedClusterRef}
          subjectsRef={subjectsRef}
          selectedSubjectRef={selectedSubjectRef}
        />
      </Card>
    </div>
  );
}
