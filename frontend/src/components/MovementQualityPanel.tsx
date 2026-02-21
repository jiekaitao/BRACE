"use client";

import { useEffect, useState } from "react";
import type { SubjectState, FrameQuality, FatigueTimeline } from "@/lib/types";
import Card from "./ui/Card";

const JOINT_NAMES = [
  "L Shoulder", "R Shoulder", "L Elbow", "R Elbow",
  "L Wrist", "R Wrist", "L Hip", "R Hip",
  "L Knee", "R Knee", "L Ankle", "R Ankle",
  "L Foot", "R Foot",
];

const ANGULAR_VEL_LABELS: Record<string, string> = {
  left_knee: "L Knee",
  right_knee: "R Knee",
  left_hip: "L Hip",
  right_hip: "R Hip",
  left_elbow: "L Elbow",
  right_elbow: "R Elbow",
};

function scoreColor(score: number): string {
  if (score >= 80) return "#58CC02";
  if (score >= 50) return "#FF9600";
  return "#EA2B2B";
}

function fatigueColor(fatigue: number): string {
  if (fatigue < 0.3) return "#58CC02";
  if (fatigue < 0.6) return "#FF9600";
  return "#EA2B2B";
}

function severityColor(severity: string): string {
  if (severity === "high") return "#EA2B2B";
  if (severity === "medium") return "#FF9600";
  return "#FFD900";
}

function anomalyColor(score: number): string {
  if (score < 0.3) return "#58CC02";
  if (score < 0.6) return "#FF9600";
  return "#EA2B2B";
}

/** Collapsible section header with chevron toggle. */
function SectionToggle({ label, open, onToggle, badge }: {
  label: string;
  open: boolean;
  onToggle: () => void;
  badge?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex items-center justify-between w-full group"
    >
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em]">
          {label}
        </h3>
        {badge}
      </div>
      <svg
        width="12"
        height="12"
        viewBox="0 0 12 12"
        className="text-[#AFAFAF] transition-transform duration-200"
        style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
      >
        <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
}

interface MovementQualityPanelProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  showDebug?: boolean;
}

export default function MovementQualityPanel({
  subjectsRef,
  selectedSubjectRef,
  showDebug = false,
}: MovementQualityPanelProps) {
  const [quality, setQuality] = useState<FrameQuality>({});
  const [showRisks, setShowRisks] = useState(true);
  const [showAnomaly, setShowAnomaly] = useState(false);
  const [showCoM, setShowCoM] = useState(false);
  const [showAngVel, setShowAngVel] = useState(false);

  // Poll subjectsRef at ~4Hz to update quality data
  useEffect(() => {
    const interval = setInterval(() => {
      const sel = selectedSubjectRef.current;
      if (sel === null) return;
      const s = subjectsRef.current.get(sel);
      if (s?.quality) {
        setQuality({ ...s.quality });
      }
    }, 250);
    return () => clearInterval(interval);
  }, [subjectsRef, selectedSubjectRef]);

  const joints = quality.joint_quality;
  const bio = quality.biomechanics;
  const risks = quality.injury_risks;
  const timeline = quality.fatigue_timeline;
  const guideline = quality.active_guideline;

  // Don't render until we have some data
  if (!bio && !guideline && !risks?.length) return null;

  const riskCount = risks?.length ?? 0;
  const highCount = risks?.filter(r => r.severity === "high").length ?? 0;

  return (
    <div className="flex flex-col gap-2 w-full">
      {/* Guideline Card */}
      {guideline && (
        <Card>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em]">
                Guideline
              </h3>
              <span
                className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                style={{
                  backgroundColor: guideline.name === "generic" ? "#E5E5E5" : "#E8F5E9",
                  color: guideline.name === "generic" ? "#777777" : "#2E7D32",
                }}
              >
                {guideline.display_name}
              </span>
            </div>
            {guideline.form_cues.length > 0 && (
              <ul className="flex flex-col gap-1">
                {guideline.form_cues.map((cue, i) => (
                  <li key={i} className="text-[11px] text-[#555555] flex items-start gap-1.5">
                    <span className="text-[#AFAFAF] flex-shrink-0 mt-0.5">{"\u2022"}</span>
                    {cue}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Card>
      )}

      {/* Injury Risks — collapsible, scrollable */}
      <Card>
        <SectionToggle
          label="Injury Risks"
          open={showRisks}
          onToggle={() => setShowRisks(v => !v)}
          badge={
            riskCount > 0 ? (
              <span
                className="text-[10px] font-bold px-1.5 py-0.5 rounded-full text-white"
                style={{ backgroundColor: highCount > 0 ? "#EA2B2B" : "#FF9600" }}
              >
                {riskCount}
              </span>
            ) : (
              <span className="text-[10px] text-[#AFAFAF]">none</span>
            )
          }
        />
        {showRisks && (
          <div className="mt-3 h-[100px] overflow-y-auto flex flex-col gap-1.5">
            {riskCount === 0 && (
              <span className="text-[11px] text-[#AFAFAF]">No risks detected</span>
            )}
            {risks?.map((r, i) => (
              <div
                key={i}
                className="flex items-center justify-between px-2 py-1 rounded"
                style={{ backgroundColor: severityColor(r.severity) + "18" }}
              >
                <div className="flex items-center gap-1.5">
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: severityColor(r.severity) }}
                  />
                  <span className="text-[11px] font-bold text-[#4B4B4B]">
                    {r.risk}
                  </span>
                  {guideline && guideline.name !== "generic" && (
                    <span className="text-[9px] text-[#AFAFAF]">
                      ({guideline.display_name.toLowerCase()})
                    </span>
                  )}
                </div>
                <span
                  className="text-[10px] font-bold uppercase flex-shrink-0"
                  style={{ color: severityColor(r.severity) }}
                >
                  {r.severity}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Joint Quality */}
      {joints && joints.scores.length > 0 && (
        <Card>
          <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-3">
            Joint Quality
          </h3>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
            {joints.scores.map((score, i) => {
              const isDegrading = joints.degrading.includes(i);
              return (
                <div key={i} className="flex items-center gap-2">
                  <span
                    className="text-[10px] font-bold w-16 truncate"
                    style={{
                      color: isDegrading ? "#EA2B2B" : "#777777",
                    }}
                  >
                    {isDegrading ? "\u26A0 " : ""}{JOINT_NAMES[i]}
                  </span>
                  <div className="flex-1 h-1.5 rounded-full bg-[#E5E5E5] overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-200"
                      style={{
                        width: `${Math.min(score, 100)}%`,
                        backgroundColor: scoreColor(score),
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Ratings */}
      {(quality.concussion_rating !== undefined || quality.fatigue_rating !== undefined) && (
        <Card>
          <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-3">
            Health Ratings
          </h3>
          <div className="flex flex-col gap-2">
            {quality.concussion_rating !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#777777]">Concussion Risk</span>
                <span
                  className="text-xs font-bold"
                  style={{
                    color: quality.concussion_rating > 70 ? "#EA2B2B" : quality.concussion_rating > 40 ? "#F5A623" : "#58CC02"
                  }}
                >
                  {quality.concussion_rating.toFixed(1)} / 100
                </span>
              </div>
            )}
            {quality.fatigue_rating !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#777777]">Fatigue Score</span>
                <span
                  className="text-xs font-bold"
                  style={{
                    color: quality.fatigue_rating > 60 ? "#EA2B2B" : quality.fatigue_rating > 30 ? "#F5A623" : "#58CC02"
                  }}
                >
                  {quality.fatigue_rating.toFixed(1)} / 100
                </span>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Biomechanics */}
      {bio && (
        <Card>
          <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-3">
            Biomechanics
          </h3>
          <div className="flex flex-col gap-2">
            <BioRow label="Knee Valgus (L)" value={bio.fppa_left} unit={"\u00B0"} warn={Math.abs(bio.fppa_left) > 15} />
            <BioRow label="Knee Valgus (R)" value={bio.fppa_right} unit={"\u00B0"} warn={Math.abs(bio.fppa_right) > 15} />
            <BioRow label="Hip Drop" value={bio.hip_drop} unit={"\u00B0"} warn={Math.abs(bio.hip_drop) > 8} />
            <BioRow label="Trunk Lean" value={bio.trunk_lean} unit={"\u00B0"} warn={Math.abs(bio.trunk_lean) > 15} />
            <BioRow label="Asymmetry" value={bio.asymmetry} unit="%" warn={bio.asymmetry > 10} />
          </div>
        </Card>
      )}

      {/* Anomaly Score — debug only, collapsible */}
      {showDebug && bio?.anomaly_score !== undefined && (
        <Card>
          <SectionToggle
            label="Anomaly"
            open={showAnomaly}
            onToggle={() => setShowAnomaly(v => !v)}
            badge={
              <span
                className="text-[10px] font-bold"
                style={{ color: anomalyColor(bio.anomaly_score) }}
              >
                {(bio.anomaly_score * 100).toFixed(0)}%
              </span>
            }
          />
          {showAnomaly && (
            <div className="mt-3">
              <div className="w-full h-2 rounded-full bg-[#E5E5E5] overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min(bio.anomaly_score * 100, 100)}%`,
                    backgroundColor: anomalyColor(bio.anomaly_score),
                  }}
                />
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Center of Mass — collapsible */}
      {bio && (bio.com_velocity !== undefined || bio.com_sway !== undefined) && (
        <Card>
          <SectionToggle
            label="Center of Mass"
            open={showCoM}
            onToggle={() => setShowCoM(v => !v)}
          />
          {showCoM && (
            <div className="mt-3 flex flex-col gap-2">
              {bio.com_velocity !== undefined && (
                <BioRow label="Velocity" value={bio.com_velocity} unit="" warn={false} />
              )}
              {bio.com_sway !== undefined && (
                <BioRow label="Sway" value={bio.com_sway} unit="" warn={bio.com_sway > 15} />
              )}
            </div>
          )}
        </Card>
      )}

      {/* Angular Velocities — collapsible */}
      {bio?.angular_velocities && Object.keys(bio.angular_velocities).length > 0 && (
        <Card>
          <SectionToggle
            label="Angular Velocity"
            open={showAngVel}
            onToggle={() => setShowAngVel(v => !v)}
          />
          {showAngVel && (
            <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5">
              {Object.entries(bio.angular_velocities).map(([key, vel]) => {
                const isSpike = vel > 500;
                return (
                  <div key={key} className="flex items-center justify-between">
                    <span
                      className="text-[10px] font-bold w-14 truncate"
                      style={{ color: isSpike ? "#EA2B2B" : "#777777" }}
                    >
                      {ANGULAR_VEL_LABELS[key] ?? key}
                    </span>
                    <span
                      className="text-[10px] font-bold tabular-nums"
                      style={{ color: isSpike ? "#EA2B2B" : "#4B4B4B" }}
                    >
                      {vel.toFixed(0)}{"\u00B0"}/s
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      )}

      {/* Fatigue Timeline */}
      {timeline && timeline.timestamps.length > 1 && (
        <Card>
          <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-3">
            Fatigue Timeline
          </h3>
          <MiniChart timeline={timeline} />
        </Card>
      )}
    </div>
  );
}

function BioRow({ label, value, unit, warn }: { label: string; value: number; unit: string; warn: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-[#777777]">{label}</span>
      <span
        className="text-xs font-bold"
        style={{ color: warn ? "#EA2B2B" : "#4B4B4B" }}
      >
        {value > 0 ? "+" : ""}{value.toFixed(1)}{unit}
      </span>
    </div>
  );
}

function MiniChart({ timeline }: { timeline: FatigueTimeline }) {
  const { timestamps, fatigue, form_scores } = timeline;
  const n = timestamps.length;
  if (n < 2) return null;

  const w = 240;
  const h = 60;
  const pad = 4;

  const tMin = timestamps[0];
  const tMax = timestamps[n - 1];
  const tRange = Math.max(tMax - tMin, 1);

  function x(t: number) { return pad + ((t - tMin) / tRange) * (w - 2 * pad); }

  // Fatigue line (0-1 range)
  const fatiguePath = fatigue.map((f, i) => {
    const px = x(timestamps[i]);
    const py = h - pad - f * (h - 2 * pad);
    return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(" ");

  // Form score line (0-100 range, normalized to 0-1)
  const formPath = form_scores.map((f, i) => {
    const px = x(timestamps[i]);
    const py = h - pad - (f / 100) * (h - 2 * pad);
    return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(" ");

  const lastFatigue = fatigue[n - 1];

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: h }}>
        {/* Form score line (green) */}
        <path d={formPath} fill="none" stroke="#58CC02" strokeWidth="1.5" opacity={0.5} />
        {/* Fatigue line (red/orange) */}
        <path d={fatiguePath} fill="none" stroke={fatigueColor(lastFatigue)} strokeWidth="2" />
      </svg>
      <div className="flex justify-between text-[10px] text-[#AFAFAF] mt-1">
        <span>{timestamps[0].toFixed(0)}s</span>
        <div className="flex gap-3">
          <span style={{ color: fatigueColor(lastFatigue) }}>
            Fatigue {(lastFatigue * 100).toFixed(0)}%
          </span>
          <span style={{ color: "#58CC02" }}>
            Form {form_scores[n - 1]?.toFixed(0) ?? "--"}%
          </span>
        </div>
        <span>{timestamps[n - 1].toFixed(0)}s</span>
      </div>
    </div>
  );
}
