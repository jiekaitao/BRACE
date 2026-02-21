"use client";

import type { RiskMoment } from "@/lib/riskTimeline";

const RISK_LABELS: Record<string, string> = {
  acl_valgus: "ACL Risk",
  hip_drop: "Hip Drop",
  trunk_lean: "Trunk Lean",
  asymmetry: "Asymmetry",
  angular_velocity_spike: "Velocity Spike",
};

interface RiskMarkerProps {
  moment: RiskMoment;
  position: number; // percentage 0-100
  onClick?: () => void;
}

export default function RiskMarker({ moment, position, onClick }: RiskMarkerProps) {
  const color = moment.severity === "high" ? "#EA2B2B" : "#FF9600";
  const label = RISK_LABELS[moment.riskType] ?? moment.riskType;

  return (
    <div
      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 cursor-pointer group z-10"
      style={{ left: `${position}%` }}
      onClick={onClick}
    >
      {/* Diamond marker */}
      <div
        className="w-3 h-3 rotate-45 border-2 border-white shadow-sm transition-transform duration-150 group-hover:scale-150"
        style={{ backgroundColor: color }}
      />

      {/* Tooltip on hover */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20">
        <div className="bg-black/85 text-white rounded px-2 py-1 text-[11px]">
          <div className="font-semibold" style={{ color }}>{label}</div>
          <div className="text-white/70">
            {moment.joint} · {moment.peakValue.toFixed(1)} (threshold: {moment.threshold.toFixed(1)})
          </div>
          <div className="text-white/50">
            {moment.duration.toFixed(1)}s duration
          </div>
        </div>
      </div>
    </div>
  );
}
