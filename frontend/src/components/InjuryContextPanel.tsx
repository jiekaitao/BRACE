"use client";

import type { InjuryProfile, RiskModifiers } from "@/lib/types";
import Card from "@/components/ui/Card";

const INJURY_LABELS: Record<string, string> = {
  acl: "ACL",
  shoulder: "Shoulder",
  ankle: "Ankle",
  lower_back: "Lower Back",
  knee_general: "Knee",
  hip: "Hip",
  hamstring: "Hamstring",
  wrist: "Wrist",
};

const SEVERITY_COLORS: Record<string, string> = {
  mild: "#58CC02",
  moderate: "#FF9600",
  severe: "#EA2B2B",
};

const METRIC_LABELS: Record<string, string> = {
  fppa_scale: "FPPA",
  hip_drop_scale: "Hip Drop",
  trunk_lean_scale: "Trunk Lean",
  asymmetry_scale: "Asymmetry",
  angular_velocity_scale: "Ang. Vel.",
};

interface Props {
  injuryProfile: InjuryProfile;
  riskModifiers?: RiskModifiers | null;
}

export default function InjuryContextPanel({ injuryProfile, riskModifiers }: Props) {
  if (!injuryProfile?.injuries?.length) return null;

  // Find which metrics are lowered (scale < 1.0)
  const loweredMetrics: { label: string; pct: number }[] = [];
  if (riskModifiers) {
    const modsAny = riskModifiers as unknown as Record<string, unknown>;
    for (const [key, label] of Object.entries(METRIC_LABELS)) {
      const val = modsAny[key];
      if (typeof val === "number" && val < 1.0) {
        loweredMetrics.push({ label, pct: Math.round(val * 100) });
      }
    }
  }

  return (
    <Card className="mt-2">
      <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-2">
        Your Injuries
      </h3>
      {/* Compact injury badges */}
      <div className="flex flex-wrap gap-1.5 mb-2">
        {injuryProfile.injuries.map((injury, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#F7F7F7] border border-[#E5E5E5]"
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ backgroundColor: SEVERITY_COLORS[injury.severity] ?? "#AFAFAF" }}
            />
            {INJURY_LABELS[injury.type] ?? injury.type}
            {injury.side !== "unknown" && ` (${injury.side})`}
          </span>
        ))}
      </div>
      {/* Monitoring section */}
      {loweredMetrics.length > 0 && (
        <div>
          <div className="text-[10px] font-bold text-[#AFAFAF] uppercase tracking-wider mb-1">
            Monitoring
          </div>
          <div className="flex flex-col gap-0.5">
            {loweredMetrics.map(({ label, pct }) => (
              <div key={label} className="flex items-center justify-between text-[10px]">
                <span className="text-[#777777]">{label}</span>
                <span className="font-bold text-[#CE82FF]">{pct}% threshold</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
