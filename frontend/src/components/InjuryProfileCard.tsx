"use client";

import type { InjuryProfile, InjuryEntry } from "@/lib/types";
import Card from "@/components/ui/Card";
import DuoButton from "@/components/ui/DuoButton";

const INJURY_LABELS: Record<string, string> = {
  acl: "ACL Tear",
  shoulder: "Shoulder Injury",
  ankle: "Ankle Sprain",
  lower_back: "Lower Back Pain",
  knee_general: "Knee Issue",
  hip: "Hip Injury",
  hamstring: "Hamstring Strain",
  wrist: "Wrist Injury",
};

const SEVERITY_COLORS: Record<string, string> = {
  mild: "#58CC02",
  moderate: "#FF9600",
  severe: "#EA2B2B",
};

export function InjuryBadge({ injury }: { injury: InjuryEntry }) {
  const label = INJURY_LABELS[injury.type] ?? injury.type;
  const color = SEVERITY_COLORS[injury.severity] ?? "#AFAFAF";

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-[10px] bg-[#F7F7F7] border border-[#E5E5E5]">
      <div
        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
        style={{ backgroundColor: color }}
      />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-bold text-[#3C3C3C]">{label}</div>
        <div className="text-[11px] text-[#AFAFAF]">
          {injury.side !== "unknown" && `${injury.side} · `}
          {injury.severity} · {injury.timeframe}
        </div>
      </div>
    </div>
  );
}

interface Props {
  profile: InjuryProfile;
  onConfirm?: () => void;
  onEdit?: () => void;
  confirmed?: boolean;
}

export default function InjuryProfileCard({ profile, onConfirm, onEdit, confirmed }: Props) {
  if (!profile.injuries.length) {
    return (
      <Card className="text-center">
        <p className="text-sm text-[#777777] py-4">
          No injuries identified. You&apos;re good to go!
        </p>
        {onConfirm && (
          <DuoButton variant="primary" onClick={onConfirm}>
            Looks Good
          </DuoButton>
        )}
      </Card>
    );
  }

  return (
    <Card>
      <h3 className="text-base font-extrabold text-[#3C3C3C] mb-3">
        {confirmed ? "Your Injury Profile" : "Detected Injuries"}
      </h3>
      <div className="flex flex-col gap-2 mb-4">
        {profile.injuries.map((injury, i) => (
          <InjuryBadge key={i} injury={injury} />
        ))}
      </div>
      {!confirmed && (
        <div className="flex gap-2">
          {onConfirm && (
            <DuoButton variant="primary" fullWidth onClick={onConfirm}>
              Looks Good
            </DuoButton>
          )}
          {onEdit && (
            <DuoButton variant="secondary" fullWidth onClick={onEdit}>
              Edit
            </DuoButton>
          )}
        </div>
      )}
      {confirmed && (
        <div className="flex items-center gap-2 text-sm text-[#58CC02] font-bold">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M6 8l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5"/>
          </svg>
          Profile saved
        </div>
      )}
    </Card>
  );
}
