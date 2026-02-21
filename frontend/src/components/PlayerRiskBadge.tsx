"use client";

import type { PlayerRiskLevel } from "@/lib/types";

interface PlayerRiskBadgeProps {
  status: PlayerRiskLevel | null;
  pullRecommended?: boolean;
}

const STATUS_CONFIG: Record<
  PlayerRiskLevel,
  { label: string; bg: string; text: string }
> = {
  GREEN: { label: "LOW RISK", bg: "#58CC02", text: "#FFFFFF" },
  YELLOW: { label: "CAUTION", bg: "#FF9600", text: "#FFFFFF" },
  RED: { label: "HIGH RISK", bg: "#EA2B2B", text: "#FFFFFF" },
};

export default function PlayerRiskBadge({
  status,
  pullRecommended = false,
}: PlayerRiskBadgeProps) {
  if (!status) return null;

  if (pullRecommended) {
    return (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          padding: "2px 10px",
          borderRadius: 9999,
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.05em",
          color: "#FFFFFF",
          background: "#EA2B2B",
          animation: "pulse-badge 1.2s ease-in-out infinite",
        }}
      >
        PULL
        <style>{`
          @keyframes pulse-badge {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        `}</style>
      </span>
    );
  }

  const config = STATUS_CONFIG[status];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "2px 10px",
        borderRadius: 9999,
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: "0.05em",
        color: config.text,
        background: config.bg,
      }}
    >
      {config.label}
    </span>
  );
}
