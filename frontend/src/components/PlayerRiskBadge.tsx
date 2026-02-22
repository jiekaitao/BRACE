"use client";

interface PlayerRiskBadgeProps {
  status: "GREEN" | "YELLOW" | "RED" | string;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

const STATUS_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  GREEN: { bg: "#E8F5E9", text: "#2E7D32", border: "#A5D6A7" },
  YELLOW: { bg: "#FFF8E1", text: "#F57F17", border: "#FFE082" },
  RED: { bg: "#FFEBEE", text: "#C62828", border: "#EF9A9A" },
};

const SIZE_CLASSES = {
  sm: "text-xs px-2 py-0.5",
  md: "text-sm px-3 py-1",
  lg: "text-base px-4 py-1.5",
};

export default function PlayerRiskBadge({
  status,
  size = "md",
  showLabel = true,
}: PlayerRiskBadgeProps) {
  const colors = STATUS_COLORS[status] ?? STATUS_COLORS.GREEN;
  const sizeClass = SIZE_CLASSES[size];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${sizeClass}`}
      style={{
        backgroundColor: colors.bg,
        color: colors.text,
        border: `1px solid ${colors.border}`,
      }}
    >
      <span
        className="w-2 h-2 rounded-full"
        style={{ backgroundColor: colors.text }}
      />
      {showLabel && status}
    </span>
  );
}
