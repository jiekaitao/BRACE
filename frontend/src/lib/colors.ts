/** Phase colors for bounding box / status indicators. */
export const PHASE_COLORS = {
  calibrating: "#1CB0F6", // blue
  normal: "#58CC02",      // green
  anomaly: "#EA2B2B",     // red
} as const;

/** Phase colors with lower opacity for fills. */
export const PHASE_FILLS = {
  calibrating: "rgba(28, 176, 246, 0.15)",
  normal: "rgba(88, 204, 2, 0.15)",
  anomaly: "rgba(234, 43, 43, 0.15)",
} as const;

/** Cluster colors (cycle through for different clusters). */
export const CLUSTER_COLORS = [
  "#58CC02", // green
  "#1CB0F6", // blue
  "#FF9600", // orange
  "#CE82FF", // purple
  "#EA2B2B", // red
  "#1899D6", // dark blue
  "#46A302", // dark green
  "#E58600", // dark orange
] as const;

/**
 * Map severity to a green→light-red gradient color for skeleton overlay.
 * Returns a subtle tint — same thickness/size, just color shift.
 */
export function riskColor(severity: "medium" | "high"): string {
  // medium → warm yellow-green, high → soft red
  return severity === "high" ? "#E07060" : "#C0A040";
}


/** Skeleton wireframe color. */
export const SKELETON_COLOR = "rgba(255, 255, 255, 0.7)";

/** Feature joint dot color. */
export const JOINT_DOT_COLOR = "#FFFFFF";

/** Subject colors for multi-person bboxes and labels. */
const SUBJECT_COLORS = [
  "#1CB0F6", // blue
  "#FF9600", // orange
  "#CE82FF", // purple
  "#58CC02", // green
  "#EA2B2B", // red
  "#1899D6", // dark blue
  "#E58600", // dark orange
  "#46A302", // dark green
] as const;

/** Get a consistent color for a subject by track ID. */
export function getSubjectColor(trackId: number): string {
  return SUBJECT_COLORS[trackId % SUBJECT_COLORS.length];
}

/** Map jersey color names to CSS hex colors. */
export const JERSEY_COLOR_MAP: Record<string, string> = {
  red: "#DC2626",
  blue: "#2563EB",
  green: "#16A34A",
  white: "#E5E7EB",
  black: "#1F2937",
  yellow: "#EAB308",
  orange: "#EA580C",
  purple: "#9333EA",
  pink: "#EC4899",
  gray: "#6B7280",
  grey: "#6B7280",
  navy: "#1E3A5F",
  maroon: "#7F1D1D",
  gold: "#CA8A04",
};

/**
 * Build a jersey tag string, e.g. "G23" (first letter of color + number).
 * Returns null if no jersey number is available.
 */
export function jerseyTagText(
  jerseyNumber: number | null | undefined,
  jerseyColor: string | null | undefined,
): string | null {
  if (jerseyNumber == null) return null;
  const prefix = jerseyColor ? jerseyColor.charAt(0).toUpperCase() : "";
  return `${prefix}${jerseyNumber}`;
}

/** Look up CSS hex color for a jersey color name, with fallback. */
export function jerseyDisplayColor(jerseyColor: string | null | undefined): string {
  if (!jerseyColor) return "#999999";
  return JERSEY_COLOR_MAP[jerseyColor.toLowerCase()] ?? "#999999";
}
