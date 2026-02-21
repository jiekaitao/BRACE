/**
 * Synthetic motion data utilities for the animated skeleton demo.
 * Coordinates and logic ported from tests/test_movement_quality.py and
 * backend/risk_profile.py.
 */

import type { RiskModifiers } from "./types";

/** 14-joint standing pose in SRP space (2D).
 *  Index mapping (FEATURE_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]):
 *  0:L_shoulder 1:R_shoulder 2:L_elbow 3:R_elbow 4:L_wrist 5:R_wrist
 *  6:L_hip 7:R_hip 8:L_knee 9:R_knee 10:L_ankle 11:R_ankle 12:L_foot 13:R_foot
 */
export const BASE_POSE_2D: [number, number][] = [
  [-0.5, 2.0],   // 0: L shoulder
  [0.5, 2.0],    // 1: R shoulder
  [-0.7, 1.2],   // 2: L elbow
  [0.7, 1.2],    // 3: R elbow
  [-0.8, 0.4],   // 4: L wrist
  [0.8, 0.4],    // 5: R wrist
  [-0.5, 0.0],   // 6: L hip
  [0.5, 0.0],    // 7: R hip
  [-0.5, -1.5],  // 8: L knee
  [0.5, -1.5],   // 9: R knee
  [-0.5, -3.0],  // 10: L ankle
  [0.5, -3.0],   // 11: R ankle
  [-0.5, -3.2],  // 12: L foot
  [0.5, -3.2],   // 13: R foot
];

/**
 * Generate a squat cycle trajectory.
 * Ported from tests/test_movement_quality.py:_make_squat_trajectory_2d
 */
export function generateSquatCycle(
  numFrames: number = 120,
  depth: number = 1.0,
): [number, number][][] {
  const frames: [number, number][][] = [];
  for (let t = 0; t < numFrames; t++) {
    const phase = Math.sin((2 * Math.PI * t) / numFrames);
    const frame: [number, number][] = BASE_POSE_2D.map(([x, y]) => [x, y]);
    // Hips drop
    frame[6][1] -= depth * (1 + phase) * 0.3;
    frame[7][1] -= depth * (1 + phase) * 0.3;
    // Knees bend outward slightly and drop
    frame[8][1] -= depth * (1 + phase) * 0.6;
    frame[9][1] -= depth * (1 + phase) * 0.6;
    frame[8][0] -= depth * (1 + phase) * 0.1;
    frame[9][0] += depth * (1 + phase) * 0.1;
    frames.push(frame);
  }
  return frames;
}

/**
 * Compute the angle at vertex p2 formed by vectors p2→p1 and p2→p3.
 * Returns degrees (0-180).
 */
export function computeJointAngle(
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
): number {
  const v1x = p1[0] - p2[0];
  const v1y = p1[1] - p2[1];
  const v2x = p3[0] - p2[0];
  const v2y = p3[1] - p2[1];
  const dot = v1x * v2x + v1y * v2y;
  const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
  if (mag1 < 1e-8 || mag2 < 1e-8) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Maps injury type to relevant joint chains (indices into the 14-joint array)
 * and associated metric name for threshold lookup.
 * Each entry has `chains` (pairs of [p1, vertex, p3]) and `metric`.
 */
export const INJURY_JOINT_CHAINS: Record<
  string,
  { chains: { joints: [number, number, number]; side: "left" | "right" }[]; metric: string }
> = {
  acl: {
    chains: [
      { joints: [6, 8, 10], side: "left" },
      { joints: [7, 9, 11], side: "right" },
    ],
    metric: "fppa",
  },
  shoulder: {
    chains: [
      { joints: [0, 2, 4], side: "left" },
      { joints: [1, 3, 5], side: "right" },
    ],
    metric: "angular_velocity",
  },
  lower_back: {
    chains: [
      { joints: [0, 6, 8], side: "left" },
      { joints: [1, 7, 9], side: "right" },
    ],
    metric: "trunk_lean",
  },
  hip: {
    chains: [
      { joints: [0, 6, 8], side: "left" },
      { joints: [1, 7, 9], side: "right" },
    ],
    metric: "hip_drop",
  },
  hamstring: {
    chains: [
      { joints: [6, 8, 10], side: "left" },
      { joints: [7, 9, 11], side: "right" },
    ],
    metric: "asymmetry",
  },
  knee_general: {
    chains: [
      { joints: [6, 8, 10], side: "left" },
      { joints: [7, 9, 11], side: "right" },
    ],
    metric: "fppa",
  },
  ankle: {
    chains: [
      { joints: [8, 10, 12], side: "left" },
      { joints: [9, 11, 13], side: "right" },
    ],
    metric: "angular_velocity",
  },
};

/** Default injury risk thresholds (from backend/risk_profile.py). */
export const DEFAULT_THRESHOLDS: Record<string, { medium: number; high?: number }> = {
  fppa: { medium: 15.0, high: 25.0 },
  hip_drop: { medium: 8.0, high: 12.0 },
  trunk_lean: { medium: 15.0, high: 25.0 },
  asymmetry: { medium: 15.0, high: 25.0 },
  angular_velocity: { medium: 500.0 },
};

/**
 * Apply risk modifiers to default thresholds (mirrors backend/risk_profile.py:apply_modifiers).
 */
export function computeModifiedThresholds(
  riskModifiers: RiskModifiers | null | undefined,
): Record<string, { medium: number; high?: number }> {
  if (!riskModifiers) {
    return Object.fromEntries(
      Object.entries(DEFAULT_THRESHOLDS).map(([k, v]) => [k, { ...v }]),
    );
  }
  const scaleMap: Record<string, number> = {
    fppa: riskModifiers.fppa_scale,
    hip_drop: riskModifiers.hip_drop_scale,
    trunk_lean: riskModifiers.trunk_lean_scale,
    asymmetry: riskModifiers.asymmetry_scale,
    angular_velocity: riskModifiers.angular_velocity_scale,
  };
  const result: Record<string, { medium: number; high?: number }> = {};
  for (const [metric, thresholds] of Object.entries(DEFAULT_THRESHOLDS)) {
    const scale = scaleMap[metric] ?? 1.0;
    result[metric] = {
      medium: thresholds.medium * scale,
      ...(thresholds.high !== undefined ? { high: thresholds.high * scale } : {}),
    };
  }
  return result;
}
