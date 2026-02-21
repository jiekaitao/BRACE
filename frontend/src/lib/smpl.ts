/** SMPL parameter interpolation utilities for smooth 10fps -> 60fps rendering. */

import type { SmplParams } from "./types";

/** Linearly interpolate between two SMPL parameter sets. */
export function interpolateSmplParams(
  prev: SmplParams,
  current: SmplParams,
  t: number
): SmplParams {
  const clampedT = Math.max(0, Math.min(1, t));

  // Betas (shape) don't change frame-to-frame, use current
  const betas = current.betas;

  // Lerp pose parameters
  const pose = new Array(current.pose.length);
  for (let i = 0; i < current.pose.length; i++) {
    const p = i < prev.pose.length ? prev.pose[i] : 0;
    pose[i] = p + (current.pose[i] - p) * clampedT;
  }

  // Lerp translation
  const trans = [
    prev.trans[0] + (current.trans[0] - prev.trans[0]) * clampedT,
    prev.trans[1] + (current.trans[1] - prev.trans[1]) * clampedT,
    prev.trans[2] + (current.trans[2] - prev.trans[2]) * clampedT,
  ];

  return { betas, pose, trans };
}

/** Convert axis-angle rotation to quaternion [x, y, z, w] for Three.js. */
export function axisAngleToQuat(
  ax: number,
  ay: number,
  az: number
): [number, number, number, number] {
  const angle = Math.sqrt(ax * ax + ay * ay + az * az);
  if (angle < 1e-8) {
    return [0, 0, 0, 1]; // identity quaternion
  }
  const halfAngle = angle / 2;
  const s = Math.sin(halfAngle) / angle;
  return [ax * s, ay * s, az * s, Math.cos(halfAngle)];
}
