/** 3D rotation and projection utilities for the interactive skeleton viewer. */

/** Rotate SRP joints (2D or 3D) by yaw (Y-axis) and pitch (X-axis). */
export function rotateYX(
  joints: ([number, number] | [number, number, number])[],
  yaw: number,
  pitch: number
): [number, number, number][] {
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw);
  const cp = Math.cos(pitch),
    sp = Math.sin(pitch);
  return joints.map((j) => {
    const x = j[0], y = j[1], z = j.length >= 3 ? (j as [number, number, number])[2] : 0;
    // Rotate around Y axis (yaw)
    const rx = cy * x + sy * z;
    const rz = -sy * x + cy * z;
    // Rotate around X axis (pitch)
    const ry = cp * y - sp * rz;
    const rz2 = sp * y + cp * rz;
    return [rx, ry, rz2];
  });
}

/** Perspective projection from 3D to 2D screen coordinates. */
export function project(
  points3d: [number, number, number][],
  fov: number,
  distance: number
): [number, number, number][] {
  return points3d.map(([x, y, z]) => {
    const scale = fov / (distance + z);
    return [x * scale, y * scale, z];
  });
}
