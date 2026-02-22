/** MediaPipe bone connections for skeleton drawing. */
export const MP_BONES: [number, number][] = [
  // Head
  [0, 1], [1, 2], [1, 3], [2, 4],                    // eyes, ears, nose
  [0, 11], [0, 12],                                  // nose to shoulders
  // Torso & Limbs
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  // torso + arms
  [11, 23], [12, 24], [23, 24],                      // torso to hips
  [23, 25], [25, 27], [27, 31],                      // left leg
  [24, 26], [26, 28], [28, 32],                      // right leg
  [15, 17], [15, 19], [15, 21],                      // left hand
  [16, 18], [16, 20], [16, 22],                      // right hand
  [27, 29], [28, 30],                                // heels
];

/** 14 feature joint indices used for SRP analysis (matches backend FEATURE_INDICES). */
export const FEATURE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32];

/** Feature joint names (keyed by MediaPipe index). */
export const FEATURE_NAMES: Record<number, string> = {
  11: "L Shoulder",
  12: "R Shoulder",
  13: "L Elbow",
  14: "R Elbow",
  15: "L Wrist",
  16: "R Wrist",
  23: "L Hip",
  24: "R Hip",
  25: "L Knee",
  26: "R Knee",
  27: "L Ankle",
  28: "R Ankle",
  31: "L Foot",
  32: "R Foot",
};

/**
 * Risk joint name → MediaPipe indices (for AnalysisCanvas video overlay).
 * Maps injury_risk.joint to the MP joint indices that should be colored.
 */
export const RISK_JOINT_TO_MP: Record<string, number[]> = {
  left_knee: [23, 25, 27],          // L hip, knee, ankle
  right_knee: [24, 26, 28],         // R hip, knee, ankle
  left_elbow: [11, 13, 15],         // L shoulder, elbow, wrist
  right_elbow: [12, 14, 16],        // R shoulder, elbow, wrist
  left_hip: [11, 23, 25],           // L shoulder, hip, knee
  right_hip: [12, 24, 26],          // R shoulder, hip, knee
  pelvis: [23, 24],                 // both hips
  trunk: [11, 12, 23, 24],          // shoulders + hips
  bilateral: [23, 24, 25, 26, 27, 28],
};

/**
 * Risk joint name → feature indices (for SkeletonGraph isolated view).
 * Maps injury_risk.joint to positions in the 14-joint srp_joints array.
 */
export const RISK_JOINT_TO_FEAT: Record<string, number[]> = {
  left_knee: [6, 8, 10],
  right_knee: [7, 9, 11],
  left_elbow: [0, 2, 4],
  right_elbow: [1, 3, 5],
  left_hip: [0, 6, 8],
  right_hip: [1, 7, 9],
  pelvis: [6, 7],
  trunk: [0, 1, 6, 7],
  bilateral: [6, 7, 8, 9, 10, 11],
};

/**
 * Map from MediaPipe feature index to position in 14-joint srp_joints array.
 * FEATURE_INDICES[i] -> i
 */
export const FEATURE_INDEX_MAP: Record<number, number> = {};
FEATURE_INDICES.forEach((mpIdx, i) => {
  FEATURE_INDEX_MAP[mpIdx] = i;
});

/**
 * Bone connections for the SRP skeleton graph, using positions in the
 * 14-joint srp_joints array (0=L Shoulder, 1=R Shoulder, ..., 13=R Foot).
 * Auto-generated from MP_BONES filtered to FEATURE_INDICES.
 */
export const FEATURE_BONES: [number, number][] = [];

// Filter MP_BONES to only include connections between feature joints,
// and remap indices to srp_joints positions.
for (const [a, b] of MP_BONES) {
  if (a in FEATURE_INDEX_MAP && b in FEATURE_INDEX_MAP) {
    FEATURE_BONES.push([FEATURE_INDEX_MAP[a], FEATURE_INDEX_MAP[b]]);
  }
}
