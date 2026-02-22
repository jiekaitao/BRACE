import Foundation

// MARK: - Sparse Landmark (backend sends 19 of 33 MediaPipe joints)

/// Indices sent by the backend: [0,1,2,3,4,11,12,13,14,15,16,23,24,25,26,27,28,31,32]
let kSendIndices: [Int] = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

struct SparseLandmark: Codable, Sendable {
    let i: Int      // MediaPipe joint index
    let x: Double   // normalised 0-1
    let y: Double   // normalised 0-1
    let v: Double   // visibility (0-1, threshold 0.3)
}

// MARK: - Bounding Box (normalised 0-1)

struct BBox: Codable, Sendable {
    let x1: Double
    let y1: Double
    let x2: Double
    let y2: Double
}

// MARK: - Biomechanics

struct Biomechanics: Codable, Sendable {
    let fppa_left: Double?
    let fppa_right: Double?
    let hip_drop: Double?
    let trunk_lean: Double?
    let asymmetry: Double?
    let curvature: Double?
    let jerk: Double?
    let angular_velocities: [String: Double]?
    let anomaly_score: Double?
    let com_velocity: Double?
    let com_sway: Double?
}

// MARK: - Injury Risk

struct InjuryRisk: Codable, Sendable {
    let joint: String
    let risk: String
    let severity: String   // "low", "medium", "high"
    let value: Double
    let threshold: Double
}

// MARK: - Movement Phase

struct MovementPhase: Codable, Sendable {
    let label: String      // "ascending", "descending", "transition"
    let progress: Double
    let cycle_count: Int
}

// MARK: - Joint Quality

struct JointQuality: Codable, Sendable {
    let scores: [Double]
    let degrading: [Int]
    let deviations: [Double]
}

// MARK: - Fatigue Timeline

struct FatigueTimeline: Codable, Sendable {
    let timestamps: [Double]
    let fatigue: [Double]
    let form_scores: [Double]
}

// MARK: - Active Guideline

struct ActiveGuideline: Codable, Sendable {
    let name: String
    let display_name: String
    let form_cues: [String]
}

// MARK: - Frame Quality (per-frame movement quality)

struct FrameQuality: Codable, Sendable {
    let movement_phase: MovementPhase?
    let form_score: Double?
    let joint_quality: JointQuality?
    let biomechanics: Biomechanics?
    let injury_risks: [InjuryRisk]?
    let fatigue_timeline: FatigueTimeline?
    let active_guideline: ActiveGuideline?
    let concussion_rating: Double?
    let fatigue_rating: Double?
}

// MARK: - Cluster Info

struct ClusterInfo: Codable, Sendable {
    let count: Int?
    let mean_score: Double?
    let anomaly_count: Int?
    let activity_label: String?
    let composite_fatigue: Double?
}

// MARK: - Per-Subject Data (multi-subject response)

struct SubjectData: Codable, Sendable {
    let label: String
    let phase: String          // "calibrating", "normal", "anomaly"
    let landmarks: [SparseLandmark]?
    let bbox: BBox?
    let cluster_id: Int?
    let consistency_score: Double?
    let is_anomaly: Bool
    let n_segments: Int
    let n_clusters: Int
    let cluster_summary: [String: ClusterInfo]?
    let velocity: Double?
    let rolling_velocity: Double?
    let fatigue_index: Double?
    let peak_velocity: Double?
    let quality: FrameQuality?
}

// MARK: - Multi-Subject Frame Response (top-level WS message)

struct MultiSubjectFrameResponse: Codable, Sendable {
    let frame_index: Int
    let video_time: Double?
    let subjects: [String: SubjectData]
    let active_track_ids: [Int]
}

// MARK: - Full 33-joint Landmark Array

struct Landmark: Equatable, Sendable {
    var x: Double = 0
    var y: Double = 0
    var visibility: Double = 0
}

/// Expand sparse landmarks into a full 33-element array.
func expandLandmarks(_ sparse: [SparseLandmark]) -> [Landmark] {
    var full = [Landmark](repeating: Landmark(), count: 33)
    for lm in sparse {
        guard lm.i >= 0 && lm.i < 33 else { continue }
        full[lm.i] = Landmark(x: lm.x, y: lm.y, visibility: lm.v)
    }
    return full
}

// MARK: - Bone Connections for Skeleton Drawing

/// MediaPipe pose connections for the 19 joints we receive.
let kBoneConnections: [(Int, Int)] = [
    // Head
    (0, 1), (1, 2), (0, 3), (3, 4),
    // Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    // Left arm
    (11, 13), (13, 15),
    // Right arm
    (12, 14), (14, 16),
    // Left leg
    (23, 25), (25, 27), (27, 31),
    // Right leg
    (24, 26), (26, 28), (28, 32),
]
