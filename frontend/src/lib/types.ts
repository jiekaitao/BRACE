/** SMPL body model parameters from backend. */
export interface SmplParams {
  betas: number[];   // 10 shape params
  pose: number[];    // 72 pose params (24 joints * 3 axis-angle)
  trans: number[];   // 3 translation
}

/** SMPL frame for interpolation (prev + current). */
export interface SmplFrame {
  prev: SmplParams | null;
  current: SmplParams | null;
  prevTime: number;
  currentTime: number;
}

/** Landmark point from the backend (normalized 0-1). */
export interface Landmark {
  x: number;
  y: number;
  visibility: number;
}

/** Bounding box (normalized 0-1). */
export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Per-cluster summary entry. */
export interface ClusterInfo {
  count: number;
  mean_score: number;
  anomaly_count: number;
  activity_label?: string;
  composite_fatigue?: number;
  cusum_onset_rep?: number | null;
  ewma_alarming_joints?: number[];
}

/** Movement phase within current segment. */
export interface MovementPhase {
  label: "ascending" | "descending" | "transition";
  progress: number;
  cycle_count: number;
}

/** Per-joint quality scores and degradation tracking. */
export interface JointQuality {
  scores: number[];        // 0-100 per joint (14 joints)
  degrading: number[];     // indices of degrading joints
  deviations: number[];    // raw deviation per joint
}

/** Injury risk flag from rule-based clinical thresholds. */
export interface InjuryRisk {
  joint: string;
  risk: string;
  severity: "low" | "medium" | "high";
  value: number;
  threshold: number;
}

/** Biomechanical metrics (per-frame). */
export interface Biomechanics {
  fppa_left: number;
  fppa_right: number;
  hip_drop: number;
  trunk_lean: number;
  asymmetry: number;
  curvature: number;
  jerk: number;
  angular_velocities?: Record<string, number>;
  anomaly_score?: number;
  com_velocity?: number;
  com_sway?: number;
}

/** Fatigue timeline sampled at ~1Hz. */
export interface FatigueTimeline {
  timestamps: number[];
  fatigue: number[];
  form_scores: number[];
}

/** Active movement guideline from matched profile. */
export interface ActiveGuideline {
  name: string;          // "squat", "lunge", etc.
  display_name: string;  // "Squat", "Forward Lunge", etc.
  form_cues: string[];   // coaching tips
}

/** Per-frame movement quality from backend. */
export interface FrameQuality {
  movement_phase?: MovementPhase;
  form_score?: number;
  joint_quality?: JointQuality;
  biomechanics?: Biomechanics;
  injury_risks?: InjuryRisk[];
  fatigue_timeline?: FatigueTimeline;
  active_guideline?: ActiveGuideline;
  concussion_rating?: number;
  fatigue_rating?: number;
}

/** Per-frame response from the backend WebSocket (legacy single-subject). */
export interface FrameResponse {
  frame_index: number;
  phase: "calibrating" | "normal" | "anomaly";
  n_segments: number;
  n_clusters: number;
  landmarks: Landmark[] | null;
  bbox: BBox | null;
  cluster_id: number | null;
  consistency_score: number | null;
  is_anomaly: boolean;
  cluster_summary: Record<string, ClusterInfo>;
}

/** Per-subject velocity state maintained on frontend. */
export interface VelocityState {
  values: number[];         // raw velocity per frame
  rolling: number[];        // EMA-smoothed velocity
  timestamps: number[];     // video_time for each sample
  fatigueIndex: number;     // 0.0-1.0
  peakVelocity: number;
}

/** UMAP embedding update from backend. */
export interface EmbeddingUpdate {
  type: "full" | "append" | "current_only";
  points?: [number, number, number][];
  new_points?: [number, number, number][];
  cluster_ids?: (number | null)[];
  new_cluster_ids?: (number | null)[];
  current_idx: number;
}

/** Per-subject data in multi-subject frame response. */
export interface SubjectData {
  label: string;
  phase: "calibrating" | "normal" | "anomaly";
  landmarks: Landmark[] | null;
  bbox: BBox | null;
  cluster_id: number | null;
  consistency_score: number | null;
  is_anomaly: boolean;
  n_segments: number;
  n_clusters: number;
  cluster_summary: Record<string, ClusterInfo>;
  srp_joints: ([number, number] | [number, number, number])[] | null;
  joint_visibility?: number[] | null;
  representative_joints?: ([number, number] | [number, number, number])[] | null;
  embedding_update?: EmbeddingUpdate;
  identity_status?: "unknown" | "tentative" | "confirmed";
  identity_confidence?: number;
  smpl_params?: SmplParams;
  uv_texture?: string;  // base64 JPEG
  cluster_representatives?: Record<string, ([number, number] | [number, number, number])[][]>;
  velocity?: number;
  rolling_velocity?: number;
  fatigue_index?: number;
  peak_velocity?: number;
  quality?: FrameQuality;
  jersey_number?: number | null;
  jersey_color?: string | null;
  jersey_crop_b64?: string;
  jersey_gemini_response?: string;
  team_id?: number | null;
  team_color?: string | null;
  similar_movements?: SimilarMovement[];
}

/** Gemini API usage statistics from backend. */
export interface GeminiStats {
  api_calls: number;
  cache_hits: number;
  estimated_cost_usd: number;
}

/** Equipment tracking data from Gemini ER. */
export interface EquipmentTracking {
  box: [number, number, number, number] | null;
  momentum: number;
  held_by_id: string | null;
}

/** Pairwise closing speed between two subjects. */
export interface ProximityPair {
  a: number;
  b: number;
  closing_speed: number;
  distance: number;
}

/** Proximity / collision detection data from backend. */
export interface ProximityData {
  pairs: ProximityPair[];
  max_closing_speed: number;
  collision_warning: boolean;
}

/** Multi-subject frame response from backend. */
export interface MultiSubjectFrameResponse {
  frame_index: number;
  video_time?: number;
  subjects: Record<string, SubjectData>;
  active_track_ids: number[];
  equipment?: EquipmentTracking;
  proximity?: ProximityData;
  gemini_stats?: GeminiStats;
}

/** Snapshot of per-frame analysis data cached for replay on video loop. */
export interface ReplaySnapshot {
  t: number;  // video_time
  quality: FrameQuality;
  clusterId: number | null;
  consistencyScore: number | null;
  isAnomaly: boolean;
  phase: "calibrating" | "normal" | "anomaly";
  nSegments: number;
  nClusters: number;
  clusterSummary: Record<string, ClusterInfo>;
}

/** UMAP embedding state maintained per subject on frontend. */
export interface EmbeddingState {
  points: [number, number, number][];
  clusterIds: (number | null)[];
  currentIdx: number;
}

/** Full per-subject state maintained on frontend. */
export interface SubjectState {
  trackId: number;
  label: string;
  landmarkFrame: LandmarkFrame;
  bbox: BBox | null;
  phase: "calibrating" | "normal" | "anomaly";
  nSegments: number;
  nClusters: number;
  clusterId: number | null;
  consistencyScore: number | null;
  isAnomaly: boolean;
  clusterSummary: Record<string, ClusterInfo>;
  srpJoints: ([number, number] | [number, number, number])[] | null;
  jointVisibility: number[] | null;
  representativeJoints: ([number, number] | [number, number, number])[] | null;
  embedding: EmbeddingState;
  velocity: VelocityState;
  identityStatus: "unknown" | "tentative" | "confirmed";
  identityConfidence: number;
  smplFrame: SmplFrame;
  uvTexture: string | null;
  clusterRepresentatives: Record<string, ([number, number] | [number, number, number])[][]>;
  /** Per-frame movement quality metrics. */
  quality: FrameQuality;
  /** Jersey detection results. */
  jerseyNumber: number | null;
  jerseyColor: string | null;
  jerseyCropBase64: string | null;
  jerseyGeminiResponse: string | null;
  /** Similar movements from VectorAI cross-session search. */
  similarMovements?: SimilarMovement[];
  /** Visual K-Means team clustering results. */
  teamId: number | null;
  teamColor: string | null;
  /** Cached analysis snapshots for replay on video loop, indexed by video_time. */
  replayTimeline: ReplaySnapshot[];
  /** Number of velocity samples recorded during first pass (used during replay). */
  firstPassVelocityLen: number;
  /** Timestamp (performance.now()) of the last frame where this subject had data. */
  lastSeenTime: number;
}

/** A contiguous segment belonging to one cluster. */
export interface ClusterSegment {
  clusterId: number;
  startTime: number;
  endTime: number;
  activityLabel?: string;
  guidelineName?: string;
}

/** A contiguous segment where an injury risk was flagged. */
export interface RiskSegment {
  riskType: string;
  severity: "medium" | "high";
  startTime: number;
  endTime: number;
  joint: string;
}

/** Derived timeline data for the WorkoutTimeline component. */
export interface TimelineData {
  duration: number;
  clusterSegments: ClusterSegment[];
  riskSegments: RiskSegment[];
}

/** Video progress message. */
export interface VideoProgress {
  type: "video_progress";
  progress: number;
  frame: number;
  total: number;
}

/** Video complete message. */
export interface VideoComplete {
  type: "video_complete";
  final_summary: Record<string, {
    label: string;
    total_frames: number;
    valid_frames: number;
    n_segments: number;
    n_clusters: number;
    cluster_summary: Record<string, ClusterInfo>;
  }>;
}

/** Analysis update message. */
export interface AnalysisUpdate {
  type: "analysis_update";
  message: string;
}

/** Config message sent from client. */
export interface ConfigMessage {
  type: "config";
  fps?: number;
  cluster_threshold?: number;
}

/** Landmark frame for canvas rendering (prev + current for interpolation). */
export interface LandmarkFrame {
  prev: Landmark[] | null;
  current: Landmark[] | null;
  prevTime: number;
  currentTime: number;
  prevVideoTime: number;
  currentVideoTime: number;
}

/** Debug statistics for the debug panel. */
export interface DebugStats {
  fps_out: number;
  fps_in: number;
  kbps_out: number;
  kbps_in: number;
  rtt_ms: number;
  activeSubjects: number;
  serverFrameIndex: number;
  geminiStats?: GeminiStats;
  history: {
    fps_out: number[];
    fps_in: number[];
    kbps_out: number[];
    kbps_in: number[];
    rtt_ms: number[];
    subjects: number[];
  };
}

/** GPU statistics from backend /api/gpu-stats endpoint. */
export interface GpuStats {
  available: boolean;
  name?: string;
  gpu_util?: number;
  mem_util?: number;
  vram_used_gb?: number;
  vram_total_gb?: number;
  temp_c?: number;
  power_w?: number;
  error?: string;
}

/** Union of all server->client messages. */
export type ServerMessage =
  | MultiSubjectFrameResponse
  | VideoProgress
  | VideoComplete
  | AnalysisUpdate
  | { type: "error"; message: string };

/** User identity from auth system. */
export interface User {
  user_id: string;
  username: string;
  injury_profile?: Record<string, unknown> | null;
  risk_modifiers?: Record<string, unknown> | null;
}

/** Auth state for the AuthContext. */
export interface AuthState {
  user: User | null;
  loading: boolean;
}

/** Chat message in the injury intake conversation. */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/** Injury entry from the intake chat. */
export interface InjuryEntry {
  type: string;        // "acl", "shoulder", "ankle", "lower_back", etc.
  side: string;        // "left", "right", "bilateral", "unknown"
  severity: string;    // "mild", "moderate", "severe"
  timeframe: string;   // "acute", "chronic", "recovered"
}

/** Extracted injury profile from the intake chat. */
export interface InjuryProfile {
  injuries: InjuryEntry[];
  complete?: boolean;
}

/** Chat response from the backend. */
export interface ChatResponse {
  response: string;
  extracted_profile?: InjuryProfile | null;
  profile_complete?: boolean;
}

/** Per-metric threshold multipliers from the risk profile system. */
export interface RiskModifiers {
  fppa_scale: number;
  hip_drop_scale: number;
  trunk_lean_scale: number;
  asymmetry_scale: number;
  angular_velocity_scale: number;
  monitor_joints: string[];
}

// ---------------------------------------------------------------------------
// Basketball / Game Analysis Types
// ---------------------------------------------------------------------------

/** Risk status for a player during game analysis. */
export type PlayerRiskStatus = "GREEN" | "YELLOW" | "RED";

/** Player info from game analysis. */
export interface GamePlayerInfo {
  subject_id: number;
  label: string;
  jersey_number?: number | null;
  jersey_color?: string | null;
  risk_status: PlayerRiskStatus;
  total_frames: number;
  injury_events: InjuryRisk[];
  workload: {
    total_frames: number;
    active_seconds: number;
    high_intensity_seconds: number;
    rest_seconds: number;
    intensity_ratio: number;
    fatigue_estimate: number;
  };
}

/** Game processing progress message from WebSocket. */
export interface GameProgressMessage {
  type: "progress";
  progress: number;
  data: {
    frame_index: number;
    total_frames: number;
    player_count: number;
    video_time: number;
  };
}

/** Game complete message from WebSocket. */
export interface GameCompleteMessage {
  type: "complete";
  data: {
    game_id: string;
    status: string;
    total_frames: number;
    duration_sec: number;
    player_count: number;
    players: Record<string, GamePlayerInfo>;
  };
}

/** Union of game WebSocket messages. */
export type GameWsMessage =
  | GameProgressMessage
  | GameCompleteMessage
  | { type: "heartbeat" };

// ---------------------------------------------------------------------------
// VectorAI Dashboard Types
// ---------------------------------------------------------------------------

/** Similar movement match from VectorAI. */
export interface SimilarMovement {
  activity_label: string;
  score: number;
  session_id: string;
}

/** Stats for a single VectorAI collection. */
export interface CollectionStats {
  count: number;
  total_vectors: number;
  indexed_vectors: number;
  deleted_vectors: number;
  storage_bytes: number;
  index_memory_bytes: number;
  error?: string;
}

/** Stats response for all collections. */
export type VectorStats = Record<string, CollectionStats>;

/** A single vector entry from the MongoDB mirror. */
export interface VectorEntry {
  vector_uuid: string;
  collection: string;
  metadata: Record<string, unknown>;
  timestamp: number;
  created_at: string;
  person_id?: string;
  session_id?: string;
  activity_label?: string;
  person_crop_b64?: string;
}

/** Paginated response from /api/vectorai/entries. */
export interface VectorEntriesResponse {
  collection: string;
  total: number;
  offset: number;
  limit: number;
  entries: VectorEntry[];
  error?: string;
}
