"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type {
  SubjectState,
  EmbeddingState,
  VelocityState,
  LandmarkFrame,
  SmplFrame,
  ClusterInfo,
  FrameQuality,
  DebugStats,
  MultiSubjectFrameResponse,
} from "@/lib/types";
import { getApiBase } from "@/lib/api";
import type { ConnectionStatus } from "./useAnalysisWebSocket";

const EMPTY_LANDMARK_FRAME: LandmarkFrame = {
  prev: null,
  current: null,
  prevTime: 0,
  currentTime: 0,
  prevVideoTime: 0,
  currentVideoTime: 0,
};

const EMPTY_EMBEDDING: EmbeddingState = {
  points: [],
  clusterIds: [],
  currentIdx: -1,
};

const EMPTY_SMPL_FRAME: SmplFrame = {
  prev: null,
  current: null,
  prevTime: 0,
  currentTime: 0,
};

/** Precomputed data fetched from backend */
interface PrecomputedData {
  video_filename: string;
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  frames: MultiSubjectFrameResponse[];
}

export interface UsePrecomputedReplayResult {
  connected: ConnectionStatus;
  replaying: boolean;
  phase: "calibrating" | "normal" | "anomaly";
  nSegments: number;
  nClusters: number;
  clusterId: number | null;
  consistencyScore: number | null;
  isAnomaly: boolean;
  clusterSummary: Record<string, ClusterInfo>;
  fatigueIndex: number;
  peakVelocity: number;
  videoProgress: number | null;
  videoComplete: boolean;
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  equipmentRef: React.MutableRefObject<import("@/lib/types").EquipmentTracking | undefined>;
  selectedSubjectId: number | null;
  activeTrackIds: number[];
  selectSubject: (trackId: number) => void;
  startCapture: (videoElement: HTMLVideoElement) => void;
  stopCapture: () => void;
  uploadVideo: (file: File) => Promise<void>;
  debugStatsRef: React.MutableRefObject<DebugStats>;
  sessionId: string | null;
  proximityRef: React.MutableRefObject<import("@/lib/types").ProximityData | null>;
  collisionWarning: boolean;
  closingSpeed: number;
  // Extra: loading state for the precomputed data
  dataLoading: boolean;
  dataReady: boolean;
}

export function usePrecomputedReplay(
  jobId: string | null,
  videoFilename: string | null,
): UsePrecomputedReplayResult {
  // Precomputed frame data
  const dataRef = useRef<PrecomputedData | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataReady, setDataReady] = useState(false);

  // Video element
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameRef = useRef(0);

  // Subject state (same as useAnalysisWebSocket)
  const subjectsRef = useRef<Map<number, SubjectState>>(new Map());
  const selectedSubjectRef = useRef<number | null>(null);
  const highlightedClusterRef = useRef<number | null>(null);
  const equipmentRef = useRef<import("@/lib/types").EquipmentTracking | undefined>(undefined);
  const userExplicitlySelectedRef = useRef(false);

  // React state for UI
  const [phase, setPhase] = useState<"calibrating" | "normal" | "anomaly">("calibrating");
  const [nSegments, setNSegments] = useState(0);
  const [nClusters, setNClusters] = useState(0);
  const [clusterId, setClusterId] = useState<number | null>(null);
  const [consistencyScore, setConsistencyScore] = useState<number | null>(null);
  const [isAnomaly, setIsAnomaly] = useState(false);
  const [clusterSummary, setClusterSummary] = useState<Record<string, ClusterInfo>>({});
  const [fatigueIndex, setFatigueIndex] = useState(0);
  const [peakVelocity, setPeakVelocity] = useState(0);
  const [selectedSubjectId, setSelectedSubjectId] = useState<number | null>(null);
  const [activeTrackIds, setActiveTrackIds] = useState<number[]>([]);

  const debugStatsRef = useRef<DebugStats>({
    fps_out: 0,
    fps_in: 0,
    kbps_out: 0,
    kbps_in: 0,
    rtt_ms: 0,
    activeSubjects: 0,
    serverFrameIndex: 0,
    history: {
      fps_out: [],
      fps_in: [],
      kbps_out: [],
      kbps_in: [],
      rtt_ms: [],
      subjects: [],
    },
  });

  // Fetch precomputed data
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    async function fetchData() {
      setDataLoading(true);
      const apiBase = getApiBase();

      // Poll until ready
      let attempts = 0;
      while (!cancelled && attempts < 600) {
        try {
          const res = await fetch(`${apiBase}/api/precompute/${jobId}/status`);
          const status = await res.json();

          if (status.status === "complete") break;
          if (status.status === "error") {
            console.error("Precompute failed:", status.error);
            setDataLoading(false);
            return;
          }
        } catch {
          // Retry
        }
        await new Promise((r) => setTimeout(r, 500));
        attempts++;
      }

      if (cancelled) return;

      // Fetch full data
      try {
        const res = await fetch(`${apiBase}/api/precompute/${jobId}/data`);
        if (!res.ok) {
          console.error("Failed to fetch precomputed data:", res.status);
          setDataLoading(false);
          return;
        }
        const data: PrecomputedData = await res.json();
        dataRef.current = data;

        // Build initial subject state from all frames
        buildSubjectState(data);

        setDataLoading(false);
        setDataReady(true);
      } catch (err) {
        console.error("Error fetching precomputed data:", err);
        setDataLoading(false);
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, [jobId]);

  /** Build SubjectState entries from precomputed frames. */
  function buildSubjectState(data: PrecomputedData) {
    const subjects = new Map<number, SubjectState>();

    // Collect per-subject velocity and replay timeline across all frames
    const velocityAccum = new Map<number, {
      values: number[];
      rolling: number[];
      timestamps: number[];
      fatigueIndex: number;
      peakVelocity: number;
    }>();

    const replayAccum = new Map<number, SubjectState["replayTimeline"]>();

    // Scan all frames to collect embedding data and final state
    for (const frame of data.frames) {
      for (const [trackIdStr, sd] of Object.entries(frame.subjects)) {
        const trackId = Number(trackIdStr);

        if (!velocityAccum.has(trackId)) {
          velocityAccum.set(trackId, {
            values: [], rolling: [], timestamps: [],
            fatigueIndex: 0, peakVelocity: 0,
          });
        }
        if (!replayAccum.has(trackId)) {
          replayAccum.set(trackId, []);
        }

        // Accumulate velocity
        const vel = velocityAccum.get(trackId)!;
        if (sd.velocity !== undefined) {
          vel.values.push(sd.velocity);
          vel.rolling.push(sd.rolling_velocity ?? 0);
          vel.timestamps.push(frame.video_time ?? 0);
          vel.fatigueIndex = sd.fatigue_index ?? 0;
          vel.peakVelocity = sd.peak_velocity ?? 0;
        }

        // Accumulate replay timeline
        const vt = frame.video_time ?? 0;
        if (vt > 0) {
          replayAccum.get(trackId)!.push({
            t: vt,
            quality: sd.quality ?? {},
            clusterId: sd.cluster_id,
            consistencyScore: sd.consistency_score,
            isAnomaly: sd.is_anomaly,
            phase: sd.phase,
            nSegments: sd.n_segments,
            nClusters: sd.n_clusters,
            clusterSummary: sd.cluster_summary,
          });
        }

        // Build/update subject state with latest frame's data
        if (!subjects.has(trackId)) {
          subjects.set(trackId, {
            trackId,
            label: sd.label,
            landmarkFrame: { ...EMPTY_LANDMARK_FRAME },
            bbox: null,
            phase: "calibrating",
            nSegments: 0,
            nClusters: 0,
            clusterId: null,
            consistencyScore: null,
            isAnomaly: false,
            clusterSummary: {},
            srpJoints: null,
            jointVisibility: null,
            representativeJoints: null,
            clusterRepresentatives: {},
            embedding: { ...EMPTY_EMBEDDING },
            velocity: { values: [], rolling: [], timestamps: [], fatigueIndex: 0, peakVelocity: 0 },
            identityStatus: "unknown",
            identityConfidence: 0,
            smplFrame: { ...EMPTY_SMPL_FRAME },
            uvTexture: null,
            quality: {},
            jerseyNumber: null,
            jerseyColor: null,
            jerseyCropBase64: null,
            jerseyGeminiResponse: null,
            teamId: null,
            teamColor: null,
            replayTimeline: [],
            firstPassVelocityLen: 0,
            lastSeenTime: performance.now(),
          });
        }

        // Update with latest data
        const state = subjects.get(trackId)!;
        state.label = sd.label;
        state.phase = sd.phase;
        state.nSegments = sd.n_segments;
        state.nClusters = sd.n_clusters;
        state.clusterId = sd.cluster_id;
        state.consistencyScore = sd.consistency_score;
        state.isAnomaly = sd.is_anomaly;
        state.clusterSummary = sd.cluster_summary;
        state.srpJoints = sd.srp_joints ?? null;
        state.identityStatus = sd.identity_status ?? "unknown";
        state.identityConfidence = sd.identity_confidence ?? 0;
        if (sd.quality) state.quality = sd.quality;
        if (sd.embedding_update) {
          const eu = sd.embedding_update;
          if (eu.type === "full" && eu.points && eu.cluster_ids) {
            state.embedding = {
              points: eu.points,
              clusterIds: eu.cluster_ids,
              currentIdx: eu.current_idx,
            };
          } else if (eu.type === "append" && eu.new_points && eu.new_cluster_ids) {
            state.embedding.points.push(...eu.new_points);
            state.embedding.clusterIds.push(...eu.new_cluster_ids);
            state.embedding.currentIdx = eu.current_idx;
          }
        }
      }
    }

    // Apply accumulated data
    for (const [trackId, state] of subjects) {
      const vel = velocityAccum.get(trackId);
      if (vel) {
        state.velocity = vel;
        state.firstPassVelocityLen = vel.values.length;
      }
      const timeline = replayAccum.get(trackId);
      if (timeline) {
        state.replayTimeline = timeline;
      }
    }

    subjectsRef.current = subjects;

    // Auto-select first subject
    const allIds = Array.from(subjects.keys());
    if (allIds.length > 0) {
      selectedSubjectRef.current = allIds[0];
      setSelectedSubjectId(allIds[0]);
    }
    setActiveTrackIds(allIds);
  }

  // Animation loop: sync video.currentTime to precomputed frame data
  useEffect(() => {
    if (!dataReady) return;

    let lastUiPush = 0;

    function tick() {
      const video = videoRef.current;
      const data = dataRef.current;
      if (!video || !data || data.frames.length === 0) {
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      const currentTime = video.currentTime;
      const fps = data.fps;
      const now = performance.now();

      // Map currentTime to frame index
      const frameIdx = Math.min(
        Math.round(currentTime * fps),
        data.frames.length - 1,
      );
      const frame = data.frames[Math.max(0, frameIdx)];
      if (!frame) {
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      // Apply frame data to subjects
      const currentSubjects = subjectsRef.current;
      const activeInFrame = new Set(frame.active_track_ids);

      for (const [trackIdStr, sd] of Object.entries(frame.subjects)) {
        const trackId = Number(trackIdStr);
        const state = currentSubjects.get(trackId);
        if (!state) continue;

        // Update landmarks for skeleton rendering (perfect sync!)
        if (sd.landmarks) {
          const lms: { x: number; y: number; visibility: number }[] =
            Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
          for (const lm of sd.landmarks as any[]) {
            lms[lm.i] = { x: lm.x, y: lm.y, visibility: lm.v };
          }
          state.landmarkFrame = {
            prev: state.landmarkFrame.current,
            current: lms,
            prevTime: state.landmarkFrame.currentTime,
            currentTime: now,
            prevVideoTime: state.landmarkFrame.currentVideoTime,
            currentVideoTime: currentTime,
          };
        }

        state.bbox = sd.bbox ?? null;
        state.label = sd.label;
        state.lastSeenTime = now;
        state.srpJoints = sd.srp_joints ?? null;
        state.identityStatus = sd.identity_status ?? "unknown";
        state.identityConfidence = sd.identity_confidence ?? 0;

        // Apply timeline snapshot for analysis data
        const tl = state.replayTimeline;
        if (tl.length > 0 && currentTime > 0) {
          let lo = 0, hi = tl.length - 1, best = 0;
          while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (tl[mid].t <= currentTime) {
              best = mid;
              lo = mid + 1;
            } else {
              hi = mid - 1;
            }
          }
          const snap = tl[best];
          const fullTimeline = tl[tl.length - 1].quality.fatigue_timeline;
          state.quality = { ...snap.quality, fatigue_timeline: fullTimeline };
          state.clusterId = snap.clusterId;
          state.consistencyScore = snap.consistencyScore;
          state.isAnomaly = snap.isAnomaly;
          state.phase = snap.phase;
          state.nSegments = snap.nSegments;
          state.nClusters = snap.nClusters;
          state.clusterSummary = snap.clusterSummary;
        }

        // Animate embedding
        const frac = video.duration > 0 ? currentTime / video.duration : 0;
        const n = state.embedding.points.length;
        if (n > 0) {
          state.embedding.currentIdx = Math.min(Math.floor(frac * n), n - 1);
        }
      }

      // Push React state at ~4/sec
      if (now - lastUiPush > 250) {
        lastUiPush = now;
        // Update active track IDs based on current frame
        setActiveTrackIds(frame.active_track_ids);

        const sel = selectedSubjectRef.current;
        if (sel !== null) {
          const s = currentSubjects.get(sel);
          if (s) {
            setPhase(s.phase);
            setNSegments(s.nSegments);
            setNClusters(s.nClusters);
            setClusterId(s.clusterId);
            setConsistencyScore(s.consistencyScore);
            setIsAnomaly(s.isAnomaly);
            setClusterSummary(s.clusterSummary);
            setFatigueIndex(s.velocity.fatigueIndex);
            setPeakVelocity(s.velocity.peakVelocity);
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(tick);
    }

    animFrameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [dataReady]);

  const selectSubject = useCallback((trackId: number) => {
    selectedSubjectRef.current = trackId;
    userExplicitlySelectedRef.current = true;
    setSelectedSubjectId(trackId);
  }, []);

  const startCapture = useCallback((videoElement: HTMLVideoElement) => {
    videoRef.current = videoElement;
  }, []);

  const stopCapture = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    videoRef.current = null;
  }, []);

  const uploadVideo = useCallback(async () => {
    // No-op for precomputed mode
  }, []);

  return {
    connected: dataReady ? "connected" : dataLoading ? "connecting" : "disconnected",
    replaying: true, // Always in replay mode
    phase,
    nSegments,
    nClusters,
    clusterId,
    consistencyScore,
    isAnomaly,
    clusterSummary,
    fatigueIndex,
    peakVelocity,
    videoProgress: null,
    videoComplete: false,
    subjectsRef,
    selectedSubjectRef,
    highlightedClusterRef,
    equipmentRef,
    selectedSubjectId,
    activeTrackIds,
    selectSubject,
    startCapture,
    stopCapture,
    uploadVideo,
    debugStatsRef,
    sessionId: null,
    proximityRef: { current: null } as React.MutableRefObject<import("@/lib/types").ProximityData | null>,
    collisionWarning: false,
    closingSpeed: 0,
    dataLoading,
    dataReady,
  };
}
