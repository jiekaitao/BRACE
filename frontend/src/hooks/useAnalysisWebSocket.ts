"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type {
  MultiSubjectFrameResponse,
  SubjectState,
  EmbeddingState,
  VelocityState,
  LandmarkFrame,
  SmplFrame,
  BBox,
  ClusterInfo,
  FrameQuality,
  ReplaySnapshot,
  ServerMessage,
  DebugStats,
} from "@/lib/types";
import { getApiBase, getWsBase } from "@/lib/api";

const CAPTURE_H = 480;
const JPEG_QUALITY = 0.65;
const TARGET_FPS = 30;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

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

const EMPTY_VELOCITY: VelocityState = {
  values: [],
  rolling: [],
  timestamps: [],
  fatigueIndex: 0,
  peakVelocity: 0,
};

const EMPTY_SMPL_FRAME: SmplFrame = {
  prev: null,
  current: null,
  prevTime: 0,
  currentTime: 0,
};

export interface UseAnalysisWebSocketResult {
  connected: boolean;
  replaying: boolean;
  // Selected subject's data (for React state-driven UI)
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
  // Multi-subject refs (for canvas rendering — no re-renders)
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  // React state for subject selection
  selectedSubjectId: number | null;
  activeTrackIds: number[];
  selectSubject: (trackId: number) => void;
  // Controls
  startCapture: (videoElement: HTMLVideoElement) => void;
  stopCapture: () => void;
  uploadVideo: (file: File) => Promise<void>;
  // Debug
  debugStatsRef: React.MutableRefObject<DebugStats>;
}

export function useAnalysisWebSocket(
  active: boolean,
  mode: "webcam" | "video"
): UseAnalysisWebSocketResult {
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef(0);
  const replayAnimRef = useRef(0);
  const lastSendTimeRef = useRef(0);
  const lastAppliedFrameRef = useRef(-1);
  const framesInFlightRef = useRef(0);
  const MAX_IN_FLIGHT = 5; // Allow pipeline overlap for higher throughput
  const rollingRttRef = useRef<number[]>([]);
  const sendTimesQueueRef = useRef<number[]>([]);
  const videoTimeRef = useRef(0);
  const MAX_RTT_SAMPLES = 10;
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [connected, setConnected] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const replayingRef = useRef(false);
  const [phase, setPhase] = useState<"calibrating" | "normal" | "anomaly">("calibrating");
  const [nSegments, setNSegments] = useState(0);
  const [nClusters, setNClusters] = useState(0);
  const [clusterId, setClusterId] = useState<number | null>(null);
  const [consistencyScore, setConsistencyScore] = useState<number | null>(null);
  const [isAnomaly, setIsAnomaly] = useState(false);
  const [clusterSummary, setClusterSummary] = useState<Record<string, ClusterInfo>>({});
  const [fatigueIndex, setFatigueIndex] = useState(0);
  const [peakVelocity, setPeakVelocity] = useState(0);
  const [videoProgress, setVideoProgress] = useState<number | null>(null);
  const [videoComplete, setVideoComplete] = useState(false);
  const [selectedSubjectId, setSelectedSubjectId] = useState<number | null>(null);
  const [activeTrackIds, setActiveTrackIds] = useState<number[]>([]);

  const lastUiUpdateRef = useRef(0);
  const subjectsRef = useRef<Map<number, SubjectState>>(new Map());
  const selectedSubjectRef = useRef<number | null>(null);
  const highlightedClusterRef = useRef<number | null>(null);

  // Debug stats tracking
  const HISTORY_LEN = 60;
  const debugStatsRef = useRef<DebugStats>({
    fps_out: 0, fps_in: 0, kbps_out: 0, kbps_in: 0,
    rtt_ms: 0, activeSubjects: 0, serverFrameIndex: 0,
    history: {
      fps_out: [], fps_in: [], kbps_out: [], kbps_in: [],
      rtt_ms: [], subjects: [],
    },
  });
  const debugAccRef = useRef({
    framesSent: 0,
    msgsReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    rttSum: 0,
    rttCount: 0,
  });

  const selectSubject = useCallback((trackId: number) => {
    selectedSubjectRef.current = trackId;
    setSelectedSubjectId(trackId);
    // Immediately push selected subject's data to React state
    const subject = subjectsRef.current.get(trackId);
    if (subject) {
      setPhase(subject.phase);
      setNSegments(subject.nSegments);
      setNClusters(subject.nClusters);
      setClusterId(subject.clusterId);
      setConsistencyScore(subject.consistencyScore);
      setIsAnomaly(subject.isAnomaly);
      setClusterSummary(subject.clusterSummary);
    }
  }, []);

  const getWsUrl = useCallback(
    (sessionId?: string) => {
      const base = getWsBase();
      if (mode === "video" && sessionId) {
        return `${base}/ws/analyze?mode=video&session_id=${sessionId}`;
      }
      return `${base}/ws/analyze?mode=webcam`;
    },
    [mode]
  );

  // Send a single frame as binary blob with 8-byte Float64 video timestamp prefix
  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!video || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.paused || video.ended || !video.videoWidth) return;

    // Backpressure: don't queue frames faster than the network can deliver
    if (framesInFlightRef.current >= MAX_IN_FLIGHT) return;

    // Rate limit to TARGET_FPS
    const now = performance.now();
    if (now - lastSendTimeRef.current < FRAME_INTERVAL_MS) return;
    lastSendTimeRef.current = now;

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
    }
    const canvas = captureCanvasRef.current;

    // Preserve video aspect ratio to avoid stretching
    const videoAR = video.videoWidth / video.videoHeight;
    const captureH = CAPTURE_H;
    const captureW = Math.round(captureH * videoAR);
    canvas.width = captureW;
    canvas.height = captureH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, captureW, captureH);

    // Detect video loop (currentTime jumps backward by more than 1 second)
    const prevVT = videoTimeRef.current;
    const videoTime = video.currentTime;
    if (prevVT > 0 && videoTime < prevVT - 1.0 && !replayingRef.current) {
      // Video looped — keep all cached analysis data, just reset frame counters
      framesInFlightRef.current = 0;
      lastAppliedFrameRef.current = -1;
      // Reset timing accumulators for fresh RTT measurements
      rollingRttRef.current = [];
      sendTimesQueueRef.current = [];
      // Clear landmark frames so canvas doesn't show stale end-of-video skeleton
      // and snapshot velocity length for replay trimming
      for (const subject of subjectsRef.current.values()) {
        subject.landmarkFrame = { ...EMPTY_LANDMARK_FRAME };
        // On first loop, record how many velocity samples the first pass produced
        if (subject.firstPassVelocityLen === 0) {
          subject.firstPassVelocityLen = subject.velocity.values.length;
        }
        // Trim velocity back to first-pass length (remove stale replay appends)
        const fpLen = subject.firstPassVelocityLen;
        if (subject.velocity.values.length > fpLen) {
          subject.velocity.values.length = fpLen;
          subject.velocity.rolling.length = fpLen;
          subject.velocity.timestamps.length = fpLen;
        }
      }
      replayingRef.current = true;
      setReplaying(true);
    }
    videoTimeRef.current = videoTime;

    // Send as binary: [8-byte Float64 timestamp] + [JPEG bytes]
    canvas.toBlob(
      (blob) => {
        if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((jpeg) => {
          const header = new Float64Array([videoTime]);
          const msg = new Uint8Array(8 + jpeg.byteLength);
          msg.set(new Uint8Array(header.buffer), 0);
          msg.set(new Uint8Array(jpeg), 8);
          sendTimesQueueRef.current.push(performance.now());
          debugAccRef.current.framesSent++;
          debugAccRef.current.bytesSent += msg.byteLength;
          framesInFlightRef.current++;
          wsRef.current!.send(msg);
        });
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }, []);

  // Connect WebSocket
  useEffect(() => {
    if (!active) {
      setConnected(false);
      return;
    }

    // For video mode, wait until we have a session ID
    if (mode === "video" && !sessionId) return;

    let stopped = false;
    let retryDelay = 1000;
    const MAX_RETRY_DELAY = 10000;

    function tryConnect() {
      if (stopped) return;

      const url = getWsUrl(sessionId ?? undefined);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      const timeout = setTimeout(() => ws.close(), 5000);

      ws.onopen = () => {
        clearTimeout(timeout);
        retryDelay = 1000;
        subjectsRef.current.clear();
        selectedSubjectRef.current = null;
        lastAppliedFrameRef.current = -1;
        framesInFlightRef.current = 0;
        replayingRef.current = false;
        setSelectedSubjectId(null);
        setReplaying(false);
        setConnected(true);
      };

      ws.onmessage = (event) => {
        // Debug stats: track RTT and incoming bytes
        const acc = debugAccRef.current;
        const sendTime = sendTimesQueueRef.current.shift();
        if (sendTime !== undefined) {
          const rtt = performance.now() - sendTime;
          acc.rttSum += rtt;
          acc.rttCount++;
          rollingRttRef.current.push(rtt);
          if (rollingRttRef.current.length > MAX_RTT_SAMPLES) rollingRttRef.current.shift();
        }
        acc.msgsReceived++;
        acc.bytesReceived += typeof event.data === "string" ? event.data.length : 0;

        let data: ServerMessage;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }

        // Handle typed messages
        if ("type" in data) {
          if (data.type === "video_progress") {
            setVideoProgress(data.progress);
            return;
          }
          if (data.type === "video_complete") {
            setVideoComplete(true);
            setVideoProgress(1.0);
            return;
          }
          if (data.type === "analysis_update") {
            return;
          }
          if (data.type === "error") {
            console.error("WS error:", data.message);
            return;
          }
        }

        // Multi-subject frame response
        const frame = data as MultiSubjectFrameResponse;
        if (frame.frame_index === undefined || !frame.subjects) return;

        // Decrement in-flight counter (response received)
        framesInFlightRef.current = Math.max(0, framesInFlightRef.current - 1);

        // Adjust video playback rate to match inference speed
        const video = videoRef.current;
        if (video && rollingRttRef.current.length >= 3) {
          const avgRtt = rollingRttRef.current.reduce((a, b) => a + b, 0) / rollingRttRef.current.length;
          // Account for pipelining: with N in-flight, effective throughput = N / avgRtt
          const pipelineThroughput = MAX_IN_FLIGHT / (avgRtt / 1000);
          const targetRate = Math.min(1.0, pipelineThroughput / TARGET_FPS);
          const clampedTarget = Math.max(0.8, targetRate);
          video.playbackRate += (clampedTarget - video.playbackRate) * 0.25;
        }

        // Discard out-of-order responses
        if (frame.frame_index < lastAppliedFrameRef.current) return;
        lastAppliedFrameRef.current = frame.frame_index;

        debugStatsRef.current.serverFrameIndex = frame.frame_index;
        debugStatsRef.current.activeSubjects = frame.active_track_ids?.length ?? 0;

        const now = performance.now();
        const currentSubjects = subjectsRef.current;
        const subjectIds = Object.keys(frame.subjects).map(Number);

        // Update each subject
        for (const [trackIdStr, subjectData] of Object.entries(frame.subjects)) {
          const trackId = Number(trackIdStr);
          let state = currentSubjects.get(trackId);

          if (!state) {
            // New subject
            state = {
              trackId,
              label: subjectData.label,
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
              alertText: null,
              replayTimeline: [],
              firstPassVelocityLen: 0,
              lastSeenTime: performance.now(),
            };
            currentSubjects.set(trackId, state);
          }

          // Update landmark frame with interpolation
          if (subjectData.landmarks) {
            // Rebuild sparse landmark list into full 33-element array
            const lms: { x: number; y: number; visibility: number }[] =
              Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
            for (const lm of subjectData.landmarks as any[]) {
              lms[lm.i] = { x: lm.x, y: lm.y, visibility: lm.v };
            }
            const videoTime = frame.video_time ?? 0;
            state.landmarkFrame = {
              prev: state.landmarkFrame.current,
              current: lms,
              prevTime: state.landmarkFrame.currentTime,
              currentTime: now,
              prevVideoTime: state.landmarkFrame.currentVideoTime,
              currentVideoTime: videoTime,
            };
          }

          state.label = subjectData.label;
          state.lastSeenTime = performance.now();
          state.bbox = subjectData.bbox;
          state.phase = subjectData.phase;
          state.nSegments = subjectData.n_segments;
          state.nClusters = subjectData.n_clusters;
          state.clusterId = subjectData.cluster_id;
          state.consistencyScore = subjectData.consistency_score;
          state.isAnomaly = subjectData.is_anomaly;
          state.clusterSummary = subjectData.cluster_summary;
          state.srpJoints = subjectData.srp_joints;
          state.jointVisibility = subjectData.joint_visibility ?? null;
          state.representativeJoints = subjectData.representative_joints ?? null;
          if (subjectData.cluster_representatives) {
            state.clusterRepresentatives = subjectData.cluster_representatives;
          }
          state.identityStatus = subjectData.identity_status ?? "unknown";
          state.identityConfidence = subjectData.identity_confidence ?? 0;

          // Update SMPL frame with interpolation
          if (subjectData.smpl_params) {
            state.smplFrame = {
              prev: state.smplFrame.current,
              current: subjectData.smpl_params,
              prevTime: state.smplFrame.currentTime,
              currentTime: now,
            };
          }

          // Update UV texture
          if (subjectData.uv_texture) {
            state.uvTexture = subjectData.uv_texture;
          }

          // Accumulate velocity history (skip during replay — use cached first-pass data)
          if (subjectData.velocity !== undefined && !replayingRef.current) {
            state.velocity.values.push(subjectData.velocity);
            state.velocity.rolling.push(subjectData.rolling_velocity ?? 0);
            state.velocity.timestamps.push(frame.video_time ?? 0);
            state.velocity.fatigueIndex = subjectData.fatigue_index ?? 0;
            state.velocity.peakVelocity = subjectData.peak_velocity ?? 0;
            // Trim to last 600 samples (~20s at 30fps)
            const MAX_VEL = 600;
            if (state.velocity.values.length > MAX_VEL) {
              state.velocity.values.splice(0, state.velocity.values.length - MAX_VEL);
              state.velocity.rolling.splice(0, state.velocity.rolling.length - MAX_VEL);
              state.velocity.timestamps.splice(0, state.velocity.timestamps.length - MAX_VEL);
            }
          }

          // Update movement quality metrics
          if (subjectData.quality) {
            // Merge incrementally: fatigue_timeline only sent when changed,
            // so preserve existing timeline if not in current frame
            const prev = state.quality;
            state.quality = {
              ...subjectData.quality,
              fatigue_timeline: subjectData.quality.fatigue_timeline ?? prev.fatigue_timeline,
            };
          }

          // Store voice alert text for voice coaching
          if (subjectData.alert_text) {
            state.alertText = subjectData.alert_text;
          }

          // Record replay timeline during first pass
          const vt = frame.video_time ?? 0;
          if (!replayingRef.current && vt > 0) {
            state.replayTimeline.push({
              t: vt,
              quality: { ...state.quality },
              clusterId: state.clusterId,
              consistencyScore: state.consistencyScore,
              isAnomaly: state.isAnomaly,
              phase: state.phase,
              nSegments: state.nSegments,
              nClusters: state.nClusters,
              clusterSummary: state.clusterSummary,
            });
          }

          // Process embedding update
          if (subjectData.embedding_update) {
            const eu = subjectData.embedding_update;
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

        // During replay, override analysis data from cached timeline
        if (replayingRef.current) {
          const video = videoRef.current;
          const currentTime = video?.currentTime ?? 0;
          for (const state of currentSubjects.values()) {
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
              // Use the full fatigue_timeline from the final snapshot
              // (each snapshot only has partial data up to that moment)
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
          }
        }

        // Update Gemini stats
        if (frame.gemini_stats) {
          debugStatsRef.current.geminiStats = frame.gemini_stats;
        }

        // Remove subjects not in active_track_ids
        const activeSet = new Set(frame.active_track_ids);
        for (const existingId of currentSubjects.keys()) {
          if (!activeSet.has(existingId)) {
            currentSubjects.delete(existingId);
          }
        }

        // Auto-select first subject if none selected
        if (
          selectedSubjectRef.current === null ||
          !currentSubjects.has(selectedSubjectRef.current)
        ) {
          if (subjectIds.length > 0) {
            const firstId = subjectIds[0];
            selectedSubjectRef.current = firstId;
            setSelectedSubjectId(firstId);
          }
        }

        // Throttle React state updates to ~4/sec
        const uiNow = performance.now();
        if (uiNow - lastUiUpdateRef.current > 250) {
          setActiveTrackIds(frame.active_track_ids);

          // Push selected subject's data to React state
          const selected = selectedSubjectRef.current;
          if (selected !== null) {
            const s = currentSubjects.get(selected);
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
          lastUiUpdateRef.current = uiNow;
        }
      };

      ws.onclose = () => {
        clearTimeout(timeout);
        setConnected(false);
        wsRef.current = null;
        if (!stopped && mode === "webcam") {
          setTimeout(tryConnect, retryDelay);
          retryDelay = Math.min(retryDelay * 1.5, MAX_RETRY_DELAY);
        }
      };

      ws.onerror = () => {
        clearTimeout(timeout);
        ws.close();
      };
    }

    tryConnect();

    return () => {
      stopped = true;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnected(false);
    };
  }, [active, mode, getWsUrl, sessionId]);

  // Debug stats: 1-second interval to compute rates
  useEffect(() => {
    if (!active) return;
    const interval = setInterval(() => {
      const acc = debugAccRef.current;
      const stats = debugStatsRef.current;
      const h = stats.history;

      stats.fps_out = acc.framesSent;
      stats.fps_in = acc.msgsReceived;
      stats.kbps_out = acc.bytesSent / 1024;
      stats.kbps_in = acc.bytesReceived / 1024;
      stats.rtt_ms = acc.rttCount > 0 ? acc.rttSum / acc.rttCount : stats.rtt_ms;

      h.fps_out.push(stats.fps_out);
      h.fps_in.push(stats.fps_in);
      h.kbps_out.push(stats.kbps_out);
      h.kbps_in.push(stats.kbps_in);
      h.rtt_ms.push(stats.rtt_ms);
      h.subjects.push(stats.activeSubjects);

      // Trim to HISTORY_LEN
      if (h.fps_out.length > HISTORY_LEN) h.fps_out.shift();
      if (h.fps_in.length > HISTORY_LEN) h.fps_in.shift();
      if (h.kbps_out.length > HISTORY_LEN) h.kbps_out.shift();
      if (h.kbps_in.length > HISTORY_LEN) h.kbps_in.shift();
      if (h.rtt_ms.length > HISTORY_LEN) h.rtt_ms.shift();
      if (h.subjects.length > HISTORY_LEN) h.subjects.shift();

      // Reset accumulators
      acc.framesSent = 0;
      acc.msgsReceived = 0;
      acc.bytesSent = 0;
      acc.bytesReceived = 0;
      acc.rttSum = 0;
      acc.rttCount = 0;
    }, 1000);
    return () => clearInterval(interval);
  }, [active]);

  // Animation loop for webcam frame capture
  useEffect(() => {
    if (!connected || !active || mode !== "webcam") return;

    function loop() {
      sendFrame();
      // During replay, animate all cached data based on video time
      if (replayingRef.current) {
        const video = videoRef.current;
        if (video && video.duration > 0 && isFinite(video.duration)) {
          const currentTime = video.currentTime;
          const frac = currentTime / video.duration;
          for (const subject of subjectsRef.current.values()) {
            // Animate embedding currentIdx
            const n = subject.embedding.points.length;
            if (n > 0) {
              subject.embedding.currentIdx = Math.min(
                Math.floor(frac * n),
                n - 1
              );
            }

            // Apply cached analysis snapshot for current video time
            const tl = subject.replayTimeline;
            if (tl.length > 0) {
              // Binary search for last snapshot with t <= currentTime
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
              subject.quality = { ...snap.quality, fatigue_timeline: fullTimeline };
              subject.clusterId = snap.clusterId;
              subject.consistencyScore = snap.consistencyScore;
              subject.isAnomaly = snap.isAnomaly;
              subject.phase = snap.phase;
              subject.nSegments = snap.nSegments;
              subject.nClusters = snap.nClusters;
              subject.clusterSummary = snap.clusterSummary;
            }
          }
        }
      }
      animFrameRef.current = requestAnimationFrame(loop);
    }
    animFrameRef.current = requestAnimationFrame(loop);

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [connected, active, mode, sendFrame]);

  // Replay animation loop for demo/video modes (webcam has its own in the sendFrame loop)
  // This ensures quality/cluster/embedding data updates at 60fps during replay
  // even when the user pauses or scrubs the video (no new WS messages arriving)
  useEffect(() => {
    if (!active || mode === "webcam") return;
    let lastUiPush = 0;

    function replayTick() {
      if (replayingRef.current) {
        const video = videoRef.current;
        if (video && video.duration > 0 && isFinite(video.duration)) {
          const currentTime = video.currentTime;
          const frac = currentTime / video.duration;
          for (const subject of subjectsRef.current.values()) {
            // Animate embedding currentIdx
            const n = subject.embedding.points.length;
            if (n > 0) {
              subject.embedding.currentIdx = Math.min(
                Math.floor(frac * n),
                n - 1
              );
            }

            // Apply cached analysis snapshot for current video time
            const tl = subject.replayTimeline;
            if (tl.length > 0) {
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
              subject.quality = { ...snap.quality, fatigue_timeline: fullTimeline };
              subject.clusterId = snap.clusterId;
              subject.consistencyScore = snap.consistencyScore;
              subject.isAnomaly = snap.isAnomaly;
              subject.phase = snap.phase;
              subject.nSegments = snap.nSegments;
              subject.nClusters = snap.nClusters;
              subject.clusterSummary = snap.clusterSummary;
            }
          }

          // Push React state at ~4/sec so UI panels update during scrub
          const now = performance.now();
          if (now - lastUiPush > 250) {
            lastUiPush = now;
            const sel = selectedSubjectRef.current;
            if (sel !== null) {
              const s = subjectsRef.current.get(sel);
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
        }
      }
      replayAnimRef.current = requestAnimationFrame(replayTick);
    }
    replayAnimRef.current = requestAnimationFrame(replayTick);
    return () => cancelAnimationFrame(replayAnimRef.current);
  }, [active, mode]);

  const startCapture = useCallback((videoElement: HTMLVideoElement) => {
    videoRef.current = videoElement;
  }, []);

  const stopCapture = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    videoRef.current = null;
  }, []);

  const uploadVideo = useCallback(async (file: File) => {
    const apiBase = getApiBase();
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${apiBase}/api/upload`, {
      method: "POST",
      body: formData,
    });
    const json = await res.json();
    setSessionId(json.session_id);
    setVideoProgress(0);
    setVideoComplete(false);
  }, []);

  return {
    connected,
    replaying,
    phase,
    nSegments,
    nClusters,
    clusterId,
    consistencyScore,
    isAnomaly,
    clusterSummary,
    fatigueIndex,
    peakVelocity,
    videoProgress,
    videoComplete,
    subjectsRef,
    selectedSubjectRef,
    highlightedClusterRef,
    selectedSubjectId,
    activeTrackIds,
    selectSubject,
    startCapture,
    stopCapture,
    uploadVideo,
    debugStatsRef,
  };
}
