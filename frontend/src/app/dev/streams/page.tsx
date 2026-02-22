"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Card from "@/components/ui/Card";
import { getApiBase } from "@/lib/api";

interface StreamInfo {
  stream_id: string;
  mode: string;
  client_type: string;
  connected_at: number;
  uptime_sec: number;
  frame_count: number;
  subject_count: number;
  fps: number;
  last_thumbnail: string | null;
  resolution: [number, number] | null;
  vr_selected_subject: number | null;
}

interface StreamsResponse {
  count: number;
  streams: StreamInfo[];
}

/* eslint-disable @typescript-eslint/no-explicit-any */
interface StreamData {
  frame_index?: number;
  video_time?: number;
  active_track_ids?: number[];
  timing?: Record<string, number>;
  vr_selected_subject?: number | null;
  subjects?: Record<string, SubjectEntry>;
  [key: string]: any;
}

interface SubjectEntry {
  label?: string;
  phase?: string;
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  n_segments?: number;
  n_clusters?: number;
  cluster_id?: number;
  consistency_score?: number;
  velocity?: number;
  rolling_velocity?: number;
  fatigue_index?: number;
  identity_status?: string;
  identity_confidence?: number;
  quality?: any;
  [key: string]: any;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/* ── Small card in the grid ─────────────────────────────────────────── */

function StreamCard({
  stream,
  onClick,
  selected,
}: {
  stream: StreamInfo;
  onClick: () => void;
  selected: boolean;
}) {
  const isVR = stream.client_type === "vr";
  return (
    <Card
      interactive
      className={`flex flex-col gap-3 ${selected ? "ring-2 ring-[#1CB0F6]" : ""}`}
      onClick={onClick}
    >
      <div className="relative w-full aspect-video bg-[#1A1A2E] rounded-lg overflow-hidden">
        {stream.last_thumbnail ? (
          <img
            src={`data:image/jpeg;base64,${stream.last_thumbnail}`}
            alt="Stream preview"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-[#AFAFAF] text-sm">
            Waiting for frames...
          </div>
        )}
        <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-black/70 rounded-full px-2 py-0.5">
          <span className="w-2 h-2 rounded-full bg-[#FF4B4B] animate-pulse" />
          <span className="text-white text-[10px] font-bold uppercase">Live</span>
        </div>
        {stream.subject_count > 0 && (
          <div className="absolute top-2 right-2 bg-[#1CB0F6] text-white text-[10px] font-bold rounded-full px-2 py-0.5">
            {stream.subject_count} {stream.subject_count === 1 ? "person" : "people"}
          </div>
        )}
        {isVR && (
          <div className="absolute bottom-2 left-2 bg-[#7B2FF7]/90 text-white text-[10px] font-bold rounded px-1.5 py-0.5">
            Quest 3 VR
          </div>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-[#3C3C3C] uppercase tracking-wide">
            {stream.mode === "video" ? "Video" : "Webcam"}
          </span>
          <span
            className={`text-[10px] font-bold rounded px-1.5 py-0.5 ${
              isVR
                ? "text-white bg-[#7B2FF7]"
                : "text-[#AFAFAF] bg-[#F7F7F7]"
            }`}
          >
            {stream.client_type.toUpperCase()}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-[#4B4B4B]">
          <div>
            <span className="text-[#AFAFAF]">Uptime: </span>
            <span className="font-bold">{formatUptime(stream.uptime_sec)}</span>
          </div>
          <div>
            <span className="text-[#AFAFAF]">Frames: </span>
            <span className="font-bold">{stream.frame_count.toLocaleString()}</span>
          </div>
        </div>
        {isVR && (
          <div className="text-[10px] text-[#7B2FF7] font-bold">
            VR Selection:{" "}
            {stream.vr_selected_subject != null
              ? `S${stream.vr_selected_subject}`
              : "None"}
          </div>
        )}
        <div className="text-[10px] text-[#AFAFAF] font-mono truncate">
          {stream.stream_id.slice(0, 8)}...
        </div>
      </div>
    </Card>
  );
}

/* ── Subject detail card ──────────────────────────────────────────── */

function SubjectCard({
  sid,
  data,
  isSelected,
}: {
  sid: string;
  data: SubjectEntry;
  isSelected: boolean;
}) {
  const phase = data.phase ?? "unknown";
  const phaseColor =
    phase === "normal"
      ? "text-[#58CC02]"
      : phase === "calibrating"
        ? "text-[#FFB020]"
        : phase === "anomaly"
          ? "text-[#FF4B4B]"
          : "text-[#AFAFAF]";

  return (
    <div
      className={`rounded-lg p-3 text-xs ${
        isSelected
          ? "bg-[#58CC02]/10 border border-[#58CC02]/40"
          : "bg-[#F7F7F7] border border-transparent"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-black text-sm text-[#3C3C3C]">
            {data.label ?? `S${sid}`}
          </span>
          <span className={`font-bold uppercase text-[10px] ${phaseColor}`}>
            {phase}
          </span>
        </div>
        {isSelected && (
          <span className="text-[10px] font-bold text-white bg-[#58CC02] rounded-full px-2 py-0.5">
            VR SELECTED
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-[#4B4B4B]">
        {data.identity_status && (
          <div>
            <span className="text-[#AFAFAF]">Identity: </span>
            <span className="font-bold">
              {data.identity_status}
              {data.identity_confidence != null &&
                ` (${(data.identity_confidence * 100).toFixed(0)}%)`}
            </span>
          </div>
        )}
        {data.n_segments != null && (
          <div>
            <span className="text-[#AFAFAF]">Segments: </span>
            <span className="font-bold">{data.n_segments}</span>
            <span className="text-[#AFAFAF]"> Clusters: </span>
            <span className="font-bold">{data.n_clusters ?? 0}</span>
          </div>
        )}
        {(data.velocity ?? 0) > 0.001 && (
          <div>
            <span className="text-[#AFAFAF]">Velocity: </span>
            <span className="font-bold">{data.velocity?.toFixed(2)}</span>
            <span className="text-[#AFAFAF]"> Rolling: </span>
            <span className="font-bold">
              {data.rolling_velocity?.toFixed(2)}
            </span>
          </div>
        )}
        {(data.consistency_score ?? 0) > 0.001 && (
          <div>
            <span className="text-[#AFAFAF]">Consistency: </span>
            <span className="font-bold">
              {data.consistency_score?.toFixed(2)}
            </span>
          </div>
        )}
        {(data.fatigue_index ?? 0) > 0.001 && (
          <div>
            <span className="text-[#AFAFAF]">Fatigue: </span>
            <span
              className={`font-bold ${
                (data.fatigue_index ?? 0) > 0.6
                  ? "text-[#FF4B4B]"
                  : (data.fatigue_index ?? 0) > 0.3
                    ? "text-[#FFB020]"
                    : "text-[#58CC02]"
              }`}
            >
              {((data.fatigue_index ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Quality / biomechanics */}
      {data.quality && (
        <div className="mt-2 pt-2 border-t border-[#E5E5E5]">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-[#4B4B4B]">
            {data.quality.form_score != null && (
              <div>
                <span className="text-[#AFAFAF]">Form: </span>
                <span
                  className={`font-bold ${
                    data.quality.form_score >= 80
                      ? "text-[#58CC02]"
                      : data.quality.form_score >= 60
                        ? "text-[#FFB020]"
                        : "text-[#FF4B4B]"
                  }`}
                >
                  {Math.round(data.quality.form_score)}/100
                </span>
              </div>
            )}
            {data.quality.biomechanics && (
              <>
                <div>
                  <span className="text-[#AFAFAF]">FPPA: </span>
                  <span className="font-bold">
                    L{data.quality.biomechanics.fppa_left?.toFixed(1)}° R
                    {data.quality.biomechanics.fppa_right?.toFixed(1)}°
                  </span>
                </div>
                <div>
                  <span className="text-[#AFAFAF]">Hip Drop: </span>
                  <span className="font-bold">
                    {data.quality.biomechanics.hip_drop?.toFixed(1)}°
                  </span>
                  <span className="text-[#AFAFAF]"> Trunk: </span>
                  <span className="font-bold">
                    {data.quality.biomechanics.trunk_lean?.toFixed(1)}°
                  </span>
                </div>
                <div>
                  <span className="text-[#AFAFAF]">Asymmetry: </span>
                  <span className="font-bold">
                    {data.quality.biomechanics.asymmetry?.toFixed(1)}%
                  </span>
                </div>
              </>
            )}
            {data.quality.injury_risks?.length > 0 && (
              <div className="col-span-2">
                <span className="text-[#FF4B4B] font-bold">Risks: </span>
                {data.quality.injury_risks.map(
                  (
                    r: { risk: string; joint: string; severity: string },
                    i: number,
                  ) => (
                    <span
                      key={i}
                      className={`inline-block mr-1 px-1 py-0.5 rounded text-[10px] font-bold ${
                        r.severity === "high"
                          ? "bg-[#FF4B4B]/20 text-[#FF4B4B]"
                          : "bg-[#FFB020]/20 text-[#FFB020]"
                      }`}
                    >
                      {r.risk} ({r.joint})
                    </span>
                  ),
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Expanded detail panel ──────────────────────────────────────────── */

function StreamDetail({
  stream,
  onClose,
}: {
  stream: StreamInfo;
  onClose: () => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dataIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [streamData, setStreamData] = useState<StreamData | null>(null);
  const [showRawJson, setShowRawJson] = useState(false);

  const isVR = stream.client_type === "vr";

  // Poll the high-res frame with bounding-box overlay
  useEffect(() => {
    const base = getApiBase();
    const load = () => {
      if (imgRef.current) {
        imgRef.current.src = `${base}/api/streams/${stream.stream_id}/frame?w=960&overlay=true&_t=${Date.now()}`;
      }
    };
    load();
    frameIntervalRef.current = setInterval(load, 500);
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, [stream.stream_id]);

  // Poll the data endpoint
  useEffect(() => {
    const base = getApiBase();
    const load = async () => {
      try {
        const res = await fetch(
          `${base}/api/streams/${stream.stream_id}/data`,
        );
        if (!res.ok) return;
        const data: StreamData = await res.json();
        setStreamData(data);
      } catch {
        // ignore transient errors
      }
    };
    load();
    dataIntervalRef.current = setInterval(load, 500);
    return () => {
      if (dataIntervalRef.current) clearInterval(dataIntervalRef.current);
    };
  }, [stream.stream_id]);

  const subjects = streamData?.subjects ?? {};
  const subjectIds = Object.keys(subjects);
  const vrSel = streamData?.vr_selected_subject;

  return (
    <Card className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#FF4B4B] animate-pulse" />
            <span className="text-sm font-black text-[#3C3C3C]">
              {stream.mode === "video" ? "Video" : "Webcam"} Stream
            </span>
          </div>
          <span
            className={`text-[10px] font-bold rounded px-1.5 py-0.5 ${
              isVR
                ? "text-white bg-[#7B2FF7]"
                : "text-[#AFAFAF] bg-[#F7F7F7]"
            }`}
          >
            {stream.client_type.toUpperCase()}
          </span>
          {stream.subject_count > 0 && (
            <span className="text-[10px] font-bold text-white bg-[#1CB0F6] rounded-full px-2 py-0.5">
              {stream.subject_count}{" "}
              {stream.subject_count === 1 ? "person" : "people"}
            </span>
          )}
          {isVR && (
            <span
              className={`text-[10px] font-bold rounded-full px-2 py-0.5 ${
                vrSel != null
                  ? "text-white bg-[#58CC02]"
                  : "text-[#AFAFAF] bg-[#F7F7F7]"
              }`}
            >
              VR Select: {vrSel != null ? `S${vrSel}` : "None"}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowRawJson((v) => !v)}
            className="text-xs font-bold text-[#1CB0F6] hover:text-[#1899D6] transition-colors px-2 py-1"
          >
            {showRawJson ? "Cards" : "Raw JSON"}
          </button>
          <button
            onClick={onClose}
            className="text-xs font-bold text-[#AFAFAF] hover:text-[#3C3C3C] transition-colors px-2 py-1"
          >
            Close
          </button>
        </div>
      </div>

      {/* Two-column: video + data */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_480px] gap-4">
        {/* Left: frame with bbox overlay */}
        <div className="flex flex-col gap-2">
          <div className="relative w-full bg-[#0A0A0A] rounded-xl overflow-hidden">
            <img
              ref={imgRef}
              alt="Live stream with detection overlay"
              className="w-full h-auto block"
              style={{ minHeight: 200 }}
            />
          </div>
          {/* Stats bar */}
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-[#4B4B4B]">
            <div>
              <span className="text-[#AFAFAF]">Uptime </span>
              <span className="font-bold">
                {formatUptime(stream.uptime_sec)}
              </span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">Frames </span>
              <span className="font-bold">
                {stream.frame_count.toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">Target FPS </span>
              <span className="font-bold">{stream.fps}</span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">Source </span>
              <span className="font-bold">
                {stream.resolution
                  ? `${stream.resolution[0]}x${stream.resolution[1]}`
                  : "--"}
              </span>
            </div>
            {streamData?.timing && (
              <div>
                <span className="text-[#AFAFAF]">Pipeline </span>
                <span className="font-bold">
                  {streamData.timing.total_ms?.toFixed(0)}ms
                </span>
              </div>
            )}
            <div>
              <span className="text-[#AFAFAF]">ID </span>
              <span className="font-bold font-mono">
                {stream.stream_id.slice(0, 12)}...
              </span>
            </div>
          </div>
        </div>

        {/* Right: subject cards or raw JSON */}
        <div className="flex flex-col gap-2 overflow-auto" style={{ maxHeight: 600 }}>
          {showRawJson ? (
            <pre
              className="bg-[#1A1A2E] text-[#A8D8A8] text-[10px] leading-[1.5] font-mono p-3 rounded-xl overflow-auto"
              style={{ maxHeight: 560, minHeight: 300 }}
            >
              {streamData
                ? JSON.stringify(streamData, null, 2)
                : "Loading..."}
            </pre>
          ) : subjectIds.length === 0 ? (
            <div className="flex items-center justify-center py-12 text-sm text-[#AFAFAF]">
              No subjects detected
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {subjectIds.map((sid) => (
                <SubjectCard
                  key={sid}
                  sid={sid}
                  data={subjects[sid]}
                  isSelected={vrSel != null && String(vrSel) === sid}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ── Page ────────────────────────────────────────────────────────────── */

export default function DevStreamsPage() {
  const [streams, setStreams] = useState<StreamInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStreams = useCallback(async () => {
    try {
      const res = await fetch(`${getApiBase()}/api/streams`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: StreamsResponse = await res.json();
      setStreams(data.streams);
      setError(null);
      if (data.streams.length > 0 && selectedId) {
        if (!data.streams.some((s) => s.stream_id === selectedId)) {
          setSelectedId(null);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }, [selectedId]);

  useEffect(() => {
    fetchStreams();
    intervalRef.current = setInterval(fetchStreams, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchStreams]);

  const selectedStream =
    streams.find((s) => s.stream_id === selectedId) ?? null;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-black text-[#3C3C3C]">
            Active Streams
          </h1>
          <p className="text-sm text-[#AFAFAF] mt-1">
            Live view of all WebSocket analysis connections
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#58CC02] animate-pulse" />
            <span className="text-sm font-bold text-[#58CC02]">
              {streams.length} active
            </span>
          </div>
          <button
            onClick={fetchStreams}
            className="text-xs font-bold text-[#1CB0F6] hover:text-[#1899D6] transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {selectedStream && (
        <div className="mb-6">
          <StreamDetail
            stream={selectedStream}
            onClose={() => setSelectedId(null)}
          />
        </div>
      )}

      {loading && streams.length === 0 ? (
        <Card className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-8 h-8 border-3 border-[#E5E5E5] border-t-[#1CB0F6] rounded-full animate-spin mx-auto mb-3" />
            <p className="text-sm text-[#AFAFAF]">Loading streams...</p>
          </div>
        </Card>
      ) : error ? (
        <Card className="flex items-center justify-center py-16">
          <div className="text-center">
            <p className="text-sm font-bold text-[#FF4B4B] mb-1">
              Connection Error
            </p>
            <p className="text-xs text-[#AFAFAF]">{error}</p>
            <p className="text-xs text-[#AFAFAF] mt-2">
              Make sure the backend is running on port 8001
            </p>
          </div>
        </Card>
      ) : streams.length === 0 ? (
        <Card className="flex items-center justify-center py-16">
          <div className="text-center">
            <p className="text-sm font-bold text-[#3C3C3C] mb-1">
              No Active Streams
            </p>
            <p className="text-xs text-[#AFAFAF]">
              Open the{" "}
              <a
                href="/analyze"
                className="text-[#1CB0F6] hover:underline"
              >
                Analyze
              </a>{" "}
              page to start a stream
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {streams.map((stream) => (
            <StreamCard
              key={stream.stream_id}
              stream={stream}
              selected={stream.stream_id === selectedId}
              onClick={() =>
                setSelectedId(
                  stream.stream_id === selectedId ? null : stream.stream_id,
                )
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
