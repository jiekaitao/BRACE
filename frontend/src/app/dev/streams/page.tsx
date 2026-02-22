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
}

interface StreamsResponse {
  count: number;
  streams: StreamInfo[];
}

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
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-[#3C3C3C] uppercase tracking-wide">
            {stream.mode === "video" ? "Video" : "Webcam"}
          </span>
          <span className="text-[10px] font-bold text-[#AFAFAF] bg-[#F7F7F7] rounded px-1.5 py-0.5">
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
        <div className="text-[10px] text-[#AFAFAF] font-mono truncate">
          {stream.stream_id.slice(0, 8)}...
        </div>
      </div>
    </Card>
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
  const jsonRef = useRef<HTMLPreElement>(null);

  const [jsonData, setJsonData] = useState<string>("Loading...");
  const [autoScroll, setAutoScroll] = useState(true);

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
        const res = await fetch(`${base}/api/streams/${stream.stream_id}/data`);
        if (!res.ok) return;
        const data = await res.json();
        setJsonData(JSON.stringify(data, null, 2));
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

  // Auto-scroll JSON panel to bottom
  useEffect(() => {
    if (autoScroll && jsonRef.current) {
      jsonRef.current.scrollTop = jsonRef.current.scrollHeight;
    }
  }, [jsonData, autoScroll]);

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
          <span className="text-[10px] font-bold text-[#AFAFAF] bg-[#F7F7F7] rounded px-1.5 py-0.5">
            {stream.client_type.toUpperCase()}
          </span>
          {stream.subject_count > 0 && (
            <span className="text-[10px] font-bold text-white bg-[#1CB0F6] rounded-full px-2 py-0.5">
              {stream.subject_count} {stream.subject_count === 1 ? "person" : "people"}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-xs font-bold text-[#AFAFAF] hover:text-[#3C3C3C] transition-colors px-2 py-1"
        >
          Close
        </button>
      </div>

      {/* Two-column: video + data */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-4">
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
              <span className="font-bold">{formatUptime(stream.uptime_sec)}</span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">Frames </span>
              <span className="font-bold">{stream.frame_count.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">Target FPS </span>
              <span className="font-bold">{stream.fps}</span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">Source </span>
              <span className="font-bold">
                {stream.resolution ? `${stream.resolution[0]}x${stream.resolution[1]}` : "--"}
              </span>
            </div>
            <div>
              <span className="text-[#AFAFAF]">ID </span>
              <span className="font-bold font-mono">{stream.stream_id.slice(0, 12)}...</span>
            </div>
          </div>
        </div>

        {/* Right: live JSON data */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#3C3C3C]">Response Data</span>
            <label className="flex items-center gap-1.5 text-[10px] text-[#AFAFAF] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="w-3 h-3"
              />
              Auto-scroll
            </label>
          </div>
          <pre
            ref={jsonRef}
            className="flex-1 bg-[#1A1A2E] text-[#A8D8A8] text-[10px] leading-[1.5] font-mono p-3 rounded-xl overflow-auto"
            style={{ maxHeight: 480, minHeight: 300 }}
          >
            {jsonData}
          </pre>
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

  const selectedStream = streams.find((s) => s.stream_id === selectedId) ?? null;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-black text-[#3C3C3C]">Active Streams</h1>
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
            <p className="text-sm font-bold text-[#FF4B4B] mb-1">Connection Error</p>
            <p className="text-xs text-[#AFAFAF]">{error}</p>
            <p className="text-xs text-[#AFAFAF] mt-2">
              Make sure the backend is running on port 8001
            </p>
          </div>
        </Card>
      ) : streams.length === 0 ? (
        <Card className="flex items-center justify-center py-16">
          <div className="text-center">
            <p className="text-sm font-bold text-[#3C3C3C] mb-1">No Active Streams</p>
            <p className="text-xs text-[#AFAFAF]">
              Open the <a href="/analyze" className="text-[#1CB0F6] hover:underline">Analyze</a> page to start a stream
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
                setSelectedId(stream.stream_id === selectedId ? null : stream.stream_id)
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
