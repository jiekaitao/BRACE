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

function StreamCard({ stream }: { stream: StreamInfo }) {
  return (
    <Card className="flex flex-col gap-3">
      {/* Thumbnail */}
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
        {/* Live badge */}
        <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-black/70 rounded-full px-2 py-0.5">
          <span className="w-2 h-2 rounded-full bg-[#FF4B4B] animate-pulse" />
          <span className="text-white text-[10px] font-bold uppercase">Live</span>
        </div>
        {/* Subject count badge */}
        {stream.subject_count > 0 && (
          <div className="absolute top-2 right-2 bg-[#1CB0F6] text-white text-[10px] font-bold rounded-full px-2 py-0.5">
            {stream.subject_count} {stream.subject_count === 1 ? "person" : "people"}
          </div>
        )}
      </div>

      {/* Info */}
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
          <div>
            <span className="text-[#AFAFAF]">FPS: </span>
            <span className="font-bold">{stream.fps}</span>
          </div>
          <div>
            <span className="text-[#AFAFAF]">Res: </span>
            <span className="font-bold">
              {stream.resolution ? `${stream.resolution[0]}x${stream.resolution[1]}` : "--"}
            </span>
          </div>
        </div>

        <div className="text-[10px] text-[#AFAFAF] font-mono truncate">
          {stream.stream_id.slice(0, 8)}...
        </div>
      </div>
    </Card>
  );
}

export default function DevStreamsPage() {
  const [streams, setStreams] = useState<StreamInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStreams = useCallback(async () => {
    try {
      const res = await fetch(`${getApiBase()}/api/streams`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: StreamsResponse = await res.json();
      setStreams(data.streams);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStreams();
    intervalRef.current = setInterval(fetchStreams, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchStreams]);

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
            <p className="text-4xl mb-3">📡</p>
            <p className="text-sm font-bold text-[#3C3C3C] mb-1">No Active Streams</p>
            <p className="text-xs text-[#AFAFAF]">
              Open the <a href="/analyze" className="text-[#1CB0F6] hover:underline">Analyze</a> page to start a stream
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {streams.map((stream) => (
            <StreamCard key={stream.stream_id} stream={stream} />
          ))}
        </div>
      )}
    </div>
  );
}
