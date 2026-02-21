"use client";

import { useEffect, useRef, useState } from "react";
import type { DebugStats, GeminiStats, GpuStats } from "@/lib/types";
import { getApiBase } from "@/lib/api";

interface DebugPanelProps {
  debugStatsRef: React.MutableRefObject<DebugStats>;
}

interface ChartConfig {
  label: string;
  unit: string;
  color: string;
  key: keyof DebugStats["history"];
  valueKey: keyof DebugStats;
}

const CHARTS: ChartConfig[] = [
  { label: "Capture FPS", unit: "fps", color: "#1CB0F6", key: "fps_out", valueKey: "fps_out" },
  { label: "Pipeline FPS", unit: "fps", color: "#58CC02", key: "fps_in", valueKey: "fps_in" },
  { label: "KB/s Out", unit: "KB/s", color: "#FF9600", key: "kbps_out", valueKey: "kbps_out" },
  { label: "KB/s In", unit: "KB/s", color: "#CE82FF", key: "kbps_in", valueKey: "kbps_in" },
  { label: "RTT", unit: "ms", color: "#EA2B2B", key: "rtt_ms", valueKey: "rtt_ms" },
  { label: "Subjects", unit: "", color: "#1899D6", key: "subjects", valueKey: "activeSubjects" },
];

function MiniChart({ config, debugStatsRef }: { config: ChartConfig; debugStatsRef: React.MutableRefObject<DebugStats> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function draw() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const stats = debugStatsRef.current;
      const data = stats.history[config.key];
      const currentVal = stats[config.valueKey] as number;

      // Background
      ctx.fillStyle = "#1A1A1A";
      ctx.fillRect(0, 0, w, h);

      // Grid lines
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const gy = (h * i) / 4;
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
      }

      if (data.length < 2) {
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      // Compute Y scale
      const maxVal = Math.max(...data, 1);
      const yScale = (h - 8) / maxVal;
      const xStep = w / 59; // 60 data points

      // Fill area
      ctx.beginPath();
      ctx.moveTo(w - (data.length - 1) * xStep, h);
      for (let i = 0; i < data.length; i++) {
        const x = w - (data.length - 1 - i) * xStep;
        const y = h - data[i] * yScale - 4;
        if (i === 0) ctx.lineTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = config.color + "15";
      ctx.fill();

      // Line
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = w - (data.length - 1 - i) * xStep;
        const y = h - data[i] * yScale - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = config.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Current value dot
      if (data.length > 0) {
        const lastY = h - data[data.length - 1] * yScale - 4;
        ctx.beginPath();
        ctx.arc(w, lastY, 3, 0, Math.PI * 2);
        ctx.fillStyle = config.color;
        ctx.fill();
      }

      // Label + value overlay
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fillText(config.label, 6, 14);

      const valText = currentVal < 10
        ? currentVal.toFixed(1)
        : Math.round(currentVal).toString();
      const fullText = config.unit ? `${valText} ${config.unit}` : valText;
      ctx.font = "bold 14px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillStyle = config.color;
      const tm = ctx.measureText(fullText);
      ctx.fillText(fullText, w - tm.width - 6, 14);

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [config, debugStatsRef]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-[8px]"
      style={{ height: 80 }}
    />
  );
}

function GpuStatsRow() {
  const [gpu, setGpu] = useState<GpuStats | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const apiBase = getApiBase();
        const res = await fetch(`${apiBase}/api/gpu-stats`);
        const data: GpuStats = await res.json();
        if (!cancelled) setGpu(data);
      } catch {
        // ignore fetch errors
      }
    }
    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (!gpu || !gpu.available) return null;

  return (
    <div className="col-span-2 sm:col-span-3 bg-[#1A1A1A] rounded-[8px] px-4 py-3 flex items-center gap-6 text-xs">
      <span className="text-[#58CC02] font-bold">{gpu.name}</span>
      <span className="text-[#1CB0F6]">
        GPU <span className="font-bold">{gpu.gpu_util}%</span>
      </span>
      <span className="text-[#FF9600]">
        VRAM <span className="font-bold">{gpu.vram_used_gb}/{gpu.vram_total_gb} GB</span>
      </span>
      <span className="text-[#EA2B2B]">
        <span className="font-bold">{gpu.temp_c}&deg;C</span>
      </span>
      <span className="text-[#CE82FF]">
        <span className="font-bold">{gpu.power_w}W</span>
      </span>
    </div>
  );
}

function GeminiStatsRow({ debugStatsRef }: { debugStatsRef: React.MutableRefObject<DebugStats> }) {
  const [stats, setStats] = useState<GeminiStats | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      const gs = debugStatsRef.current.geminiStats;
      if (gs) setStats({ ...gs });
    }, 1000);
    return () => clearInterval(interval);
  }, [debugStatsRef]);

  if (!stats) return null;

  return (
    <div className="col-span-2 sm:col-span-3 bg-[#1A1A1A] rounded-[8px] px-4 py-3 flex items-center gap-6 text-xs">
      <span className="text-[#58CC02] font-bold">Gemini 2.0 Flash</span>
      <span className="text-[#1CB0F6]">
        API Calls <span className="font-bold">{stats.api_calls}</span>
      </span>
      <span className="text-[#FF9600]">
        Cache Hits <span className="font-bold">{stats.cache_hits}</span>
      </span>
      <span className="text-[#CE82FF]">
        Cost <span className="font-bold">${stats.estimated_cost_usd.toFixed(4)}</span>
      </span>
    </div>
  );
}

export default function DebugPanel({ debugStatsRef }: DebugPanelProps) {
  return (
    <div className="bg-[#111111] rounded-[16px] border-2 border-[#333333] p-4">
      <h3 className="text-sm font-bold text-[#AFAFAF] mb-3 tracking-wider uppercase">
        Debug Stats
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <GpuStatsRow />
        <GeminiStatsRow debugStatsRef={debugStatsRef} />
        {CHARTS.map((chart) => (
          <MiniChart
            key={chart.key}
            config={chart}
            debugStatsRef={debugStatsRef}
          />
        ))}
      </div>
    </div>
  );
}
