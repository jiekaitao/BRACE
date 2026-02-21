"use client";

import { useEffect, useRef } from "react";
import type { SubjectState } from "@/lib/types";
import { CLUSTER_COLORS } from "@/lib/colors";
import Card from "./ui/Card";

interface VelocityGraphProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  compact?: boolean;
}

export default function VelocityGraph({
  subjectsRef,
  selectedSubjectRef,
  highlightedClusterRef,
  compact = false,
}: VelocityGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const compactRef = useRef(compact);
  compactRef.current = compact;

  const PL = compact ? 28 : 40;
  const PR = compact ? 8 : 12;
  const PT = compact ? 8 : 16;
  const PB = compact ? 14 : 24;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function draw() {
      const c = canvasRef.current;
      if (!c) return;
      const cx = c.getContext("2d");
      if (!cx) return;
      const isCompact = compactRef.current;

      // Handle DPI scaling
      const rect = c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = rect.width;
      const h = rect.height;
      if (c.width !== w * dpr || c.height !== h * dpr) {
        c.width = w * dpr;
        c.height = h * dpr;
      }
      cx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Clear
      cx.clearRect(0, 0, w, h);

      // Get selected subject
      const trackId = selectedSubjectRef.current;
      if (trackId === null) {
        animRef.current = requestAnimationFrame(draw);
        return;
      }
      const subject = subjectsRef.current.get(trackId);
      if (!subject || subject.velocity.values.length < 2) {
        // Draw placeholder text
        cx.fillStyle = "#AFAFAF";
        cx.font = `${isCompact ? 10 : 12}px system-ui, sans-serif`;
        cx.textAlign = "center";
        cx.fillText("Collecting velocity data...", w / 2, h / 2);
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      const vel = subject.velocity;
      const plotW = w - PL - PR;
      const plotH = h - PT - PB;
      const n = vel.values.length;

      // Y-axis: auto-scale to max of rolling values (smoother scaling)
      let yMax = 0;
      for (let i = 0; i < n; i++) {
        if (vel.rolling[i] > yMax) yMax = vel.rolling[i];
        if (vel.values[i] > yMax) yMax = vel.values[i];
      }
      yMax = Math.max(yMax * 1.15, 0.1); // 15% headroom, min 0.1

      const xScale = plotW / Math.max(n - 1, 1);
      const yScale = plotH / yMax;

      // Helper: data index to canvas coords
      const toX = (i: number) => PL + i * xScale;
      const toY = (v: number) => PT + plotH - v * yScale;

      // Draw cluster background bands
      if (subject.clusterSummary && subject.nClusters > 0) {
        const highlightedCluster = highlightedClusterRef.current;
        if (highlightedCluster !== null) {
          const color = CLUSTER_COLORS[highlightedCluster % CLUSTER_COLORS.length];
          cx.fillStyle = color + "15"; // very light
          cx.fillRect(PL, PT, plotW, plotH);
        }
      }

      // Draw grid lines
      cx.strokeStyle = "#E5E5E5";
      cx.lineWidth = 0.5;
      const gridLines = isCompact ? 2 : 4;
      for (let i = 0; i <= gridLines; i++) {
        const yVal = (yMax / gridLines) * i;
        const y = toY(yVal);
        cx.beginPath();
        cx.moveTo(PL, y);
        cx.lineTo(w - PR, y);
        cx.stroke();

        // Y-axis labels
        cx.fillStyle = "#AFAFAF";
        cx.font = `${isCompact ? 8 : 9}px system-ui, sans-serif`;
        cx.textAlign = "right";
        cx.fillText(yVal.toFixed(1), PL - 4, y + 3);
      }

      // Draw raw velocity (thin semi-transparent line)
      cx.beginPath();
      cx.strokeStyle = "rgba(28, 176, 246, 0.25)";
      cx.lineWidth = 1;
      for (let i = 0; i < n; i++) {
        const x = toX(i);
        const y = toY(vel.values[i]);
        if (i === 0) cx.moveTo(x, y);
        else cx.lineTo(x, y);
      }
      cx.stroke();

      // Draw rolling average (solid line)
      cx.beginPath();
      cx.strokeStyle = "#1CB0F6";
      cx.lineWidth = isCompact ? 1.5 : 2;
      for (let i = 0; i < n; i++) {
        const x = toX(i);
        const y = toY(vel.rolling[i]);
        if (i === 0) cx.moveTo(x, y);
        else cx.lineTo(x, y);
      }
      cx.stroke();

      // Draw peak velocity line (dashed)
      if (vel.peakVelocity > 0) {
        const peakY = toY(vel.peakVelocity);
        if (peakY > PT && peakY < PT + plotH) {
          cx.setLineDash([4, 4]);
          cx.strokeStyle = "#FF9600";
          cx.lineWidth = 1;
          cx.beginPath();
          cx.moveTo(PL, peakY);
          cx.lineTo(w - PR, peakY);
          cx.stroke();
          cx.setLineDash([]);

          // Peak label (skip in compact mode)
          if (!isCompact) {
            cx.fillStyle = "#FF9600";
            cx.font = "9px system-ui, sans-serif";
            cx.textAlign = "left";
            cx.fillText("peak", w - PR - 28, peakY - 4);
          }
        }
      }

      // Draw fatigue trend line if fatigue > 0.1
      if (vel.fatigueIndex > 0.1 && n >= 60) {
        const startRolling = vel.rolling[0];
        const endRolling = vel.rolling[n - 1];
        cx.setLineDash([6, 3]);
        cx.strokeStyle = vel.fatigueIndex > 0.6
          ? "#EA2B2B"
          : vel.fatigueIndex > 0.3
            ? "#F5A623"
            : "#58CC02";
        cx.lineWidth = 1.5;
        cx.beginPath();
        cx.moveTo(toX(0), toY(startRolling));
        cx.lineTo(toX(n - 1), toY(endRolling));
        cx.stroke();
        cx.setLineDash([]);
      }

      // Current velocity dot at the right edge
      if (n > 0) {
        const lastRolling = vel.rolling[n - 1];
        const cx2 = toX(n - 1);
        const cy2 = toY(lastRolling);
        cx.beginPath();
        cx.arc(cx2, cy2, isCompact ? 3 : 4, 0, Math.PI * 2);
        cx.fillStyle = "#1CB0F6";
        cx.fill();
        cx.strokeStyle = "#FFFFFF";
        cx.lineWidth = 1.5;
        cx.stroke();
      }

      // X-axis label (skip in compact mode)
      if (!isCompact) {
        cx.fillStyle = "#AFAFAF";
        cx.font = "9px system-ui, sans-serif";
        cx.textAlign = "center";
        cx.fillText(`${n} frames`, w / 2, h - 4);
      }

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [subjectsRef, selectedSubjectRef, highlightedClusterRef, PL, PR, PT, PB]);

  return (
    <Card>
      <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-2">
        Velocity
      </h3>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: compact ? 80 : 160 }}
      />
    </Card>
  );
}
