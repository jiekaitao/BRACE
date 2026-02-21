"use client";

import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import type { ClusterInfo, SubjectState } from "@/lib/types";
import { CLUSTER_COLORS } from "@/lib/colors";
import { FEATURE_BONES } from "@/lib/skeleton";

interface ClusterSummaryProps {
  clusters: Record<string, ClusterInfo>;
  activeCluster: number | null;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
}

export default function ClusterSummary({
  clusters,
  activeCluster,
  highlightedClusterRef,
  subjectsRef,
  selectedSubjectRef,
}: ClusterSummaryProps) {
  const [popoverCluster, setPopoverCluster] = useState<number | null>(null);
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  });

  const popoverCanvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef(0);
  const frameCounterRef = useRef(0);
  const lastFrameTimeRef = useRef(0);

  const entries = Object.entries(clusters).sort(
    ([a], [b]) => Number(a) - Number(b)
  );

  // Animated skeleton canvas for popover
  useEffect(() => {
    if (popoverCluster === null) {
      cancelAnimationFrame(animFrameRef.current);
      frameCounterRef.current = 0;
      lastFrameTimeRef.current = 0;
      return;
    }

    const canvas = popoverCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = (time: number) => {
      // Advance frame at ~15fps
      if (time - lastFrameTimeRef.current >= 67) {
        lastFrameTimeRef.current = time;
        frameCounterRef.current++;
      }

      // Get trajectory from subjectsRef
      const selectedId = selectedSubjectRef.current;
      if (selectedId === null) {
        ctx.clearRect(0, 0, 200, 200);
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }
      const subject = subjectsRef.current.get(selectedId);
      const trajectory =
        subject?.clusterRepresentatives?.[String(popoverCluster)];

      if (!trajectory || trajectory.length === 0) {
        ctx.clearRect(0, 0, 200, 200);
        ctx.fillStyle = "#999";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No trajectory data", 100, 100);
        animFrameRef.current = requestAnimationFrame(draw);
        return;
      }

      const frameIdx = frameCounterRef.current % trajectory.length;
      const joints = trajectory[frameIdx];

      ctx.clearRect(0, 0, 200, 200);

      // Find bounds for auto-fitting
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const j of joints) {
        if (j[0] < minX) minX = j[0];
        if (j[0] > maxX) maxX = j[0];
        if (j[1] < minY) minY = j[1];
        if (j[1] > maxY) maxY = j[1];
      }

      // Map to canvas with padding
      const pad = 20;
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const scale = Math.min(
        (200 - 2 * pad) / rangeX,
        (200 - 2 * pad) / rangeY
      );
      const toX = (v: number) =>
        pad + (v - minX) * scale + ((200 - 2 * pad) - rangeX * scale) / 2;
      const toY = (v: number) =>
        pad + (v - minY) * scale + ((200 - 2 * pad) - rangeY * scale) / 2;

      // Draw bones
      const clusterColor =
        CLUSTER_COLORS[popoverCluster % CLUSTER_COLORS.length];
      ctx.strokeStyle = clusterColor;
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      for (const [a, b] of FEATURE_BONES) {
        if (a < joints.length && b < joints.length) {
          ctx.beginPath();
          ctx.moveTo(toX(joints[a][0]), toY(joints[a][1]));
          ctx.lineTo(toX(joints[b][0]), toY(joints[b][1]));
          ctx.stroke();
        }
      }

      // Draw joints
      ctx.fillStyle = clusterColor;
      for (const j of joints) {
        ctx.beginPath();
        ctx.arc(toX(j[0]), toY(j[1]), 4, 0, Math.PI * 2);
        ctx.fill();
      }

      animFrameRef.current = requestAnimationFrame(draw);
    };

    animFrameRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [popoverCluster, subjectsRef, selectedSubjectRef]);

  if (entries.length === 0) {
    return (
      <div className="text-sm text-[#AFAFAF] text-center py-3">
        No clusters detected yet
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {entries.map(([id, info]) => {
        const colorIdx = Number(id) % CLUSTER_COLORS.length;
        const color = CLUSTER_COLORS[colorIdx];
        const isActive = Number(id) === activeCluster;

        return (
          <div
            key={id}
            className={`
              flex items-center justify-between
              px-3 py-2 rounded-[12px] border-2
              transition-all duration-100
              ${isActive ? "border-current" : "border-[#E5E5E5]"}
            `}
            style={{
              borderColor: isActive ? color : undefined,
              backgroundColor: isActive ? `${color}15` : "white",
              cursor: "pointer",
            }}
            onMouseEnter={() => {
              highlightedClusterRef.current = Number(id);
              setPopoverCluster(Number(id));
            }}
            onMouseLeave={() => {
              highlightedClusterRef.current = null;
              setPopoverCluster(null);
            }}
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              setPopoverPos({ x: rect.left - 210, y: rect.top - 50 });
            }}
          >
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-sm font-bold text-[#4B4B4B]">
                {info.activity_label || `Motion #${Number(id) + 1}`}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-[#777777]">
                {info.count} rep{info.count !== 1 ? "s" : ""}
              </span>
              {info.anomaly_count > 0 && (
                <span className="text-xs font-bold text-white bg-[#EA2B2B] px-2 py-0.5 rounded-full">
                  {info.anomaly_count} flagged
                </span>
              )}
            </div>
          </div>
        );
      })}

      {/* Skeleton Popover — rendered via Portal to escape ancestor transforms */}
      {popoverCluster !== null &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            style={{
              position: "fixed",
              left: popoverPos.x,
              top: popoverPos.y,
              width: 200,
              height: 200,
              backgroundColor: "white",
              borderRadius: 12,
              boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
              zIndex: 50,
              pointerEvents: "none",
              transition: "opacity 150ms ease-in",
              opacity: 1,
              overflow: "hidden",
            }}
          >
            <canvas
              ref={popoverCanvasRef}
              width={200}
              height={200}
              style={{ width: 200, height: 200 }}
            />
          </div>,
          document.body
        )}
    </div>
  );
}
