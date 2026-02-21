"use client";

import { useEffect, useRef, useMemo } from "react";
import type { InjuryProfile, InjuryEntry, RiskModifiers } from "@/lib/types";
import { FEATURE_BONES } from "@/lib/skeleton";
import {
  generateSquatCycle,
  computeJointAngle,
  INJURY_JOINT_CHAINS,
  DEFAULT_THRESHOLDS,
  computeModifiedThresholds,
} from "@/lib/syntheticMotion";
import Card from "./ui/Card";

const PADDING = 32;
const SAFE_COLOR = "#58CC02";
const RISK_COLOR = "#EA2B2B";
const BONE_COLOR = "#3C3C3C";
const JOINT_COLOR = "#3C3C3C";
const ARC_RADIUS = 25;
const LABEL_OFFSET = 14;

interface Props {
  injuryProfile: InjuryProfile;
  riskModifiers: RiskModifiers | null;
}

/** Resolve which chains to draw based on injury side. */
function getChainsForInjury(
  injury: InjuryEntry,
): { joints: [number, number, number]; side: "left" | "right" }[] {
  const mapping = INJURY_JOINT_CHAINS[injury.type];
  if (!mapping) return [];
  const side = injury.side || "bilateral";
  if (side === "left") return mapping.chains.filter((c) => c.side === "left");
  if (side === "right") return mapping.chains.filter((c) => c.side === "right");
  return mapping.chains; // bilateral or unknown
}

export default function AnimatedSkeletonDemo({
  injuryProfile,
  riskModifiers,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const frameRef = useRef(0);
  const reducedMotion = useRef(false);

  const frames = useMemo(() => generateSquatCycle(120, 1.0), []);
  const thresholds = useMemo(
    () => computeModifiedThresholds(riskModifiers),
    [riskModifiers],
  );

  // Collect all chains to draw, deduplicating by vertex index
  const chainsToRender = useMemo(() => {
    const seen = new Map<string, { joints: [number, number, number]; metric: string; threshold: number }>();
    for (const injury of injuryProfile.injuries) {
      const mapping = INJURY_JOINT_CHAINS[injury.type];
      if (!mapping) continue;
      const chains = getChainsForInjury(injury);
      for (const chain of chains) {
        const key = chain.joints.join(",");
        const t = thresholds[mapping.metric];
        const thresholdVal = t?.medium ?? 180;
        const existing = seen.get(key);
        if (!existing || thresholdVal < existing.threshold) {
          seen.set(key, { joints: chain.joints, metric: mapping.metric, threshold: thresholdVal });
        }
      }
    }
    return Array.from(seen.values());
  }, [injuryProfile, thresholds]);

  // Explanatory messages
  const messages = useMemo(() => {
    const msgs: string[] = [];
    for (const injury of injuryProfile.injuries) {
      const mapping = INJURY_JOINT_CHAINS[injury.type];
      if (!mapping) continue;
      const label =
        injury.type === "acl"
          ? "ACL"
          : injury.type === "lower_back"
            ? "Lower Back"
            : injury.type === "knee_general"
              ? "Knee"
              : injury.type.charAt(0).toUpperCase() + injury.type.slice(1);
      const sideStr = injury.side !== "unknown" ? ` (${injury.side}` : " (";
      const sevStr = injury.severity ? `, ${injury.severity})` : ")";
      const t = thresholds[mapping.metric];
      const tVal = t?.medium ?? 0;
      const defaultT = DEFAULT_THRESHOLDS[mapping.metric]?.medium ?? 0;
      const pct = defaultT > 0 ? Math.round(((defaultT - tVal) / defaultT) * 100) : 0;
      const sensitivityNote =
        pct > 0
          ? ` (your threshold is ${pct}% more sensitive)`
          : "";
      msgs.push(
        `${label}${sideStr}${sevStr}: Monitoring ${mapping.metric.replace("_", " ")} — threshold ${tVal.toFixed(1)}${sensitivityNote}`,
      );
    }
    return msgs;
  }, [injuryProfile, thresholds]);

  useEffect(() => {
    reducedMotion.current = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // If reduced motion, show static mid-cycle frame
    if (reducedMotion.current) {
      frameRef.current = 30; // quarter cycle (mid-squat)
    }

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
      ctx.clearRect(0, 0, w, h);

      const joints = frames[frameRef.current];

      // Auto-fit joints to canvas
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const [x, y] of joints) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const drawW = w - 2 * PADDING;
      const drawH = h - 2 * PADDING;
      const scale = Math.min(drawW / rangeX, drawH / rangeY) * 0.85;
      const midX = (minX + maxX) / 2;
      const midY = (minY + maxY) / 2;
      const cx = w / 2;
      const cy = h / 2;

      const toX = (px: number) => cx + (px - midX) * scale;
      // Flip Y so positive Y is up (SRP space has Y up)
      const toY = (py: number) => cy - (py - midY) * scale;

      // Draw bones
      ctx.lineCap = "round";
      ctx.lineWidth = 3;
      ctx.strokeStyle = BONE_COLOR;
      for (const [a, b] of FEATURE_BONES) {
        if (a >= joints.length || b >= joints.length) continue;
        ctx.beginPath();
        ctx.moveTo(toX(joints[a][0]), toY(joints[a][1]));
        ctx.lineTo(toX(joints[b][0]), toY(joints[b][1]));
        ctx.stroke();
      }

      // Draw joints
      for (let i = 0; i < joints.length; i++) {
        const jx = toX(joints[i][0]);
        const jy = toY(joints[i][1]);
        ctx.fillStyle = JOINT_COLOR;
        ctx.beginPath();
        ctx.arc(jx, jy, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw angle arcs for injury chains
      for (const chain of chainsToRender) {
        const [i1, i2, i3] = chain.joints;
        const p1 = joints[i1];
        const p2 = joints[i2]; // vertex
        const p3 = joints[i3];
        const angle = computeJointAngle(p1, p2, p3);
        const atRisk = angle < chain.threshold; // smaller angle = more bent = more risk for joint stress

        drawAngleArc(
          ctx,
          toX(p2[0]),
          toY(p2[1]),
          toX(p1[0]),
          toY(p1[1]),
          toX(p3[0]),
          toY(p3[1]),
          angle,
          atRisk,
        );
      }

      // Advance frame
      if (!reducedMotion.current) {
        frameRef.current = (frameRef.current + 1) % frames.length;
      }

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [frames, chainsToRender]);

  return (
    <Card>
      <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-2">
        Your Movement Monitor
      </h3>
      <div
        className="relative w-full bg-[#FAFAFA] rounded-[12px] overflow-hidden"
        style={{ aspectRatio: "4/3" }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      </div>
      {injuryProfile.injuries.length === 0 ? (
        <p className="text-sm text-[#58CC02] font-bold mt-3">
          All clear — no injury-specific thresholds to monitor.
        </p>
      ) : (
        <div className="mt-3 flex flex-col gap-1">
          {messages.map((msg, i) => (
            <p key={i} className="text-xs text-[#777777] leading-relaxed">
              {msg}
            </p>
          ))}
        </div>
      )}
    </Card>
  );
}

function drawAngleArc(
  ctx: CanvasRenderingContext2D,
  vx: number,
  vy: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  angleDeg: number,
  atRisk: boolean,
) {
  const angleA = Math.atan2(ay - vy, ax - vx);
  const angleB = Math.atan2(by - vy, bx - vx);

  // Determine sweep direction (always draw the smaller arc)
  let startAngle = angleA;
  let endAngle = angleB;
  let diff = endAngle - startAngle;
  if (diff > Math.PI) {
    startAngle = angleB;
    endAngle = angleA;
  } else if (diff < -Math.PI) {
    // keep as is
  } else if (diff < 0) {
    startAngle = angleB;
    endAngle = angleA;
  }

  const color = atRisk ? RISK_COLOR : SAFE_COLOR;

  // Draw arc
  ctx.beginPath();
  ctx.arc(vx, vy, ARC_RADIUS, startAngle, endAngle);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.setLineDash([]);
  ctx.stroke();

  // If at risk, draw dashed threshold arc
  if (atRisk) {
    ctx.beginPath();
    ctx.arc(vx, vy, ARC_RADIUS + 4, startAngle, endAngle);
    ctx.strokeStyle = "rgba(234,43,43,0.3)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Label at arc midpoint
  const midAngle = (startAngle + endAngle) / 2;
  const labelDist = ARC_RADIUS + LABEL_OFFSET;
  const lx = vx + Math.cos(midAngle) * labelDist;
  const ly = vy + Math.sin(midAngle) * labelDist;

  ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(`${Math.round(angleDeg)}°`, lx, ly);
}
