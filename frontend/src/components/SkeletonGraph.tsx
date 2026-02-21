"use client";

import { useEffect, useRef, useState } from "react";
import type { SubjectState } from "@/lib/types";
import { FEATURE_BONES, RISK_JOINT_TO_FEAT } from "@/lib/skeleton";
import { CLUSTER_COLORS, PHASE_COLORS, riskColor } from "@/lib/colors";
import { rotateYX, project } from "@/lib/skeleton3d";
import Card from "./ui/Card";

interface SkeletonGraphProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
}

const PADDING = 24;
const FOV = 200;
const DISTANCE = 4;
const SMOOTHING = 0.12;

// Arrow constants (from GatorMotion PoseOverlay)
const ARROW_ACTIVATE_THRESHOLD = 0.12;
const ARROW_CLEAR_THRESHOLD = 0.06;
const EMA_LERP_FROM = 0.25;
const EMA_LERP_TO = 0.12;

// Reference skeleton color
const REF_COLOR = "rgba(234, 43, 43, 0.35)";
const REF_JOINT_COLOR = "rgba(234, 43, 43, 0.25)";

interface SmoothedArrow {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  distance: number;
  active: boolean;
  opacity: number;
}

export default function SkeletonGraph({
  subjectsRef,
  selectedSubjectRef,
}: SkeletonGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const prevColorRef = useRef<string | null>(null);
  const colorLerpRef = useRef(1.0);

  // Auto-fit smoothed refs (operate on projected 2D coords)
  const smoothedScaleRef = useRef(0); // 0 = uninitialized, snap on first frame
  const smoothedCxRef = useRef(0);
  const smoothedCyRef = useRef(0);

  // Guide toggle
  const [showGuide, setShowGuide] = useState(false);
  const showGuideRef = useRef(false);

  // Smoothed arrow state (persists across frames)
  const smoothedArrowsRef = useRef<Map<number, SmoothedArrow>>(new Map());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function draw() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // DPR-aware sizing
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      // Get selected subject
      const selectedId = selectedSubjectRef.current;
      if (selectedId === null) {
        drawPlaceholder(ctx, w, h, "No subject selected");
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      const subject = subjectsRef.current.get(selectedId);
      if (!subject || !subject.srpJoints) {
        drawPlaceholder(ctx, w, h, "No pose detected");
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      const joints = subject.srpJoints;

      // Center SRP joints at their midpoint before rotation
      let rawMinX = Infinity, rawMaxX = -Infinity,
        rawMinY = Infinity, rawMaxY = -Infinity;
      for (const j of joints) {
        if (j[0] < rawMinX) rawMinX = j[0];
        if (j[0] > rawMaxX) rawMaxX = j[0];
        if (j[1] < rawMinY) rawMinY = j[1];
        if (j[1] > rawMaxY) rawMaxY = j[1];
      }
      const rawMidX = (rawMinX + rawMaxX) / 2;
      const rawMidY = (rawMinY + rawMaxY) / 2;

      const centered: ([number, number] | [number, number, number])[] = joints.map((j) => {
        if (j.length >= 3) return [j[0] - rawMidX, j[1] - rawMidY, (j as [number, number, number])[2]] as [number, number, number];
        return [j[0] - rawMidX, j[1] - rawMidY] as [number, number];
      });

      // 3D rotation then perspective projection
      const rotated = rotateYX(centered, 0, 0);
      const projected = project(rotated, FOV, DISTANCE);

      // Auto-fit using ALL projected joints so limbs are never cut off
      let minX = Infinity, maxX = -Infinity,
        minY = Infinity, maxY = -Infinity;
      for (const [px, py] of projected) {
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }
      const rangeX = maxX - minX;
      const rangeY = maxY - minY;
      const projRange = Math.max(rangeX, rangeY, 1);
      const projMidX = (minX + maxX) / 2;
      const projMidY = (minY + maxY) / 2;

      // Target scale: fit projected range into canvas with generous padding
      const drawW = w - 2 * PADDING;
      const drawH = h - 2 * PADDING;
      const targetScale = Math.min(drawW, drawH) / (projRange * 1.8);

      // Snap on first frame, then smooth
      if (smoothedScaleRef.current === 0) {
        smoothedScaleRef.current = targetScale;
        smoothedCxRef.current = projMidX;
        smoothedCyRef.current = projMidY;
      } else {
        // Asymmetric smoothing: zoom out fast (avoid clipping), zoom in slow (avoid jitter)
        const delta = targetScale - smoothedScaleRef.current;
        const scaleAlpha = delta < 0 ? 0.3 : 0.02;
        smoothedScaleRef.current += delta * scaleAlpha;
        smoothedCxRef.current += (projMidX - smoothedCxRef.current) * SMOOTHING;
        smoothedCyRef.current += (projMidY - smoothedCyRef.current) * SMOOTHING;
      }

      const sc = smoothedScaleRef.current;
      const canvasCX = w / 2;
      const canvasCY = h / 2;

      const toCanvasX = (px: number) => canvasCX + (px - smoothedCxRef.current) * sc;
      const toCanvasY = (py: number) => canvasCY + (py - smoothedCyRef.current) * sc;

      // Determine color from cluster
      const cid = subject.clusterId;
      let targetColor: string;
      if (cid !== null && cid !== undefined) {
        targetColor = CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
      } else {
        targetColor = PHASE_COLORS[subject.phase];
      }

      // Smooth color transitions
      if (prevColorRef.current !== targetColor) {
        if (prevColorRef.current === null) {
          prevColorRef.current = targetColor;
          colorLerpRef.current = 1.0;
        } else {
          colorLerpRef.current = 0.0;
        }
      }
      colorLerpRef.current = Math.min(colorLerpRef.current + 0.02, 1.0);
      const color =
        colorLerpRef.current >= 1.0
          ? targetColor
          : lerpColor(prevColorRef.current!, targetColor, colorLerpRef.current);
      if (colorLerpRef.current >= 1.0) {
        prevColorRef.current = targetColor;
      }

      // --- Draw reference skeleton overlay (when guide active) ---
      const guideOn = showGuideRef.current;
      let refProjected: [number, number, number][] | null = null;

      if (guideOn && subject.representativeJoints) {
        const repJoints = subject.representativeJoints;

        // Center representative joints using SAME midpoint as current joints
        const refCentered: ([number, number] | [number, number, number])[] = repJoints.map((j) => {
          if (j.length >= 3) return [j[0] - rawMidX, j[1] - rawMidY, (j as [number, number, number])[2]] as [number, number, number];
          return [j[0] - rawMidX, j[1] - rawMidY] as [number, number];
        });

        const refRotated = rotateYX(refCentered, 0, 0);
        refProjected = project(refRotated, FOV, DISTANCE);

        // Draw reference bones (semi-transparent)
        const refSortedBones = [...FEATURE_BONES]
          .filter(([a, b]) => a < refProjected!.length && b < refProjected!.length)
          .map(([a, b]) => ({
            a,
            b,
            avgZ: (refProjected![a][2] + refProjected![b][2]) / 2,
          }))
          .sort((a, b) => a.avgZ - b.avgZ);

        ctx.lineCap = "round";
        for (const bone of refSortedBones) {
          const pa = refProjected![bone.a];
          const pb = refProjected![bone.b];
          ctx.strokeStyle = REF_COLOR;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(toCanvasX(pa[0]), toCanvasY(pa[1]));
          ctx.lineTo(toCanvasX(pb[0]), toCanvasY(pb[1]));
          ctx.stroke();
        }

        // Draw reference joints
        for (let i = 0; i < refProjected.length; i++) {
          const cx = toCanvasX(refProjected[i][0]);
          const cy = toCanvasY(refProjected[i][1]);
          ctx.fillStyle = REF_JOINT_COLOR;
          ctx.beginPath();
          ctx.arc(cx, cy, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Per-joint visibility alpha: fade to 0.1 when out of frame
      const vis = subject.jointVisibility;
      const visAlpha = (i: number) => {
        if (!vis || i >= vis.length) return 1.0;
        return vis[i] >= 0.3 ? 1.0 : 0.1;
      };

      // Sort bones by average z-depth (back-to-front)
      const sortedBones = [...FEATURE_BONES]
        .filter(([a, b]) => a < projected.length && b < projected.length)
        .map(([a, b]) => ({
          a,
          b,
          avgZ: (projected[a][2] + projected[b][2]) / 2,
        }))
        .sort((a, b) => a.avgZ - b.avgZ);

      // Build risk color map (feature joint index → color)
      const riskColorMap = new Map<number, string>();
      if (subject.quality?.injury_risks) {
        for (const risk of subject.quality.injury_risks) {
          if (risk.severity !== "medium" && risk.severity !== "high") continue;
          const rc = riskColor(risk.severity);
          const featIndices = RISK_JOINT_TO_FEAT[risk.joint];
          if (!featIndices) continue;
          for (const idx of featIndices) {
            const existing = riskColorMap.get(idx);
            if (!existing || risk.severity === "high") {
              riskColorMap.set(idx, rc);
            }
          }
        }
      }

      // Draw bones
      ctx.lineCap = "round";
      for (const bone of sortedBones) {
        const pa = projected[bone.a];
        const pb = projected[bone.b];
        const df = depthAlpha(bone.avgZ);
        const va = Math.min(visAlpha(bone.a), visAlpha(bone.b));

        const boneRisk = riskColorMap.get(bone.a) || riskColorMap.get(bone.b);
        ctx.strokeStyle = boneRisk ? withAlpha(boneRisk, df * va) : withAlpha(color, df * va);
        ctx.lineWidth = 2 + df * 2;

        ctx.beginPath();
        ctx.moveTo(toCanvasX(pa[0]), toCanvasY(pa[1]));
        ctx.lineTo(toCanvasX(pb[0]), toCanvasY(pb[1]));
        ctx.stroke();
      }

      // Sort joints by z-depth
      const sortedJoints = projected
        .map((p, i) => ({ i, x: p[0], y: p[1], z: p[2] }))
        .sort((a, b) => a.z - b.z);

      // Draw joints
      for (const joint of sortedJoints) {
        const cx = toCanvasX(joint.x);
        const cy = toCanvasY(joint.y);
        const df = depthAlpha(joint.z);
        const va = visAlpha(joint.i);
        const jointRisk = riskColorMap.get(joint.i);
        const radius = 3 + df * 3;

        ctx.fillStyle = jointRisk ? withAlpha(jointRisk, df * va) : withAlpha(color, df * va);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = withAlpha("#FFFFFF", df * 0.8 * va);
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // --- Draw correction arrows (when guide active + reference available) ---
      if (guideOn && refProjected) {
        const smoothed = smoothedArrowsRef.current;
        const seenKeys = new Set<number>();

        for (let i = 0; i < projected.length && i < refProjected.length; i++) {
          seenKeys.add(i);

          const curCX = toCanvasX(projected[i][0]);
          const curCY = toCanvasY(projected[i][1]);
          const refCX = toCanvasX(refProjected[i][0]);
          const refCY = toCanvasY(refProjected[i][1]);

          // Compute distance in projected (pre-canvas) space for threshold comparison
          const dx = projected[i][0] - refProjected[i][0];
          const dy = projected[i][1] - refProjected[i][1];
          const dist = Math.sqrt(dx * dx + dy * dy);

          const existing = smoothed.get(i);

          if (existing) {
            // EMA smooth
            existing.fromX = lerp(existing.fromX, curCX, EMA_LERP_FROM);
            existing.fromY = lerp(existing.fromY, curCY, EMA_LERP_FROM);
            existing.toX = lerp(existing.toX, refCX, EMA_LERP_TO);
            existing.toY = lerp(existing.toY, refCY, EMA_LERP_TO);
            existing.distance = lerp(existing.distance, dist, 0.15);

            // Hysteresis
            if (!existing.active && dist > ARROW_ACTIVATE_THRESHOLD) {
              existing.active = true;
            } else if (existing.active && dist < ARROW_CLEAR_THRESHOLD) {
              existing.active = false;
            }

            // Animate opacity
            if (existing.active) {
              existing.opacity = Math.min(1.0, existing.opacity + 0.06);
            } else {
              existing.opacity = Math.max(0, existing.opacity - 0.03);
            }
          } else {
            const isActive = dist > ARROW_ACTIVATE_THRESHOLD;
            smoothed.set(i, {
              fromX: curCX,
              fromY: curCY,
              toX: refCX,
              toY: refCY,
              distance: dist,
              active: isActive,
              opacity: isActive ? 0.2 : 0,
            });
          }
        }

        // Fade out arrows for joints no longer present
        for (const [key, arrow] of smoothed) {
          if (!seenKeys.has(key)) {
            arrow.active = false;
            arrow.opacity = Math.max(0, arrow.opacity - 0.03);
            if (arrow.opacity <= 0) {
              smoothed.delete(key);
            }
          }
        }

        // Draw all visible arrows
        for (const arrow of smoothed.values()) {
          if (arrow.opacity <= 0.01) continue;
          drawCorrectionArrow(ctx, arrow);
        }
      } else {
        // Clear smoothed arrows when guide is off
        smoothedArrowsRef.current.clear();
      }

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [subjectsRef, selectedSubjectRef]);

  const toggleGuide = () => {
    const next = !showGuideRef.current;
    showGuideRef.current = next;
    setShowGuide(next);
    if (!next) {
      smoothedArrowsRef.current.clear();
    }
  };

  return (
    <Card>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em]">
          Skeleton
        </h3>
        <button
          onClick={toggleGuide}
          className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
            showGuide
              ? "bg-[#3C3C3C] text-white border-[#3C3C3C]"
              : "bg-white text-[#3C3C3C] border-[#D0D0D0] hover:border-[#3C3C3C]"
          }`}
        >
          {showGuide ? "Hide Guide" : "Show Guide"}
        </button>
      </div>
      <div
        className="relative w-full bg-[#FAFAFA] rounded-[12px] overflow-hidden"
        style={{ aspectRatio: "1/1" }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      </div>
    </Card>
  );
}

function drawPlaceholder(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  text: string
) {
  ctx.fillStyle = "#AFAFAF";
  ctx.font = "13px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
}

/** Map z-depth to alpha (0.3-1.0), farther = more transparent. */
function depthAlpha(z: number): number {
  return Math.max(0.3, Math.min(1.0, 0.65 + z * 0.35));
}

/** Apply alpha to a hex color string. */
function withAlpha(color: string, alpha: number): string {
  if (color.startsWith("#")) {
    const c = color.replace("#", "");
    const r = parseInt(c.slice(0, 2), 16);
    const g = parseInt(c.slice(2, 4), 16);
    const b = parseInt(c.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  if (color.startsWith("rgb(")) {
    return color.replace("rgb(", "rgba(").replace(")", `,${alpha})`);
  }
  return color;
}

function lerpColor(a: string, b: string, t: number): string {
  const parseHex = (hex: string) => {
    const c = hex.replace("#", "");
    return [
      parseInt(c.slice(0, 2), 16),
      parseInt(c.slice(2, 4), 16),
      parseInt(c.slice(4, 6), 16),
    ];
  };
  const [r1, g1, b1] = parseHex(a);
  const [r2, g2, b2] = parseHex(b);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const bl = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r},${g},${bl})`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Color gradient based on divergence magnitude. */
function arrowColor(distance: number): string {
  if (distance < 0.10) return "rgba(80, 190, 255, 0.85)";
  if (distance < 0.20) return "rgba(0, 190, 255, 0.85)";
  if (distance < 0.35) return "rgba(255, 150, 0, 0.9)";
  return "rgba(234, 43, 43, 0.95)";
}

function arrowGlowColor(distance: number): string {
  if (distance < 0.10) return "rgba(80, 190, 255, 0.3)";
  if (distance < 0.20) return "rgba(0, 190, 255, 0.3)";
  if (distance < 0.35) return "rgba(255, 150, 0, 0.3)";
  return "rgba(234, 43, 43, 0.3)";
}

/** Draw a single correction arrow with glow, arrowhead, and origin circle. */
function drawCorrectionArrow(
  ctx: CanvasRenderingContext2D,
  arrow: SmoothedArrow,
) {
  const { fromX, fromY, toX, toY, distance, opacity } = arrow;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 2) return;

  const angle = Math.atan2(dy, dx);
  const headLen = Math.max(6, len * 0.24);
  const headAngle = Math.PI / 6;
  const color = arrowColor(distance);
  const glow = arrowGlowColor(distance);
  const lineWidth = 2 + Math.min(distance * 4, 2);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.shadowColor = glow;
  ctx.shadowBlur = 6;

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Shaft
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - headLen * Math.cos(angle - headAngle),
    toY - headLen * Math.sin(angle - headAngle),
  );
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - headLen * Math.cos(angle + headAngle),
    toY - headLen * Math.sin(angle + headAngle),
  );
  ctx.stroke();

  // Small circle at origin
  ctx.beginPath();
  ctx.arc(fromX, fromY, 4, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.restore();
}
