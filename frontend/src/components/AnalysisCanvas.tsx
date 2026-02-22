"use client";

import { useEffect, useRef, useCallback } from "react";
import type { SubjectState, BBox, EquipmentTracking } from "@/lib/types";
import { MP_BONES, FEATURE_INDICES, RISK_JOINT_TO_MP } from "@/lib/skeleton";
import { PHASE_COLORS, SKELETON_COLOR, JOINT_DOT_COLOR, riskColor, getSubjectColor, jerseyTagText } from "@/lib/colors";

interface AnalysisCanvasProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  equipmentRef?: React.MutableRefObject<EquipmentTracking | undefined>;
  onSelectSubject?: (trackId: number) => void;
  mirrored?: boolean;
  showRisks?: boolean;
}

export default function AnalysisCanvas({
  subjectsRef,
  selectedSubjectRef,
  equipmentRef,
  onSelectSubject,
  mirrored = false,
  showRisks = false,
}: AnalysisCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const smoothedBboxRef = useRef<Record<number, { x1: number; y1: number; x2: number; y2: number }>>({});
  // Minecraft-style step-function label positions — only snap on large movements
  const committedLabelRef = useRef<Record<number, { x: number; y: number }>>({});
  const showRisksRef = useRef(showRisks);
  showRisksRef.current = showRisks;

  // Click-to-select handler
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onSelectSubject) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left) / rect.width;
      const clickY = (e.clientY - rect.top) / rect.height;

      // Normalize click to [0, 1]
      const nx = mirrored ? 1 - clickX : clickX;
      const ny = clickY;

      // Hit-test against all subject bboxes, sorted by area ascending (smaller = preferred)
      const hits: { trackId: number; area: number }[] = [];
      for (const [trackId, subject] of subjectsRef.current) {
        const bbox = subject.bbox;
        if (!bbox) continue;
        if (nx >= bbox.x1 && nx <= bbox.x2 && ny >= bbox.y1 && ny <= bbox.y2) {
          const area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
          hits.push({ trackId, area });
        }
      }

      if (hits.length > 0) {
        hits.sort((a, b) => a.area - b.area);
        onSelectSubject(hits[0].trackId);
      }
    },
    [onSelectSubject, mirrored, subjectsRef]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

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

      const selectedId = selectedSubjectRef.current;
      const subjects = subjectsRef.current;

      if (subjects.size === 0) {
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      // Mirror if needed
      const toX = (nx: number) => (mirrored ? (1 - nx) * w : nx * w);
      const toY = (ny: number) => ny * h;

      // Draw each subject
      for (const [trackId, subject] of subjects) {
        // Skip stale subjects (off-screen for >500ms)
        const STALE_MS = 500;
        if (performance.now() - subject.lastSeenTime > STALE_MS) continue;

        const isSelected = trackId === selectedId;
        const bbox = subject.bbox;
        const subjectColor = isSelected ? getSubjectColor(trackId) : "#999999";

        // Non-selected: only draw a small label chip above head (Minecraft-style step-function)
        if (!isSelected) {
          // Compute raw head position from shoulders
          const lms = subject.landmarkFrame.current;
          let rawX: number | null = null, rawY: number | null = null;
          if (lms && lms[11]?.visibility > 0.3 && lms[12]?.visibility > 0.3) {
            rawX = (lms[11].x + lms[12].x) / 2;
            rawY = Math.min(lms[11].y, lms[12].y) - 0.04;
          } else if (bbox) {
            rawX = (bbox.x1 + bbox.x2) / 2;
            rawY = bbox.y1 - 0.02;
          }
          if (rawX === null || rawY === null) continue;

          // Step-function: only update committed position on significant movement
          const SNAP_THRESHOLD = 0.035; // ~3.5% of canvas — about 50px on 1080p
          const committed = committedLabelRef.current[trackId];
          if (!committed || Math.abs(rawX - committed.x) > SNAP_THRESHOLD || Math.abs(rawY - committed.y) > SNAP_THRESHOLD) {
            committedLabelRef.current[trackId] = { x: rawX, y: rawY };
          }
          const labelPos = committedLabelRef.current[trackId];
          const labelX = toX(labelPos.x);
          const labelY = toY(labelPos.y);

          const isUnknown = subject.identityStatus === "unknown";
          const jerseyTag = jerseyTagText(subject.jerseyNumber, subject.jerseyColor);
          const chipLabel = jerseyTag ? `${subject.label} / ${jerseyTag}` : subject.label;
          ctx.globalAlpha = isUnknown ? 0.3 : 0.5;
          ctx.fillStyle = subjectColor;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
          const tm = ctx.measureText(chipLabel);
          const chipW = tm.width + 8;
          ctx.beginPath();
          ctx.roundRect(labelX - chipW / 2, labelY - 18, chipW, 16, 4);
          ctx.fill();
          ctx.fillStyle = "#FFF";
          ctx.fillText(chipLabel, labelX - chipW / 2 + 4, labelY - 5);
          ctx.globalAlpha = 1.0;
          continue;
        }

        // Selected subject: full rendering
        ctx.globalAlpha = 1.0;

        const frame = subject.landmarkFrame;
        const phase = subject.phase;
        const landmarks = frame.current;

        if (!landmarks) continue;

        // Interpolation + extrapolation between prev and current
        // Uses receipt time (performance.now) for smooth 60fps sub-frame interpolation
        // Loop handling is done by clearing landmarks in useAnalysisWebSocket
        let lms = landmarks;
        if (frame.prev && frame.currentTime > frame.prevTime) {
          const elapsed = performance.now() - frame.currentTime;
          const interval = frame.currentTime - frame.prevTime;
          const t = Math.min(elapsed / interval, 1.5);
          lms = landmarks.map((curr, i) => {
            const prev = frame.prev![i];
            if (!prev) return curr;
            return {
              x: prev.x + (curr.x - prev.x) * t,
              y: prev.y + (curr.y - prev.y) * t,
              visibility: curr.visibility,
            };
          });
        }

        const phaseColor = PHASE_COLORS[phase];

        // Compute torso-based display bbox (exclude wrists/hands to prevent jitter)
        const torsoIndices = [11, 12, 23, 24, 25, 26, 27, 28]; // shoulders, hips, knees, ankles
        let tMinX = 1, tMaxX = 0, tMinY = 1, tMaxY = 0;
        let hasTorso = false;
        for (const idx of torsoIndices) {
          if (idx < lms.length && lms[idx].visibility > 0.3) {
            tMinX = Math.min(tMinX, lms[idx].x);
            tMaxX = Math.max(tMaxX, lms[idx].x);
            tMinY = Math.min(tMinY, lms[idx].y);
            tMaxY = Math.max(tMaxY, lms[idx].y);
            hasTorso = true;
          }
        }

        let displayBbox = hasTorso ? {
          x1: tMinX - 0.02,
          y1: tMinY - 0.04,  // extra space above for head
          x2: tMaxX + 0.02,
          y2: tMaxY + 0.02,
        } : (subject.bbox ? { ...subject.bbox } : null);

        // Smooth the display bbox to prevent frame-to-frame jitter
        const BBOX_SMOOTH = 0.15;
        const prevBbox = smoothedBboxRef.current[trackId];
        if (prevBbox && displayBbox) {
          displayBbox.x1 = prevBbox.x1 + (displayBbox.x1 - prevBbox.x1) * BBOX_SMOOTH;
          displayBbox.y1 = prevBbox.y1 + (displayBbox.y1 - prevBbox.y1) * BBOX_SMOOTH;
          displayBbox.x2 = prevBbox.x2 + (displayBbox.x2 - prevBbox.x2) * BBOX_SMOOTH;
          displayBbox.y2 = prevBbox.y2 + (displayBbox.y2 - prevBbox.y2) * BBOX_SMOOTH;
        }
        if (displayBbox) smoothedBboxRef.current[trackId] = { ...displayBbox };

        // Draw bounding box from torso-based display bbox
        if (displayBbox) {
          const bx = toX(mirrored ? displayBbox.x2 : displayBbox.x1);
          const by = toY(displayBbox.y1);
          const bw = Math.abs(toX(displayBbox.x2) - toX(displayBbox.x1));
          const bh = toY(displayBbox.y2) - toY(displayBbox.y1);
          const r = 12;

          ctx.strokeStyle = phaseColor;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(bx + r, by);
          ctx.lineTo(bx + bw - r, by);
          ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r);
          ctx.lineTo(bx + bw, by + bh - r);
          ctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - r, by + bh);
          ctx.lineTo(bx + r, by + bh);
          ctx.quadraticCurveTo(bx, by + bh, bx, by + bh - r);
          ctx.lineTo(bx, by + r);
          ctx.quadraticCurveTo(bx, by, bx + r, by);
          ctx.closePath();
          ctx.stroke();

          // Subject label - Minecraft-style step-function position
          const selJerseyTag = jerseyTagText(subject.jerseyNumber, subject.jerseyColor);
          const labelText = selJerseyTag ? `${subject.label} / ${selJerseyTag}` : subject.label;
          ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, sans-serif";
          const tm = ctx.measureText(labelText);
          const labelW = tm.width + 10;
          const labelH = 20;

          // Compute raw head position
          let rawHX: number | null = null, rawHY: number | null = null;
          if (lms[11]?.visibility > 0.3 && lms[12]?.visibility > 0.3) {
            rawHX = (lms[11].x + lms[12].x) / 2;
            rawHY = Math.min(lms[11].y, lms[12].y) - 0.05;
          }

          // Step-function: snap only on large movement
          const LABEL_SNAP = 0.035;
          if (rawHX !== null && rawHY !== null) {
            const prev = committedLabelRef.current[trackId];
            if (!prev || Math.abs(rawHX - prev.x) > LABEL_SNAP || Math.abs(rawHY - prev.y) > LABEL_SNAP) {
              committedLabelRef.current[trackId] = { x: rawHX, y: rawHY };
            }
          }
          const cLabel = committedLabelRef.current[trackId];
          let headLabelX: number, headLabelY: number;
          if (cLabel) {
            headLabelX = toX(cLabel.x) - labelW / 2;
            headLabelY = toY(cLabel.y) - labelH;
          } else {
            headLabelX = bx;
            headLabelY = by - labelH - 4;
          }

          ctx.fillStyle = phaseColor;
          ctx.beginPath();
          ctx.roundRect(headLabelX, headLabelY, labelW, labelH, 6);
          ctx.fill();
          ctx.fillStyle = "#FFFFFF";
          ctx.fillText(labelText, headLabelX + 5, headLabelY + 14);
        }

        // Build risk color map for this subject (MP joint index → color)
        const riskColorMap = new Map<number, string>();
        if (showRisksRef.current && subject.quality?.injury_risks) {
          for (const risk of subject.quality.injury_risks) {
            if (risk.severity !== "medium" && risk.severity !== "high") continue;
            const color = riskColor(risk.severity);
            const mpIndices = RISK_JOINT_TO_MP[risk.joint];
            if (!mpIndices) continue;
            for (const idx of mpIndices) {
              // Higher severity wins
              const existing = riskColorMap.get(idx);
              if (!existing || risk.severity === "high") {
                riskColorMap.set(idx, color);
              }
            }
          }
        }

        // Draw skeleton wireframe
        ctx.lineWidth = 2;
        for (const [a, b] of MP_BONES) {
          if (a >= lms.length || b >= lms.length) continue;
          const la = lms[a];
          const lb = lms[b];
          if (la.visibility < 0.3 || lb.visibility < 0.3) continue;

          const boneRisk = riskColorMap.get(a) || riskColorMap.get(b);
          ctx.strokeStyle = boneRisk || SKELETON_COLOR;

          ctx.beginPath();
          ctx.moveTo(toX(la.x), toY(la.y));
          ctx.lineTo(toX(lb.x), toY(lb.y));
          ctx.stroke();
        }

        // Draw feature joint dots
        const featureSet = new Set(FEATURE_INDICES);
        for (let i = 0; i < lms.length; i++) {
          const lm = lms[i];
          if (lm.visibility < 0.3) continue;

          const isFeature = featureSet.has(i);
          const radius = isFeature ? 5 : 3;
          const jointRisk = riskColorMap.get(i);

          ctx.fillStyle = jointRisk || (isFeature ? JOINT_DOT_COLOR : "rgba(255,255,255,0.4)");
          ctx.beginPath();
          ctx.arc(toX(lm.x), toY(lm.y), radius, 0, Math.PI * 2);
          ctx.fill();

          if (isFeature) {
            ctx.strokeStyle = jointRisk || phaseColor;
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }
      }

      // Clean up committed labels for subjects no longer visible
      for (const id of Object.keys(committedLabelRef.current).map(Number)) {
        if (!subjects.has(id)) delete committedLabelRef.current[id];
      }

      // Draw equipment (football)
      const eq = equipmentRef?.current;
      if (eq?.box) {
        const [ymin, xmin, ymax, xmax] = eq.box;
        const bx = toX(mirrored ? 1 - xmax : xmin);
        const by = toY(ymin);
        const bw = toX(mirrored ? 1 - xmin : xmax) - bx;
        const bh = toY(ymax) - by;

        ctx.strokeStyle = "#F5A623"; // Orange for football
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);

        // Draw momentum/holder label
        const momentumText = `${Math.round(eq.momentum)} Mmt`;
        let holderText = "";
        if (eq.held_by_id) {
          const heldId = Number(eq.held_by_id);
          const holderSubject = subjectsRef.current.get(heldId);
          if (holderSubject) {
            holderText = ` (${holderSubject.label})`;
          } else {
            holderText = ` (ID: ${eq.held_by_id})`;
          }
        }

        const text = `Football: ${momentumText}${holderText}`;
        ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
        const tm = ctx.measureText(text);

        ctx.fillStyle = "#F5A623";
        ctx.beginPath();
        ctx.roundRect(bx, by - 24, tm.width + 12, 20, 4);
        ctx.fill();

        ctx.fillStyle = "#FFF";
        ctx.fillText(text, bx + 6, by - 9);
      }

      ctx.globalAlpha = 1.0;
      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);

    return () => cancelAnimationFrame(animRef.current);
  }, [subjectsRef, selectedSubjectRef, equipmentRef, mirrored]);

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="absolute inset-0 w-full h-full cursor-pointer"
      style={{ zIndex: 10 }}
    />
  );
}
