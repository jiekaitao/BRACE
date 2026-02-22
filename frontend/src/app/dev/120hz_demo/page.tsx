"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAnalysisWebSocket } from "@/hooks/useAnalysisWebSocket";
import CameraFeed from "@/components/CameraFeed";
import AnalysisCanvas from "@/components/AnalysisCanvas";

/** Body joints useful for acceleration tracking (skip face landmarks). */
const JOINT_OPTIONS = [
  { idx: 11, name: "Left Shoulder" },
  { idx: 12, name: "Right Shoulder" },
  { idx: 13, name: "Left Elbow" },
  { idx: 14, name: "Right Elbow" },
  { idx: 15, name: "Left Wrist" },
  { idx: 16, name: "Right Wrist" },
  { idx: 23, name: "Left Hip" },
  { idx: 24, name: "Right Hip" },
  { idx: 25, name: "Left Knee" },
  { idx: 26, name: "Right Knee" },
  { idx: 27, name: "Left Ankle" },
  { idx: 28, name: "Right Ankle" },
];

const MAX_SAMPLES = 300; // ~10s at 30fps

export default function HighFpsDemo() {
  const [selectedJoint, setSelectedJoint] = useState(25); // L Knee default
  const selectedJointRef = useRef(selectedJoint);
  selectedJointRef.current = selectedJoint;

  const {
    connected,
    subjectsRef,
    selectedSubjectRef,
    equipmentRef,
    selectSubject,
    startCapture,
    debugStatsRef,
  } = useAnalysisWebSocket(true, "webcam");

  const onCameraReady = useCallback(
    (videoEl: HTMLVideoElement) => {
      startCapture(videoEl);
    },
    [startCapture]
  );

  // FPS display (updates via interval since debugStatsRef is a ref)
  const [fpsLabel, setFpsLabel] = useState("");
  useEffect(() => {
    const iv = setInterval(() => {
      const s = debugStatsRef.current;
      setFpsLabel(
        `${s.fps_out} fps out / ${s.fps_in} fps in / ${Math.round(s.rtt_ms)}ms RTT`
      );
    }, 500);
    return () => clearInterval(iv);
  }, [debugStatsRef]);

  // Acceleration tracking state (ref-based, no re-renders)
  const accelCanvasRef = useRef<HTMLCanvasElement>(null);
  const accelAnimRef = useRef(0);
  const jointHistoryRef = useRef<{
    // Raw per-frame velocity components (for computing acceleration)
    vx: number[];
    vy: number[];
    // Acceleration magnitude per frame
    accel: number[];
    // Smoothed acceleration (EMA)
    smoothed: number[];
    // Last landmark receipt time to detect new frames
    lastLmTime: number;
    // Previous position for velocity delta
    prevX: number;
    prevY: number;
    prevT: number;
  }>({
    vx: [],
    vy: [],
    accel: [],
    smoothed: [],
    lastLmTime: 0,
    prevX: 0,
    prevY: 0,
    prevT: 0,
  });

  // Reset history when joint changes
  useEffect(() => {
    jointHistoryRef.current = {
      vx: [],
      vy: [],
      accel: [],
      smoothed: [],
      lastLmTime: 0,
      prevX: 0,
      prevY: 0,
      prevT: 0,
    };
  }, [selectedJoint]);

  // Acceleration canvas draw loop
  useEffect(() => {
    const canvas = accelCanvasRef.current;
    if (!canvas) return;

    function draw() {
      const c = accelCanvasRef.current;
      if (!c) return;
      const ctx = c.getContext("2d");
      if (!ctx) return;

      // DPR-aware sizing
      const rect = c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = rect.width;
      const h = rect.height;
      if (c.width !== w * dpr || c.height !== h * dpr) {
        c.width = w * dpr;
        c.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // --- Sample new data from current landmarks ---
      const selId = selectedSubjectRef.current;
      if (selId !== null) {
        const subject = subjectsRef.current.get(selId);
        if (subject?.landmarkFrame.current) {
          const lms = subject.landmarkFrame.current;
          const jointIdx = selectedJointRef.current;
          const lm = lms[jointIdx];
          const lmTime = subject.landmarkFrame.currentTime;
          const hist = jointHistoryRef.current;

          if (lm && lm.visibility > 0.3 && lmTime !== hist.lastLmTime) {
            const now = performance.now();

            if (hist.lastLmTime > 0 && hist.prevT > 0) {
              const dt = (now - hist.prevT) / 1000;
              if (dt > 0 && dt < 1) {
                // Per-axis velocity
                const vx = (lm.x - hist.prevX) / dt;
                const vy = (lm.y - hist.prevY) / dt;
                hist.vx.push(vx);
                hist.vy.push(vy);

                // Per-axis acceleration
                const nv = hist.vx.length;
                if (nv >= 2) {
                  const ax = (hist.vx[nv - 1] - hist.vx[nv - 2]) / dt;
                  const ay = (hist.vy[nv - 1] - hist.vy[nv - 2]) / dt;
                  const mag = Math.sqrt(ax * ax + ay * ay);
                  hist.accel.push(mag);

                  // EMA smoothing (alpha = 0.3)
                  const prev =
                    hist.smoothed.length > 0
                      ? hist.smoothed[hist.smoothed.length - 1]
                      : mag;
                  hist.smoothed.push(prev + 0.3 * (mag - prev));
                }

                // Trim
                if (hist.vx.length > MAX_SAMPLES) hist.vx.shift();
                if (hist.vy.length > MAX_SAMPLES) hist.vy.shift();
                if (hist.accel.length > MAX_SAMPLES) hist.accel.shift();
                if (hist.smoothed.length > MAX_SAMPLES) hist.smoothed.shift();
              }
            }

            hist.prevX = lm.x;
            hist.prevY = lm.y;
            hist.prevT = now;
            hist.lastLmTime = lmTime;
          }
        }
      }

      // --- Draw chart ---
      const hist = jointHistoryRef.current;
      const raw = hist.accel;
      const smooth = hist.smoothed;
      const n = raw.length;

      const PL = 48;
      const PR = 12;
      const PT = 8;
      const PB = 20;
      const plotW = w - PL - PR;
      const plotH = h - PT - PB;

      if (n < 3) {
        ctx.fillStyle = "#AFAFAF";
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Collecting acceleration data...", w / 2, h / 2);
        accelAnimRef.current = requestAnimationFrame(draw);
        return;
      }

      // Auto-scale Y
      let yMax = 0;
      for (let i = 0; i < n; i++) {
        if (smooth[i] > yMax) yMax = smooth[i];
        if (raw[i] > yMax) yMax = raw[i];
      }
      yMax = Math.max(yMax * 1.15, 0.5);

      const xScale = plotW / Math.max(n - 1, 1);
      const toX = (i: number) => PL + i * xScale;
      const toY = (v: number) => PT + plotH - (v / yMax) * plotH;

      // Grid lines
      ctx.strokeStyle = "#E5E5E5";
      ctx.lineWidth = 0.5;
      const gridLines = 3;
      for (let i = 0; i <= gridLines; i++) {
        const yVal = (yMax / gridLines) * i;
        const y = toY(yVal);
        ctx.beginPath();
        ctx.moveTo(PL, y);
        ctx.lineTo(w - PR, y);
        ctx.stroke();

        ctx.fillStyle = "#AFAFAF";
        ctx.font = "9px system-ui, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(yVal.toFixed(1), PL - 4, y + 3);
      }

      // Gradient fill under smoothed line
      const gradient = ctx.createLinearGradient(0, PT, 0, PT + plotH);
      gradient.addColorStop(0, "rgba(255, 107, 53, 0.25)");
      gradient.addColorStop(1, "rgba(255, 107, 53, 0.02)");
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(0));
      for (let i = 0; i < n; i++) {
        ctx.lineTo(toX(i), toY(smooth[i]));
      }
      ctx.lineTo(toX(n - 1), toY(0));
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      // Raw acceleration (thin semi-transparent)
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 107, 53, 0.2)";
      ctx.lineWidth = 1;
      for (let i = 0; i < n; i++) {
        const x = toX(i);
        const y = toY(raw[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Smoothed acceleration (solid)
      ctx.beginPath();
      ctx.strokeStyle = "#FF6B35";
      ctx.lineWidth = 2;
      for (let i = 0; i < n; i++) {
        const x = toX(i);
        const y = toY(smooth[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Current value dot
      if (n > 0) {
        const last = smooth[n - 1];
        ctx.beginPath();
        ctx.arc(toX(n - 1), toY(last), 4, 0, Math.PI * 2);
        ctx.fillStyle = "#FF6B35";
        ctx.fill();
        ctx.strokeStyle = "#FFF";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Value label next to dot
        ctx.fillStyle = "#FF6B35";
        ctx.font = "bold 10px system-ui, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(last.toFixed(1), toX(n - 1) - 8, toY(last) - 6);
      }

      // X-axis label
      ctx.fillStyle = "#AFAFAF";
      ctx.font = "9px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${n} samples`, w / 2, h - 4);

      accelAnimRef.current = requestAnimationFrame(draw);
    }

    accelAnimRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(accelAnimRef.current);
  }, [subjectsRef, selectedSubjectRef]);

  return (
    <div className="h-dvh h-screen flex flex-col bg-[#0A0A0A] overflow-hidden">
      {/* Camera view with skeleton overlay */}
      <div className="flex-1 relative min-h-0">
        <CameraFeed onVideoReady={onCameraReady} mirrored />
        <AnalysisCanvas
          subjectsRef={subjectsRef}
          selectedSubjectRef={selectedSubjectRef}
          equipmentRef={equipmentRef}
          onSelectSubject={selectSubject}
          mirrored
        />

        {/* Connection badge */}
        <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 rounded-full px-2.5 py-1 z-20">
          <div
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-[#58CC02]" : "bg-[#EA2B2B]"
            }`}
          />
          <span className="text-[11px] font-bold text-white/80">
            {connected ? "Live" : "Connecting..."}
          </span>
        </div>

        {/* FPS counter */}
        <div className="absolute top-3 left-3 bg-black/60 rounded-full px-2.5 py-1 z-20">
          <span className="text-[11px] font-mono text-white/80">{fpsLabel}</span>
        </div>
      </div>

      {/* Acceleration chart */}
      <div className="h-[33vh] flex flex-col bg-[#111111] border-t border-[#333]">
        {/* Header row with title + joint selector */}
        <div className="flex items-center justify-between px-3 py-2">
          <h3 className="text-xs font-bold text-[#AFAFAF] uppercase tracking-wider">
            Joint Acceleration
          </h3>
          <select
            value={selectedJoint}
            onChange={(e) => setSelectedJoint(Number(e.target.value))}
            className="text-xs border border-[#444] rounded-lg px-2 py-1 bg-[#1A1A1A] text-white font-medium focus:outline-none focus:border-[#FF6B35]"
          >
            {JOINT_OPTIONS.map(({ idx, name }) => (
              <option key={idx} value={idx}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <canvas ref={accelCanvasRef} className="w-full flex-1 min-h-0" />
      </div>
    </div>
  );
}
