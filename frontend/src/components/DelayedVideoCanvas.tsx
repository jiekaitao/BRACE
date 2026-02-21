"use client";

import { useEffect, useRef } from "react";
import type { DebugStats } from "@/lib/types";

interface DelayedVideoCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  debugStatsRef: React.MutableRefObject<DebugStats>;
}

/**
 * Renders video frames delayed by the pipeline RTT so the skeleton
 * overlay aligns with the displayed frame instead of lagging behind.
 *
 * The actual <video> element stays hidden but keeps playing for frame capture.
 * This canvas shows frames from ~RTT ms ago, matching when the backend
 * processed them.
 */
export default function DelayedVideoCanvas({
  videoRef,
  debugStatsRef,
}: DelayedVideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const buffer: { bitmap: ImageBitmap; captureTime: number; videoTime: number }[] = [];
    let lastCapturedVideoTime = -1;
    let lastVideoTime = 0;
    let pendingCapture = false;
    // Smoothed delay tracks RTT with EMA to avoid jarring jumps
    let smoothedDelay = 0;

    function loop() {
      const video = videoRef.current;
      if (!canvas || !video || !video.videoWidth) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      // Size canvas to match container (DPR-aware)
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const tw = Math.round(rect.width * dpr);
      const th = Math.round(rect.height * dpr);
      if (canvas.width !== tw || canvas.height !== th) {
        canvas.width = tw;
        canvas.height = th;
      }

      const now = performance.now();
      const vt = video.currentTime;

      // Detect video loop (currentTime jumps backward)
      if (lastVideoTime > 0 && vt < lastVideoTime - 0.5) {
        for (const f of buffer) f.bitmap.close();
        buffer.length = 0;
        lastCapturedVideoTime = -1;
      }
      lastVideoTime = vt;

      // Capture new frame when video advances
      if (
        !video.paused &&
        !pendingCapture &&
        video.readyState >= 2 &&
        vt !== lastCapturedVideoTime
      ) {
        pendingCapture = true;
        const captureTime = now;
        const captureVideoTime = vt;
        createImageBitmap(video)
          .then((bitmap) => {
            buffer.push({ bitmap, captureTime, videoTime: captureVideoTime });
            lastCapturedVideoTime = captureVideoTime;
            // Keep buffer bounded (~30 frames ≈ 1s at 30fps)
            while (buffer.length > 30) {
              buffer[0].bitmap.close();
              buffer.shift();
            }
            pendingCapture = false;
          })
          .catch(() => {
            pendingCapture = false;
          });
      }

      // Track RTT with EMA smoothing (avoids jarring jumps when RTT spikes)
      const rtt = debugStatsRef.current.rtt_ms || 100;
      const targetDelay = Math.max(50, rtt);
      smoothedDelay = smoothedDelay === 0
        ? targetDelay
        : smoothedDelay + (targetDelay - smoothedDelay) * 0.05;
      const delayMs = smoothedDelay;

      // Find the frame from ~delayMs ago
      const targetTime = now - delayMs;
      let frame: (typeof buffer)[0] | null = null;
      for (let i = buffer.length - 1; i >= 0; i--) {
        if (buffer[i].captureTime <= targetTime) {
          frame = buffer[i];
          break;
        }
      }
      // If delay buffer hasn't filled yet, show oldest available
      if (!frame && buffer.length > 0) frame = buffer[0];

      if (frame && ctx) {
        // object-contain: aspect-fit the frame into the canvas
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const cw = canvas.width;
        const ch = canvas.height;
        const scale = Math.min(cw / vw, ch / vh);
        const dw = vw * scale;
        const dh = vh * scale;
        const dx = (cw - dw) / 2;
        const dy = (ch - dh) / 2;

        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, cw, ch);
        ctx.drawImage(frame.bitmap, dx, dy, dw, dh);
      }

      animRef.current = requestAnimationFrame(loop);
    }

    animRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animRef.current);
      for (const f of buffer) f.bitmap.close();
      buffer.length = 0;
    };
  }, [videoRef, debugStatsRef]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ zIndex: 5 }}
    />
  );
}
