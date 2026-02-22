"use client";

import { useState, useRef, useCallback } from "react";
import { getApiBase, getWsBase } from "@/lib/api";
import type { CrashAnalysisResult } from "@/lib/types";

type Phase = "idle" | "uploading" | "processing" | "complete" | "error";

interface CrashAnalysisState {
  phase: Phase;
  analysisId: string | null;
  progress: number;
  collisionCount: number;
  subjectsTracked: number;
  result: CrashAnalysisResult | null;
  error: string | null;
  startAnalysis: (file: File) => void;
  reset: () => void;
}

export function useCrashAnalysis(): CrashAnalysisState {
  const [phase, setPhase] = useState<Phase>("idle");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [collisionCount, setCollisionCount] = useState(0);
  const [subjectsTracked, setSubjectsTracked] = useState(0);
  const [result, setResult] = useState<CrashAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const reset = useCallback(() => {
    setPhase("idle");
    setAnalysisId(null);
    setProgress(0);
    setCollisionCount(0);
    setSubjectsTracked(0);
    setResult(null);
    setError(null);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connectWs = useCallback((aid: string) => {
    const wsBase = getWsBase();
    const ws = new WebSocket(`${wsBase}/ws/crash/${aid}`);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "progress") {
          setProgress(msg.progress ?? 0);
          if (msg.data) {
            setCollisionCount(msg.data.collision_count ?? 0);
            setSubjectsTracked(msg.data.subjects_tracked ?? 0);
          }
        } else if (msg.type === "complete") {
          const data = msg.data;
          if (data?.status === "error" || data?.error) {
            setError(data.error ?? "Analysis failed");
            setPhase("error");
          } else {
            setResult(data as CrashAnalysisResult);
            setPhase("complete");
            setProgress(100);
          }
          ws.close();
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      setError("WebSocket connection error");
      setPhase("error");
    };
  }, []);

  const startAnalysis = useCallback(
    async (file: File) => {
      reset();
      setPhase("uploading");

      const apiBase = getApiBase();

      try {
        // Step 1: Upload the file
        const formData = new FormData();
        formData.append("file", file);
        const uploadResp = await fetch(`${apiBase}/api/upload`, {
          method: "POST",
          body: formData,
        });
        if (!uploadResp.ok) {
          throw new Error(`Upload failed: ${uploadResp.statusText}`);
        }
        const uploadData = await uploadResp.json();
        const sessionId = uploadData.session_id;
        if (!sessionId) {
          throw new Error("No session_id returned from upload");
        }

        // Step 2: Start crash analysis
        setPhase("processing");
        const analysisResp = await fetch(
          `${apiBase}/api/crash-analysis?session_id=${sessionId}`,
          { method: "POST" }
        );
        if (!analysisResp.ok) {
          throw new Error(`Analysis start failed: ${analysisResp.statusText}`);
        }
        const analysisData = await analysisResp.json();
        if (analysisData.error) {
          throw new Error(analysisData.error);
        }
        const aid = analysisData.analysis_id;
        setAnalysisId(aid);

        // Step 3: Connect WebSocket for progress
        connectWs(aid);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setPhase("error");
      }
    },
    [reset, connectWs]
  );

  return {
    phase,
    analysisId,
    progress,
    collisionCount,
    subjectsTracked,
    result,
    error,
    startAnalysis,
    reset,
  };
}
