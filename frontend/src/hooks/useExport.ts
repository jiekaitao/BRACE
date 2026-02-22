import { useState, useCallback, useRef } from "react";
import { getApiBase, getWsBase } from "@/lib/api";

type ExportStatus = "idle" | "processing" | "complete" | "error";

export function useExport() {
  const [exportStatus, setExportStatus] = useState<ExportStatus>("idle");
  const [exportProgress, setExportProgress] = useState(0);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);
  const [skeletonUrl, setSkeletonUrl] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const startExport = useCallback(async (sessionId: string) => {
    setExportStatus("processing");
    setExportProgress(0);
    setAnnotatedUrl(null);
    setSkeletonUrl(null);

    try {
      const apiBase = getApiBase();
      const res = await fetch(
        `${apiBase}/api/export?session_id=${encodeURIComponent(sessionId)}`,
        { method: "POST" }
      );
      const data = await res.json();

      if (data.error) {
        setExportStatus("error");
        return;
      }

      const exportId = data.export_id;
      const wsBase = getWsBase();
      const ws = new WebSocket(`${wsBase}/ws/export/${exportId}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "progress") {
          setExportProgress(msg.progress);
        } else if (msg.type === "complete") {
          const base = apiBase;
          setAnnotatedUrl(`${base}${msg.annotated_url}`);
          setSkeletonUrl(`${base}${msg.skeleton_url}`);
          setExportStatus("complete");
          ws.close();
        }
      };

      ws.onerror = () => {
        setExportStatus("error");
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch {
      setExportStatus("error");
    }
  }, []);

  return { exportStatus, exportProgress, annotatedUrl, skeletonUrl, startExport };
}
