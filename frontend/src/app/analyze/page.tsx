"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { useAnalysisWebSocket } from "@/hooks/useAnalysisWebSocket";
import { getApiBase } from "@/lib/api";
import CameraFeed from "@/components/CameraFeed";
import VideoUploader from "@/components/VideoUploader";
import AnalysisCanvas from "@/components/AnalysisCanvas";
import StatusHUD from "@/components/StatusHUD";
import EmbeddingGraph from "@/components/EmbeddingGraph";
import SkeletonGraph from "@/components/SkeletonGraph";
import ProgressBar from "@/components/ui/ProgressBar";
import DuoButton from "@/components/ui/DuoButton";
import DemoVideoModal from "@/components/DemoVideoModal";
import VelocityGraph from "@/components/VelocityGraph";
import DebugPanel from "@/components/DebugPanel";
import DelayedVideoCanvas from "@/components/DelayedVideoCanvas";
import MovementQualityPanel from "@/components/MovementQualityPanel";
import RiskSummaryCard from "@/components/RiskSummaryCard";
import WorkoutTimeline from "@/components/WorkoutTimeline";

function VideoControls({ videoRef, replaying }: { videoRef: React.RefObject<HTMLVideoElement | null>; replaying: boolean }) {
  const [paused, setPaused] = useState(false);
  const [muted, setMuted] = useState(true);
  const [time, setTime] = useState("0:00 / 0:00");

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    function onTimeUpdate() {
      const v = videoRef.current;
      if (!v) return;
      const fmt = (s: number) => {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${sec.toString().padStart(2, "0")}`;
      };
      setTime(`${fmt(v.currentTime)} / ${fmt(v.duration || 0)}`);
    }
    function onPlay() { setPaused(false); }
    function onPause() { setPaused(true); }
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    setPaused(video.paused);
    setMuted(video.muted);
    return () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
    };
  }, [videoRef]);

  const togglePlay = useCallback(() => {
    if (!replaying) return; // Can't pause during first pass
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play(); else v.pause();
  }, [videoRef, replaying]);

  const toggleMute = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  }, [videoRef]);

  return (
    <div
      className="absolute bottom-2 left-2 flex items-center gap-2 bg-black/60 rounded-full px-3 py-1.5 text-white text-xs"
      style={{ zIndex: 20 }}
    >
      <button
        onClick={togglePlay}
        className={replaying ? "hover:text-[#1CB0F6] font-bold" : "text-white/40 font-bold cursor-default"}
      >
        {paused ? "\u25B6" : "\u23F8"}
      </button>
      <button onClick={toggleMute} className="hover:text-[#1CB0F6] font-bold">
        {muted ? "\uD83D\uDD07" : "\uD83D\uDD0A"}
      </button>
      <span className="text-[#AFAFAF] font-mono">{time}</span>
    </div>
  );
}

function AnalyzeContent() {
  const searchParams = useSearchParams();
  const paramMode = searchParams.get("mode");
  const paramVideo = searchParams.get("video");

  const initialMode = paramMode === "upload" ? "video" : "webcam";

  // Build demo URL only on the client to avoid SSR hydration mismatch
  // (getApiBase() returns localhost during SSR but the real hostname on client)
  const isDemoMode = paramMode === "demo" && !!paramVideo;

  const [mode, setMode] = useState<"webcam" | "video">(
    paramMode === "demo" ? "webcam" : initialMode === "video" ? "video" : "webcam"
  );
  const [active, setActive] = useState(mode === "webcam" || isDemoMode);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [showDemoModal, setShowDemoModal] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [demoVideoUrl, setDemoVideoUrl] = useState<string | null>(null);
  const videoPlaybackRef = useRef<HTMLVideoElement>(null);
  const demoVideoRef = useRef<HTMLVideoElement>(null);

  // Set demo URL on client mount (avoids SSR hydration mismatch)
  useEffect(() => {
    if (paramMode === "demo" && paramVideo) {
      setDemoVideoUrl(`${getApiBase()}/api/demo-videos/${encodeURIComponent(paramVideo)}`);
    }
  }, [paramMode, paramVideo]);

  // Demo mode uses "webcam" WebSocket mode (frame-by-frame streaming)
  const wsMode = demoVideoUrl ? "webcam" : mode;

  const {
    connected,
    replaying,
    phase,
    nSegments,
    nClusters,
    clusterId,
    isAnomaly,
    clusterSummary,
    fatigueIndex,
    peakVelocity,
    videoProgress,
    videoComplete,
    subjectsRef,
    selectedSubjectRef,
    highlightedClusterRef,
    selectedSubjectId,
    activeTrackIds,
    selectSubject,
    startCapture,
    stopCapture,
    uploadVideo,
    debugStatsRef,
    equipmentRef,
  } = useAnalysisWebSocket(active, wsMode);


  // Build subject labels and identities maps from subjectsRef
  const subjectLabels = useMemo(() => {
    const labels: Record<number, string> = {};
    for (const trackId of activeTrackIds) {
      const subject = subjectsRef.current.get(trackId);
      if (subject) {
        labels[trackId] = subject.label;
      }
    }
    return labels;
  }, [activeTrackIds, subjectsRef]);

  const subjectIdentities = useMemo(() => {
    const identities: Record<number, { label: string; identityStatus: "unknown" | "tentative" | "confirmed" }> = {};
    for (const trackId of activeTrackIds) {
      const subject = subjectsRef.current.get(trackId);
      if (subject) {
        identities[trackId] = {
          label: subject.label,
          identityStatus: subject.identityStatus,
        };
      }
    }
    return identities;
  }, [activeTrackIds, subjectsRef]);

  // Handle webcam ready
  const onCameraReady = useCallback(
    (videoEl: HTMLVideoElement) => {
      startCapture(videoEl);
    },
    [startCapture]
  );

  // Handle video file selection
  const onFileSelected = useCallback(
    async (file: File) => {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      await uploadVideo(file);
      setActive(true);
    },
    [uploadVideo]
  );

  // Handle demo video selection
  const onDemoSelect = useCallback(
    (filename: string) => {
      stopCapture();
      setActive(false);
      setVideoFile(null);
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
        setVideoUrl(null);
      }
      setShowDemoModal(false);

      const apiBase = getApiBase();
      const url = `${apiBase}/api/demo-videos/${encodeURIComponent(filename)}`;
      setDemoVideoUrl(url);
      setMode("webcam");
      setActive(true);
    },
    [stopCapture, videoUrl]
  );

  // Ref callback: start capture as soon as demo video element mounts.
  // Opens WebSocket immediately; frame sending is gated by video.videoWidth > 0.
  const demoVideoRefCallback = useCallback((el: HTMLVideoElement | null) => {
    demoVideoRef.current = el;
    if (el) {
      startCapture(el);
    }
  }, [startCapture]);

  // When demo video can play, start playback
  const onDemoVideoReady = useCallback(() => {
    if (demoVideoRef.current) {
      demoVideoRef.current.play().catch(() => { });
    }
  }, []);

  // Mode switch
  const switchMode = useCallback(
    (newMode: "webcam" | "video") => {
      stopCapture();
      setActive(false);
      setVideoFile(null);
      setDemoVideoUrl(null);
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
        setVideoUrl(null);
      }
      setMode(newMode);
      if (newMode === "webcam") {
        setActive(true);
      }
    },
    [stopCapture, videoUrl]
  );

  // Stop analysis
  const handleStop = useCallback(() => {
    stopCapture();
    setActive(false);
    setDemoVideoUrl(null);
  }, [stopCapture]);

  // Cleanup blob URL
  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  const isDemo = !!demoVideoUrl;

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F7F7]">
      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row gap-4 p-4 max-w-[1600px] mx-auto w-full">
        {/* Left column: Video + Graphs */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Video panel */}
          <div className="fade-up">
            <div
              className="relative bg-black rounded-[16px] overflow-hidden w-full"
              style={{ aspectRatio: "16/9" }}
            >
              {/* Connection status badge */}
              <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 rounded-full px-2.5 py-1 z-20">
                <div
                  className={`w-2 h-2 rounded-full ${
                    !connected ? "bg-[#EA2B2B]" : replaying ? "bg-[#F5A623]" : "bg-[#58CC02]"
                  }`}
                />
                <span className="text-[11px] font-bold text-white/80">
                  {!connected ? "Offline" : replaying ? "Replaying" : "Live"}
                </span>
              </div>

              {/* Webcam mode (not demo) */}
              {mode === "webcam" && active && !isDemo && !isDemoMode && (
                <CameraFeed onVideoReady={onCameraReady} mirrored />
              )}

              {/* Demo video playback — delayed canvas covers the video element,
                  showing frames offset by pipeline RTT so skeleton aligns.
                  Video stays visible (not opacity:0) so Chrome autoplay works. */}
              {isDemo && active && (
                <>
                  <video
                    ref={demoVideoRefCallback}
                    src={demoVideoUrl}
                    className="w-full h-full object-contain"
                    playsInline
                    muted
                    autoPlay
                    loop
                    onCanPlay={onDemoVideoReady}
                  />
                  <DelayedVideoCanvas
                    videoRef={demoVideoRef}
                    debugStatsRef={debugStatsRef}
                  />
                  <VideoControls videoRef={demoVideoRef} replaying={replaying} />
                </>
              )}

              {/* Upload mode: show uploader or playback */}
              {mode === "video" && !videoFile && !isDemo && (
                <div className="absolute inset-0 flex items-center justify-center bg-white">
                  <VideoUploader onFileSelected={onFileSelected} />
                </div>
              )}

              {mode === "video" && videoUrl && !isDemo && (
                <>
                  <video
                    ref={videoPlaybackRef}
                    src={videoUrl}
                    className="w-full h-full object-contain"
                    playsInline
                    muted
                    autoPlay
                  />
                  <VideoControls videoRef={videoPlaybackRef} replaying={replaying} />
                </>
              )}

              {active && (
                <AnalysisCanvas
                  subjectsRef={subjectsRef}
                  selectedSubjectRef={selectedSubjectRef}
                  equipmentRef={equipmentRef}
                  onSelectSubject={selectSubject}
                  mirrored={mode === "webcam" && !isDemo}
                  showRisks={showDebug}
                />
              )}

              {/* Video progress overlay */}
              {mode === "video" && videoProgress !== null && !videoComplete && (
                <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent">
                  <ProgressBar
                    value={videoProgress * 100}
                    color="#1CB0F6"
                    height={8}
                  />
                </div>
              )}

              {/* Video complete overlay */}
              {videoComplete && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                  <div className="pop-in bg-white rounded-[16px] p-6 text-center">
                    <div className="text-2xl font-extrabold text-[#58CC02] mb-2">
                      Analysis Complete
                    </div>
                    <p className="text-sm text-[#777777]">
                      {activeTrackIds.length} subjects tracked &middot; {nSegments} motions in {nClusters} clusters
                    </p>
                  </div>
                </div>
              )}

              {/* Workout timeline at bottom of video */}
              {active && (isDemo || mode === "video") && (
                <div className="absolute bottom-0 left-0 right-0" style={{ zIndex: 15 }}>
                  <WorkoutTimeline
                    subjectsRef={subjectsRef}
                    selectedSubjectRef={selectedSubjectRef}
                    videoRef={isDemo ? demoVideoRef : videoPlaybackRef}
                    replaying={replaying}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Graph panels below video */}
          {active && (
            <div
              className="grid grid-cols-1 sm:grid-cols-2 gap-4 fade-up"
              style={{ "--stagger-index": 3 } as React.CSSProperties}
            >
              <EmbeddingGraph
                subjectsRef={subjectsRef}
                selectedSubjectRef={selectedSubjectRef}
                highlightedClusterRef={highlightedClusterRef}
                nSegments={nSegments}
                nClusters={nClusters}
              />
              <SkeletonGraph
                subjectsRef={subjectsRef}
                selectedSubjectRef={selectedSubjectRef}
              />
            </div>
          )}

          {/* Debug panel */}
          {active && showDebug && (
            <div
              className="fade-up"
              style={{ "--stagger-index": 4 } as React.CSSProperties}
            >
              <DebugPanel debugStatsRef={debugStatsRef} />
            </div>
          )}
        </div>

        {/* Status panel */}
        <div
          className="w-full lg:w-80 fade-up"
          style={{ "--stagger-index": 1 } as React.CSSProperties}
        >
          <StatusHUD
            phase={phase}
            clusterId={clusterId}
            clusterSummary={clusterSummary}
            connected={connected}
            activeTrackIds={activeTrackIds}
            selectedSubjectId={selectedSubjectId}
            subjectLabels={subjectLabels}
            subjectIdentities={subjectIdentities}
            onSelectSubject={selectSubject}
            highlightedClusterRef={highlightedClusterRef}
            subjectsRef={subjectsRef}
            selectedSubjectRef={selectedSubjectRef}
            fatigueIndex={fatigueIndex}
            peakVelocity={peakVelocity}
          />
          {active && (
            <VelocityGraph
              subjectsRef={subjectsRef}
              selectedSubjectRef={selectedSubjectRef}
              highlightedClusterRef={highlightedClusterRef}
              compact
            />
          )}
          {active && (
            <MovementQualityPanel
              subjectsRef={subjectsRef}
              selectedSubjectRef={selectedSubjectRef}
              showDebug={showDebug}
            />
          )}
          {active && (
            <RiskSummaryCard
              subjectsRef={subjectsRef}
              selectedSubjectRef={selectedSubjectRef}
              videoRef={demoVideoUrl ? demoVideoRef : videoPlaybackRef}
              replaying={replaying}
            />
          )}
        </div>
      </div>

      {/* Bottom controls */}
      <footer className="flex items-center justify-center gap-4 px-5 py-4 bg-white border-t-2 border-[#E5E5E5]">
        <DuoButton
          variant={mode === "webcam" && !isDemo ? "blue" : "secondary"}
          onClick={() => switchMode("webcam")}
        >
          Webcam
        </DuoButton>
        <DuoButton
          variant={mode === "video" && !isDemo ? "blue" : "secondary"}
          onClick={() => switchMode("video")}
        >
          Upload
        </DuoButton>
        <DuoButton
          variant={isDemo ? "blue" : "secondary"}
          onClick={() => setShowDemoModal(true)}
        >
          Demo
        </DuoButton>
        {active && (
          <DuoButton
            variant={showDebug ? "blue" : "secondary"}
            onClick={() => setShowDebug((v) => !v)}
          >
            Debug
          </DuoButton>
        )}
        {active && (
          <DuoButton variant="danger" onClick={handleStop}>
            Stop
          </DuoButton>
        )}
      </footer>

      {/* Demo video modal */}
      {showDemoModal && (
        <DemoVideoModal
          onSelect={onDemoSelect}
          onClose={() => setShowDemoModal(false)}
        />
      )}
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center">
          <div className="skeleton w-64 h-8 rounded-[12px]" />
        </div>
      }
    >
      <AnalyzeContent />
    </Suspense>
  );
}
