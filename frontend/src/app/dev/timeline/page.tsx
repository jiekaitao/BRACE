"use client";

import { useRef, useMemo } from "react";
import Card from "@/components/ui/Card";
import RiskSummaryCard from "@/components/RiskSummaryCard";
import type { SubjectState, ReplaySnapshot, LandmarkFrame, EmbeddingState, VelocityState, SmplFrame, FrameQuality, InjuryRisk } from "@/lib/types";

// Generate mock replay timeline with risk events
function createMockTimeline(): ReplaySnapshot[] {
  const snapshots: ReplaySnapshot[] = [];
  for (let t = 0; t < 60; t += 0.5) {
    const risks: InjuryRisk[] = [];

    // Simulate ACL risk at 10-15s
    if (t >= 10 && t <= 15) {
      risks.push({ joint: "left_knee", risk: "acl_valgus", severity: "medium", value: 18, threshold: 15 });
    }
    // High ACL risk at 25-28s
    if (t >= 25 && t <= 28) {
      risks.push({ joint: "right_knee", risk: "acl_valgus", severity: "high", value: 28, threshold: 25 });
    }
    // Hip drop at 35-40s
    if (t >= 35 && t <= 40) {
      risks.push({ joint: "pelvis", risk: "hip_drop", severity: "medium", value: 10, threshold: 8 });
    }
    // Trunk lean at 45-48s
    if (t >= 45 && t <= 48) {
      risks.push({ joint: "trunk", risk: "trunk_lean", severity: "high", value: 30, threshold: 25 });
    }

    const quality: FrameQuality = { injury_risks: risks };

    snapshots.push({
      t,
      quality,
      clusterId: Math.floor(t / 15) % 3,
      consistencyScore: 0.85 + Math.random() * 0.1,
      isAnomaly: false,
      phase: "normal",
      nSegments: 4,
      nClusters: 3,
      clusterSummary: {},
    });
  }
  return snapshots;
}

export default function DevTimelinePage() {
  const mockTimeline = useMemo(() => createMockTimeline(), []);

  const emptyLandmarkFrame: LandmarkFrame = { prev: null, current: null, prevTime: 0, currentTime: 0, prevVideoTime: 0, currentVideoTime: 0 };
  const emptyEmbedding: EmbeddingState = { points: [], clusterIds: [], currentIdx: 0 };
  const emptyVelocity: VelocityState = { values: [], rolling: [], timestamps: [], fatigueIndex: 0.3, peakVelocity: 5 };
  const emptySmplFrame: SmplFrame = { prev: null, current: null, prevTime: 0, currentTime: 0 };

  // Create mock subject state
  const mockSubject: SubjectState = useMemo(() => ({
    trackId: 1,
    label: "S1",
    landmarkFrame: emptyLandmarkFrame,
    bbox: null,
    phase: "normal",
    nSegments: 4,
    nClusters: 3,
    clusterId: 0,
    consistencyScore: 0.9,
    isAnomaly: false,
    clusterSummary: {},
    srpJoints: null,
    jointVisibility: null,
    representativeJoints: null,
    embedding: emptyEmbedding,
    velocity: emptyVelocity,
    identityStatus: "confirmed",
    identityConfidence: 0.95,
    smplFrame: emptySmplFrame,
    uvTexture: null,
    clusterRepresentatives: {},
    quality: {},
    replayTimeline: mockTimeline,
    firstPassVelocityLen: 0,
    lastSeenTime: Date.now(),
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [mockTimeline]);

  const subjectsRef = useRef(new Map<number, SubjectState>([[1, mockSubject]]));
  const selectedSubjectRef = useRef<number | null>(1);
  const videoRef = useRef<HTMLVideoElement>(null);

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Timeline Test</h1>

      <Card className="mb-4">
        <h3 className="text-sm font-bold text-[#3C3C3C] mb-2">Mock Scenario</h3>
        <p className="text-xs text-[#777777]">
          60s workout with 4 risk events: ACL risk at 10-15s (medium), ACL risk at 25-28s (high),
          Hip drop at 35-40s (medium), Trunk lean at 45-48s (high).
        </p>
      </Card>

      <RiskSummaryCard
        subjectsRef={subjectsRef}
        selectedSubjectRef={selectedSubjectRef}
        videoRef={videoRef}
        replaying={true}
      />
    </div>
  );
}
