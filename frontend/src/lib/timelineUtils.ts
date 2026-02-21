import type { ReplaySnapshot, ClusterSegment, RiskSegment, TimelineData, SubjectState } from "./types";

const GAP_TOLERANCE = 0.5; // merge segments with gaps smaller than this (seconds)

/**
 * Extract contiguous cluster segments from a replay timeline.
 * Merges adjacent segments with the same clusterId when the gap is < GAP_TOLERANCE.
 */
export function extractClusterSegments(timeline: ReplaySnapshot[]): ClusterSegment[] {
  if (timeline.length === 0) return [];

  const segments: ClusterSegment[] = [];
  let currentId: number | null = null;
  let startTime = 0;
  let endTime = 0;
  let activityLabel: string | undefined;
  let guidelineName: string | undefined;

  for (const snap of timeline) {
    if (snap.clusterId === null) {
      // Calibration phase — flush any open segment
      if (currentId !== null) {
        segments.push({ clusterId: currentId, startTime, endTime, activityLabel, guidelineName });
        currentId = null;
      }
      continue;
    }

    if (snap.clusterId === currentId && snap.t - endTime < GAP_TOLERANCE) {
      // Extend current segment
      endTime = snap.t;
    } else {
      // Flush previous segment
      if (currentId !== null) {
        segments.push({ clusterId: currentId, startTime, endTime, activityLabel, guidelineName });
      }
      // Start new segment
      currentId = snap.clusterId;
      startTime = snap.t;
      endTime = snap.t;
      // Pull activity label from cluster summary
      const info = snap.clusterSummary[String(snap.clusterId)];
      activityLabel = info?.activity_label;
      // Pull guideline name from quality data
      const gl = snap.quality?.active_guideline;
      guidelineName = gl && gl.name !== "generic" ? gl.display_name : undefined;
    }
  }

  // Flush last segment
  if (currentId !== null) {
    segments.push({ clusterId: currentId, startTime, endTime, activityLabel, guidelineName });
  }

  return segments;
}

/**
 * Extract contiguous risk segments from a replay timeline.
 * Groups by riskType, merges gaps < GAP_TOLERANCE, keeps highest severity.
 */
export function extractRiskSegments(timeline: ReplaySnapshot[]): RiskSegment[] {
  if (timeline.length === 0) return [];

  // Track active risks: key = riskType, value = current open segment
  const active = new Map<string, { severity: "medium" | "high"; startTime: number; endTime: number; joint: string }>();
  const segments: RiskSegment[] = [];

  for (const snap of timeline) {
    const risks = snap.quality?.injury_risks;
    const frameRiskTypes = new Set<string>();

    if (risks) {
      for (const risk of risks) {
        // Only track medium and high severity
        if (risk.severity !== "medium" && risk.severity !== "high") continue;

        frameRiskTypes.add(risk.risk);

        const existing = active.get(risk.risk);
        if (existing && snap.t - existing.endTime < GAP_TOLERANCE) {
          // Extend and keep highest severity
          existing.endTime = snap.t;
          if (risk.severity === "high") existing.severity = "high";
        } else {
          // Flush previous segment for this risk type
          if (existing) {
            segments.push({ riskType: risk.risk, ...existing });
          }
          active.set(risk.risk, {
            severity: risk.severity as "medium" | "high",
            startTime: snap.t,
            endTime: snap.t,
            joint: risk.joint,
          });
        }
      }
    }

    // Flush any active risks not present in this frame (with gap tolerance)
    for (const [riskType, seg] of active) {
      if (!frameRiskTypes.has(riskType) && snap.t - seg.endTime >= GAP_TOLERANCE) {
        segments.push({ riskType, ...seg });
        active.delete(riskType);
      }
    }
  }

  // Flush remaining active segments
  for (const [riskType, seg] of active) {
    segments.push({ riskType, ...seg });
  }

  // Sort by start time
  segments.sort((a, b) => a.startTime - b.startTime);
  return segments;
}

// Memoization cache
let cachedKey = "";
let cachedData: TimelineData | null = null;

/**
 * Build complete timeline data for a subject.
 * Memoized by timeline length + subject trackId.
 */
export function buildTimelineData(subject: SubjectState): TimelineData | null {
  const tl = subject.replayTimeline;
  if (tl.length < 2) return null;

  const key = `${subject.trackId}:${tl.length}`;
  if (key === cachedKey && cachedData) return cachedData;

  const duration = tl[tl.length - 1].t;
  if (duration <= 0) return null;

  const clusterSegments = extractClusterSegments(tl);
  const riskSegments = extractRiskSegments(tl);

  cachedData = { duration, clusterSegments, riskSegments };
  cachedKey = key;
  return cachedData;
}
