import type { ReplaySnapshot, InjuryRisk } from "./types";

export interface RiskMoment {
  riskType: string;
  joint: string;
  severity: "medium" | "high";
  startTime: number;
  endTime: number;
  peakValue: number;
  threshold: number;
  duration: number; // endTime - startTime
}

/**
 * Aggregate consecutive frames with the same risk type into RiskMoments.
 * Sorts by severity (high first) then duration (longest first).
 */
export function buildRiskMoments(
  timeline: ReplaySnapshot[],
  minDurationSec: number = 0.3,
): RiskMoment[] {
  if (!timeline.length) return [];

  const moments: RiskMoment[] = [];

  // Track open risk segments: key is `${riskType}|${joint}`
  const open = new Map<
    string,
    {
      riskType: string;
      joint: string;
      severity: "medium" | "high";
      startTime: number;
      endTime: number;
      peakValue: number;
      threshold: number;
    }
  >();

  for (const snap of timeline) {
    const risks = snap.quality?.injury_risks ?? [];
    const activeKeys = new Set<string>();

    for (const risk of risks) {
      if (risk.severity !== "medium" && risk.severity !== "high") continue;

      const key = `${risk.risk}|${risk.joint}`;
      activeKeys.add(key);

      const existing = open.get(key);
      if (existing) {
        existing.endTime = snap.t;
        existing.peakValue = Math.max(existing.peakValue, risk.value);
        // Upgrade severity if needed
        if (risk.severity === "high") existing.severity = "high";
      } else {
        open.set(key, {
          riskType: risk.risk,
          joint: risk.joint,
          severity: risk.severity as "medium" | "high",
          startTime: snap.t,
          endTime: snap.t,
          peakValue: risk.value,
          threshold: risk.threshold,
        });
      }
    }

    // Close segments that are no longer active
    for (const [key, seg] of open) {
      if (!activeKeys.has(key)) {
        const duration = seg.endTime - seg.startTime;
        if (duration >= minDurationSec) {
          moments.push({ ...seg, duration });
        }
        open.delete(key);
      }
    }
  }

  // Close remaining open segments
  for (const seg of open.values()) {
    const duration = seg.endTime - seg.startTime;
    if (duration >= minDurationSec) {
      moments.push({ ...seg, duration });
    }
  }

  // Sort: high severity first, then by duration descending
  moments.sort((a, b) => {
    if (a.severity !== b.severity) return a.severity === "high" ? -1 : 1;
    return b.duration - a.duration;
  });

  return moments;
}

/**
 * Get top N risk moments for the summary card.
 */
export function getTopRiskMoments(
  timeline: ReplaySnapshot[],
  maxCount: number = 5,
): RiskMoment[] {
  return buildRiskMoments(timeline).slice(0, maxCount);
}
