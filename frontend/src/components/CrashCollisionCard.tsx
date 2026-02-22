"use client";

import type { CrashCollisionEvent } from "@/lib/types";

const RISK_COLORS: Record<string, string> = {
  LOW: "bg-green-500/20 text-green-300 border-green-500/40",
  MODERATE: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
  HIGH: "bg-orange-500/20 text-orange-300 border-orange-500/40",
  CRITICAL: "bg-red-500/20 text-red-300 border-red-500/40",
};

const ZONE_LABELS: Record<string, string> = {
  head_to_head: "Head-to-Head",
  head_to_shoulder: "Head-to-Shoulder",
  head_to_torso: "Head-to-Torso",
  head_to_limb: "Head-to-Limb",
};

export default function CrashCollisionCard({
  event,
}: {
  event: CrashCollisionEvent;
}) {
  const riskClass = RISK_COLORS[event.risk_level] ?? RISK_COLORS.LOW;
  const probPct = ((event.concussion_probability ?? 0) * 100).toFixed(1);
  const zone = ZONE_LABELS[event.contact_zone] ?? event.contact_zone ?? "Unknown";

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 space-y-3">
      {/* Header: risk badge + time */}
      <div className="flex items-center justify-between">
        <span
          className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border ${riskClass}`}
        >
          {event.risk_level}
        </span>
        <span className="text-xs text-white/50">
          {(event.video_time ?? 0).toFixed(1)}s (frame {event.frame_index ?? 0})
        </span>
      </div>

      {/* Subjects */}
      <div className="text-sm text-white/70">
        Subject {event.subject_a} &harr; Subject {event.subject_b}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <div className="text-white/40 text-xs">Concussion Prob</div>
          <div className="font-mono font-semibold text-white">{probPct}%</div>
        </div>
        <div>
          <div className="text-white/40 text-xs">Contact Zone</div>
          <div className="font-medium text-white">{zone}</div>
        </div>
        <div>
          <div className="text-white/40 text-xs">Closing Speed</div>
          <div className="font-mono text-white">
            {(event.closing_speed_ms ?? 0).toFixed(1)} m/s
          </div>
        </div>
        <div>
          <div className="text-white/40 text-xs">Peak G-force</div>
          <div className="font-mono text-white">
            {(event.peak_linear_g ?? 0).toFixed(1)}g
          </div>
        </div>
        <div>
          <div className="text-white/40 text-xs">HIC</div>
          <div className="font-mono text-white">{(event.hic ?? 0).toFixed(0)}</div>
        </div>
        <div>
          <div className="text-white/40 text-xs">Rotational Accel</div>
          <div className="font-mono text-white">
            {(event.peak_rotational_rads2 ?? 0).toFixed(0)} rad/s&sup2;
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="text-xs text-white/60 italic">{event.recommendation}</div>
    </div>
  );
}
