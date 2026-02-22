"use client";

import type { CrashCollisionEvent } from "@/lib/types";

const RISK_COLORS: Record<string, string> = {
  LOW: "bg-green-50 text-green-700 border-green-200",
  MODERATE: "bg-yellow-50 text-yellow-700 border-yellow-200",
  HIGH: "bg-orange-50 text-orange-700 border-orange-200",
  CRITICAL: "bg-red-50 text-red-700 border-red-200",
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
    <div className="rounded-xl border border-[#E5E5E5] bg-white p-4 space-y-3">
      {/* Header: risk badge + time */}
      <div className="flex items-center justify-between">
        <span
          className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border ${riskClass}`}
        >
          {event.risk_level}
        </span>
        <span className="text-xs text-[#AFAFAF]">
          {(event.video_time ?? 0).toFixed(1)}s (frame {event.frame_index ?? 0})
        </span>
      </div>

      {/* Subjects */}
      <div className="text-sm text-[#777777]">
        Subject {event.subject_a} &harr; Subject {event.subject_b}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <div className="text-[#AFAFAF] text-xs">Concussion Prob</div>
          <div className="font-mono font-semibold text-[#3C3C3C]">{probPct}%</div>
        </div>
        <div>
          <div className="text-[#AFAFAF] text-xs">Contact Zone</div>
          <div className="font-medium text-[#3C3C3C]">{zone}</div>
        </div>
        <div>
          <div className="text-[#AFAFAF] text-xs">Closing Speed</div>
          <div className="font-mono text-[#3C3C3C]">
            {(event.closing_speed_ms ?? 0).toFixed(1)} m/s
          </div>
        </div>
        <div>
          <div className="text-[#AFAFAF] text-xs">Peak G-force</div>
          <div className="font-mono text-[#3C3C3C]">
            {(event.peak_linear_g ?? 0).toFixed(1)}g
          </div>
        </div>
        <div>
          <div className="text-[#AFAFAF] text-xs">HIC</div>
          <div className="font-mono text-[#3C3C3C]">{(event.hic ?? 0).toFixed(0)}</div>
        </div>
        <div>
          <div className="text-[#AFAFAF] text-xs">Rotational Accel</div>
          <div className="font-mono text-[#3C3C3C]">
            {(event.peak_rotational_rads2 ?? 0).toFixed(0)} rad/s&sup2;
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="text-xs text-[#777777] italic">{event.recommendation}</div>
    </div>
  );
}
