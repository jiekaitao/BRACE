"use client";

interface ConsistencyBarProps {
  score: number | null; // Raw RMS score (lower = more consistent)
  maxScore?: number;
}

export default function ConsistencyBar({ score, maxScore = 3.0 }: ConsistencyBarProps) {
  if (score === null) {
    return (
      <div>
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs font-bold text-[#4B4B4B] uppercase tracking-[0.03em]">
            Consistency
          </span>
          <span className="text-xs font-bold text-[#AFAFAF]">--</span>
        </div>
        <div className="bg-[#E5E5E5] rounded-full h-4 w-full" />
      </div>
    );
  }

  // Convert RMS score to percentage (lower score = better consistency = higher bar)
  const pct = Math.max(0, Math.min(100, (1 - score / maxScore) * 100));

  // Color shifts green → orange → red
  let color: string;
  if (pct >= 70) {
    color = "#58CC02"; // green
  } else if (pct >= 40) {
    color = "#FF9600"; // orange
  } else {
    color = "#EA2B2B"; // red
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-bold text-[#4B4B4B] uppercase tracking-[0.03em]">
          Consistency
        </span>
        <span className="text-xs font-bold" style={{ color }}>
          {Math.round(pct)}%
        </span>
      </div>
      <div className="bg-[#E5E5E5] rounded-full h-4 w-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            backgroundColor: color,
            width: `${pct}%`,
            transition: "width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1), background-color 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}
