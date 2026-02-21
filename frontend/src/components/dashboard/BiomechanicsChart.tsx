"use client";

import Card from "@/components/ui/Card";

interface TimePoint {
  t: number;
  fppa_left: number;
  fppa_right: number;
  hip_drop: number;
  trunk_lean: number;
  asymmetry: number;
}

const LINES: { key: keyof TimePoint; color: string; label: string }[] = [
  { key: "fppa_left", color: "#1CB0F6", label: "FPPA L" },
  { key: "fppa_right", color: "#CE82FF", label: "FPPA R" },
  { key: "hip_drop", color: "#FF9600", label: "Hip Drop" },
  { key: "trunk_lean", color: "#EA2B2B", label: "Trunk Lean" },
];

interface Props {
  data: TimePoint[];
  width?: number;
  height?: number;
}

export default function BiomechanicsChart({
  data,
  width = 400,
  height = 120,
}: Props) {
  if (data.length < 2) {
    return (
      <Card>
        <h3 className="text-base font-extrabold text-[#3C3C3C] mb-2">Biomechanics</h3>
        <p className="text-xs text-[#AFAFAF]">Not enough data points</p>
      </Card>
    );
  }

  const pad = 6;
  const tMin = data[0].t;
  const tMax = data[data.length - 1].t;
  const tRange = Math.max(tMax - tMin, 0.1);

  // Find value range across all lines
  let vMin = Infinity;
  let vMax = -Infinity;
  for (const pt of data) {
    for (const line of LINES) {
      const v = pt[line.key] as number;
      if (v < vMin) vMin = v;
      if (v > vMax) vMax = v;
    }
  }
  const vRange = Math.max(vMax - vMin, 1);

  function x(t: number) {
    return pad + ((t - tMin) / tRange) * (width - 2 * pad);
  }
  function y(v: number) {
    return height - pad - ((v - vMin) / vRange) * (height - 2 * pad);
  }

  return (
    <Card>
      <h3 className="text-base font-extrabold text-[#3C3C3C] mb-2">Biomechanics Timeline</h3>
      <svg
        width={width}
        height={height}
        className="w-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
      >
        {/* Zero line */}
        {vMin <= 0 && vMax >= 0 && (
          <line
            x1={pad}
            x2={width - pad}
            y1={y(0)}
            y2={y(0)}
            stroke="#E5E5E5"
            strokeWidth="1"
            strokeDasharray="4 2"
          />
        )}
        {LINES.map((line) => {
          const path = data
            .map((pt, i) => {
              const px = x(pt.t);
              const py = y(pt[line.key] as number);
              return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
            })
            .join(" ");
          return (
            <path
              key={line.key}
              d={path}
              fill="none"
              stroke={line.color}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity={0.8}
            />
          );
        })}
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-2">
        {LINES.map((line) => (
          <div key={line.key} className="flex items-center gap-1">
            <div
              className="w-2.5 h-0.5 rounded-full"
              style={{ backgroundColor: line.color }}
            />
            <span className="text-[11px] text-[#777777]">{line.label}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
