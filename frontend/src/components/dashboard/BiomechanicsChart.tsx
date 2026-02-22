"use client";

import { useMemo } from "react";

interface DataPoint {
  video_time: number;
  value: number;
}

interface BiomechanicsChartProps {
  title: string;
  data: DataPoint[];
  thresholdMedium?: number;
  thresholdHigh?: number;
  unit?: string;
  color?: string;
  height?: number;
}

export default function BiomechanicsChart({
  title,
  data,
  thresholdMedium,
  thresholdHigh,
  unit = "",
  color = "#4FC3F7",
  height = 120,
}: BiomechanicsChartProps) {
  const { points, maxVal, maxTime } = useMemo(() => {
    if (data.length === 0) return { points: "", maxVal: 1, maxTime: 1 };
    const maxV = Math.max(...data.map((d) => d.value), thresholdHigh ?? 0, 1);
    const maxT = Math.max(...data.map((d) => d.video_time), 1);
    const w = 400;
    const h = height;
    const pts = data
      .map((d) => {
        const x = (d.video_time / maxT) * w;
        const y = h - (d.value / maxV) * h;
        return `${x},${y}`;
      })
      .join(" ");
    return { points: pts, maxVal: maxV, maxTime: maxT };
  }, [data, thresholdHigh, height]);

  if (data.length === 0) {
    return (
      <div className="bg-[#F7F7F7] rounded-lg p-3 border border-[#E5E5E5]">
        <p className="text-xs font-bold text-[#3C3C3C] mb-1">{title}</p>
        <p className="text-xs text-[#AFAFAF]">No data</p>
      </div>
    );
  }

  return (
    <div className="bg-[#F7F7F7] rounded-lg p-3 border border-[#E5E5E5]">
      <p className="text-xs font-bold text-[#3C3C3C] mb-2">
        {title}
        {unit && <span className="font-normal text-[#AFAFAF] ml-1">({unit})</span>}
      </p>
      <svg viewBox={`0 0 400 ${height}`} className="w-full" preserveAspectRatio="none">
        {/* Threshold lines */}
        {thresholdMedium != null && (
          <line
            x1={0}
            y1={height - (thresholdMedium / maxVal) * height}
            x2={400}
            y2={height - (thresholdMedium / maxVal) * height}
            stroke="#FFB74D"
            strokeDasharray="4 4"
            strokeWidth={1}
          />
        )}
        {thresholdHigh != null && (
          <line
            x1={0}
            y1={height - (thresholdHigh / maxVal) * height}
            x2={400}
            y2={height - (thresholdHigh / maxVal) * height}
            stroke="#EF5350"
            strokeDasharray="4 4"
            strokeWidth={1}
          />
        )}
        {/* Data line */}
        <polyline
          fill="none"
          stroke={color}
          strokeWidth={2}
          points={points}
        />
      </svg>
    </div>
  );
}
