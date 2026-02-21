"use client";

import type { TrendData } from "@/lib/types";
import Card from "@/components/ui/Card";

interface SparklineProps {
  values: (number | null)[];
  color: string;
  height?: number;
  width?: number;
  min?: number;
  max?: number;
  label: string;
  formatValue?: (v: number) => string;
}

function Sparkline({
  values,
  color,
  height = 48,
  width = 200,
  min: forceMin,
  max: forceMax,
  label,
  formatValue = (v) => v.toFixed(1),
}: SparklineProps) {
  const valid = values.filter((v): v is number => v !== null);
  if (valid.length < 2) {
    return (
      <div className="text-xs text-[#AFAFAF]">Not enough data</div>
    );
  }

  const pad = 4;
  const mn = forceMin ?? Math.min(...valid);
  const mx = forceMax ?? Math.max(...valid);
  const range = Math.max(mx - mn, 0.01);

  const points = values
    .map((v, i) => {
      if (v === null) return null;
      const x = pad + (i / (values.length - 1)) * (width - 2 * pad);
      const y = height - pad - ((v - mn) / range) * (height - 2 * pad);
      return { x, y, v };
    })
    .filter(Boolean) as { x: number; y: number; v: number }[];

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(" ");

  const last = points[points.length - 1];

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-bold text-[#777777]">{label}</span>
        {last && (
          <span className="text-xs font-bold" style={{ color }}>
            {formatValue(last.v)}
          </span>
        )}
      </div>
      <svg width={width} height={height} className="w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        {last && (
          <circle cx={last.x} cy={last.y} r="3" fill={color} />
        )}
      </svg>
    </div>
  );
}

interface BarChartProps {
  values: number[];
  labels: string[];
  color: string;
  height?: number;
  width?: number;
  label: string;
}

function BarChart({
  values,
  labels,
  color,
  height = 48,
  width = 200,
  label,
}: BarChartProps) {
  if (values.length === 0) {
    return <div className="text-xs text-[#AFAFAF]">No data</div>;
  }

  const mx = Math.max(...values, 1);
  const barW = Math.max(4, (width - 8) / values.length - 2);

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-bold text-[#777777]">{label}</span>
        <span className="text-xs font-bold" style={{ color }}>
          {values.reduce((a, b) => a + b, 0)} total
        </span>
      </div>
      <svg width={width} height={height} className="w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {values.map((v, i) => {
          const bh = (v / mx) * (height - 8);
          const x = 4 + i * ((width - 8) / values.length);
          return (
            <rect
              key={i}
              x={x}
              y={height - 4 - bh}
              width={barW}
              height={bh}
              rx={2}
              fill={color}
              opacity={0.7}
            >
              <title>{`${labels[i]}: ${v}`}</title>
            </rect>
          );
        })}
      </svg>
    </div>
  );
}

interface Props {
  trends: TrendData;
}

export default function TrendCharts({ trends }: Props) {
  const dateLabels = trends.dates.map((d) => {
    const date = new Date(d);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  });

  return (
    <Card>
      <h3 className="text-base font-extrabold text-[#3C3C3C] mb-3">Trends</h3>
      <div className="flex flex-col gap-4">
        <Sparkline
          values={trends.form_scores}
          color="#58CC02"
          label="Form Score"
          min={0}
          max={100}
          formatValue={(v) => Math.round(v).toString()}
        />
        <Sparkline
          values={trends.fatigue_scores}
          color="#FF9600"
          label="Fatigue"
          min={0}
          max={1}
        />
        <BarChart
          values={trends.injury_counts}
          labels={dateLabels}
          color="#EA2B2B"
          label="Injury Risks"
        />
      </div>
    </Card>
  );
}
