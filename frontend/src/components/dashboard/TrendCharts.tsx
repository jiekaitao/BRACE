"use client";

import { useEffect, useState } from "react";
import { fetchTrends, fetchInjuryRiskTrends, type TrendPoint, type InjuryRiskTrendPoint } from "@/lib/dashboard";

interface TrendChartsProps {
  userId: string;
}

export default function TrendCharts({ userId }: TrendChartsProps) {
  const [fatigueTrend, setFatigueTrend] = useState<TrendPoint[]>([]);
  const [riskTrend, setRiskTrend] = useState<InjuryRiskTrendPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      fetchTrends(userId, "fatigue_score"),
      fetchInjuryRiskTrends(userId),
    ]).then(([fatigue, risks]) => {
      if (!cancelled) {
        setFatigueTrend(fatigue);
        setRiskTrend(risks);
        setLoading(false);
      }
    }).catch(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [userId]);

  if (loading) {
    return <p className="text-xs text-[#AFAFAF]">Loading trends...</p>;
  }

  const hasFatigue = fatigueTrend.length > 0;
  const hasRisks = riskTrend.length > 0;

  if (!hasFatigue && !hasRisks) {
    return (
      <div className="bg-[#F7F7F7] rounded-xl p-4 border border-[#E5E5E5]">
        <p className="text-xs font-bold text-[#3C3C3C] mb-1">Trends</p>
        <p className="text-xs text-[#AFAFAF]">
          Complete more workouts to see trends.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {hasFatigue && (
        <div className="bg-[#F7F7F7] rounded-xl p-4 border border-[#E5E5E5]">
          <p className="text-xs font-bold text-[#3C3C3C] mb-2">Fatigue Trend</p>
          <div className="flex items-end gap-1 h-16">
            {fatigueTrend.map((pt, i) => {
              const maxVal = Math.max(...fatigueTrend.map((p) => p.value), 1);
              const h = (pt.value / maxVal) * 100;
              return (
                <div
                  key={i}
                  className="flex-1 bg-[#4FC3F7] rounded-t"
                  style={{ height: `${h}%`, minHeight: 2 }}
                  title={`${pt.date}: ${pt.value.toFixed(2)}`}
                />
              );
            })}
          </div>
        </div>
      )}

      {hasRisks && (
        <div className="bg-[#F7F7F7] rounded-xl p-4 border border-[#E5E5E5]">
          <p className="text-xs font-bold text-[#3C3C3C] mb-2">Injury Risk Frequency</p>
          <div className="flex items-end gap-1 h-16">
            {riskTrend.map((pt, i) => {
              const maxVal = Math.max(...riskTrend.map((p) => p.total_risks), 1);
              const h = (pt.total_risks / maxVal) * 100;
              const color = pt.total_risks > 5 ? "#EF5350" : pt.total_risks > 2 ? "#FFB74D" : "#66BB6A";
              return (
                <div
                  key={i}
                  className="flex-1 rounded-t"
                  style={{ height: `${h}%`, minHeight: 2, backgroundColor: color }}
                  title={`${pt.date}: ${pt.total_risks} risks`}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
