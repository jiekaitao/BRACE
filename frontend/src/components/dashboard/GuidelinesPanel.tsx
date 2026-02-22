"use client";

import { useEffect, useState } from "react";
import { fetchGuidelines, type Guideline } from "@/lib/dashboard";

interface GuidelinesPanelProps {
  userId: string;
  activity?: string;
}

export default function GuidelinesPanel({ userId, activity }: GuidelinesPanelProps) {
  const [guidelines, setGuidelines] = useState<Guideline[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchGuidelines(userId, activity).then((data) => {
      if (!cancelled) {
        setGuidelines(data);
        setLoading(false);
      }
    }).catch(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [userId, activity]);

  if (loading) {
    return (
      <div className="bg-[#F7F7F7] rounded-xl p-4 border border-[#E5E5E5]">
        <p className="text-xs text-[#AFAFAF]">Loading guidelines...</p>
      </div>
    );
  }

  if (guidelines.length === 0) {
    return (
      <div className="bg-[#F7F7F7] rounded-xl p-4 border border-[#E5E5E5]">
        <p className="text-xs font-bold text-[#3C3C3C] mb-1">Movement Guidelines</p>
        <p className="text-xs text-[#AFAFAF]">
          No personalized guidelines yet. Complete a workout to generate guidelines.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-[#F7F7F7] rounded-xl p-4 border border-[#E5E5E5]">
      <p className="text-xs font-bold text-[#3C3C3C] mb-3 uppercase tracking-wider">
        Movement Guidelines
      </p>
      <div className="flex flex-col gap-3">
        {guidelines.map((g) => (
          <div key={g._id} className="bg-white rounded-lg p-3 border border-[#E5E5E5]">
            <p className="text-sm font-semibold text-[#3C3C3C] mb-1 capitalize">
              {g.activity}
            </p>
            {g.injury_context && (
              <p className="text-xs text-[#FF8A65] mb-2">
                Context: {g.injury_context}
              </p>
            )}
            <ul className="list-disc list-inside space-y-1">
              {g.guidelines.map((tip, i) => (
                <li key={i} className="text-xs text-[#777777]">{tip}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
