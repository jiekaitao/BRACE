"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { getApiBase } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { InjuryProfile } from "@/lib/types";
import Card from "@/components/ui/Card";

interface Guideline {
  injury_type: string;
  metric: string;
  title: string;
  explanation: string;
  precautions: string[];
}

interface ResearchResult {
  guidelines: Guideline[];
  summary: string;
}

const INJURY_LABELS: Record<string, string> = {
  acl: "ACL",
  shoulder: "Shoulder",
  ankle: "Ankle",
  lower_back: "Lower Back",
  knee_general: "Knee",
  hip: "Hip",
  hamstring: "Hamstring",
  wrist: "Wrist",
};

const METRIC_COLORS: Record<string, string> = {
  FPPA: "#1CB0F6",
  "Hip Drop": "#FF9600",
  "Trunk Lean": "#CE82FF",
  "Bilateral Asymmetry": "#58CC02",
  "Angular Velocity": "#EA2B2B",
};

export default function GeminiResearchPanel() {
  const { user, refreshUser } = useAuth();
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);

  // Load saved guidelines from user profile
  useEffect(() => {
    if (user?.research_guidelines) {
      setResult(user.research_guidelines as unknown as ResearchResult);
    }
  }, [user?.research_guidelines]);

  const generateGuidelines = async () => {
    if (!user?.injury_profile) return;
    setGenerating(true);

    try {
      const token = getToken();
      const res = await fetch(`${getApiBase()}/api/chat/research`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ injury_profile: user.injury_profile }),
      });

      if (!res.ok) throw new Error("Failed");
      const data: ResearchResult = await res.json();
      setResult(data);

      // Save to profile
      if (user.user_id) {
        await fetch(`${getApiBase()}/api/chat/save-research`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            user_id: user.user_id,
            research_guidelines: data,
          }),
        });
        refreshUser();
      }
    } catch {
      // silent fail
    } finally {
      setGenerating(false);
    }
  };

  // No saved guidelines — show generate option
  if (!result) {
    return (
      <Card>
        <div className="flex items-center gap-2 mb-3">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="flex-shrink-0">
            <path
              d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z"
              fill="#CE82FF"
            />
          </svg>
          <h3 className="text-base font-extrabold text-[#3C3C3C]">
            Research & Recommendations
          </h3>
        </div>
        {generating ? (
          <p className="text-sm text-[#777777]">Generating guidelines...</p>
        ) : (
          <>
            <p className="text-sm text-[#777777] mb-3">
              Get personalized monitoring guidelines based on your injury profile.
            </p>
            <button
              onClick={generateGuidelines}
              className="text-sm font-bold text-[#CE82FF] hover:text-[#B060E0] transition-colors"
            >
              Generate Guidelines
            </button>
          </>
        )}
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="flex-shrink-0">
            <path
              d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z"
              fill="#CE82FF"
            />
          </svg>
          <h3 className="text-sm font-extrabold text-[#3C3C3C]">
            Your Guidelines
          </h3>
        </div>
        <button
          onClick={generateGuidelines}
          disabled={generating}
          className="text-[10px] font-bold text-[#AFAFAF] hover:text-[#CE82FF] transition-colors"
        >
          {generating ? "..." : "Regenerate"}
        </button>
      </div>

      <div className="flex flex-col gap-3">
        {result.guidelines.map((g, i) => (
          <div key={i} className="border-l-2 border-[#E5E5E5] pl-3">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-[#F7F7F7]">
                {INJURY_LABELS[g.injury_type] ?? g.injury_type}
              </span>
              <span
                className="text-[9px] font-bold px-1.5 py-0.5 rounded text-white"
                style={{ backgroundColor: METRIC_COLORS[g.metric] ?? "#777777" }}
              >
                {g.metric}
              </span>
            </div>
            <p className="text-[11px] font-bold text-[#3C3C3C] mb-0.5">{g.title}</p>
            <p className="text-[10px] text-[#777777] leading-relaxed">{g.explanation}</p>
          </div>
        ))}
      </div>

      {result.summary && (
        <p className="text-[10px] text-[#AFAFAF] mt-3 leading-relaxed">
          {result.summary}
        </p>
      )}
    </Card>
  );
}
