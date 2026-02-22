"use client";

import { useEffect, useState } from "react";
import { getApiBase } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { InjuryProfile } from "@/lib/types";
import Card from "@/components/ui/Card";
import DuoButton from "@/components/ui/DuoButton";

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

interface Props {
  injuryProfile: InjuryProfile;
  userId?: string;
  onContinue: () => void;
}

export default function ResearchStep({ injuryProfile, userId, onContinue }: Props) {
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchResearch() {
      setLoading(true);
      setError(null);

      try {
        const token = getToken();
        const res = await fetch(`${getApiBase()}/api/chat/research`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({ injury_profile: injuryProfile }),
        });

        if (!res.ok) throw new Error("Failed to generate guidelines");

        const data: ResearchResult = await res.json();
        if (cancelled) return;
        setResult(data);

        // Save to user profile
        if (userId) {
          fetch(`${getApiBase()}/api/chat/save-research`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: JSON.stringify({
              user_id: userId,
              research_guidelines: data,
            }),
          }).catch(() => {}); // best-effort
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Something went wrong");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchResearch();
    return () => { cancelled = true; };
  }, [injuryProfile, userId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-4">
        <div className="relative">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" className="animate-spin-slow">
            <path
              d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z"
              fill="#CE82FF"
            />
          </svg>
        </div>
        <p className="text-sm font-bold text-[#3C3C3C]">
          Researching guidelines for your injuries...
        </p>
        <p className="text-xs text-[#AFAFAF]">
          Powered by Gemini
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-sm text-[#EA2B2B] mb-4">{error}</p>
        <DuoButton variant="primary" onClick={onContinue}>
          Skip & Continue
        </DuoButton>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="flex flex-col gap-4">
      {result.guidelines.map((g, i) => (
        <Card key={i}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#F7F7F7] border border-[#E5E5E5]">
              {INJURY_LABELS[g.injury_type] ?? g.injury_type}
            </span>
            <span
              className="text-[10px] font-bold px-2 py-0.5 rounded-full text-white"
              style={{ backgroundColor: METRIC_COLORS[g.metric] ?? "#777777" }}
            >
              {g.metric}
            </span>
          </div>
          <h4 className="text-sm font-bold text-[#3C3C3C] mb-1">{g.title}</h4>
          <p className="text-xs text-[#555555] leading-relaxed mb-2">
            {g.explanation}
          </p>
          {g.precautions.length > 0 && (
            <ul className="flex flex-col gap-1">
              {g.precautions.map((p, j) => (
                <li key={j} className="text-[11px] text-[#777777] flex items-start gap-1.5">
                  <span className="text-[#AFAFAF] flex-shrink-0 mt-0.5">{"\u2022"}</span>
                  {p}
                </li>
              ))}
            </ul>
          )}
        </Card>
      ))}

      {result.summary && (
        <p className="text-xs text-[#777777] leading-relaxed px-1">
          {result.summary}
        </p>
      )}

      <DuoButton variant="primary" onClick={onContinue}>
        Continue
      </DuoButton>
    </div>
  );
}
