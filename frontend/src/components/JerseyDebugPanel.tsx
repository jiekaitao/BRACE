"use client";

import { useEffect, useState } from "react";
import type { SubjectState } from "@/lib/types";
import { jerseyDisplayColor, jerseyTagText } from "@/lib/colors";
import Card from "./ui/Card";

interface JerseyDebugPanelProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
}

export default function JerseyDebugPanel({
  subjectsRef,
  selectedSubjectRef,
}: JerseyDebugPanelProps) {
  const [, setTick] = useState(0);

  // Poll at 2Hz to pick up jersey data changes
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 500);
    return () => clearInterval(interval);
  }, []);

  const selectedId = selectedSubjectRef.current;
  if (selectedId === null) return null;

  const subject = subjectsRef.current.get(selectedId);
  if (!subject) return null;

  const { jerseyNumber, jerseyColor, jerseyCropBase64, jerseyGeminiResponse, teamId, teamColor } = subject;
  const tag = jerseyTagText(jerseyNumber, jerseyColor);

  // Don't render if no jersey data at all
  if (jerseyNumber == null && !jerseyColor && !jerseyCropBase64 && teamId == null) return null;

  return (
    <Card>
      <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-1.5">
        Jersey Detection
      </h3>
      <div className="flex items-center gap-3 mb-2">
        {/* Color swatch */}
        {jerseyColor && (
          <div
            className="w-8 h-8 rounded-lg border border-[#E5E5E5] flex-shrink-0"
            style={{ backgroundColor: jerseyDisplayColor(jerseyColor) }}
            title={jerseyColor}
          />
        )}
        <div>
          {tag && (
            <div className="text-lg font-extrabold text-[#3C3C3C]">{tag}</div>
          )}
          <div className="text-[11px] text-[#AFAFAF]">
            {jerseyNumber != null ? `#${jerseyNumber}` : "No number"}
            {jerseyColor ? ` \u00b7 ${jerseyColor}` : ""}
          </div>
        </div>
      </div>

      {/* Team clustering */}
      {teamId != null && (
        <div className="flex items-center gap-2 mb-2">
          {teamColor && (
            <div
              className="w-5 h-5 rounded border border-[#E5E5E5] flex-shrink-0"
              style={{ backgroundColor: teamColor }}
            />
          )}
          <div className="text-[11px] text-[#AFAFAF]">
            Team {teamId + 1}{teamColor ? ` \u00b7 ${teamColor}` : ""}
          </div>
        </div>
      )}

      {/* Crop image */}
      {jerseyCropBase64 && (
        <div className="mb-2">
          <div className="text-[10px] text-[#AFAFAF] mb-0.5 uppercase tracking-wider">Crop sent to Gemini</div>
          <img
            src={`data:image/jpeg;base64,${jerseyCropBase64}`}
            alt="Player crop"
            className="rounded-lg border border-[#E5E5E5] max-w-full"
            style={{ maxHeight: 160 }}
          />
        </div>
      )}

      {/* Gemini response */}
      {jerseyGeminiResponse && (
        <div>
          <div className="text-[10px] text-[#AFAFAF] mb-0.5 uppercase tracking-wider">Gemini Response</div>
          <pre className="text-[11px] text-[#777] bg-[#F7F7F7] rounded-lg p-2 overflow-x-auto whitespace-pre-wrap break-all">
            {jerseyGeminiResponse}
          </pre>
        </div>
      )}
    </Card>
  );
}
