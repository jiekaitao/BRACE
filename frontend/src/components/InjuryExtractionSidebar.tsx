"use client";

import type { InjuryEntry, InjuryProfile } from "@/lib/types";
import { InjuryBadge } from "@/components/InjuryProfileCard";
import { INJURY_METRIC_MAP } from "@/hooks/useChat";

interface Props {
  partialInjuries: InjuryEntry[];
  extractedProfile: InjuryProfile | null;
}

export default function InjuryExtractionSidebar({ partialInjuries, extractedProfile }: Props) {
  // Use real extracted profile if available, otherwise partial
  const injuries = extractedProfile?.injuries ?? partialInjuries;
  const isPartial = !extractedProfile;

  return (
    <div className="flex flex-col h-full">
      <h3 className="text-sm font-bold text-[#3C3C3C] mb-3">
        Detected Injuries
      </h3>

      {injuries.length === 0 ? (
        <div className="flex-1 flex items-center justify-center px-3">
          <p className="text-xs text-[#AFAFAF] text-center leading-relaxed">
            Tell me about any past injuries or areas of concern...
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2 flex-1 overflow-y-auto">
          {injuries.map((injury, i) => (
            <div key={`${injury.type}-${i}`}>
              <InjuryBadge injury={injury} />
              {INJURY_METRIC_MAP[injury.type] && (
                <div className="text-[10px] text-[#CE82FF] font-bold mt-0.5 ml-5">
                  BRACE → {INJURY_METRIC_MAP[injury.type]}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {isPartial && injuries.length > 0 && (
        <p className="text-[10px] text-[#AFAFAF] mt-2 italic">
          Keep chatting for more detail...
        </p>
      )}

      {extractedProfile && (
        <div className="flex items-center gap-1.5 mt-2 text-[11px] text-[#58CC02] font-bold">
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
            <path d="M6 8l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5"/>
          </svg>
          Profile extracted
        </div>
      )}
    </div>
  );
}
