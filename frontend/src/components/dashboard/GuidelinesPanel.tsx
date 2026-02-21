"use client";

import { useState } from "react";
import type { InjuryGuidelines, InjuryEntry } from "@/lib/types";
import { fetchGuidelines } from "@/lib/dashboard";
import Card from "@/components/ui/Card";
import DuoButton from "@/components/ui/DuoButton";

interface Props {
  injury: InjuryEntry;
}

export default function GuidelinesPanel({ injury }: Props) {
  const [guidelines, setGuidelines] = useState<InjuryGuidelines | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const handleFetch = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchGuidelines(
        injury.type,
        injury.severity,
        injury.side !== "unknown" ? injury.side : "general",
      );
      setGuidelines(data);
      setExpanded(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load guidelines");
    } finally {
      setLoading(false);
    }
  };

  if (!guidelines) {
    return (
      <div className="mt-2">
        <DuoButton
          variant="secondary"
          onClick={handleFetch}
          disabled={loading}
        >
          {loading ? "Researching..." : "Get Guidelines"}
        </DuoButton>
        {error && (
          <p className="text-xs text-[#EA2B2B] mt-1">{error}</p>
        )}
      </div>
    );
  }

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-sm font-bold text-[#1CB0F6] hover:underline"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          className={`transition-transform ${expanded ? "rotate-90" : ""}`}
        >
          <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
        </svg>
        Guidelines
      </button>
      {expanded && (
        <Card className="mt-2 bg-[#F7F7F7]">
          {/* Summary */}
          <p className="text-sm text-[#4B4B4B] mb-3">{guidelines.summary}</p>

          {/* Red Flags */}
          {guidelines.red_flags.length > 0 && (
            <Section title="Red Flags">
              <ul className="list-disc list-inside text-xs text-[#EA2B2B] space-y-0.5">
                {guidelines.red_flags.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </Section>
          )}

          {/* Safe ROM */}
          {guidelines.safe_rom.length > 0 && (
            <Section title="Safe Range of Motion">
              <div className="space-y-1">
                {guidelines.safe_rom.map((r, i) => (
                  <div key={i} className="flex justify-between text-xs">
                    <span className="text-[#777777]">{r.joint} - {r.motion}</span>
                    <span className="font-bold text-[#4B4B4B]">{r.min_degrees}-{r.max_degrees}</span>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* Rehab Protocols */}
          {guidelines.rehab_protocols.length > 0 && (
            <Section title="Rehab Protocols">
              <div className="space-y-2">
                {guidelines.rehab_protocols.map((p, i) => (
                  <div key={i} className="bg-white rounded-lg p-2 border border-[#E5E5E5]">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs font-bold text-[#3C3C3C]">{p.phase}</span>
                      <span className="text-[11px] text-[#AFAFAF]">{p.duration}</span>
                    </div>
                    <p className="text-xs text-[#777777] mb-1">{p.goals}</p>
                    <ul className="text-xs text-[#4B4B4B] list-disc list-inside">
                      {p.exercises.map((e, j) => <li key={j}>{e}</li>)}
                    </ul>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* Recommended Exercises */}
          {guidelines.recommended_exercises.length > 0 && (
            <Section title="Recommended Exercises">
              <div className="space-y-1">
                {guidelines.recommended_exercises.map((ex, i) => (
                  <div key={i} className="flex justify-between items-start text-xs gap-2">
                    <span className="font-bold text-[#3C3C3C]">{ex.name}</span>
                    <span className="text-[#777777] flex-shrink-0">
                      {ex.sets}x{ex.reps}
                    </span>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* Activities to Avoid */}
          {guidelines.activities_to_avoid.length > 0 && (
            <Section title="Avoid">
              <ul className="list-disc list-inside text-xs text-[#FF9600] space-y-0.5">
                {guidelines.activities_to_avoid.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </Section>
          )}

          {/* References */}
          {guidelines.references.length > 0 && (
            <Section title="References">
              <ul className="text-[11px] text-[#AFAFAF] space-y-0.5">
                {guidelines.references.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </Section>
          )}
        </Card>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <h4 className="text-xs font-bold text-[#3C3C3C] mb-1 uppercase tracking-wide">
        {title}
      </h4>
      {children}
    </div>
  );
}
