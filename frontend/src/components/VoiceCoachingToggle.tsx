"use client";

interface Props {
  enabled: boolean;
  onToggle: () => void;
}

export default function VoiceCoachingToggle({ enabled, onToggle }: Props) {
  return (
    <button
      onClick={onToggle}
      className={`
        flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold
        border-2 transition-all duration-150
        ${enabled
          ? "bg-[#58CC02] text-white border-[#46A302] shadow-[0_2px_0_#46A302]"
          : "bg-white text-[#AFAFAF] border-[#E5E5E5] shadow-[0_2px_0_#E5E5E5]"
        }
        hover:brightness-105
        active:shadow-none active:translate-y-[2px]
      `}
      title={enabled ? "Voice coaching ON" : "Voice coaching OFF"}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        {enabled ? (
          <>
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
          </>
        ) : (
          <>
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <line x1="23" y1="9" x2="17" y2="15" />
            <line x1="17" y1="9" x2="23" y2="15" />
          </>
        )}
      </svg>
      Voice
    </button>
  );
}
