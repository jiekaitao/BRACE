"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { SPORTS } from "@/lib/teamSportsData";

const SPORT_ICONS: Record<string, React.ReactNode> = {
  basketball: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="32" cy="32" r="28" />
      <path d="M32 4v56" />
      <path d="M4 32h56" />
      <path d="M8.5 12C18 20 18 44 8.5 52" />
      <path d="M55.5 12C46 20 46 44 55.5 52" />
      <path d="M12 8.5C20 18 44 18 52 8.5" />
      <path d="M12 55.5C20 46 44 46 52 55.5" />
    </svg>
  ),
  football: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="32" cy="32" rx="28" ry="16" transform="rotate(-30 32 32)" />
      <path d="M18 46L46 18" />
      <path d="M24 28l4 4" />
      <path d="M28 24l4 4" />
      <path d="M32 20l4 4" />
      <path d="M36 16l4 4" />
      <path d="M10 42c-2-6-2-12 0-18" />
      <path d="M54 42c2-6 2-12 0-18" />
    </svg>
  ),
  soccer: (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="32" cy="32" r="28" />
      <polygon points="32,14 40,20 38,30 26,30 24,20" />
      <line x1="32" y1="4" x2="32" y2="14" />
      <line x1="40" y1="20" x2="56" y2="14" />
      <line x1="38" y1="30" x2="54" y2="40" />
      <line x1="26" y1="30" x2="10" y2="40" />
      <line x1="24" y1="20" x2="8" y2="14" />
      <line x1="32" y1="60" x2="32" y2="48" />
      <line x1="10" y1="40" x2="16" y2="48" />
      <line x1="54" y1="40" x2="48" y2="48" />
      <line x1="16" y1="48" x2="32" y2="48" />
      <line x1="48" y1="48" x2="32" y2="48" />
    </svg>
  ),
};

interface TeamSportBrowserProps {
  onClose: () => void;
  showContent: boolean;
}

export default function TeamSportBrowser({ onClose, showContent }: TeamSportBrowserProps) {
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <motion.div
      animate={{ opacity: showContent ? 1 : 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col h-full"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[#222]">
        <h2 className="text-xl font-extrabold text-white m-0">
          Choose a Sport
        </h2>
        <button
          onClick={onClose}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-[#1A1A1A] border border-[#333] text-white/60 font-bold text-lg cursor-pointer hover:bg-[#2A2A2A] transition-colors"
        >
          &times;
        </button>
      </div>

      {/* Sport grid */}
      <div className="flex-1 px-6 py-5 overflow-y-auto">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2 }}
          className="grid grid-cols-3 gap-4"
        >
          {SPORTS.map((s) => (
            <button
              key={s.id}
              onClick={() => {
                if (s.demoVideo) {
                  router.push(`/analyze?mode=demo&video=${encodeURIComponent(s.demoVideo)}`);
                }
              }}
              disabled={!s.demoVideo}
              className={`group relative overflow-hidden rounded-[16px] border bg-[#111] aspect-[4/3] flex flex-col items-center justify-center gap-3 transition-all duration-100 ${
                s.demoVideo
                  ? "border-[#333] cursor-pointer shadow-[0_4px_0_#222] hover:border-white/40 hover:shadow-[0_4px_0_#444] active:shadow-none active:translate-y-[4px]"
                  : "border-[#222] cursor-default opacity-40"
              }`}
            >
              <div className={`transition-opacity duration-200 ${s.demoVideo ? "opacity-40 group-hover:opacity-80" : "opacity-30"}`}>
                {SPORT_ICONS[s.id]}
              </div>
              <span className="text-white font-extrabold text-sm">
                {s.name}
              </span>
              {!s.demoVideo && (
                <span className="text-white/30 text-[10px] font-bold">Coming soon</span>
              )}
            </button>
          ))}
        </motion.div>
      </div>
    </motion.div>
  );
}
