"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SPORTS } from "@/lib/teamSportsData";
import VideoButton from "@/components/ui/VideoButton";

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
  const [selectedSport, setSelectedSport] = useState<string | null>(null);

  // Escape key: go back to sport list if a sport is selected, otherwise close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedSport) {
          setSelectedSport(null);
        } else {
          onClose();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedSport, onClose]);

  const sport = SPORTS.find((s) => s.id === selectedSport);

  return (
    <motion.div
      animate={{ opacity: showContent ? 1 : 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col h-full"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[#222]">
        <div className="flex items-center gap-3">
          {selectedSport && (
            <button
              onClick={() => setSelectedSport(null)}
              className="w-9 h-9 flex items-center justify-center rounded-full bg-[#1A1A1A] border border-[#333] text-white/60 font-bold text-lg cursor-pointer hover:bg-[#2A2A2A] transition-colors"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5" />
                <path d="M12 19l-7-7 7-7" />
              </svg>
            </button>
          )}
          <h2 className="text-xl font-extrabold text-white m-0">
            {sport ? sport.name : "Choose a Sport"}
          </h2>
        </div>
        <button
          onClick={onClose}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-[#1A1A1A] border border-[#333] text-white/60 font-bold text-lg cursor-pointer hover:bg-[#2A2A2A] transition-colors"
        >
          &times;
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 px-6 py-5 overflow-y-auto">
        <AnimatePresence mode="wait">
          {!selectedSport ? (
            <motion.div
              key="sports-grid"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-3 gap-4"
            >
              {SPORTS.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedSport(s.id)}
                  className="group relative overflow-hidden rounded-[16px] border border-[#333] bg-[#111] cursor-pointer aspect-[4/3] shadow-[0_4px_0_#222] hover:border-white/40 hover:shadow-[0_4px_0_#444] active:shadow-none active:translate-y-[4px] transition-all duration-100 flex flex-col items-center justify-center gap-3"
                >
                  <div className="opacity-40 group-hover:opacity-80 transition-opacity duration-200">
                    {SPORT_ICONS[s.id]}
                  </div>
                  <span className="text-white font-extrabold text-sm">
                    {s.name}
                  </span>
                </button>
              ))}
            </motion.div>
          ) : (
            <motion.div
              key={`videos-${selectedSport}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col gap-3"
            >
              {sport!.videos.map((video, i) => (
                <motion.div
                  key={video.id}
                  initial={{ opacity: 0, x: -30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.25, delay: i * 0.06 }}
                >
                  <VideoButton
                    title={video.title}
                    thumbnail={video.thumbnail}
                  />
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
