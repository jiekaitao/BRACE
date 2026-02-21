"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SPORTS } from "@/lib/teamSportsData";
import VideoButton from "@/components/ui/VideoButton";

interface TeamSportBrowserProps {
  onClose: () => void;
  showContent: boolean;
}

export default function TeamSportBrowser({ onClose, showContent }: TeamSportBrowserProps) {
  const [selectedSport, setSelectedSport] = useState<string | null>(null);

  const sport = SPORTS.find((s) => s.id === selectedSport);

  return (
    <motion.div
      animate={{ opacity: showContent ? 1 : 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col h-full"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          {selectedSport && (
            <button
              onClick={() => setSelectedSport(null)}
              className="w-9 h-9 flex items-center justify-center rounded-full bg-white/20 border-2 border-white/30 text-white font-bold text-lg cursor-pointer hover:bg-white/30 transition-colors"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5" />
                <path d="M12 19l-7-7 7-7" />
              </svg>
            </button>
          )}
          <h2 className="text-xl font-extrabold text-white m-0">
            {sport ? sport.name : "Team Performance Monitor"}
          </h2>
        </div>
        <button
          onClick={onClose}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-[#F7F7F7] border-2 border-[#E5E5E5] text-[#777777] font-bold text-lg cursor-pointer hover:bg-[#E5E5E5] transition-colors"
        >
          &times;
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 px-6 pb-6 overflow-y-auto">
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
                  className="group relative overflow-hidden rounded-[16px] border-2 border-white/20 cursor-pointer aspect-[4/3] hover:border-white/50 hover:scale-[1.03] active:scale-[0.98] transition-all duration-200"
                >
                  <img
                    src={s.image}
                    alt={s.name}
                    className="absolute inset-0 w-full h-full object-cover"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent" />
                  <div className="absolute bottom-0 left-0 right-0 p-3">
                    <span className="text-white font-extrabold text-sm sm:text-base">
                      {s.name}
                    </span>
                  </div>
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
