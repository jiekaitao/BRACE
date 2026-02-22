"use client";

import { motion } from "framer-motion";
import Image from "next/image";

interface PlayerCardProps {
  id: number;
  name: string;
  position: string;
  number: number;
  concussionProb: number;
  fatigue: number;
  x: number;
  y: number;
  vectorData: number[];
  onClick: () => void;
  isSelected: boolean;
}

export function PlayerCard({ name, number, concussionProb, fatigue, x, y, onClick, isSelected }: PlayerCardProps) {
  const isHighRisk = concussionProb > 70;

  return (
    <motion.div
      className="absolute cursor-pointer"
      style={{ left: `${x}%`, top: `${y}%`, transform: 'translate(-50%, -50%)' }}
      onClick={onClick}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      animate={isHighRisk ? {
        opacity: [1, 0.3, 1],
        scale: [1, 1.1, 1]
      } : {}}
      transition={isHighRisk ? {
        duration: 0.8,
        repeat: Infinity,
        ease: "easeInOut"
      } : {}}
    >
      <div className={`relative ${isHighRisk ? 'animate-pulse' : ''}`}>
        {/* Red glow for high-risk players */}
        {isHighRisk && (
          <motion.div
            className="absolute inset-0 bg-red-600 rounded-full blur-xl opacity-60"
            animate={{
              opacity: [0.6, 0.3, 0.6],
            }}
            transition={{
              duration: 0.8,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}

        {/* Selection ring */}
        {isSelected && (
          <motion.div
            className="absolute inset-0 bg-yellow-400 rounded-full blur-lg opacity-80"
            animate={{
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}

        {/* Player sprite */}
        <div className="relative flex flex-col items-center">
          <Image
            src="/player-sprite.png"
            alt={name}
            width={64}
            height={64}
            className={`pixelated ${isHighRisk ? 'drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]' : ''} ${isSelected ? 'drop-shadow-[0_0_12px_rgba(250,204,21,1)]' : ''}`}
            style={{ imageRendering: 'pixelated' }}
          />

          {/* Player number badge - below sprite */}
          <div
            className={`mt-1 border-2 border-white text-white text-xs px-2 py-0.5 rounded shadow-lg ${isSelected ? 'bg-yellow-500' : 'bg-blue-600'}`}
            style={{ fontFamily: "'Press Start 2P', cursive" }}
          >
            #{number}
          </div>
        </div>
      </div>

      {/* Hover card */}
      <motion.div
        className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-gray-900 border-4 border-white p-3 rounded shadow-2xl z-50 min-w-[200px]"
        initial={{ opacity: 0, y: -10 }}
        whileHover={{ opacity: 1, y: 0 }}
        style={{ pointerEvents: 'none' }}
      >
        <div className="text-white text-xs space-y-1" style={{ fontFamily: "'Press Start 2P', cursive" }}>
          <div className="text-yellow-400 mb-2">#{number} {name}</div>
          <div className="h-px bg-gray-600 my-2"></div>
          <div className={concussionProb > 70 ? 'text-red-500' : concussionProb > 40 ? 'text-yellow-400' : 'text-green-400'}>
            INJURY: {concussionProb}%
          </div>
          <div className={fatigue > 70 ? 'text-red-500' : fatigue > 40 ? 'text-yellow-400' : 'text-green-400'}>
            FATIGUE: {fatigue}%
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
