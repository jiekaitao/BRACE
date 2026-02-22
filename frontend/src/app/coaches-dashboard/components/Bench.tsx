"use client";

import { motion } from "framer-motion";

interface BenchPlayer {
  id: number;
  name: string;
  position: string;
  number: number;
  concussionProb: number;
  fatigue: number;
}

interface BenchProps {
  players: BenchPlayer[];
  onPlayerClick: (player: BenchPlayer) => void;
  selectedPlayerId?: number;
}

export function Bench({ players, onPlayerClick, selectedPlayerId }: BenchProps) {
  return (
    <div className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl">
      <div className="bg-yellow-600 px-4 py-2 border-b-4 border-white">
        <h3 className="text-white text-xs">BENCH</h3>
        <p className="text-yellow-200 text-xs mt-1">Click to view stats</p>
      </div>
      <div className="p-4 bg-gray-900">
        <div className="grid grid-cols-2 gap-2">
          {players.map((player) => {
            const isHighRisk = player.concussionProb > 70;
            const isSelected = selectedPlayerId === player.id;

            return (
              <motion.button
                key={player.id}
                onClick={() => onPlayerClick(player)}
                className={`
                  relative p-3 border-2 rounded-lg text-left transition-all
                  ${isSelected
                    ? 'border-blue-400 bg-blue-900/50'
                    : 'border-gray-600 bg-gray-800 hover:border-white hover:bg-gray-700'
                  }
                `}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                animate={isHighRisk ? {
                  backgroundColor: ['rgb(31, 41, 55)', 'rgb(127, 29, 29)', 'rgb(31, 41, 55)']
                } : {}}
                transition={isHighRisk ? {
                  duration: 1,
                  repeat: Infinity,
                  ease: 'easeInOut'
                } : {}}
              >
                <div className={`text-lg mb-1 ${isHighRisk ? 'text-red-400' : 'text-white'}`}>
                  #{player.number}
                </div>
                <div className="text-xs text-gray-300 mb-2">{player.name}</div>
                <div className="text-xs text-gray-400 mb-2">{player.position}</div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">INJ:</span>
                    <span className={player.concussionProb > 70 ? 'text-red-400' : player.concussionProb > 40 ? 'text-yellow-400' : 'text-green-400'}>
                      {player.concussionProb}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">FAT:</span>
                    <span className={player.fatigue > 70 ? 'text-red-400' : player.fatigue > 40 ? 'text-orange-400' : 'text-blue-400'}>
                      {player.fatigue}%
                    </span>
                  </div>
                </div>
                {isHighRisk && (
                  <motion.div
                    className="absolute top-2 right-2 text-red-400 text-xs"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 0.8, repeat: Infinity }}
                  >
                    !!
                  </motion.div>
                )}
              </motion.button>
            );
          })}
        </div>
        {players.length === 0 && (
          <div className="text-center text-gray-500 text-xs py-4">
            NO PLAYERS ON BENCH
          </div>
        )}
      </div>
    </div>
  );
}
