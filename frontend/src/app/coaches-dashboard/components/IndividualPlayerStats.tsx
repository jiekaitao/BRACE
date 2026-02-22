"use client";

import { motion } from "framer-motion";
import Image from "next/image";

interface Player {
  id: number;
  name: string;
  position: string;
  number: number;
  concussionProb: number;
  fatigue: number;
  recovery: number;
  movementQuality: number;
  performance: number;
  heartRate: number;
  collision_velocity: number;
  hydration: number;
  vectorData: number[];
  reasoning?: string;
}

interface IndividualPlayerStatsProps {
  player: Player;
}

export function IndividualPlayerStats({ player }: IndividualPlayerStatsProps) {
  const isHighRisk = player.concussionProb > 70;
  const isFatigued = player.fatigue > 70;

  return (
    <motion.div
      className="space-y-4"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      key={player.id}
    >
      {/* Player Header */}
      <div className={`bg-gray-950 rounded-lg border-4 overflow-hidden shadow-2xl ${
        isHighRisk ? 'border-red-500' : 'border-white'
      }`}>
        <div className={`px-4 py-3 border-b-4 border-white flex items-center justify-between ${
          isHighRisk ? 'bg-red-600' : 'bg-blue-600'
        }`}>
          <div>
            <h3 className="text-white text-base" style={{ fontFamily: "'Press Start 2P', cursive" }}>
              #{player.number} {player.name}
            </h3>
            <p className="text-blue-200 text-xs mt-1" style={{ fontFamily: "'Press Start 2P', cursive" }}>
              {player.position}
            </p>
          </div>
          <motion.div
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
            <Image
              src="/player-sprite.png"
              alt={player.name}
              width={64}
              height={64}
              className="pixelated"
              style={{ imageRendering: 'pixelated' }}
            />
          </motion.div>
        </div>

        {/* Critical Alerts */}
        {(isHighRisk || isFatigued) && (
          <div className="bg-red-900 border-b-4 border-red-700 px-4 py-3">
            <div className="flex items-center gap-3">
              <motion.div
                className="w-3 h-3 bg-white rounded-full"
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              />
              <div className="text-white text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
                {isHighRisk && "HIGH CONCUSSION RISK"}
                {isHighRisk && isFatigued && " | "}
                {isFatigued && "FATIGUED"}
              </div>
            </div>
          </div>
        )}

        <div className="p-6 bg-gray-900">
          <div className="grid grid-cols-2 gap-6">
            {/* Concussion Risk */}
            <div>
              <div className="text-gray-400 text-xs mb-2" style={{ fontFamily: "'Press Start 2P', cursive" }}>
                CONCUSSION RISK
              </div>
              <div className="relative">
                <div className="bg-gray-800 h-8 rounded overflow-hidden border-2 border-gray-700">
                  <motion.div
                    className={`h-full ${
                      player.concussionProb > 70 ? 'bg-red-500' :
                      player.concussionProb > 40 ? 'bg-yellow-400' :
                      'bg-green-400'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${player.concussionProb}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
                <div className={`text-center mt-2 text-2xl ${
                  player.concussionProb > 70 ? 'text-red-500' :
                  player.concussionProb > 40 ? 'text-yellow-400' :
                  'text-green-400'
                }`} style={{ fontFamily: "'Press Start 2P', cursive" }}>
                  {player.concussionProb}%
                </div>
              </div>
            </div>

            {/* Fatigue */}
            <div>
              <div className="text-gray-400 text-xs mb-2" style={{ fontFamily: "'Press Start 2P', cursive" }}>
                FATIGUE
              </div>
              <div className="relative">
                <div className="bg-gray-800 h-8 rounded overflow-hidden border-2 border-gray-700">
                  <motion.div
                    className={`h-full ${
                      player.fatigue > 70 ? 'bg-red-500' :
                      player.fatigue > 40 ? 'bg-orange-400' :
                      'bg-blue-400'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${player.fatigue}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
                <div className={`text-center mt-2 text-2xl ${
                  player.fatigue > 70 ? 'text-red-500' :
                  player.fatigue > 40 ? 'text-orange-400' :
                  'text-blue-400'
                }`} style={{ fontFamily: "'Press Start 2P', cursive" }}>
                  {player.fatigue}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl">
        <div className="bg-purple-600 px-4 py-2 border-b-4 border-white">
          <h3 className="text-white text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
            PERFORMANCE DATA
          </h3>
        </div>
        <div className="p-4 bg-gray-900">
          <div className="grid grid-cols-2 gap-4 text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <span className="text-gray-400 block mb-1">COLLISION VEL</span>
              <span className={`text-sm ${player.collision_velocity > 28 ? 'text-red-400' : player.collision_velocity > 16 ? 'text-yellow-400' : 'text-cyan-400'}`}>
                {player.collision_velocity} MPH
              </span>
            </div>
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <span className="text-gray-400 block mb-1">PERFORMANCE</span>
              <span className={`text-sm ${player.performance > 70 ? 'text-green-400' : player.performance > 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                {player.performance}%
              </span>
            </div>
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <span className="text-gray-400 block mb-1">HEART RATE</span>
              <span className={`text-sm ${player.heartRate > 180 ? 'text-red-400' : player.heartRate > 150 ? 'text-orange-400' : 'text-cyan-400'}`}>
                {player.heartRate} BPM
              </span>
            </div>
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <span className="text-gray-400 block mb-1">RECOVERY</span>
              <span className={`text-sm ${player.recovery < 30 ? 'text-green-400' : player.recovery < 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                {player.recovery}%
              </span>
            </div>
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <span className="text-gray-400 block mb-1">HYDRATION</span>
              <span className={`text-sm ${player.hydration > 70 ? 'text-green-400' : player.hydration > 40 ? 'text-yellow-400' : 'text-orange-400'}`}>
                {player.hydration}%
              </span>
            </div>
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <span className="text-gray-400 block mb-1">MOVEMENT</span>
              <span className={`text-sm ${player.movementQuality > 70 ? 'text-green-400' : player.movementQuality > 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                {player.movementQuality}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl">
        <div className={`px-4 py-2 border-b-4 border-white ${
          isHighRisk || isFatigued ? 'bg-orange-600' : 'bg-green-600'
        }`}>
          <h3 className="text-white text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
            COACH RECOMMENDATION
          </h3>
        </div>
        <div className="p-4 bg-gray-900">
          <div className="mb-3">
            <p className="text-white text-xs font-bold mb-2" style={{ fontFamily: "'Press Start 2P', cursive" }}>
              {isHighRisk && isFatigued && "BENCH NOW"}
              {isHighRisk && !isFatigued && "LIMIT PLAY"}
              {!isHighRisk && isFatigued && "REST NEEDED"}
              {!isHighRisk && !isFatigued && "CLEARED"}
            </p>
          </div>
          {player.reasoning && (
            <div className="bg-gray-800 p-3 rounded border-2 border-gray-700">
              <p className="text-gray-300 text-xs leading-relaxed" style={{ fontFamily: "'Press Start 2P', cursive", fontSize: '8px', lineHeight: '1.6' }}>
                {player.reasoning}
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
