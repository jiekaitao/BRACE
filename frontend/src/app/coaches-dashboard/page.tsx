"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FootballField } from "./components/FootballField";
import { IndividualPlayerStats } from "./components/IndividualPlayerStats";
import { PlayerStats } from "./components/PlayerStats";
import { Vector3DVisualizer } from "./components/Vector3DVisualizer";
import { Bench } from "./components/Bench";
import teamsData from "./data/teams.json";

// Derive metrics from 50D vector data
// The vector represents raw sensor/biometric data from which we calculate meaningful metrics
const deriveMetricsFromVector = (vectorData: number[]) => {
  // Dimensions 0-9: Head impact sensors (concussion risk)
  const headImpactData = vectorData.slice(0, 10);
  const avgHeadImpact = headImpactData.reduce((a, b) => a + b, 0) / headImpactData.length;
  const maxHeadImpact = Math.max(...headImpactData);
  const concussionProb = Math.floor((avgHeadImpact * 0.6 + maxHeadImpact * 0.4) * 100);

  // Dimensions 10-19: Physical exertion sensors (fatigue)
  const exertionData = vectorData.slice(10, 20);
  const avgExertion = exertionData.reduce((a, b) => a + b, 0) / exertionData.length;
  const fatigue = Math.floor(avgExertion * 100);

  // Dimensions 20-29: Heart rate variability and recovery metrics
  const hrvData = vectorData.slice(20, 30);
  const avgHRV = hrvData.reduce((a, b) => a + b, 0) / hrvData.length;
  const recovery = Math.floor((1 - avgHRV) * 100); // Lower HRV = worse recovery

  // Dimensions 30-39: Movement quality and biomechanics
  const movementData = vectorData.slice(30, 40);
  const avgMovement = movementData.reduce((a, b) => a + b, 0) / movementData.length;
  const movementQuality = Math.floor(avgMovement * 100);

  // Dimensions 40-49: Performance metrics (speed, agility, power)
  const performanceData = vectorData.slice(40, 50);
  const avgPerformance = performanceData.reduce((a, b) => a + b, 0) / performanceData.length;
  const performance = Math.floor(avgPerformance * 100);

  return {
    concussionProb,
    fatigue,
    recovery,
    movementQuality,
    performance,
    // Additional derived metrics
    heartRate: Math.floor(60 + (avgExertion * 140)), // 60-200 bpm
    collision_velocity: parseFloat((maxHeadImpact * 40).toFixed(1)), // 0-40 mph relative collision velocity
    hydration: Math.floor((1 - avgExertion * 0.5) * 100), // Decreases with exertion
  };
};

export default function CoachesDashboard() {
  const [selectedTeam, setSelectedTeam] = useState<'blue' | 'red' | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<any>(null);
  const [players, setPlayers] = useState<any[]>([]);
  const [benchPlayers, setBenchPlayers] = useState<any[]>([]);
  const [teamName, setTeamName] = useState("");
  const [teamColor, setTeamColor] = useState("");

  // Load team data when team is selected
  useEffect(() => {
    if (!selectedTeam) return;

    const teamData = teamsData[selectedTeam];
    setTeamName(teamData.name);
    setTeamColor(teamData.color);

    // Process players with derived metrics
    const allFieldPlayers = teamData.players.map((player: any) => ({
      ...player,
      ...deriveMetricsFromVector(player.vectorData)
    }));

    // Process bench players
    const bench = teamData.bench.map((player: any) => ({
      ...player,
      ...deriveMetricsFromVector(player.vectorData)
    }));

    setPlayers(allFieldPlayers);
    setBenchPlayers(bench);
    setSelectedPlayer(null); // Reset selection when switching teams
  }, [selectedTeam]);

  const allPlayers = [...players, ...benchPlayers];
  const highRiskPlayers = allPlayers.filter(p => p.concussionProb > 70);

  const handlePlayerClick = (player: any) => {
    // Toggle: if clicking the same player, deselect and show team view
    if (selectedPlayer?.id === player.id) {
      setSelectedPlayer(null);
    } else {
      setSelectedPlayer(player);
    }
  };

  // Team Selection Screen
  if (!selectedTeam) {
    return (
      <div
        className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-4 flex items-center justify-center"
        style={{ fontFamily: "'Press Start 2P', cursive" }}
      >
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap"
          rel="stylesheet"
        />
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="max-w-2xl w-full"
        >
          <div className="bg-gray-950 border-4 border-white rounded-lg overflow-hidden shadow-2xl">
            <div className="bg-purple-600 px-6 py-4 border-b-4 border-white">
              <h1 className="text-white text-xl mb-2">RETRO BRACE ML</h1>
              <p className="text-purple-200 text-xs">COACHES ANALYTICS TABLET</p>
            </div>
            <div className="p-8 bg-gray-900">
              <h2 className="text-white text-lg mb-6 text-center">SELECT YOUR TEAM</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Blue Team */}
                <motion.button
                  onClick={() => setSelectedTeam('blue')}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-blue-600 border-4 border-white rounded-lg p-6 hover:bg-blue-500 transition-colors"
                >
                  <div className="text-white text-base mb-2">{teamsData.blue.name}</div>
                  <div className="text-blue-200 text-xs">11 PLAYERS</div>
                  <div className="text-blue-200 text-xs mt-1">6 BENCH</div>
                  <motion.div
                    className="mt-4 w-16 h-16 mx-auto bg-blue-500 rounded-full"
                    animate={{
                      boxShadow: [
                        "0 0 20px rgba(59, 130, 246, 0.5)",
                        "0 0 40px rgba(59, 130, 246, 0.8)",
                        "0 0 20px rgba(59, 130, 246, 0.5)"
                      ]
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </motion.button>

                {/* Red Team */}
                <motion.button
                  onClick={() => setSelectedTeam('red')}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-red-600 border-4 border-white rounded-lg p-6 hover:bg-red-500 transition-colors"
                >
                  <div className="text-white text-base mb-2">{teamsData.red.name}</div>
                  <div className="text-red-200 text-xs">11 PLAYERS</div>
                  <div className="text-red-200 text-xs mt-1">6 BENCH</div>
                  <motion.div
                    className="mt-4 w-16 h-16 mx-auto bg-red-500 rounded-full"
                    animate={{
                      boxShadow: [
                        "0 0 20px rgba(239, 68, 68, 0.5)",
                        "0 0 40px rgba(239, 68, 68, 0.8)",
                        "0 0 20px rgba(239, 68, 68, 0.5)"
                      ]
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </motion.button>
              </div>
              <div className="mt-8 text-center text-gray-400 text-xs">
                <p>As a coach, you can only access</p>
                <p>your team&apos;s biometric data</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-4"
      style={{ fontFamily: "'Press Start 2P', cursive" }}
    >
      {/* Google Font */}
      {/* eslint-disable-next-line @next/next/no-page-custom-font */}
      <link
        href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap"
        rel="stylesheet"
      />

      {/* Header */}
      <motion.div
        className="border-4 border-white rounded-lg p-4 mb-4 shadow-2xl"
        style={{ backgroundColor: teamColor }}
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-white text-xl mb-2">RETRO BRACE ML</h1>
            <p className="text-white text-xs opacity-80">{teamName} - COACHES TABLET</p>
          </div>
          <div className="text-right">
            <div className="text-white text-xs mb-1">GAME TIME</div>
            <div className="text-yellow-400 text-sm">Q4 - 2:34</div>
            <button
              onClick={() => setSelectedTeam(null)}
              className="mt-2 text-white text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded border-2 border-white transition-colors"
            >
              SWITCH TEAM
            </button>
          </div>
        </div>
      </motion.div>

      {/* Alert Banner for High Risk Players */}
      {highRiskPlayers.length > 0 && (
        <motion.div
          className="bg-red-600 border-4 border-white rounded-lg p-3 mb-4 shadow-2xl"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <div className="flex items-center gap-3">
            <motion.div
              className="w-4 h-4 bg-white rounded-full"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <div className="text-white text-xs">
              HIGH RISK: {highRiskPlayers.length} PLAYER(S) -
              {highRiskPlayers.map(p => ` #${p.number} ${p.name}`).join(',')}
            </div>
          </div>
        </motion.div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Football Field - Takes up 2 columns */}
        <motion.div
          className="lg:col-span-2"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="bg-gray-950 border-4 border-white rounded-lg overflow-hidden shadow-2xl">
            <div className="bg-green-700 px-4 py-2 border-b-4 border-white">
              <h2 className="text-white text-sm">FIELD VIEW - {teamName}</h2>
              <p className="text-green-200 text-xs mt-1">
                {selectedPlayer ? 'Click again to return' : 'Click player for details'}
              </p>
            </div>
            <div className="aspect-[3/4] p-4 bg-gray-900">
              <FootballField
                players={players}
                onPlayerClick={handlePlayerClick}
                selectedPlayerId={selectedPlayer?.id}
              />
            </div>
          </div>
        </motion.div>

        {/* Right Sidebar */}
        <motion.div
          className="space-y-4"
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          {selectedPlayer ? (
            /* Individual Player Stats */
            <IndividualPlayerStats player={selectedPlayer} />
          ) : (
            <>
              {/* Team Overview - Player Stats */}
              <PlayerStats players={players} />

              {/* Team Summary */}
              <div className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl">
                <div className="bg-purple-600 px-4 py-2 border-b-4 border-white">
                  <h3 className="text-white text-xs">TEAM STATUS</h3>
                </div>
                <div className="p-4 bg-gray-900 space-y-3">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">AVG INJURY:</span>
                    <span className={`${players.length > 0 && players.reduce((a: number, b: any) => a + b.concussionProb, 0) / players.length > 50 ? 'text-red-500' : 'text-green-400'}`}>
                      {players.length > 0 ? Math.round(players.reduce((a: number, b: any) => a + b.concussionProb, 0) / players.length) : 0}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">AVG FATIGUE:</span>
                    <span className={`${players.length > 0 && players.reduce((a: number, b: any) => a + b.fatigue, 0) / players.length > 50 ? 'text-orange-500' : 'text-blue-400'}`}>
                      {players.length > 0 ? Math.round(players.reduce((a: number, b: any) => a + b.fatigue, 0) / players.length) : 0}%
                    </span>
                  </div>
                  <div className="h-px bg-gray-700 my-2"></div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">HIGH RISK:</span>
                    <span className="text-red-500">{highRiskPlayers.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">ACTIVE:</span>
                    <span className="text-blue-400">{players.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">BENCH:</span>
                    <span className="text-gray-400">{benchPlayers.length}</span>
                  </div>
                </div>
              </div>

              {/* Bench */}
              <Bench
                players={benchPlayers}
                onPlayerClick={handlePlayerClick}
                selectedPlayerId={selectedPlayer?.id}
              />
            </>
          )}
        </motion.div>
      </div>

      {/* Bottom Section - 3D Vector Visualization */}
      {selectedPlayer && (
        <motion.div
          className="mt-4"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Vector3DVisualizer
            vectorData={selectedPlayer.vectorData}
            playerName={`#${selectedPlayer.number} ${selectedPlayer.name}`}
          />
        </motion.div>
      )}

      {/* Footer */}
      <motion.div
        className="mt-4 bg-gray-950 border-4 border-white rounded-lg p-3 shadow-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <div className="flex justify-between items-center text-xs text-gray-400">
          <div>RETRO BRACE ML ANALYTICS v2.0 - {teamName}</div>
          <div className="flex items-center gap-2">
            <motion.div
              className="w-2 h-2 bg-green-500 rounded-full"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
            <span>LIVE</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
