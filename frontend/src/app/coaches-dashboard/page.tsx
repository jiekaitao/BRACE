"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { FootballField } from "./components/FootballField";
import { IndividualPlayerStats } from "./components/IndividualPlayerStats";
import { PlayerStats } from "./components/PlayerStats";
import { Vector3DVisualizer } from "./components/Vector3DVisualizer";
import { Bench } from "./components/Bench";

interface Player {
  id: number;
  name: string;
  position: string;
  number: number;
  concussionProb: number;
  fatigue: number;
  x: number;
  y: number;
  vectorData: number[];
}

// Generate mock field player data
const generatePlayers = (): Player[] => {
  const positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S'];
  const names = [
    'SMITH', 'JOHNSON', 'WILLIAMS', 'BROWN', 'JONES',
    'GARCIA', 'MILLER', 'DAVIS', 'RODRIGUEZ', 'MARTINEZ',
    'HERNANDEZ'
  ];

  return Array.from({ length: 11 }, (_, i) => ({
    id: i + 1,
    name: names[i % names.length],
    position: positions[i % positions.length],
    number: i + 10,
    concussionProb: Math.floor(Math.random() * 100),
    fatigue: Math.floor(Math.random() * 100),
    x: i < 5 ? 25 + (i * 10) : 25 + ((i - 5) * 10),
    y: i < 5 ? 35 : 65,
    vectorData: Array.from({ length: 50 }, () => Math.random())
  }));
};

// Generate bench players
const generateBenchPlayers = (): Player[] => {
  const positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S'];
  const names = [
    'WILSON', 'TAYLOR', 'ANDERSON', 'THOMAS', 'JACKSON', 'WHITE'
  ];

  return Array.from({ length: 6 }, (_, i) => ({
    id: i + 100,
    name: names[i % names.length],
    position: positions[i % positions.length],
    number: i + 50,
    concussionProb: Math.floor(Math.random() * 100),
    fatigue: Math.floor(Math.random() * 100),
    x: 0,
    y: 0,
    vectorData: Array.from({ length: 50 }, () => Math.random())
  }));
};

export default function CoachesDashboard() {
  const [players] = useState(generatePlayers);
  const [benchPlayers] = useState(generateBenchPlayers);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);

  const allPlayers = [...players, ...benchPlayers];
  const highRiskPlayers = allPlayers.filter(p => p.concussionProb > 70);

  const handlePlayerClick = (player: Player) => {
    if (selectedPlayer?.id === player.id) {
      setSelectedPlayer(null);
    } else {
      setSelectedPlayer(player);
    }
  };

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
        className="bg-blue-600 border-4 border-white rounded-lg p-4 mb-4 shadow-2xl"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-white text-xl mb-2">RETRO BOWL</h1>
            <p className="text-blue-200 text-xs">COACHES ANALYTICS TABLET</p>
          </div>
          <div className="text-right">
            <div className="text-white text-xs mb-1">GAME TIME</div>
            <div className="text-yellow-400 text-sm">Q4 - 2:34</div>
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
              {highRiskPlayers.map(p => ` #${p.number}`).join(',')}
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
              <h2 className="text-white text-sm">FIELD VIEW</h2>
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
                    <span className={`${players.reduce((a, b) => a + b.concussionProb, 0) / players.length > 50 ? 'text-red-500' : 'text-green-400'}`}>
                      {Math.round(players.reduce((a, b) => a + b.concussionProb, 0) / players.length)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">AVG FATIGUE:</span>
                    <span className={`${players.reduce((a, b) => a + b.fatigue, 0) / players.length > 50 ? 'text-orange-500' : 'text-blue-400'}`}>
                      {Math.round(players.reduce((a, b) => a + b.fatigue, 0) / players.length)}%
                    </span>
                  </div>
                  <div className="h-px bg-gray-700 my-2"></div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">HIGH RISK:</span>
                    <span className="text-red-500">{highRiskPlayers.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">PLAYERS:</span>
                    <span className="text-blue-400">{players.length}</span>
                  </div>
                </div>
              </div>

              {/* Bench */}
              <Bench
                players={benchPlayers}
                onPlayerClick={handlePlayerClick}
                selectedPlayerId={undefined}
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
          <div>BRACE ANALYTICS v2.0</div>
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
