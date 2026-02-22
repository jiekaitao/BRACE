"use client";

import { PlayerCard } from "./PlayerCard";

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

interface FootballFieldProps {
  players: Player[];
  onPlayerClick: (player: Player) => void;
  selectedPlayerId?: number;
}

export function FootballField({ players, onPlayerClick, selectedPlayerId }: FootballFieldProps) {
  return (
    <div className="relative w-full h-full bg-gradient-to-b from-green-700 via-green-600 to-green-700 rounded-lg overflow-hidden border-8 border-white shadow-2xl">
      {/* Field lines */}
      <div className="absolute inset-0">
        {/* Yard lines */}
        {[10, 20, 30, 40, 50, 60, 70, 80, 90].map((yard) => (
          <div
            key={yard}
            className="absolute w-full border-t-4 border-white border-dashed opacity-40"
            style={{ top: `${yard}%` }}
          />
        ))}

        {/* Center line */}
        <div className="absolute top-1/2 w-full border-t-4 border-white opacity-60" />

        {/* Side lines */}
        <div className="absolute left-0 h-full border-l-8 border-white" />
        <div className="absolute right-0 h-full border-r-8 border-white" />

        {/* End zones */}
        <div className="absolute top-0 w-full h-[10%] bg-blue-900 opacity-30" />
        <div className="absolute bottom-0 w-full h-[10%] bg-blue-900 opacity-30" />
      </div>

      {/* Hash marks pattern */}
      <div className="absolute inset-0 opacity-20">
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={`left-${i}`}
            className="absolute left-1/4 w-2 h-1 bg-white"
            style={{ top: `${5 + i * 5}%` }}
          />
        ))}
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={`right-${i}`}
            className="absolute right-1/4 w-2 h-1 bg-white"
            style={{ top: `${5 + i * 5}%` }}
          />
        ))}
      </div>

      {/* Players */}
      {players.map((player) => (
        <PlayerCard
          key={player.id}
          {...player}
          onClick={() => onPlayerClick(player)}
          isSelected={player.id === selectedPlayerId}
        />
      ))}

      {/* Field texture overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-transparent via-green-800 to-transparent opacity-20 pointer-events-none" />
    </div>
  );
}
