"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface PlayerStatsProps {
  players: Array<{
    id: number;
    name: string;
    number: number;
    concussionProb: number;
    fatigue: number;
  }>;
}

export function PlayerStats({ players }: PlayerStatsProps) {
  const concussionData = players.map(p => ({
    name: `#${p.number}`,
    value: p.concussionProb,
    isHighRisk: p.concussionProb > 70
  }));

  const fatigueData = players.map(p => ({
    name: `#${p.number}`,
    value: p.fatigue,
    isHighRisk: p.fatigue > 70
  }));

  return (
    <div className="space-y-4">
      {/* Concussion Risk Chart */}
      <div className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl">
        <div className="bg-red-600 px-4 py-2 border-b-4 border-white">
          <h3 className="text-white text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
            CONCUSSION PROBABILITY
          </h3>
        </div>
        <div className="p-4 bg-gray-900">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={concussionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                stroke="#9ca3af"
                angle={-45}
                textAnchor="end"
                height={70}
                interval={0}
                style={{ fontFamily: "'Press Start 2P', cursive", fontSize: '6px' }}
              />
              <YAxis
                stroke="#9ca3af"
                style={{ fontFamily: "'Press Start 2P', cursive", fontSize: '8px' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '2px solid white',
                  borderRadius: '4px',
                  fontFamily: "'Press Start 2P', cursive",
                  fontSize: '8px'
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {concussionData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isHighRisk ? '#ef4444' : entry.value > 40 ? '#eab308' : '#22c55e'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Fatigue Chart */}
      <div className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl">
        <div className="bg-orange-600 px-4 py-2 border-b-4 border-white">
          <h3 className="text-white text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
            FATIGUE LEVELS
          </h3>
        </div>
        <div className="p-4 bg-gray-900">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={fatigueData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                stroke="#9ca3af"
                angle={-45}
                textAnchor="end"
                height={70}
                interval={0}
                style={{ fontFamily: "'Press Start 2P', cursive", fontSize: '6px' }}
              />
              <YAxis
                stroke="#9ca3af"
                style={{ fontFamily: "'Press Start 2P', cursive", fontSize: '8px' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '2px solid white',
                  borderRadius: '4px',
                  fontFamily: "'Press Start 2P', cursive",
                  fontSize: '8px'
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {fatigueData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isHighRisk ? '#ef4444' : entry.value > 40 ? '#f97316' : '#3b82f6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
