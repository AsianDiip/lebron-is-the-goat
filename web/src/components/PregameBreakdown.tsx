"use client";

import { useEffect, useState } from "react";
import type { PregameBreakdownData } from "@/types/game";
import { fetchPregameBreakdown } from "@/lib/api";
import { teamColor } from "@/lib/teamMeta";
import { formatWinProb } from "@/lib/formatters";

interface Props {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
}

export default function PregameBreakdown({ gameId, homeTeam, awayTeam }: Props) {
  const [data, setData] = useState<PregameBreakdownData | null>(null);
  const [error, setError] = useState(false);
  const homeCol = teamColor(homeTeam);
  const awayCol = teamColor(awayTeam);

  useEffect(() => {
    fetchPregameBreakdown(gameId)
      .then(setData)
      .catch(() => setError(true));
  }, [gameId]);

  if (error) {
    return <div className="text-center text-gray-500 py-8">Pre-game data unavailable.</div>;
  }

  if (!data) {
    return (
      <div className="text-center text-gray-500 py-8">
        <div className="inline-block w-5 h-5 border-2 border-gray-600 border-t-gray-300 rounded-full animate-spin" />
      </div>
    );
  }

  // Find max absolute value for scaling bars
  const entries = Object.entries(data.features);
  const maxVal = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div className="space-y-5">
      {/* Pre-game probability */}
      <div className="text-center">
        <p className="text-xs text-gray-400 uppercase tracking-wider">Pre-game Win Probability</p>
        <p className="text-2xl font-bold text-white mt-1">
          <span style={{ color: homeCol }}>{homeTeam} {formatWinProb(data.pre_game_prob)}</span>
          <span className="text-gray-500 mx-2">vs</span>
          <span style={{ color: awayCol }}>{awayTeam} {formatWinProb(1 - data.pre_game_prob)}</span>
        </p>
      </div>

      {/* Feature bars */}
      <div className="space-y-2">
        <h3 className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
          Feature Breakdown
        </h3>
        {entries.map(([key, value]) => {
          const label = data.labels[key] ?? key;
          const barPct = (Math.abs(value) / maxVal) * 50;
          const favorsHome = value > 0;

          return (
            <div key={key} className="flex items-center gap-2">
              <span className="text-xs text-gray-400 w-36 text-right shrink-0 truncate">
                {label}
              </span>
              <div className="flex-1 flex items-center h-5">
                {/* Center-aligned bar: away side (left) | center | home side (right) */}
                <div className="w-1/2 flex justify-end">
                  {!favorsHome && (
                    <div
                      className="h-4 rounded-l transition-all duration-300"
                      style={{
                        width: `${barPct}%`,
                        background: awayCol,
                        opacity: 0.7,
                      }}
                    />
                  )}
                </div>
                <div className="w-px h-5 bg-gray-600 shrink-0" />
                <div className="w-1/2">
                  {favorsHome && (
                    <div
                      className="h-4 rounded-r transition-all duration-300"
                      style={{
                        width: `${barPct}%`,
                        background: homeCol,
                        opacity: 0.7,
                      }}
                    />
                  )}
                </div>
              </div>
              <span className="text-xs font-mono text-gray-400 w-14 shrink-0">
                {value > 0 ? "+" : ""}{value.toFixed(2)}
              </span>
            </div>
          );
        })}
        <div className="flex justify-between text-xs text-gray-500 px-36">
          <span style={{ color: awayCol }}>Favors {awayTeam}</span>
          <span style={{ color: homeCol }}>Favors {homeTeam}</span>
        </div>
      </div>
    </div>
  );
}
