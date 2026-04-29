"use client";

import type { PlayEvent } from "@/types/game";
import { formatPeriod, formatWinProb } from "@/lib/formatters";
import { teamColor } from "@/lib/teamMeta";

interface Props {
  playLog: PlayEvent[];
  homeTeam: string;
  awayTeam: string;
}

export default function PlayByPlayFeed({ playLog, homeTeam, awayTeam }: Props) {
  // Most recent first
  const rows = [...playLog].reverse();

  if (rows.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        No play-by-play data yet.
      </div>
    );
  }

  return (
    <div className="overflow-y-auto max-h-[400px] scrollbar-thin">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-zinc-950/95 backdrop-blur-sm z-10">
          <tr className="text-gray-400 text-xs uppercase tracking-wider border-b border-white/5">
            <th className="py-2 px-3 text-left w-16">Qtr</th>
            <th className="py-2 px-3 text-left w-16">Clock</th>
            <th className="py-2 px-3 text-left w-16">Team</th>
            <th className="py-2 px-3 text-left">Play</th>
            <th className="py-2 px-3 text-center w-20">Score</th>
            <th className="py-2 px-3 text-right w-16">Win%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((play, i) => {
            const col = play.team_abbrev ? teamColor(play.team_abbrev) : "#888";
            const clockMin = Math.floor(play.clock_seconds / 60);
            const clockSec = play.clock_seconds % 60;
            return (
              <tr
                key={`${play.action_number}-${i}`}
                className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
              >
                <td className="py-2 px-3 text-gray-400 font-mono text-xs">
                  {formatPeriod(play.period)}
                </td>
                <td className="py-2 px-3 text-gray-400 font-mono text-xs">
                  {clockMin}:{clockSec < 10 ? "0" : ""}{clockSec}
                </td>
                <td className="py-2 px-3">
                  <span className="font-semibold text-xs" style={{ color: col }}>
                    {play.team_abbrev || "—"}
                  </span>
                </td>
                <td className="py-2 px-3 text-gray-300 truncate max-w-[300px]">
                  {play.description}
                </td>
                <td className="py-2 px-3 text-center text-gray-300 font-mono text-xs tabular-nums">
                  {play.away_score} - {play.home_score}
                </td>
                <td className="py-2 px-3 text-right text-gray-300 font-mono text-xs tabular-nums">
                  {formatWinProb(play.home_win_prob)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
