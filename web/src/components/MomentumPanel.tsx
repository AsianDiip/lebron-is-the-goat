"use client";

import { useMemo } from "react";
import type { ProbPoint, PlayEvent } from "@/types/game";
import { topSwings, countLeadChanges, largestRun } from "@/lib/momentum";
import { formatPeriod, formatWinProb } from "@/lib/formatters";
import { teamColor } from "@/lib/teamMeta";

interface Props {
  probHistory: ProbPoint[];
  playLog: PlayEvent[];
  homeTeam: string;
  awayTeam: string;
}

export default function MomentumPanel({ probHistory, playLog, homeTeam, awayTeam }: Props) {
  const homeCol = teamColor(homeTeam);
  const awayCol = teamColor(awayTeam);

  const swings = useMemo(() => topSwings(probHistory, playLog, 5), [probHistory, playLog]);
  const leadChanges = useMemo(() => countLeadChanges(playLog), [playLog]);
  const bigRun = useMemo(() => largestRun(playLog), [playLog]);

  if (probHistory.length < 2) {
    return (
      <div className="text-center text-gray-500 py-8">
        Not enough data for momentum analysis.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-lg bg-white/[0.03] border border-white/5 p-4 text-center">
          <p className="text-3xl font-bold text-white">{leadChanges}</p>
          <p className="text-xs text-gray-400 uppercase tracking-wider mt-1">Lead Changes</p>
        </div>
        {bigRun && (
          <div className="rounded-lg bg-white/[0.03] border border-white/5 p-4 text-center">
            <p className="text-3xl font-bold text-white">{bigRun.points}-0</p>
            <p className="text-xs uppercase tracking-wider mt-1" style={{
              color: bigRun.team === "home" ? homeCol : awayCol,
            }}>
              Largest Run ({bigRun.team === "home" ? homeTeam : awayTeam})
            </p>
          </div>
        )}
      </div>

      {/* Top momentum swings */}
      <div>
        <h3 className="text-xs text-gray-400 uppercase tracking-wider mb-3 font-semibold">
          Biggest Momentum Swings
        </h3>
        <div className="space-y-2">
          {swings.map((swing, i) => {
            const direction = swing.home_win_prob > 0.5 ? "home" : "away";
            const barColor = direction === "home" ? homeCol : awayCol;
            const barWidth = Math.min(swing.delta * 500, 100); // scale for visual

            return (
              <div
                key={i}
                className="rounded-lg bg-white/[0.03] border border-white/5 p-3 flex items-center gap-3"
              >
                <div className="shrink-0 w-12 text-center">
                  <span className="text-xs font-mono text-gray-400">
                    {formatPeriod(swing.period)}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-200 truncate">{swing.description}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${barWidth}%`, background: barColor }}
                      />
                    </div>
                    <span className="text-xs font-mono text-gray-400 shrink-0">
                      {(swing.delta * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="shrink-0 text-right">
                  <span className="text-xs text-gray-500">
                    {swing.away_score}-{swing.home_score}
                  </span>
                  <br />
                  <span className="text-xs font-mono" style={{ color: barColor }}>
                    {formatWinProb(swing.home_win_prob)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
