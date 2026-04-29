"use client";

import { teamLogoUrl, teamColor, teamName } from "@/lib/teamMeta";
import { formatClock, formatWinProb } from "@/lib/formatters";

interface Props {
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  homeWinProb: number;
  period?: number;
  clockSeconds?: number;
  gameStatus?: string;
}

export default function Scoreboard({
  homeTeam,
  awayTeam,
  homeScore,
  awayScore,
  homeWinProb,
  period,
  clockSeconds,
  gameStatus,
}: Props) {
  const homeLogo = teamLogoUrl(homeTeam);
  const awayLogo = teamLogoUrl(awayTeam);
  const homeCol = teamColor(homeTeam);
  const awayCol = teamColor(awayTeam);
  const awayProb = 1 - homeWinProb;
  const probPct = Math.round(homeWinProb * 100);

  const clock =
    period != null && clockSeconds != null
      ? formatClock(period, clockSeconds)
      : gameStatus ?? "";

  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/10 backdrop-blur-sm p-6 mb-6">
      {/* Teams + Score row */}
      <div className="flex items-center justify-between mb-5">
        {/* Away team */}
        <div className="flex items-center gap-4 flex-1">
          {awayLogo && (
            <img src={awayLogo} alt={awayTeam} className="w-14 h-14 object-contain" />
          )}
          <div>
            <p className="text-xs text-gray-400 tracking-wider uppercase">{awayTeam}</p>
            <p className="text-xl font-bold" style={{ color: awayCol }}>
              {teamName(awayTeam)}
            </p>
          </div>
        </div>

        {/* Score */}
        <div className="text-center px-8">
          <p className="text-5xl font-bold tracking-tight text-white tabular-nums">
            {awayScore}
            <span className="text-gray-500 mx-3">&ndash;</span>
            {homeScore}
          </p>
          <p className="text-sm text-gray-400 mt-1 font-mono">{clock}</p>
        </div>

        {/* Home team */}
        <div className="flex items-center gap-4 flex-1 justify-end">
          <div className="text-right">
            <p className="text-xs text-gray-400 tracking-wider uppercase">{homeTeam}</p>
            <p className="text-xl font-bold" style={{ color: homeCol }}>
              {teamName(homeTeam)}
            </p>
          </div>
          {homeLogo && (
            <img src={homeLogo} alt={homeTeam} className="w-14 h-14 object-contain" />
          )}
        </div>
      </div>

      {/* Win probability bar */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-semibold w-20 text-right" style={{ color: awayCol }}>
          {formatWinProb(awayProb)}
        </span>
        <div className="flex-1 h-3 rounded-full overflow-hidden bg-white/10 relative">
          <div
            className="absolute inset-y-0 left-0 rounded-full transition-all duration-500 ease-out"
            style={{
              width: `${100 - probPct}%`,
              background: awayCol,
              opacity: 0.8,
            }}
          />
          <div
            className="absolute inset-y-0 right-0 rounded-full transition-all duration-500 ease-out"
            style={{
              width: `${probPct}%`,
              background: homeCol,
              opacity: 0.8,
            }}
          />
        </div>
        <span className="text-xs font-semibold w-20" style={{ color: homeCol }}>
          {formatWinProb(homeWinProb)}
        </span>
      </div>
    </div>
  );
}
