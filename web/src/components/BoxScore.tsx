"use client";

import type { BoxScoreData } from "@/types/game";
import { teamColor } from "@/lib/teamMeta";

interface Props {
  boxScore: BoxScoreData;
  homeTeam: string;
  awayTeam: string;
}

interface StatRow {
  label: string;
  away: string;
  home: string;
  awayVal: number;
  homeVal: number;
  higherIsBetter: boolean;
}

export default function BoxScore({ boxScore, homeTeam, awayTeam }: Props) {
  const homeCol = teamColor(homeTeam);
  const awayCol = teamColor(awayTeam);
  const h = boxScore.home;
  const a = boxScore.away;

  const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;

  const stats: StatRow[] = [
    { label: "FG", away: `${a.fgm}/${a.fga} (${fmtPct(a.fg_pct)})`, home: `${h.fgm}/${h.fga} (${fmtPct(h.fg_pct)})`, awayVal: a.fg_pct, homeVal: h.fg_pct, higherIsBetter: true },
    { label: "2PT", away: `${a.two_pm}/${a.two_pa} (${fmtPct(a.two_pt_pct)})`, home: `${h.two_pm}/${h.two_pa} (${fmtPct(h.two_pt_pct)})`, awayVal: a.two_pt_pct, homeVal: h.two_pt_pct, higherIsBetter: true },
    { label: "3PT", away: `${a.three_pm}/${a.three_pa} (${fmtPct(a.three_pt_pct)})`, home: `${h.three_pm}/${h.three_pa} (${fmtPct(h.three_pt_pct)})`, awayVal: a.three_pt_pct, homeVal: h.three_pt_pct, higherIsBetter: true },
    { label: "FT", away: `${a.ftm}/${a.fta} (${fmtPct(a.ft_pct)})`, home: `${h.ftm}/${h.fta} (${fmtPct(h.ft_pct)})`, awayVal: a.ft_pct, homeVal: h.ft_pct, higherIsBetter: true },
    { label: "Fouls", away: `${a.fouls}`, home: `${h.fouls}`, awayVal: a.fouls, homeVal: h.fouls, higherIsBetter: false },
    { label: "Turnovers", away: `${a.turnovers}`, home: `${h.turnovers}`, awayVal: a.turnovers, homeVal: h.turnovers, higherIsBetter: false },
    { label: "Timeouts", away: `${a.timeouts_remaining}`, home: `${h.timeouts_remaining}`, awayVal: a.timeouts_remaining, homeVal: h.timeouts_remaining, higherIsBetter: true },
  ];

  return (
    <div className="overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs uppercase tracking-wider border-b border-white/10">
            <th className="py-3 px-4 text-left" style={{ color: awayCol }}>
              {awayTeam}
            </th>
            <th className="py-3 px-4 text-center text-gray-500">Stat</th>
            <th className="py-3 px-4 text-right" style={{ color: homeCol }}>
              {homeTeam}
            </th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => {
            const awayLeads = s.higherIsBetter
              ? s.awayVal > s.homeVal
              : s.awayVal < s.homeVal;
            const homeLeads = s.higherIsBetter
              ? s.homeVal > s.awayVal
              : s.homeVal < s.awayVal;
            const tied = s.awayVal === s.homeVal;

            return (
              <tr
                key={s.label}
                className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
              >
                <td className={`py-2.5 px-4 text-left tabular-nums ${awayLeads && !tied ? "text-white font-semibold" : "text-gray-400"}`}>
                  {s.away}
                </td>
                <td className="py-2.5 px-4 text-center text-gray-500 text-xs uppercase">
                  {s.label}
                </td>
                <td className={`py-2.5 px-4 text-right tabular-nums ${homeLeads && !tied ? "text-white font-semibold" : "text-gray-400"}`}>
                  {s.home}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
