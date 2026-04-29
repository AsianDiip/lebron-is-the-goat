"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ReferenceLine,
  ReferenceArea,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { ProbPoint, PlayEvent } from "@/types/game";
import { teamColor } from "@/lib/teamMeta";
import { formatWinProb } from "@/lib/formatters";

interface Props {
  probHistory: ProbPoint[];
  playLog: PlayEvent[];
  homeTeam: string;
  awayTeam: string;
}

interface ChartRow {
  seconds_remaining: number;
  y_espn: number; // 0 = home certain, 100 = away certain
  home_win_prob: number;
  description?: string;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ChartRow }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  const secAbs = Math.abs(d.seconds_remaining);
  const min = Math.floor(secAbs / 60);
  const sec = secAbs % 60;
  return (
    <div className="rounded-lg bg-zinc-900/95 border border-white/10 px-3 py-2 text-xs shadow-xl">
      <p className="text-white font-semibold">{formatWinProb(d.home_win_prob)} home</p>
      <p className="text-gray-400">{min}:{sec < 10 ? "0" : ""}{sec} remaining</p>
      {d.description && (
        <p className="text-gray-300 mt-1 max-w-[200px] truncate">{d.description}</p>
      )}
    </div>
  );
}

export default function WinProbChart({ probHistory, playLog, homeTeam, awayTeam }: Props) {
  const homeCol = teamColor(homeTeam);
  const awayCol = teamColor(awayTeam);

  const data = useMemo<ChartRow[]>(() => {
    return probHistory.map((p, i) => ({
      seconds_remaining: p.seconds_remaining,
      y_espn: (1 - p.home_win_prob) * 100,
      home_win_prob: p.home_win_prob,
      description: playLog[i]?.description,
    }));
  }, [probHistory, playLog]);

  if (data.length === 0) {
    return (
      <div className="rounded-xl bg-white/[0.03] border border-white/10 p-8 mb-6 text-center text-gray-500">
        Waiting for play-by-play data...
      </div>
    );
  }

  const maxSec = Math.max(data[0]?.seconds_remaining ?? 2880, 2880);
  const minSec = data[data.length - 1]?.seconds_remaining ?? 0;

  // Quarter boundaries
  const quarters = [
    { sec: 2160, label: "" },
    { sec: 1440, label: "" },
    { sec: 720, label: "" },
    { sec: 0, label: "" },
  ].filter((q) => q.sec >= minSec && q.sec <= maxSec);

  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/10 p-4 mb-6 relative">
      {/* Team labels */}
      <div className="absolute top-6 left-12 z-10">
        <span className="text-xs font-bold tracking-wider" style={{ color: awayCol }}>
          {awayTeam}
        </span>
      </div>
      <div className="absolute bottom-10 left-12 z-10">
        <span className="text-xs font-bold tracking-wider" style={{ color: homeCol }}>
          {homeTeam}
        </span>
      </div>

      {/* Quarter labels */}
      <div className="flex justify-around text-xs text-gray-500 font-medium px-12 mb-1">
        {["1st", "2nd", "3rd", "4th"].map((q, i) => {
          const center = 2880 - i * 720 - 360;
          return center >= minSec ? <span key={q}>{q}</span> : <span key={q} />;
        })}
        {minSec < 0 && (
          <span className="text-amber-500">OT</span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <AreaChart
          data={data}
          margin={{ top: 10, right: 10, bottom: 5, left: 10 }}
        >
          {/* Gradient definitions */}
          <defs>
            <linearGradient id="homeGrad" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0%" stopColor={homeCol} stopOpacity={0.3} />
              <stop offset="100%" stopColor={homeCol} stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="awayGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={awayCol} stopOpacity={0.3} />
              <stop offset="100%" stopColor={awayCol} stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <XAxis
            dataKey="seconds_remaining"
            type="number"
            domain={[maxSec + 20, minSec - 20]}
            reversed
            hide
          />
          <YAxis
            domain={[0, 100]}
            hide
          />

          {/* Toss-up band */}
          <ReferenceArea
            y1={35}
            y2={65}
            fill="white"
            fillOpacity={0.03}
          />

          {/* 50% midline */}
          <ReferenceLine y={50} stroke="#555" strokeDasharray="3 3" strokeWidth={0.8} />

          {/* Quarter boundaries */}
          {quarters.map((q) => (
            <ReferenceLine
              key={q.sec}
              x={q.sec}
              stroke="#333"
              strokeWidth={0.8}
            />
          ))}

          {/* Home region (below 50) */}
          <Area
            type="monotone"
            dataKey="y_espn"
            stroke={homeCol}
            strokeWidth={2}
            fill="url(#homeGrad)"
            baseLine={50}
            isAnimationActive={false}
          />

          <Tooltip content={<CustomTooltip />} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
