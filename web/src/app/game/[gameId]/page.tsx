"use client";

import { useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { useGameWebSocket } from "@/hooks/useGameWebSocket";
import Scoreboard from "@/components/Scoreboard";
import WinProbChart from "@/components/WinProbChart";
import PlayByPlayFeed from "@/components/PlayByPlayFeed";
import BoxScore from "@/components/BoxScore";
import MomentumPanel from "@/components/MomentumPanel";
import PregameBreakdown from "@/components/PregameBreakdown";

const TABS = ["Play-by-Play", "Box Score", "Key Moments", "Pre-Game"] as const;
type Tab = (typeof TABS)[number];

export default function GamePage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const gameId = params.gameId as string;
  const mode = (searchParams.get("mode") ?? "replay") as "live" | "replay";

  const { gameState, status, isLive } = useGameWebSocket(gameId, mode);
  const [activeTab, setActiveTab] = useState<Tab>("Play-by-Play");

  // Loading state
  if (!gameState) {
    return (
      <div className="flex items-center justify-center py-32">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-2 border-gray-600 border-t-gray-300 rounded-full animate-spin mb-4" />
          <p className="text-gray-400 text-sm">
            {status === "error" ? "Failed to connect" : "Loading game data..."}
          </p>
        </div>
      </div>
    );
  }

  const { home_team_abbrev: home, away_team_abbrev: away } = gameState;

  return (
    <div>
      {/* Connection status for live games */}
      {isLive && (
        <div className="flex items-center gap-2 mb-4">
          <div
            className={`w-2 h-2 rounded-full ${
              status === "connected"
                ? "bg-green-400 animate-pulse"
                : status === "connecting"
                ? "bg-yellow-400 animate-pulse"
                : "bg-red-400"
            }`}
          />
          <span className="text-xs text-gray-500">
            {status === "connected" ? "Live" : status === "connecting" ? "Connecting..." : "Disconnected"}
          </span>
        </div>
      )}

      {/* Scoreboard */}
      <Scoreboard
        homeTeam={home}
        awayTeam={away}
        homeScore={gameState.home_score}
        awayScore={gameState.away_score}
        homeWinProb={gameState.home_win_prob}
        period={gameState.period}
        clockSeconds={gameState.clock_seconds}
        gameStatus={isLive ? undefined : "Final"}
      />

      {/* Win probability chart */}
      <WinProbChart
        probHistory={gameState.prob_history}
        playLog={gameState.play_log}
        homeTeam={home}
        awayTeam={away}
      />

      {/* Tabbed panels */}
      <div className="rounded-xl bg-white/[0.03] border border-white/10 overflow-hidden">
        {/* Tab bar */}
        <div className="flex border-b border-white/5">
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-3 text-sm font-medium transition-colors relative ${
                activeTab === tab
                  ? "text-white"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {tab}
              {activeTab === tab && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="p-5">
          {activeTab === "Play-by-Play" && (
            <PlayByPlayFeed
              playLog={gameState.play_log}
              homeTeam={home}
              awayTeam={away}
            />
          )}
          {activeTab === "Box Score" && gameState.box_score && (
            <BoxScore
              boxScore={gameState.box_score}
              homeTeam={home}
              awayTeam={away}
            />
          )}
          {activeTab === "Key Moments" && (
            <MomentumPanel
              probHistory={gameState.prob_history}
              playLog={gameState.play_log}
              homeTeam={home}
              awayTeam={away}
            />
          )}
          {activeTab === "Pre-Game" && (
            <PregameBreakdown
              gameId={gameId}
              homeTeam={home}
              awayTeam={away}
            />
          )}
        </div>
      </div>
    </div>
  );
}
