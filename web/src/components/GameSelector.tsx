"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { GameListItem, GameListResponse } from "@/types/game";
import { fetchGameList } from "@/lib/api";
import { teamColor } from "@/lib/teamMeta";

export default function GameSelector() {
  const router = useRouter();
  const [seasons, setSeasons] = useState<string[]>([]);
  const [selectedSeason, setSelectedSeason] = useState<string>("");
  const [games, setGames] = useState<GameListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [liveGameId, setLiveGameId] = useState("");

  // Fetch initial data
  useEffect(() => {
    fetchGameList()
      .then((data: GameListResponse) => {
        setSeasons(data.seasons);
        setGames(data.games);
        if (data.seasons.length > 0) {
          setSelectedSeason(data.seasons[0]);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Refetch when season changes
  useEffect(() => {
    if (!selectedSeason) return;
    setLoading(true);
    fetchGameList(selectedSeason)
      .then((data: GameListResponse) => {
        setGames(data.games);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [selectedSeason]);

  const goToGame = (gameId: string, mode: "replay" | "live") => {
    router.push(`/game/${gameId}?mode=${mode}`);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Live game entry */}
      <div className="rounded-xl bg-white/[0.03] border border-white/10 p-6">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 font-semibold mb-4">
          Live Game
        </h2>
        <div className="flex gap-3">
          <input
            type="text"
            value={liveGameId}
            onChange={(e) => setLiveGameId(e.target.value)}
            placeholder="Enter Game ID (e.g. 0022401234)"
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-white/20 focus:ring-1 focus:ring-white/10 font-mono"
          />
          <button
            onClick={() => liveGameId && goToGame(liveGameId.trim(), "live")}
            disabled={!liveGameId.trim()}
            className="px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium transition-colors"
          >
            Track Live
          </button>
        </div>
      </div>

      {/* Historical games */}
      <div className="rounded-xl bg-white/[0.03] border border-white/10 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm uppercase tracking-wider text-gray-400 font-semibold">
            Historical Games
          </h2>
          <select
            value={selectedSeason}
            onChange={(e) => setSelectedSeason(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none cursor-pointer"
          >
            {seasons.map((s) => (
              <option key={s} value={s} className="bg-zinc-900">
                {s}
              </option>
            ))}
          </select>
        </div>

        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block w-6 h-6 border-2 border-gray-600 border-t-gray-300 rounded-full animate-spin" />
          </div>
        ) : (
          <div className="overflow-y-auto max-h-[500px] scrollbar-thin">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-zinc-950/95 backdrop-blur-sm">
                <tr className="text-xs uppercase tracking-wider text-gray-500 border-b border-white/5">
                  <th className="py-2 px-3 text-left">Date</th>
                  <th className="py-2 px-3 text-left">Matchup</th>
                  <th className="py-2 px-3 text-center">Score</th>
                  <th className="py-2 px-3 text-right">Result</th>
                  <th className="py-2 px-3 text-right"></th>
                </tr>
              </thead>
              <tbody>
                {games.map((game) => (
                  <tr
                    key={game.game_id}
                    className="border-b border-white/[0.03] hover:bg-white/[0.03] transition-colors cursor-pointer group"
                    onClick={() => goToGame(game.game_id, "replay")}
                  >
                    <td className="py-2.5 px-3 text-gray-400 font-mono text-xs">
                      {game.game_date}
                    </td>
                    <td className="py-2.5 px-3">
                      <span style={{ color: teamColor(game.away_team) }} className="font-semibold">
                        {game.away_team}
                      </span>
                      <span className="text-gray-500 mx-2">@</span>
                      <span style={{ color: teamColor(game.home_team) }} className="font-semibold">
                        {game.home_team}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center text-gray-300 font-mono text-xs tabular-nums">
                      {game.away_pts} - {game.home_pts}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={`text-xs font-semibold ${game.home_wl === "W" ? "text-green-400" : "text-red-400"}`}>
                        {game.home_wl === "W" ? `${game.home_team} W` : `${game.away_team} W`}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className="text-xs text-gray-600 group-hover:text-gray-300 transition-colors">
                        Replay &rarr;
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
