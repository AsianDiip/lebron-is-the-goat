import type { GameState, GameListResponse, PregameBreakdownData } from "@/types/game";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function fetchLiveGame(gameId: string): Promise<GameState> {
  return fetchJson<GameState>(`/live/${gameId}`);
}

export function fetchReplayGame(gameId: string): Promise<GameState> {
  return fetchJson<GameState>(`/replay/${gameId}`);
}

export function fetchGameList(season?: string): Promise<GameListResponse> {
  const query = season ? `?season=${encodeURIComponent(season)}` : "";
  return fetchJson<GameListResponse>(`/games${query}`);
}

export function fetchPregameBreakdown(gameId: string): Promise<PregameBreakdownData> {
  return fetchJson<PregameBreakdownData>(`/pregame/${gameId}/breakdown`);
}

export function wsUrl(gameId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/${gameId}`;
}
