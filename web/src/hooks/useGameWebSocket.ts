"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { GameState, ConnectionStatus } from "@/types/game";
import { wsUrl, fetchLiveGame, fetchReplayGame } from "@/lib/api";

interface UseGameResult {
  gameState: GameState | null;
  status: ConnectionStatus;
  isLive: boolean;
}

/**
 * Hook that connects to the game WebSocket for live updates,
 * with REST polling fallback. For replay mode, fetches once.
 */
export function useGameWebSocket(
  gameId: string | null,
  mode: "live" | "replay" = "live",
): UseGameResult {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryCount = useRef(0);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
    }
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
    }
  }, []);

  // Replay mode: single fetch
  useEffect(() => {
    if (!gameId || mode !== "replay") return;
    setStatus("connecting");
    fetchReplayGame(gameId)
      .then((data) => {
        setGameState(data);
        setStatus("connected");
      })
      .catch(() => setStatus("error"));
  }, [gameId, mode]);

  // Live mode: WebSocket with REST fallback
  useEffect(() => {
    if (!gameId || mode !== "live") return;

    let cancelled = false;

    function connect() {
      if (cancelled) return;
      setStatus("connecting");

      const ws = new WebSocket(wsUrl(gameId!));
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setStatus("connected");
        retryCount.current = 0;
        // Send periodic pings to keep connection alive
        pollTimer.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 30_000);
      };

      ws.onmessage = (event) => {
        if (cancelled) return;
        try {
          const data = JSON.parse(event.data) as GameState;
          setGameState(data);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        setStatus("disconnected");
        if (pollTimer.current) clearInterval(pollTimer.current);
        // Reconnect with exponential backoff
        const delay = Math.min(1000 * 2 ** retryCount.current, 30_000);
        retryCount.current++;
        reconnectTimer.current = setTimeout(() => {
          if (!cancelled) connect();
        }, delay);
      };

      ws.onerror = () => {
        if (cancelled) return;
        setStatus("error");
        ws.close();
        // Fall back to REST polling
        startRestPolling();
      };
    }

    function startRestPolling() {
      if (cancelled || pollTimer.current) return;
      pollTimer.current = setInterval(async () => {
        if (cancelled) return;
        try {
          const data = await fetchLiveGame(gameId!);
          setGameState(data);
          setStatus("connected");
        } catch {
          setStatus("error");
        }
      }, 5_000);
    }

    connect();

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [gameId, mode, cleanup]);

  return {
    gameState,
    status,
    isLive: mode === "live",
  };
}
