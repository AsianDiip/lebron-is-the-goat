"""
api.py — FastAPI server for NBA win probability.

Endpoints:
    GET  /pregame/{game_id}            — pre-game win probability
    GET  /pregame/{game_id}/breakdown  — pre-game feature values + labels
    GET  /live/{game_id}               — current live probability + game state
    GET  /games                        — historical game list for replay selector
    GET  /replay/{game_id}             — full replay of a historical game
    WS   /ws/{game_id}                 — WebSocket for real-time updates

Usage:
    uvicorn live.api:app --reload
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib

from live.game_state import GameState
from live.poller import get_team_abbrev_map, lookup_pregame
from model.train_ingame import INGAME_FEATURES, StratifiedCalibrator  # noqa: F401
from model.train_pregame import PREGAME_FEATURES

# Register StratifiedCalibrator in all module namespaces that joblib/pickle
# might look up during deserialization (e.g. __main__, __mp_main__).
import __main__
__main__.StratifiedCalibrator = StratifiedCalibrator
if "__mp_main__" in sys.modules:
    sys.modules["__mp_main__"].StratifiedCalibrator = StratifiedCalibrator

MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
FIGURES_DIR = MODEL_DIR / "eval_figures"

# Loaded once at startup
_pregame_model = None
_ingame_model = None
_abbrev_map: dict[str, int] = {}

# Active game states keyed by game_id
_active_games: dict[str, GameState] = {}
_active_probs: dict[str, float] = {}  # latest probability per game
_prob_history: dict[str, list[dict]] = {}  # full probability timeline per game
_play_log: dict[str, list[dict]] = {}  # play-by-play log per game
_poll_tasks: dict[str, asyncio.Task] = {}

# WebSocket connections keyed by game_id
_ws_connections: dict[str, set[WebSocket]] = {}

# Human-readable labels for pre-game features
_PREGAME_LABELS = {
    "elo_diff": "ELO Rating Diff",
    "efg_pct_diff": "eFG% Diff",
    "ortg_diff": "Off. Rating Diff",
    "drtg_diff": "Def. Rating Diff",
    "prev_season_win_pct_diff": "Prev Season Win% Diff",
    "rest_days_diff": "Rest Days Diff",
    "home_flag": "Home Court",
    "ast_pct_diff": "Assist Rate Diff",
    "tov_pct_diff": "Turnover Rate Diff",
}


# ------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pregame_model, _ingame_model, _abbrev_map
    _pregame_model = joblib.load(MODEL_DIR / "pregame.pkl")
    _ingame_model = joblib.load(MODEL_DIR / "ingame.pkl")
    _abbrev_map = get_team_abbrev_map()
    yield


app = FastAPI(title="NBA Win Probability API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve evaluation figures as static files
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_pre_game_prob(pregame_info: dict) -> float:
    vec = np.array([[pregame_info[f] for f in PREGAME_FEATURES]], dtype=np.float64)
    return float(_pregame_model.predict_proba(vec)[0, 1])


def _reverse_abbrev_map() -> dict[int, str]:
    """Build team_id -> abbreviation lookup from the abbrev map."""
    return {v: k for k, v in _abbrev_map.items()}


def _safe_pct(made: int, attempted: int) -> float:
    return round(made / attempted, 3) if attempted > 0 else 0.0


def _build_box_score(state: GameState) -> dict:
    """Extract box score stats from a GameState."""
    return {
        "home": {
            "fgm": state.home_fgm, "fga": state.home_fga,
            "fg_pct": _safe_pct(state.home_fgm, state.home_fga),
            "two_pm": state.home_2pm, "two_pa": state.home_2pa,
            "two_pt_pct": _safe_pct(state.home_2pm, state.home_2pa),
            "three_pm": state.home_3pm, "three_pa": state.home_3pa,
            "three_pt_pct": _safe_pct(state.home_3pm, state.home_3pa),
            "ftm": state.home_ftm, "fta": state.home_fta,
            "ft_pct": _safe_pct(state.home_ftm, state.home_fta),
            "fouls": state.home_fouls,
            "turnovers": state.home_turnovers,
            "timeouts_remaining": 7 - state.home_timeouts_used,
        },
        "away": {
            "fgm": state.away_fgm, "fga": state.away_fga,
            "fg_pct": _safe_pct(state.away_fgm, state.away_fga),
            "two_pm": state.away_2pm, "two_pa": state.away_2pa,
            "two_pt_pct": _safe_pct(state.away_2pm, state.away_2pa),
            "three_pm": state.away_3pm, "three_pa": state.away_3pa,
            "three_pt_pct": _safe_pct(state.away_3pm, state.away_3pa),
            "ftm": state.away_ftm, "fta": state.away_fta,
            "ft_pct": _safe_pct(state.away_ftm, state.away_fta),
            "fouls": state.away_fouls,
            "turnovers": state.away_turnovers,
            "timeouts_remaining": 7 - state.away_timeouts_used,
        },
    }


def _build_game_payload(game_id: str) -> dict:
    """Build the full game state payload used by both REST and WebSocket."""
    state = _active_games[game_id]
    prob = _active_probs.get(game_id, state.pre_game_prob)
    id_to_abbrev = _reverse_abbrev_map()
    return {
        "game_id": game_id,
        "period": state.period,
        "clock_seconds": state.clock_seconds,
        "home_score": state.home_score,
        "away_score": state.away_score,
        "home_win_prob": round(prob, 4),
        "events_processed": len(state.play_log),
        "home_team_id": state.home_team_id,
        "away_team_id": state.away_team_id,
        "home_team_abbrev": id_to_abbrev.get(state.home_team_id, "HOME"),
        "away_team_abbrev": id_to_abbrev.get(state.away_team_id, "AWAY"),
        "pre_game_prob": round(state.pre_game_prob, 4),
        "prob_history": _prob_history.get(game_id, []),
        "play_log": _play_log.get(game_id, [])[-50:],
        "box_score": _build_box_score(state),
    }


def _ensure_game_state(game_id: str) -> tuple[GameState, float]:
    """Get or create a GameState for the given game_id."""
    if game_id in _active_games:
        return _active_games[game_id], _active_probs.get(game_id, 0.5)

    pregame_info = lookup_pregame(game_id)
    if pregame_info is None:
        raise HTTPException(404, f"No pregame data for game {game_id}")

    pre_game_prob = _get_pre_game_prob(pregame_info)

    state = GameState(
        game_id=game_id,
        home_team_id=pregame_info["home_team_id"],
        away_team_id=pregame_info["away_team_id"],
        pre_game_prob=pre_game_prob,
        team_abbrev_map=_abbrev_map,
    )
    _active_games[game_id] = state
    _active_probs[game_id] = pre_game_prob
    _prob_history[game_id] = []
    _play_log[game_id] = []
    return state, pre_game_prob


async def _broadcast_ws(game_id: str) -> None:
    """Send the current game payload to all WebSocket clients for a game."""
    clients = _ws_connections.get(game_id)
    if not clients:
        return
    payload = _build_game_payload(game_id)
    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def _poll_loop(game_id: str, interval: int = 30) -> None:
    """Background polling task for a live game."""
    from live.poller import _fetch_live_actions

    state = _active_games[game_id]
    consecutive_empty = 0

    while consecutive_empty < 20:
        # Use shorter interval when WebSocket clients are connected
        effective_interval = 15 if _ws_connections.get(game_id) else interval

        try:
            actions = await asyncio.to_thread(_fetch_live_actions, game_id)
        except Exception:
            await asyncio.sleep(effective_interval)
            continue

        new_count = 0
        id_to_abbrev = _reverse_abbrev_map()
        for event in actions:
            if not state.update(event):
                continue
            new_count += 1

            fv = state.to_feature_vector().reshape(1, -1)
            prob = float(_ingame_model.predict_proba(fv)[0, 1])
            _active_probs[game_id] = prob

            action_num = event.get("actionNumber") or event.get("action_number") or 0
            team_id = event.get("teamId") or event.get("team_id") or 0
            _prob_history[game_id].append({
                "seconds_remaining": state.seconds_remaining,
                "home_win_prob": round(prob, 4),
                "action_number": int(action_num),
            })
            _play_log[game_id].append({
                "action_number": int(action_num),
                "period": state.period,
                "clock_seconds": state.clock_seconds,
                "team_abbrev": id_to_abbrev.get(int(team_id) if team_id else 0, ""),
                "description": str(event.get("description") or ""),
                "home_score": state.home_score,
                "away_score": state.away_score,
                "home_win_prob": round(prob, 4),
            })

        # Broadcast to WebSocket clients if there are new events
        if new_count > 0:
            await _broadcast_ws(game_id)

        consecutive_empty = 0 if new_count > 0 else consecutive_empty + 1
        await asyncio.sleep(effective_interval)


# ------------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------------

@app.get("/pregame/{game_id}")
def pregame(game_id: str) -> dict:
    """Return pre-game win probability for the home team."""
    pregame_info = lookup_pregame(game_id)
    if pregame_info is None:
        raise HTTPException(404, f"No pregame data for game {game_id}")

    prob = _get_pre_game_prob(pregame_info)
    return {
        "game_id": game_id,
        "home_team_id": pregame_info["home_team_id"],
        "away_team_id": pregame_info["away_team_id"],
        "pre_game_prob": round(prob, 4),
    }


@app.get("/pregame/{game_id}/breakdown")
def pregame_breakdown(game_id: str) -> dict:
    """Return pre-game feature values with human-readable labels."""
    pregame_info = lookup_pregame(game_id)
    if pregame_info is None:
        raise HTTPException(404, f"No pregame data for game {game_id}")

    prob = _get_pre_game_prob(pregame_info)
    features = {f: round(pregame_info[f], 4) for f in PREGAME_FEATURES}

    return {
        "game_id": game_id,
        "home_team_id": pregame_info["home_team_id"],
        "away_team_id": pregame_info["away_team_id"],
        "pre_game_prob": round(prob, 4),
        "features": features,
        "labels": _PREGAME_LABELS,
    }


@app.get("/games")
def game_list(season: str | None = Query(None, description="e.g. 2024-25")) -> dict:
    """Return game list for the replay selector."""
    db_path = DATA_DIR / "raw" / "games.db"
    if not db_path.exists():
        raise HTTPException(404, "games.db not found")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if season:
        rows = conn.execute(
            """
            SELECT h.game_id, h.game_date, h.season,
                   a.team_abbrev AS away_team, h.team_abbrev AS home_team,
                   a.pts AS away_pts, h.pts AS home_pts, h.wl AS home_wl
            FROM game_logs h
            JOIN game_logs a ON h.game_id = a.game_id AND a.is_home = 0
            WHERE h.is_home = 1 AND h.season = ?
            ORDER BY h.game_date DESC, h.game_id
            """,
            (season,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT h.game_id, h.game_date, h.season,
                   a.team_abbrev AS away_team, h.team_abbrev AS home_team,
                   a.pts AS away_pts, h.pts AS home_pts, h.wl AS home_wl
            FROM game_logs h
            JOIN game_logs a ON h.game_id = a.game_id AND a.is_home = 0
            WHERE h.is_home = 1
            ORDER BY h.game_date DESC, h.game_id
            LIMIT 200
            """,
        ).fetchall()

    conn.close()

    # Get distinct seasons for the season selector
    conn2 = sqlite3.connect(db_path)
    seasons = [
        r[0] for r in conn2.execute(
            "SELECT DISTINCT season FROM game_logs ORDER BY season DESC"
        ).fetchall()
    ]
    conn2.close()

    return {
        "seasons": seasons,
        "games": [dict(r) for r in rows],
    }


@app.get("/replay/{game_id}")
def replay(game_id: str) -> dict:
    """
    Replay a historical game from pbp.db.

    Returns the full probability timeline and play log, matching the shape
    of the /live/ endpoint so the frontend can use the same components.
    """
    pregame_info = lookup_pregame(game_id)
    if pregame_info is None:
        raise HTTPException(404, f"No pregame data for game {game_id}")

    pregame_vec = np.array(
        [[pregame_info[f] for f in PREGAME_FEATURES]], dtype=np.float64
    )
    pre_game_prob = float(_pregame_model.predict_proba(pregame_vec)[0, 1])

    state = GameState(
        game_id=game_id,
        home_team_id=pregame_info["home_team_id"],
        away_team_id=pregame_info["away_team_id"],
        pre_game_prob=pre_game_prob,
        team_abbrev_map=_abbrev_map,
    )

    db_path = DATA_DIR / "raw" / "pbp.db"
    if not db_path.exists():
        raise HTTPException(404, "pbp.db not found")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT action_number, action_id, period, clock_seconds,
               team_id, action_type, sub_type, description,
               score_home, score_away, is_field_goal, shot_result
        FROM play_by_play
        WHERE game_id = ?
        ORDER BY action_number
        """,
        (game_id,),
    ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(404, f"No play-by-play data for game {game_id}")

    id_to_abbrev = _reverse_abbrev_map()
    prob_history = []
    play_log = []

    for row in rows:
        event = dict(row)
        if not state.update(event):
            continue

        fv = state.to_feature_vector().reshape(1, -1)
        prob = float(_ingame_model.predict_proba(fv)[0, 1])
        team_id = event.get("team_id") or 0

        prob_history.append({
            "seconds_remaining": state.seconds_remaining,
            "home_win_prob": round(prob, 4),
            "action_number": event.get("action_number", 0),
        })
        play_log.append({
            "action_number": event.get("action_number", 0),
            "period": state.period,
            "clock_seconds": state.clock_seconds,
            "team_abbrev": id_to_abbrev.get(int(team_id) if team_id else 0, ""),
            "description": event.get("description") or event.get("action_type") or "",
            "home_score": state.home_score,
            "away_score": state.away_score,
            "home_win_prob": round(prob, 4),
        })

    return {
        "game_id": game_id,
        "period": state.period,
        "clock_seconds": state.clock_seconds,
        "home_score": state.home_score,
        "away_score": state.away_score,
        "home_win_prob": round(prob_history[-1]["home_win_prob"], 4) if prob_history else pre_game_prob,
        "events_processed": len(play_log),
        "home_team_id": pregame_info["home_team_id"],
        "away_team_id": pregame_info["away_team_id"],
        "home_team_abbrev": id_to_abbrev.get(pregame_info["home_team_id"], "HOME"),
        "away_team_abbrev": id_to_abbrev.get(pregame_info["away_team_id"], "AWAY"),
        "pre_game_prob": round(pre_game_prob, 4),
        "prob_history": prob_history,
        "play_log": play_log,
        "box_score": _build_box_score(state),
    }


@app.get("/live/{game_id}")
async def live(game_id: str) -> dict:
    """
    Return current live probability and game state.

    On first call for a game, initializes the GameState and starts a
    background polling task.
    """
    state, prob = _ensure_game_state(game_id)

    # Start background polling if not already running
    if game_id not in _poll_tasks or _poll_tasks[game_id].done():
        _poll_tasks[game_id] = asyncio.ensure_future(_poll_loop(game_id))

    return _build_game_payload(game_id)


# ------------------------------------------------------------------
# WebSocket Endpoint
# ------------------------------------------------------------------

@app.websocket("/ws/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str) -> None:
    """
    WebSocket endpoint for real-time game updates.

    On connect: sends current state immediately, starts poll task if needed.
    Pushes updates whenever _poll_loop processes new events.
    """
    await websocket.accept()
    _ws_connections.setdefault(game_id, set()).add(websocket)

    try:
        # Initialize game state and start polling
        state, prob = _ensure_game_state(game_id)
        if game_id not in _poll_tasks or _poll_tasks[game_id].done():
            _poll_tasks[game_id] = asyncio.ensure_future(_poll_loop(game_id, interval=15))

        # Send initial state immediately
        payload = _build_game_payload(game_id)
        await websocket.send_json(payload)

        # Keep connection alive — wait for client pings or disconnect
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_connections.get(game_id, set()).discard(websocket)
