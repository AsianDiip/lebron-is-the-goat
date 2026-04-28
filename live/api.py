"""
api.py — FastAPI server for NBA win probability.

Endpoints:
    GET /pregame/{game_id}  — pre-game win probability
    GET /live/{game_id}     — current live probability + game state

Usage:
    uvicorn live.api:app --reload
    # then: curl http://localhost:8000/pregame/0022301234
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

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

app = FastAPI(title="NBA Win Probability API")

# Loaded once at startup
_pregame_model = None
_ingame_model = None
_abbrev_map: dict[str, int] = {}

# Active game states keyed by game_id
_active_games: dict[str, GameState] = {}
_active_probs: dict[str, float] = {}  # latest probability per game
_poll_tasks: dict[str, asyncio.Task] = {}


@app.on_event("startup")
def startup() -> None:
    global _pregame_model, _ingame_model, _abbrev_map
    _pregame_model = joblib.load(MODEL_DIR / "pregame.pkl")
    _ingame_model = joblib.load(MODEL_DIR / "ingame.pkl")
    _abbrev_map = get_team_abbrev_map()


def _get_pre_game_prob(pregame_info: dict) -> float:
    vec = np.array([[pregame_info[f] for f in PREGAME_FEATURES]], dtype=np.float64)
    return float(_pregame_model.predict_proba(vec)[0, 1])


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
    return state, pre_game_prob


async def _poll_loop(game_id: str, interval: int = 30) -> None:
    """Background polling task for a live game."""
    from nba_api.stats.endpoints import PlayByPlayV3

    state = _active_games[game_id]
    consecutive_empty = 0

    while consecutive_empty < 20:
        try:
            pbp = PlayByPlayV3(game_id=game_id)
            df = pbp.get_data_frames()[0]
        except Exception:
            await asyncio.sleep(interval)
            continue

        new_count = 0
        for _, row in df.iterrows():
            event = {
                "actionId": row.get("actionId"),
                "actionNumber": row.get("actionNumber"),
                "period": row.get("period"),
                "clock": str(row.get("clock", "") or ""),
                "teamId": row.get("teamId"),
                "actionType": str(row.get("actionType", "") or ""),
                "subType": row.get("subType") or "",
                "description": row.get("description") or "",
                "scoreHome": row.get("scoreHome"),
                "scoreAway": row.get("scoreAway"),
                "isFieldGoal": row.get("isFieldGoal"),
                "shotResult": row.get("shotResult") or "",
            }
            if not state.update(event):
                continue
            new_count += 1

            fv = state.to_feature_vector().reshape(1, -1)
            prob = float(_ingame_model.predict_proba(fv)[0, 1])
            _active_probs[game_id] = prob

        consecutive_empty = 0 if new_count > 0 else consecutive_empty + 1
        await asyncio.sleep(interval)


# ------------------------------------------------------------------
# Endpoints
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


@app.get("/live/{game_id}")
def live(game_id: str) -> dict:
    """
    Return current live probability and game state.

    On first call for a game, initializes the GameState and starts a
    background polling task.
    """
    state, prob = _ensure_game_state(game_id)

    # Start background polling if not already running
    if game_id not in _poll_tasks or _poll_tasks[game_id].done():
        loop = asyncio.get_event_loop()
        _poll_tasks[game_id] = loop.create_task(_poll_loop(game_id))

    return {
        "game_id": game_id,
        "period": state.period,
        "clock_seconds": state.clock_seconds,
        "home_score": state.home_score,
        "away_score": state.away_score,
        "home_win_prob": round(_active_probs.get(game_id, prob), 4),
        "events_processed": len(state.play_log),
    }
