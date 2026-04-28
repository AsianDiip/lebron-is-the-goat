"""
poller.py — Live win probability polling loop.

Two modes:
  - Live:   polls PlayByPlayV3 every 30 seconds for a running game
  - Replay: reads PBP from pbp.db for a completed game and replays events

Usage:
    python live/poller.py --game_id 0022301234           # live polling
    python live/poller.py --game_id 0022301234 --replay  # replay from DB

Outputs:
    live/{game_id}_probability.csv — one row per new event
    stdout — real-time probability updates
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.game_state import GameState

# Must import StratifiedCalibrator before loading ingame.pkl
from model.train_ingame import INGAME_FEATURES, StratifiedCalibrator  # noqa: F401
from model.train_pregame import PREGAME_FEATURES

import joblib

LIVE_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

POLL_INTERVAL = 30  # seconds between API calls in live mode

# ISO 8601 clock regex
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_models() -> tuple:
    """Load both pregame and ingame models."""
    pregame_model = joblib.load(MODEL_DIR / "pregame.pkl")
    ingame_model = joblib.load(MODEL_DIR / "ingame.pkl")
    print("Models loaded: pregame.pkl, ingame.pkl")
    return pregame_model, ingame_model


# ------------------------------------------------------------------
# Pregame feature lookup
# ------------------------------------------------------------------

def lookup_pregame(game_id: str) -> dict | None:
    """
    Look up pregame features for a game from pregame_features.parquet.

    Returns a dict with PREGAME_FEATURES keys + home_team_id, away_team_id,
    or None if the game_id is not found.
    """
    path = DATA_DIR / "processed" / "pregame_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    row = df[df["game_id"] == game_id]
    if row.empty:
        return None
    row = row.iloc[0]
    result = {feat: float(row[feat]) for feat in PREGAME_FEATURES}
    result["home_team_id"] = int(row["home_team_id"])
    result["away_team_id"] = int(row["away_team_id"])
    return result


def get_team_abbrev_map() -> dict[str, int]:
    """Build UPPER(abbrev) -> team_id map from games.db."""
    db_path = DATA_DIR / "raw" / "games.db"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT team_abbrev, team_id FROM game_logs"
    ).fetchall()
    conn.close()
    return {abbrev.upper(): int(tid) for abbrev, tid in rows}


# ------------------------------------------------------------------
# CSV logging
# ------------------------------------------------------------------

class ProbabilityLogger:
    """Append-only CSV logger for probability updates."""

    def __init__(self, game_id: str) -> None:
        self.path = LIVE_DIR / f"{game_id}_probability.csv"
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "period", "clock", "event", "home_win_prob"])
        self._file.flush()

    def log(self, period: int, clock_seconds: int, event_desc: str, prob: float) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        self._writer.writerow([ts, period, clock_seconds, event_desc, f"{prob:.4f}"])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ------------------------------------------------------------------
# Replay mode (from SQLite)
# ------------------------------------------------------------------

def replay_game(
    game_id: str,
    pregame_model,
    ingame_model,
    pregame_info: dict,
    abbrev_map: dict[str, int],
) -> None:
    """Replay a historical game from pbp.db, printing probabilities."""
    db_path = DATA_DIR / "raw" / "pbp.db"
    if not db_path.exists():
        print(f"ERROR: pbp.db not found at {db_path}")
        sys.exit(1)

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
        print(f"ERROR: No PBP data found for game {game_id}")
        sys.exit(1)

    # Compute pre_game_prob
    pregame_vec = np.array(
        [[pregame_info[f] for f in PREGAME_FEATURES]], dtype=np.float64
    )
    pre_game_prob = float(pregame_model.predict_proba(pregame_vec)[0, 1])

    state = GameState(
        game_id=game_id,
        home_team_id=pregame_info["home_team_id"],
        away_team_id=pregame_info["away_team_id"],
        pre_game_prob=pre_game_prob,
        team_abbrev_map=abbrev_map,
    )

    logger = ProbabilityLogger(game_id)

    print(f"\nReplaying game {game_id}")
    print(f"Pre-game probability (home win): {pre_game_prob:.4f}")
    print(f"{'─' * 80}")
    print(f"{'Period':>6}  {'Clock':>5}  {'Score':>9}  {'Prob':>6}  Event")
    print(f"{'─' * 80}")

    for row in rows:
        event = dict(row)
        if not state.update(event):
            continue  # duplicate

        fv = state.to_feature_vector().reshape(1, -1)
        prob = float(ingame_model.predict_proba(fv)[0, 1])

        desc = event.get("description") or event.get("action_type") or ""
        period = state.period
        clock = state.clock_seconds

        score_str = f"{state.home_score}-{state.away_score}"
        print(f"  Q{period:>1}    {clock:>5}  {score_str:>9}  {prob:>6.3f}  {desc[:50]}")

        logger.log(period, clock, desc, prob)

    print(f"{'─' * 80}")
    print(f"Final: {state.home_score}-{state.away_score}")
    print(f"Probability log saved to {logger.path}")
    logger.close()


# ------------------------------------------------------------------
# Live polling mode
# ------------------------------------------------------------------

def poll_live(
    game_id: str,
    pregame_model,
    ingame_model,
    pregame_info: dict,
    abbrev_map: dict[str, int],
    interval: int = POLL_INTERVAL,
) -> None:
    """Poll PlayByPlayV3 for a live game, updating probabilities."""
    from nba_api.stats.endpoints import PlayByPlayV3

    # Compute pre_game_prob
    pregame_vec = np.array(
        [[pregame_info[f] for f in PREGAME_FEATURES]], dtype=np.float64
    )
    pre_game_prob = float(pregame_model.predict_proba(pregame_vec)[0, 1])

    state = GameState(
        game_id=game_id,
        home_team_id=pregame_info["home_team_id"],
        away_team_id=pregame_info["away_team_id"],
        pre_game_prob=pre_game_prob,
        team_abbrev_map=abbrev_map,
    )

    logger = ProbabilityLogger(game_id)

    print(f"\nPolling game {game_id} every {interval}s")
    print(f"Pre-game probability (home win): {pre_game_prob:.4f}")
    print(f"{'─' * 80}")

    consecutive_empty = 0
    MAX_EMPTY_POLLS = 20  # stop after ~10 min with no new events (game likely over)

    while True:
        try:
            pbp = PlayByPlayV3(game_id=game_id)
            df = pbp.get_data_frames()[0]
        except Exception as e:
            print(f"  API error: {e} — retrying in {interval}s")
            time.sleep(interval)
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
            prob = float(ingame_model.predict_proba(fv)[0, 1])

            desc = event.get("description") or event.get("actionType") or ""
            score_str = f"{state.home_score}-{state.away_score}"
            print(
                f"  Q{state.period}  {state.clock_seconds:>5}s  "
                f"{score_str:>9}  {prob:.3f}  {desc[:50]}"
            )
            logger.log(state.period, state.clock_seconds, desc, prob)

        if new_count == 0:
            consecutive_empty += 1
            if consecutive_empty >= MAX_EMPTY_POLLS:
                print(f"\nNo new events for {MAX_EMPTY_POLLS * interval}s — game appears final.")
                break
        else:
            consecutive_empty = 0

        time.sleep(interval)

    print(f"Final: {state.home_score}-{state.away_score}")
    print(f"Probability log saved to {logger.path}")
    logger.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Win Probability Poller")
    parser.add_argument("--game_id", required=True, help="NBA game ID (e.g. 0022301234)")
    parser.add_argument("--replay", action="store_true", help="Replay from pbp.db instead of live polling")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds (live mode)")
    args = parser.parse_args()

    pregame_model, ingame_model = load_models()
    abbrev_map = get_team_abbrev_map()

    pregame_info = lookup_pregame(args.game_id)
    if pregame_info is None:
        print(f"ERROR: No pregame features found for game {args.game_id}")
        print("Ensure data/processed/pregame_features.parquet exists and contains this game.")
        sys.exit(1)

    if args.replay:
        replay_game(args.game_id, pregame_model, ingame_model, pregame_info, abbrev_map)
    else:
        poll_live(args.game_id, pregame_model, ingame_model, pregame_info, abbrev_map, args.interval)


if __name__ == "__main__":
    main()
