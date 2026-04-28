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
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.elo import compute_elo_ratings
from live.game_state import GameState

# Must import StratifiedCalibrator before loading ingame.pkl
from model.train_ingame import INGAME_FEATURES, StratifiedCalibrator  # noqa: F401
from model.train_pregame import PREGAME_FEATURES

import joblib

LIVE_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

# League-average fallbacks (same as features/pregame.py)
_LEAGUE_AVG = {
    "efg_pct": 0.488,
    "ast_rate": 0.575,
    "tov_rate": 0.136,
    "off_rating": 106.0,
    "def_rating": 106.0,
    "w_pct": 0.500,
}

# Previous season mapping for ORtg/DRtg/win% lookups
_PREV_SEASON: dict[str, str] = {
    "2015-16": "2014-15", "2016-17": "2015-16", "2017-18": "2016-17",
    "2018-19": "2017-18", "2019-20": "2018-19", "2020-21": "2019-20",
    "2021-22": "2020-21", "2022-23": "2021-22", "2023-24": "2022-23",
    "2024-25": "2023-24",
    "2025-26": "2024-25",
}

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
    Look up pregame features for a game.

    First checks pregame_features.parquet (historical games). If not found,
    auto-detects teams from the NBA live scoreboard and computes features
    on-the-fly from the most recent data in the local databases.

    Returns a dict with PREGAME_FEATURES keys + home_team_id, away_team_id,
    or None if the game cannot be resolved.
    """
    # Try the parquet first (fast path for historical games)
    path = DATA_DIR / "processed" / "pregame_features.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        row = df[df["game_id"] == game_id]
        if not row.empty:
            row = row.iloc[0]
            result = {feat: float(row[feat]) for feat in PREGAME_FEATURES}
            result["home_team_id"] = int(row["home_team_id"])
            result["away_team_id"] = int(row["away_team_id"])
            return result

    # Fallback: live game not in parquet — resolve teams and compute on-the-fly
    teams = lookup_live_game_teams(game_id)
    if teams is None:
        return None
    home_team_id, away_team_id = teams
    print(f"  Game {game_id} not in parquet — computing pregame features live")
    return compute_live_pregame(home_team_id, away_team_id)


def lookup_live_game_teams(game_id: str) -> tuple[int, int] | None:
    """
    Look up home/away team IDs for a game from the NBA live scoreboard API.

    Returns (home_team_id, away_team_id) or None if the game is not on
    today's scoreboard.
    """
    try:
        from nba_api.live.nba.endpoints import scoreboard
        sb = scoreboard.ScoreBoard()
        data = sb.get_dict()
        for game in data.get("scoreboard", {}).get("games", []):
            if game.get("gameId") == game_id:
                home_id = int(game["homeTeam"]["teamId"])
                away_id = int(game["awayTeam"]["teamId"])
                return home_id, away_id
    except Exception as e:
        print(f"  Scoreboard API error: {e}")
    return None


def compute_live_pregame(home_team_id: int, away_team_id: int) -> dict:
    """
    Compute pregame features on-the-fly for a live game using the most
    recent data available in the local databases.

    Returns a dict with all PREGAME_FEATURES keys + home_team_id, away_team_id.
    """
    games_db = DATA_DIR / "raw" / "games.db"
    players_db = DATA_DIR / "raw" / "players.db"

    # --- ELO ---
    home_elo, away_elo = 1500.0, 1500.0
    if games_db.exists():
        _, current_elo = compute_elo_ratings(games_db)
        home_elo = current_elo.get(home_team_id, 1500.0)
        away_elo = current_elo.get(away_team_id, 1500.0)

    # --- Rolling box stats (eFG%, AST rate, TOV rate) ---
    home_box = {"efg_pct": _LEAGUE_AVG["efg_pct"],
                "ast_rate": _LEAGUE_AVG["ast_rate"],
                "tov_rate": _LEAGUE_AVG["tov_rate"]}
    away_box = dict(home_box)

    if players_db.exists() and games_db.exists():
        for team_id, box_out in [(home_team_id, home_box), (away_team_id, away_box)]:
            stats = _query_season_box_stats(players_db, games_db, team_id, "2025-26")
            if stats is not None:
                box_out.update(stats)

    # --- ORtg, DRtg, win% from previous season ---
    prev_season = _PREV_SEASON.get("2025-26", "2024-25")
    home_eff = _query_team_efficiency(games_db, home_team_id, prev_season)
    away_eff = _query_team_efficiency(games_db, away_team_id, prev_season)

    # --- Rest days ---
    home_rest = _query_rest_days(games_db, home_team_id)
    away_rest = _query_rest_days(games_db, away_team_id)

    return {
        "elo_diff": home_elo - away_elo,
        "efg_pct_diff": home_box["efg_pct"] - away_box["efg_pct"],
        "ortg_diff": home_eff["off_rating"] - away_eff["off_rating"],
        "drtg_diff": home_eff["def_rating"] - away_eff["def_rating"],
        "prev_season_win_pct_diff": home_eff["w_pct"] - away_eff["w_pct"],
        "rest_days_diff": home_rest - away_rest,
        "home_flag": 1,
        "ast_pct_diff": home_box["ast_rate"] - away_box["ast_rate"],
        "tov_pct_diff": home_box["tov_rate"] - away_box["tov_rate"],
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
    }


def _query_season_box_stats(
    players_db: Path, games_db: Path, team_id: int, season: str,
) -> dict | None:
    """Aggregate season-to-date box stats for a team, return rolling rates."""
    conn_p = sqlite3.connect(players_db)
    conn_g = sqlite3.connect(games_db)

    # Get game_ids for this team+season from game_logs
    game_ids = [
        r[0] for r in conn_g.execute(
            "SELECT DISTINCT game_id FROM game_logs WHERE team_id = ? AND season = ?",
            (team_id, season),
        ).fetchall()
    ]
    conn_g.close()

    if not game_ids:
        conn_p.close()
        return None

    placeholders = ",".join("?" * len(game_ids))
    row = conn_p.execute(
        f"""SELECT SUM(fgm), SUM(fga), SUM(fg3m), SUM(ast), SUM(tov), SUM(fta)
            FROM player_box_scores
            WHERE team_id = ? AND game_id IN ({placeholders})""",
        [team_id] + game_ids,
    ).fetchone()
    conn_p.close()

    if row is None or row[1] is None or row[1] == 0:
        return None

    fgm, fga, fg3m, ast, tov, fta = (float(v or 0) for v in row)
    efg = (fgm + 0.5 * fg3m) / fga if fga > 0 else _LEAGUE_AVG["efg_pct"]
    ast_denom = fga + 0.44 * fta + ast + tov
    ast_rate = ast / ast_denom if ast_denom > 0 else _LEAGUE_AVG["ast_rate"]
    tov_denom = fga + 0.44 * fta + tov
    tov_rate = tov / tov_denom if tov_denom > 0 else _LEAGUE_AVG["tov_rate"]

    return {"efg_pct": efg, "ast_rate": ast_rate, "tov_rate": tov_rate}


def _query_team_efficiency(
    games_db: Path, team_id: int, season: str,
) -> dict:
    """Look up ORtg, DRtg, win% from team_efficiency table."""
    if not games_db.exists():
        return dict(_LEAGUE_AVG)
    conn = sqlite3.connect(games_db)
    row = conn.execute(
        "SELECT off_rating, def_rating, w_pct FROM team_efficiency "
        "WHERE team_id = ? AND season = ?",
        (team_id, season),
    ).fetchone()
    conn.close()
    if row is None:
        return {"off_rating": _LEAGUE_AVG["off_rating"],
                "def_rating": _LEAGUE_AVG["def_rating"],
                "w_pct": _LEAGUE_AVG["w_pct"]}
    return {"off_rating": float(row[0]), "def_rating": float(row[1]), "w_pct": float(row[2])}


def _query_rest_days(games_db: Path, team_id: int) -> float:
    """Compute days since the team's most recent completed game."""
    if not games_db.exists():
        return 7.0
    conn = sqlite3.connect(games_db)
    row = conn.execute(
        "SELECT MAX(game_date) FROM game_logs WHERE team_id = ?",
        (team_id,),
    ).fetchone()
    conn.close()
    if row is None or row[0] is None:
        return 7.0
    last_date = datetime.strptime(row[0][:10], "%Y-%m-%d").date()
    days = (date.today() - last_date).days
    return float(max(days, 1))


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

def _fetch_live_actions(game_id: str) -> list[dict]:
    """Fetch play-by-play actions from the NBA live endpoint."""
    from nba_api.live.nba.endpoints import playbyplay
    pbp = playbyplay.PlayByPlay(game_id)
    actions = pbp.get_dict().get("game", {}).get("actions", [])
    return [
        {
            "actionId": a.get("actionNumber"),  # live API has no actionId; use actionNumber
            "actionNumber": a.get("actionNumber"),
            "period": a.get("period"),
            "clock": str(a.get("clock", "") or ""),
            "teamId": a.get("teamId"),
            "actionType": str(a.get("actionType", "") or ""),
            "subType": a.get("subType") or "",
            "description": a.get("description") or "",
            "scoreHome": a.get("scoreHome"),
            "scoreAway": a.get("scoreAway"),
            "isFieldGoal": a.get("isFieldGoal"),
            "shotResult": a.get("shotResult") or "",
        }
        for a in actions
    ]


def poll_live(
    game_id: str,
    pregame_model,
    ingame_model,
    pregame_info: dict,
    abbrev_map: dict[str, int],
    interval: int = POLL_INTERVAL,
) -> None:
    """Poll the NBA live PBP endpoint for a live game, updating probabilities."""
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
            actions = _fetch_live_actions(game_id)
        except Exception as e:
            print(f"  API error: {e} — retrying in {interval}s")
            time.sleep(interval)
            continue

        new_count = 0
        for event in actions:

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
