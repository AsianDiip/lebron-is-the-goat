"""
pregame.py — Pre-game feature computation.

Produces one row per game with all pre-game features expressed as home-minus-away
differentials. All features are leakage-free: they only use information available
before tip-off.

Leakage strategy:
  - eFG%, AST rate, TOV rate: season-to-date cumulative sums from player_box_scores,
    shifted by one game so the current game is never included.
  - ORtg, DRtg: previous full-season values from team_efficiency (per-possession
    ratings cannot be derived from box scores alone).
  - prev_season_win_pct: previous full-season w_pct from team_efficiency.
  - ELO: walk-forward ratings passed in from elo.py.
  - rest_days: calendar days since each team's last game.

Usage:
    from features.elo import compute_elo_ratings
    from features.pregame import build_pregame_features

    elo = compute_elo_ratings(GAMES_DB)
    df = build_pregame_features(GAMES_DB, PLAYERS_DB, elo)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# League-average fallbacks used when a team has no prior-season data
# (e.g., expansion teams, or 2015-16 being the first season in the dataset).
_LEAGUE_AVG = {
    "efg_pct": 0.488,
    "ast_rate": 0.575,
    "tov_rate": 0.136,
    "off_rating": 106.0,
    "def_rating": 106.0,
    "w_pct": 0.500,
}

# Season ordering used for "previous season" lookups
_SEASON_ORDER = [
    "2014-15",  # only needed as a prior for 2015-16 first game
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
_PREV_SEASON: dict[str, str] = {
    s: _SEASON_ORDER[i - 1] for i, s in enumerate(_SEASON_ORDER) if i > 0
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_game_logs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return game_logs with parsed game_date as datetime."""
    df = pd.read_sql_query(
        "SELECT game_id, season, game_date, team_id, team_abbrev, is_home, wl "
        "FROM game_logs ORDER BY game_date, game_id",
        conn,
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _load_team_efficiency(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT season, team_id, off_rating, def_rating, w_pct FROM team_efficiency",
        conn,
    )


def _load_player_box_scores(players_conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT game_id, team_id, fgm, fga, fg3m, ast, tov, fta "
        "FROM player_box_scores",
        players_conn,
    )


def _compute_rolling_box_stats(
    box_df: pd.DataFrame,
    game_logs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute season-to-date eFG%, AST rate, and TOV rate per (team_id, game_id),
    excluding the current game (shift(1) within season group).

    Returns DataFrame with columns:
        team_id, game_id, roll_efg_pct, roll_ast_rate, roll_tov_rate
    """
    # Aggregate box scores to team-game level
    team_game = (
        box_df.groupby(["game_id", "team_id"], as_index=False)
        .agg(fgm=("fgm", "sum"), fga=("fga", "sum"), fg3m=("fg3m", "sum"),
             ast=("ast", "sum"), tov=("tov", "sum"), fta=("fta", "sum"))
    )

    # Join in season and game_date from game_logs (one row per team per game)
    game_meta = game_logs[["game_id", "team_id", "season", "game_date"]].drop_duplicates()
    team_game = team_game.merge(game_meta, on=["game_id", "team_id"], how="inner")
    team_game = team_game.sort_values(["team_id", "season", "game_date", "game_id"])

    # Cumulative sums within (team, season), shifted by 1 to exclude current game
    cum_cols = ["fgm", "fga", "fg3m", "ast", "tov", "fta"]
    for col in cum_cols:
        team_game[f"cum_{col}"] = (
            team_game.groupby(["team_id", "season"])[col]
            .transform(lambda s: s.cumsum().shift(1))
        )

    # eFG% = (FGM + 0.5 * FG3M) / FGA
    team_game["roll_efg_pct"] = (
        (team_game["cum_fgm"] + 0.5 * team_game["cum_fg3m"])
        / team_game["cum_fga"].replace(0, np.nan)
    )

    # AST rate = AST / (FGA + 0.44*FTA + AST + TOV)  — possession-based
    ast_denom = (
        team_game["cum_fga"]
        + 0.44 * team_game["cum_fta"]
        + team_game["cum_ast"]
        + team_game["cum_tov"]
    ).replace(0, np.nan)
    team_game["roll_ast_rate"] = team_game["cum_ast"] / ast_denom

    # TOV rate = TOV / (FGA + 0.44*FTA + TOV)
    tov_denom = (
        team_game["cum_fga"]
        + 0.44 * team_game["cum_fta"]
        + team_game["cum_tov"]
    ).replace(0, np.nan)
    team_game["roll_tov_rate"] = team_game["cum_tov"] / tov_denom

    return team_game[["team_id", "game_id", "roll_efg_pct", "roll_ast_rate", "roll_tov_rate"]]


def _compute_rest_days(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days (calendar days since last game) per (team_id, game_id).

    First game of each season defaults to 7 days.
    """
    # One row per (team, game)
    team_games = (
        game_logs[["team_id", "game_id", "game_date", "season"]]
        .drop_duplicates()
        .sort_values(["team_id", "game_date", "game_id"])
    )

    team_games["prev_game_date"] = team_games.groupby("team_id")["game_date"].shift(1)
    team_games["rest_days"] = (
        (team_games["game_date"] - team_games["prev_game_date"]).dt.days
    )
    # First game of each season: fill with 7 (reasonable rest assumption)
    team_games["rest_days"] = team_games["rest_days"].fillna(7.0)

    return team_games[["team_id", "game_id", "rest_days"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pregame_features(
    games_db_path: Path,
    players_db_path: Path,
    elo_ratings: dict[tuple[int, str], float],
) -> pd.DataFrame:
    """
    Build one row per game with all pre-game features.

    All features are home-minus-away differentials (positive = home team advantage).

    Args:
        games_db_path:  Path to data/raw/games.db
        players_db_path: Path to data/raw/players.db
        elo_ratings:    Output of features.elo.compute_elo_ratings()

    Returns:
        DataFrame with columns:
            game_id, season, game_date, home_team_id, away_team_id,
            elo_diff, efg_pct_diff, ortg_diff, drtg_diff,
            prev_season_win_pct_diff, rest_days_diff,
            home_flag, ast_pct_diff, tov_pct_diff, home_win
    """
    games_conn = sqlite3.connect(games_db_path)
    players_conn = sqlite3.connect(players_db_path)

    game_logs = _load_game_logs(games_conn)
    team_eff = _load_team_efficiency(games_conn)
    box_df = _load_player_box_scores(players_conn)

    games_conn.close()
    players_conn.close()

    # ------------------------------------------------------------------
    # 1. Rolling season-to-date stats from box scores (leakage-safe)
    # ------------------------------------------------------------------
    rolling = _compute_rolling_box_stats(box_df, game_logs)

    # ------------------------------------------------------------------
    # 2. Previous-season ORtg, DRtg, w_pct from team_efficiency
    # ------------------------------------------------------------------
    # Build a lookup: (team_id, current_season) -> prev-season values
    eff_by_season: dict[tuple[int, str], dict] = {}
    for _, row in team_eff.iterrows():
        eff_by_season[(int(row["team_id"]), row["season"])] = {
            "off_rating": row["off_rating"],
            "def_rating": row["def_rating"],
            "w_pct": row["w_pct"],
        }

    def _prev_season_stat(team_id: int, current_season: str, key: str) -> float:
        prev = _PREV_SEASON.get(current_season)
        if prev and (team_id, prev) in eff_by_season:
            return eff_by_season[(team_id, prev)][key]
        return _LEAGUE_AVG[{"off_rating": "off_rating",
                             "def_rating": "def_rating",
                             "w_pct": "w_pct"}[key]]

    # ------------------------------------------------------------------
    # 3. Rest days
    # ------------------------------------------------------------------
    rest = _compute_rest_days(game_logs)

    # ------------------------------------------------------------------
    # 4. Build one row per game
    # ------------------------------------------------------------------
    # Pivot game_logs to wide format: home and away side by side
    home = game_logs[game_logs["is_home"] == 1][
        ["game_id", "season", "game_date", "team_id", "wl"]
    ].rename(columns={"team_id": "home_team_id", "wl": "home_wl"})

    away = game_logs[game_logs["is_home"] == 0][
        ["game_id", "team_id"]
    ].rename(columns={"team_id": "away_team_id"})

    games = home.merge(away, on="game_id", how="inner")
    games["home_win"] = (games["home_wl"] == "W").astype(int)
    games = games.drop(columns=["home_wl"])

    # Join rolling stats for home and away
    rolling_home = rolling.rename(columns={
        "team_id": "home_team_id",
        "roll_efg_pct": "h_efg", "roll_ast_rate": "h_ast", "roll_tov_rate": "h_tov",
    })
    rolling_away = rolling.rename(columns={
        "team_id": "away_team_id",
        "roll_efg_pct": "a_efg", "roll_ast_rate": "a_ast", "roll_tov_rate": "a_tov",
    })

    games = games.merge(rolling_home[["home_team_id", "game_id", "h_efg", "h_ast", "h_tov"]],
                        on=["home_team_id", "game_id"], how="left")
    games = games.merge(rolling_away[["away_team_id", "game_id", "a_efg", "a_ast", "a_tov"]],
                        on=["away_team_id", "game_id"], how="left")

    # Join rest days
    rest_home = rest.rename(columns={"team_id": "home_team_id", "rest_days": "h_rest"})
    rest_away = rest.rename(columns={"team_id": "away_team_id", "rest_days": "a_rest"})
    games = games.merge(rest_home[["home_team_id", "game_id", "h_rest"]],
                        on=["home_team_id", "game_id"], how="left")
    games = games.merge(rest_away[["away_team_id", "game_id", "a_rest"]],
                        on=["away_team_id", "game_id"], how="left")

    # ------------------------------------------------------------------
    # 5. Add ELO from pre-computed ratings
    # ------------------------------------------------------------------
    games["h_elo"] = games.apply(
        lambda r: elo_ratings.get((r["home_team_id"], r["game_id"]), 1500.0),
        axis=1,
    )
    games["a_elo"] = games.apply(
        lambda r: elo_ratings.get((r["away_team_id"], r["game_id"]), 1500.0),
        axis=1,
    )

    # ------------------------------------------------------------------
    # 6. Previous-season ORtg, DRtg, w_pct
    # ------------------------------------------------------------------
    games["h_ortg"] = games.apply(
        lambda r: _prev_season_stat(r["home_team_id"], r["season"], "off_rating"), axis=1
    )
    games["a_ortg"] = games.apply(
        lambda r: _prev_season_stat(r["away_team_id"], r["season"], "off_rating"), axis=1
    )
    games["h_drtg"] = games.apply(
        lambda r: _prev_season_stat(r["home_team_id"], r["season"], "def_rating"), axis=1
    )
    games["a_drtg"] = games.apply(
        lambda r: _prev_season_stat(r["away_team_id"], r["season"], "def_rating"), axis=1
    )
    games["h_wpct"] = games.apply(
        lambda r: _prev_season_stat(r["home_team_id"], r["season"], "w_pct"), axis=1
    )
    games["a_wpct"] = games.apply(
        lambda r: _prev_season_stat(r["away_team_id"], r["season"], "w_pct"), axis=1
    )

    # ------------------------------------------------------------------
    # 7. Fill NaN rolling stats with previous-season or league-average fallback
    #    (happens for the first 1-2 games of a season where shift(1) is NaN)
    # ------------------------------------------------------------------
    games["h_efg"] = games["h_efg"].fillna(
        games.apply(lambda r: eff_by_season.get(
            (r["home_team_id"], _PREV_SEASON.get(r["season"], "")), {}
        ).get("efg_pct", _LEAGUE_AVG["efg_pct"]), axis=1)
    )
    games["a_efg"] = games["a_efg"].fillna(
        games.apply(lambda r: eff_by_season.get(
            (r["away_team_id"], _PREV_SEASON.get(r["season"], "")), {}
        ).get("efg_pct", _LEAGUE_AVG["efg_pct"]), axis=1)
    )
    # For ast/tov we don't have per-season values in team_efficiency; use league avg
    games["h_ast"] = games["h_ast"].fillna(_LEAGUE_AVG["ast_rate"])
    games["a_ast"] = games["a_ast"].fillna(_LEAGUE_AVG["ast_rate"])
    games["h_tov"] = games["h_tov"].fillna(_LEAGUE_AVG["tov_rate"])
    games["a_tov"] = games["a_tov"].fillna(_LEAGUE_AVG["tov_rate"])

    # ------------------------------------------------------------------
    # 8. Compute differentials (home minus away)
    # ------------------------------------------------------------------
    out = games[["game_id", "season", "game_date", "home_team_id", "away_team_id", "home_win"]].copy()

    out["elo_diff"] = games["h_elo"] - games["a_elo"]
    out["efg_pct_diff"] = games["h_efg"] - games["a_efg"]
    out["ortg_diff"] = games["h_ortg"] - games["a_ortg"]
    out["drtg_diff"] = games["h_drtg"] - games["a_drtg"]
    out["prev_season_win_pct_diff"] = games["h_wpct"] - games["a_wpct"]
    out["rest_days_diff"] = games["h_rest"].fillna(7.0) - games["a_rest"].fillna(7.0)
    out["home_flag"] = 1
    out["ast_pct_diff"] = games["h_ast"] - games["a_ast"]
    out["tov_pct_diff"] = games["h_tov"] - games["a_tov"]

    # Canonical column order matching the model spec
    out = out[[
        "game_id", "season", "game_date", "home_team_id", "away_team_id",
        "elo_diff", "efg_pct_diff", "ortg_diff", "drtg_diff",
        "prev_season_win_pct_diff", "rest_days_diff", "home_flag",
        "ast_pct_diff", "tov_pct_diff", "home_win",
    ]]

    return out.reset_index(drop=True)
