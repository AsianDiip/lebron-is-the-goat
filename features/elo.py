"""
elo.py — Walk-forward ELO ratings for NBA teams.

Computes each team's ELO rating as it stood *before* each game, ensuring no
future information leaks into the feature. ELO is updated after every game and
regressed toward the mean at the start of each new season.

Usage:
    from features.elo import compute_elo_ratings
    elo_ratings = compute_elo_ratings(Path("data/raw/games.db"))
    elo_before = elo_ratings[(team_id, game_id)]
"""

import sqlite3
from collections import defaultdict
from pathlib import Path


# ELO hyperparameters (frozen — do not change without retraining all features)
INITIAL_ELO: float = 1500.0
K_FACTOR: float = 100.0
# Fraction of the gap from 1500 that is *retained* at the start of each new season.
# 0.75 means teams regress 25% toward 1500 — retains inter-season signal while
# preventing runaway ratings over a 10-year window.
SEASON_REGRESSION: float = 0.75


def _expected_win(team_elo: float, opponent_elo: float) -> float:
    """Expected win probability for team given their ELO ratings."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))


def compute_elo_ratings(games_db_path: Path) -> dict[tuple[int, str], float]:
    """
    Walk-forward ELO computation over all seasons in games.db.

    Returns a dict mapping (team_id, game_id) -> elo_before_game.
    The ELO stored is the rating that team had *entering* that game, so it
    can be used as a feature without any lookahead.

    Algorithm:
      1. Load all game_logs sorted by (game_date, game_id) — deterministic order.
      2. Group rows by game_id to get both team rows for each game.
      3. For each game in chronological order:
           a. Record the current ELO for both teams (pre-game ELO).
           b. Update both teams' ELO based on the outcome.
      4. At each season boundary, regress all team ELOs toward 1500.

    Args:
        games_db_path: Path to games.db (must contain game_logs table).

    Returns:
        dict[(team_id, game_id), float] — ELO entering each game.
    """
    conn = sqlite3.connect(games_db_path)
    rows = conn.execute(
        """
        SELECT game_id, game_date, season, team_id, is_home, wl
        FROM game_logs
        ORDER BY game_date, game_id
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(
            f"No game_logs found in {games_db_path}. Run fetch_games.py first."
        )

    # Current ELO for each team. Defaults to INITIAL_ELO for unseen teams.
    elo: dict[int, float] = defaultdict(lambda: INITIAL_ELO)

    # Output: (team_id, game_id) -> elo before the game
    result: dict[tuple[int, str], float] = {}

    # Group consecutive rows by game_id while preserving chronological order.
    # Each game_id appears exactly twice (one per team).
    current_season: str | None = None
    i = 0
    while i < len(rows):
        game_id = rows[i][0]

        # Collect both rows for this game_id
        game_rows = []
        j = i
        while j < len(rows) and rows[j][0] == game_id:
            game_rows.append(rows[j])
            j += 1

        if len(game_rows) != 2:
            # Corrupted data or bye; skip rather than crash
            i = j
            continue

        # Unpack both team rows
        _, game_date, season, team_a_id, is_home_a, wl_a = game_rows[0]
        _, _, _, team_b_id, _, wl_b = game_rows[1]

        # Determine home/away — needed to look up ELO by role
        # (the actual update is symmetric; home/away labels only matter for
        # the ELO table key, not the math)
        home_id = team_a_id if is_home_a else team_b_id
        away_id = team_b_id if is_home_a else team_a_id
        home_wl = wl_a if is_home_a else wl_b

        # Season boundary: regress all team ELOs toward 1500
        if season != current_season:
            if current_season is not None:
                for tid in elo:
                    elo[tid] = INITIAL_ELO + SEASON_REGRESSION * (
                        elo[tid] - INITIAL_ELO
                    )
            current_season = season

        # Record pre-game ELO for both teams before updating
        result[(home_id, game_id)] = elo[home_id]
        result[(away_id, game_id)] = elo[away_id]

        # Compute ELO update
        home_expected = _expected_win(elo[home_id], elo[away_id])
        home_actual = 1.0 if home_wl == "W" else 0.0

        delta = K_FACTOR * (home_actual - home_expected)
        elo[home_id] += delta
        elo[away_id] -= delta

        i = j

    return result
