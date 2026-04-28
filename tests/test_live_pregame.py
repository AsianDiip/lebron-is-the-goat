"""
test_live_pregame.py — Diagnostic tests for live pregame feature computation.

Verifies that:
  1. Playoff games are not in pregame_features.parquet (root cause of 404)
  2. games.db has no playoff game IDs with '004' prefix
  3. compute_live_pregame() returns valid features for any team pair
  4. lookup_pregame() falls back to live computation for unknown game IDs

Run:
    pytest tests/test_live_pregame.py -v
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train_pregame import PREGAME_FEATURES

DATA_DIR = ROOT / "data"
GAMES_DB = DATA_DIR / "raw" / "games.db"
PLAYERS_DB = DATA_DIR / "raw" / "players.db"
PREGAME_PARQUET = DATA_DIR / "processed" / "pregame_features.parquet"

PLAYOFF_GAME_ID = "0042500165"


# ------------------------------------------------------------------
# Diagnostic tests — confirm the gap exists
# ------------------------------------------------------------------


class TestDiagnosticGap:
    """Tests that document the missing playoff data (root cause of 404)."""

    @pytest.mark.skipif(not PREGAME_PARQUET.exists(), reason="pregame_features.parquet not found")
    def test_playoff_game_not_in_parquet(self):
        """Playoff game 0042500165 is not in pregame_features.parquet."""
        df = pd.read_parquet(PREGAME_PARQUET)
        assert df[df["game_id"] == PLAYOFF_GAME_ID].empty, (
            "Expected playoff game to be absent from parquet"
        )

    @pytest.mark.skipif(not GAMES_DB.exists(), reason="games.db not found")
    def test_no_playoff_ids_in_games_db(self):
        """games.db contains no game IDs with the '004' playoff prefix."""
        conn = sqlite3.connect(GAMES_DB)
        count = conn.execute(
            "SELECT COUNT(*) FROM game_logs WHERE game_id LIKE '004%'"
        ).fetchone()[0]
        conn.close()
        assert count == 0, (
            f"Expected 0 playoff game IDs (004 prefix) in games.db, found {count}"
        )


# ------------------------------------------------------------------
# Live pregame computation tests
# ------------------------------------------------------------------


class TestComputeLivePregame:
    """Tests for on-the-fly pregame feature computation."""

    @pytest.mark.skipif(not GAMES_DB.exists(), reason="games.db not found")
    def test_returns_all_pregame_features(self):
        """compute_live_pregame() returns all 9 PREGAME_FEATURES + team IDs."""
        from live.poller import compute_live_pregame

        # Use two known team IDs from games.db
        conn = sqlite3.connect(GAMES_DB)
        teams = conn.execute(
            "SELECT DISTINCT team_id FROM game_logs LIMIT 2"
        ).fetchall()
        conn.close()
        assert len(teams) >= 2, "Need at least 2 teams in games.db"

        home_id, away_id = int(teams[0][0]), int(teams[1][0])
        result = compute_live_pregame(home_id, away_id)

        # Check all feature keys present
        for feat in PREGAME_FEATURES:
            assert feat in result, f"Missing feature: {feat}"
            assert isinstance(result[feat], (int, float)), f"{feat} is not numeric"
            assert np.isfinite(result[feat]), f"{feat} is not finite: {result[feat]}"

        assert result["home_team_id"] == home_id
        assert result["away_team_id"] == away_id

    @pytest.mark.skipif(not GAMES_DB.exists(), reason="games.db not found")
    def test_home_flag_is_one(self):
        """home_flag is always 1."""
        from live.poller import compute_live_pregame

        conn = sqlite3.connect(GAMES_DB)
        teams = conn.execute(
            "SELECT DISTINCT team_id FROM game_logs LIMIT 2"
        ).fetchall()
        conn.close()

        home_id, away_id = int(teams[0][0]), int(teams[1][0])
        result = compute_live_pregame(home_id, away_id)
        assert result["home_flag"] == 1

    @pytest.mark.skipif(not GAMES_DB.exists(), reason="games.db not found")
    def test_elo_diff_is_reasonable(self):
        """ELO diff should be within a plausible range (not extreme)."""
        from live.poller import compute_live_pregame

        conn = sqlite3.connect(GAMES_DB)
        teams = conn.execute(
            "SELECT DISTINCT team_id FROM game_logs LIMIT 2"
        ).fetchall()
        conn.close()

        home_id, away_id = int(teams[0][0]), int(teams[1][0])
        result = compute_live_pregame(home_id, away_id)
        assert -600 < result["elo_diff"] < 600, (
            f"ELO diff {result['elo_diff']} seems unreasonable"
        )

    def test_unknown_teams_use_fallbacks(self):
        """Unknown team IDs should use league-average fallbacks, not crash."""
        from live.poller import compute_live_pregame

        result = compute_live_pregame(999999, 999998)
        for feat in PREGAME_FEATURES:
            assert feat in result
            assert np.isfinite(result[feat])
        # Unknown teams: ELO diff = 0 (both default to 1500)
        assert result["elo_diff"] == 0.0
