"""
test_game_state.py — Unit tests for live/game_state.py.

Tests the GameState class for:
  - Score tracking and forward-fill
  - Clock encoding (regulation + OT)
  - Shooting stat accumulation (FG, 2PT, 3PT, FT)
  - Foul, turnover, timeout counting
  - Possession state machine and last_5_poss_swing
  - Deduplication by action_id
  - Feature vector ordering and shape
  - Clutch flag logic

Run:
    pytest tests/test_game_state.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.game_state import GameState

HOME_ID = 100
AWAY_ID = 200
ABBREV_MAP = {"HOM": HOME_ID, "AWY": AWAY_ID}


def _make_state(pre_game_prob: float = 0.6) -> GameState:
    return GameState(
        game_id="0099900001",
        home_team_id=HOME_ID,
        away_team_id=AWAY_ID,
        pre_game_prob=pre_game_prob,
        team_abbrev_map=ABBREV_MAP,
    )


def _event(
    action_id: int,
    period: int = 1,
    clock_seconds: int = 600,
    team_id: int = 0,
    action_type: str = "",
    sub_type: str = "",
    description: str = "",
    score_home=None,
    score_away=None,
    is_field_goal: int = 0,
    shot_result: str = "",
) -> dict:
    """Helper to create a snake_case event dict (SQLite replay format)."""
    return {
        "action_id": action_id,
        "action_number": action_id,
        "period": period,
        "clock_seconds": clock_seconds,
        "team_id": team_id,
        "action_type": action_type,
        "sub_type": sub_type,
        "description": description,
        "score_home": score_home,
        "score_away": score_away,
        "is_field_goal": is_field_goal,
        "shot_result": shot_result,
    }


# ------------------------------------------------------------------
# Score tracking
# ------------------------------------------------------------------

class TestScoreTracking:
    def test_initial_score_zero(self):
        state = _make_state()
        assert state.home_score == 0
        assert state.away_score == 0
        assert state.score_diff == 0

    def test_score_updates_on_scoring_play(self):
        state = _make_state()
        state.update(_event(1, team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=2, score_away=0))
        assert state.home_score == 2
        assert state.away_score == 0
        assert state.score_diff == 2

    def test_score_forward_fills(self):
        state = _make_state()
        state.update(_event(1, team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=2, score_away=0))
        # Foul event has no score fields — score should persist
        state.update(_event(2, team_id=AWAY_ID, action_type="foul"))
        assert state.home_score == 2
        assert state.away_score == 0


# ------------------------------------------------------------------
# Clock encoding
# ------------------------------------------------------------------

class TestClockEncoding:
    def test_tip_off(self):
        state = _make_state()
        assert state.seconds_remaining == (4 - 1) * 720 + 720  # 2880

    def test_regulation_mid_game(self):
        state = _make_state()
        state.update(_event(1, period=2, clock_seconds=360))
        assert state.seconds_remaining == (4 - 2) * 720 + 360  # 1800

    def test_end_of_q4(self):
        state = _make_state()
        state.update(_event(1, period=4, clock_seconds=0))
        assert state.seconds_remaining == 0

    def test_ot1_start(self):
        state = _make_state()
        state.update(_event(1, period=5, clock_seconds=300))
        # OT1 start: -((5-5)*300 + (300-300)) = 0
        assert state.seconds_remaining == 0

    def test_ot1_end(self):
        state = _make_state()
        state.update(_event(1, period=5, clock_seconds=0))
        # OT1 end: -((5-5)*300 + (300-0)) = -300
        assert state.seconds_remaining == -300

    def test_ot2(self):
        state = _make_state()
        state.update(_event(1, period=6, clock_seconds=150))
        # OT2 mid: -((6-5)*300 + (300-150)) = -(300 + 150) = -450
        assert state.seconds_remaining == -450


# ------------------------------------------------------------------
# Shooting stats
# ------------------------------------------------------------------

class TestShootingStats:
    def test_fg_pct(self):
        state = _make_state()
        # Home makes 1 of 2 FGs
        state.update(_event(1, team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=2, score_away=0))
        state.update(_event(2, team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Missed"))
        assert state.home_fgm == 1
        assert state.home_fga == 2
        vec = state.to_feature_vector()
        assert vec[3] == pytest.approx(0.5)  # home_fg_pct_live

    def test_three_pt(self):
        state = _make_state()
        state.update(_event(1, team_id=AWAY_ID, action_type="3pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=0, score_away=3))
        assert state.away_3pm == 1
        assert state.away_3pa == 1
        vec = state.to_feature_vector()
        assert vec[8] == pytest.approx(1.0)  # away_3pt_pct_live

    def test_ft_accumulation(self):
        state = _make_state()
        state.update(_event(1, team_id=HOME_ID, action_type="free throw",
                            sub_type="Free Throw 1 of 2",
                            shot_result="Made", score_home=1, score_away=0))
        state.update(_event(2, team_id=HOME_ID, action_type="free throw",
                            sub_type="Free Throw 2 of 2",
                            shot_result="Missed"))
        assert state.home_ftm == 1
        assert state.home_fta == 2
        vec = state.to_feature_vector()
        assert vec[9] == pytest.approx(0.5)  # home_ft_pct_live

    def test_zero_attempts_gives_zero_pct(self):
        state = _make_state()
        vec = state.to_feature_vector()
        # All shooting percentages should be 0.0 with no attempts
        for idx in (3, 4, 5, 6, 7, 8, 9, 10):
            assert vec[idx] == 0.0


# ------------------------------------------------------------------
# Fouls, turnovers, timeouts
# ------------------------------------------------------------------

class TestCounts:
    def test_fouls(self):
        state = _make_state()
        state.update(_event(1, team_id=HOME_ID, action_type="foul"))
        state.update(_event(2, team_id=HOME_ID, action_type="foul"))
        state.update(_event(3, team_id=AWAY_ID, action_type="foul"))
        assert state.home_fouls == 2
        assert state.away_fouls == 1

    def test_turnovers(self):
        state = _make_state()
        state.update(_event(1, team_id=HOME_ID, action_type="turnover"))
        state.update(_event(2, team_id=AWAY_ID, action_type="turnover"))
        state.update(_event(3, team_id=AWAY_ID, action_type="turnover"))
        vec = state.to_feature_vector()
        assert vec[13] == -1  # turnover_diff_live: home(1) - away(2) = -1

    def test_timeouts(self):
        state = _make_state()
        state.update(_event(1, action_type="timeout",
                            description="HOM Timeout: Regular"))
        assert state.home_timeouts_used == 1
        vec = state.to_feature_vector()
        # timeout_remaining_diff = (7-1) - (7-0) = -1
        assert vec[14] == -1


# ------------------------------------------------------------------
# Possession state machine
# ------------------------------------------------------------------

class TestPossessionSwing:
    def test_made_shot_ends_possession(self):
        state = _make_state()
        # Home scores 2
        state.update(_event(1, team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=2, score_away=0))
        assert len(state._poss_deque) == 1
        assert state._poss_deque[0] == 2  # +2 from home perspective

    def test_turnover_ends_possession(self):
        state = _make_state()
        state.update(_event(1, team_id=HOME_ID, action_type="turnover"))
        assert len(state._poss_deque) == 1
        assert state._poss_deque[0] == 0  # no points scored

    def test_last_5_poss_swing(self):
        state = _make_state()
        # 5 possessions: home scores 2 each time
        for i in range(5):
            state.update(_event(
                i + 1, team_id=HOME_ID, action_type="2pt",
                is_field_goal=1, shot_result="Made",
                score_home=2 * (i + 1), score_away=0,
            ))
        vec = state.to_feature_vector()
        assert vec[15] == 10  # 5 * 2 = 10

    def test_deque_maxlen_5(self):
        state = _make_state()
        # 6 possessions via turnovers (0 pts each)
        for i in range(6):
            state.update(_event(i + 1, team_id=HOME_ID, action_type="turnover"))
        assert len(state._poss_deque) == 5


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_event_ignored(self):
        state = _make_state()
        event = _event(1, team_id=HOME_ID, action_type="foul")
        assert state.update(event) is True
        assert state.update(event) is False
        assert state.home_fouls == 1  # not 2

    def test_different_ids_both_processed(self):
        state = _make_state()
        assert state.update(_event(1, team_id=HOME_ID, action_type="foul")) is True
        assert state.update(_event(2, team_id=HOME_ID, action_type="foul")) is True
        assert state.home_fouls == 2


# ------------------------------------------------------------------
# Feature vector
# ------------------------------------------------------------------

class TestFeatureVector:
    def test_shape_and_dtype(self):
        state = _make_state()
        vec = state.to_feature_vector()
        assert vec.shape == (18,)
        assert vec.dtype == np.float32

    def test_pre_game_prob_at_index_2(self):
        state = _make_state(pre_game_prob=0.72)
        vec = state.to_feature_vector()
        assert vec[2] == pytest.approx(0.72, abs=1e-5)

    def test_quarter_at_index_16(self):
        state = _make_state()
        state.update(_event(1, period=3, clock_seconds=500))
        vec = state.to_feature_vector()
        assert vec[16] == 3

    def test_clutch_flag_at_index_17(self):
        state = _make_state()
        # Q4, score tied → clutch
        state.update(_event(1, period=4, clock_seconds=60,
                            team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=100, score_away=98))
        vec = state.to_feature_vector()
        assert vec[17] == 1  # clutch_flag

    def test_not_clutch_when_blowout(self):
        state = _make_state()
        state.update(_event(1, period=4, clock_seconds=60,
                            team_id=HOME_ID, action_type="2pt",
                            is_field_goal=1, shot_result="Made",
                            score_home=120, score_away=90))
        vec = state.to_feature_vector()
        assert vec[17] == 0  # not clutch (diff > 5)

    def test_not_clutch_in_q2(self):
        state = _make_state()
        state.update(_event(1, period=2, clock_seconds=60,
                            score_home=50, score_away=50))
        vec = state.to_feature_vector()
        assert vec[17] == 0  # not clutch (Q2)


# ------------------------------------------------------------------
# camelCase API format
# ------------------------------------------------------------------

class TestCamelCaseInput:
    def test_accepts_camel_case_keys(self):
        state = _make_state()
        event = {
            "actionId": 1,
            "actionNumber": 1,
            "period": 2,
            "clock": "PT08M30.00S",
            "teamId": HOME_ID,
            "actionType": "2pt",
            "subType": "",
            "description": "Home layup",
            "scoreHome": 10,
            "scoreAway": 8,
            "isFieldGoal": 1,
            "shotResult": "Made",
        }
        assert state.update(event) is True
        assert state.home_score == 10
        assert state.period == 2
        assert state.clock_seconds == 510  # 8*60 + 30
