"""
game_state.py — Mutable in-game state for live win probability inference.

Tracks all in-game state incrementally (one event at a time) and produces
the 18-element feature vector expected by the in-game XGBoost model.

Reuses clock encoding, FT detection, and timeout parsing from features/ingame.py
to stay consistent with training-time feature computation.

Usage:
    from live.game_state import GameState

    state = GameState(
        game_id="0022301234",
        home_team_id=1610612747,
        away_team_id=1610612738,
        pre_game_prob=0.62,
        team_abbrev_map={"LAL": 1610612747, "BOS": 1610612738},
    )
    state.update(event_dict)
    feature_vec = state.to_feature_vector()
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

# Reuse helpers from the training-time feature pipeline
from features.ingame import (
    _FT_RE,
    _TIMEOUTS_PER_TEAM,
    _is_last_free_throw,
    _parse_timeout_team,
    compute_seconds_remaining,
)

# ISO 8601 clock regex (same as data/fetch_pbp.py)
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def _parse_clock_str(clock_str: str) -> int:
    """Parse ISO 8601 clock string (e.g. 'PT11M42.00S') to integer seconds."""
    if not clock_str:
        return 0
    m = _CLOCK_RE.match(clock_str)
    if not m:
        return 0
    return int(int(m.group(1)) * 60 + float(m.group(2)))


class GameState:
    """
    Holds all mutable in-game state. Updated incrementally after each play
    event — never recomputed from scratch.

    The 18 features produced by to_feature_vector() are in the exact order
    defined by INGAME_FEATURES in model/train_ingame.py:
        [score_diff, seconds_remaining, pre_game_prob,
         home_fg_pct_live, away_fg_pct_live,
         home_2pt_pct_live, away_2pt_pct_live,
         home_3pt_pct_live, away_3pt_pct_live,
         home_ft_pct_live, away_ft_pct_live,
         home_fouls, away_fouls,
         turnover_diff_live, timeout_remaining_diff,
         last_5_poss_swing, quarter, clutch_flag]
    """

    def __init__(
        self,
        game_id: str,
        home_team_id: int,
        away_team_id: int,
        pre_game_prob: float,
        team_abbrev_map: dict[str, int],
    ) -> None:
        self.game_id = game_id
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.pre_game_prob = pre_game_prob
        self.team_abbrev_map = team_abbrev_map

        # Score
        self.home_score: int = 0
        self.away_score: int = 0

        # Clock
        self.period: int = 1
        self.clock_seconds: int = 720

        # Shooting counts
        self.home_fgm: int = 0
        self.home_fga: int = 0
        self.away_fgm: int = 0
        self.away_fga: int = 0
        self.home_2pm: int = 0
        self.home_2pa: int = 0
        self.away_2pm: int = 0
        self.away_2pa: int = 0
        self.home_3pm: int = 0
        self.home_3pa: int = 0
        self.away_3pm: int = 0
        self.away_3pa: int = 0
        self.home_ftm: int = 0
        self.home_fta: int = 0
        self.away_ftm: int = 0
        self.away_fta: int = 0

        # Fouls, turnovers, timeouts
        self.home_fouls: int = 0
        self.away_fouls: int = 0
        self.home_turnovers: int = 0
        self.away_turnovers: int = 0
        self.home_timeouts_used: int = 0
        self.away_timeouts_used: int = 0

        # Possession state machine
        self._current_team: int | None = None
        self._poss_points: int = 0
        self._poss_deque: deque[int] = deque(maxlen=5)

        # Previous event tracking (for technical FT detection)
        self._prev_sub_type: str = ""

        # Deduplication
        self._seen_event_ids: set[int] = set()

        # Full event log
        self.play_log: list[dict] = []

        # Previous scores (for computing per-event score change)
        self._prev_home_score: int = 0
        self._prev_away_score: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, event: dict) -> bool:
        """
        Update state from a single PlayByPlayV3 event dict.

        Expected keys (camelCase, as returned by the live API):
            actionNumber, actionId, period, clock (ISO 8601 str),
            teamId, actionType, subType, description,
            scoreHome, scoreAway, isFieldGoal, shotResult

        Also accepts the lowercase/underscore form used in the SQLite DB:
            action_number, action_id, period, clock_seconds,
            team_id, action_type, sub_type, description,
            score_home, score_away, is_field_goal, shot_result

        Returns True if the event was new (processed), False if duplicate.
        """
        # Normalize field names — accept both camelCase (live API) and
        # snake_case (SQLite replay)
        action_id = event.get("actionId") or event.get("action_id")
        if action_id is not None:
            action_id = int(action_id)
            if action_id in self._seen_event_ids:
                return False
            self._seen_event_ids.add(action_id)

        # Parse fields
        period = int(event.get("period", self.period))

        # Clock: accept either raw seconds or ISO 8601 string
        if "clock_seconds" in event and event["clock_seconds"] is not None:
            clock_sec = int(event["clock_seconds"])
        elif "clock" in event and event["clock"] is not None:
            clock_sec = _parse_clock_str(str(event["clock"]))
        else:
            clock_sec = self.clock_seconds

        action_type = str(event.get("actionType") or event.get("action_type") or "").lower()
        sub_type = str(event.get("subType") or event.get("sub_type") or "")
        description = str(event.get("description") or "")
        team_id = event.get("teamId") or event.get("team_id")
        team_id = int(team_id) if team_id is not None and str(team_id) != "" else 0
        is_field_goal = int(event.get("isFieldGoal") or event.get("is_field_goal") or 0)
        shot_result = str(event.get("shotResult") or event.get("shot_result") or "")

        # Score — forward-fill (only present on scoring plays)
        score_home = event.get("scoreHome") or event.get("score_home")
        score_away = event.get("scoreAway") or event.get("score_away")
        if score_home is not None and str(score_home) != "":
            self.home_score = int(score_home)
        if score_away is not None and str(score_away) != "":
            self.away_score = int(score_away)

        # Update clock
        self.period = period
        self.clock_seconds = clock_sec

        # ---- Period start: reset possession state -----------------------
        if action_type == "period" and "start" in sub_type.lower():
            self._current_team = None
            self._poss_points = 0

        # ---- Determine team affiliation ---------------------------------
        is_home_event = (team_id == self.home_team_id)
        is_away_event = (team_id == self.away_team_id)

        # ---- Field goals ------------------------------------------------
        if is_field_goal:
            is_three = action_type == "3pt"
            made = shot_result.lower() == "made"
            if is_home_event:
                self.home_fga += 1
                if made:
                    self.home_fgm += 1
                if is_three:
                    self.home_3pa += 1
                    if made:
                        self.home_3pm += 1
                else:
                    self.home_2pa += 1
                    if made:
                        self.home_2pm += 1
            elif is_away_event:
                self.away_fga += 1
                if made:
                    self.away_fgm += 1
                if is_three:
                    self.away_3pa += 1
                    if made:
                        self.away_3pm += 1
                else:
                    self.away_2pa += 1
                    if made:
                        self.away_2pm += 1

        # ---- Fouls ------------------------------------------------------
        if action_type == "foul":
            if is_home_event:
                self.home_fouls += 1
            elif is_away_event:
                self.away_fouls += 1

        # ---- Turnovers --------------------------------------------------
        if action_type == "turnover":
            if is_home_event:
                self.home_turnovers += 1
            elif is_away_event:
                self.away_turnovers += 1

        # ---- Timeouts ---------------------------------------------------
        if action_type == "timeout":
            to_team = _parse_timeout_team(
                description, self.team_abbrev_map,
                self.home_team_id, self.away_team_id,
            )
            if to_team == self.home_team_id:
                self.home_timeouts_used += 1
            elif to_team == self.away_team_id:
                self.away_timeouts_used += 1

        # ---- Possession state machine -----------------------------------
        if self._current_team is None and team_id in (self.home_team_id, self.away_team_id):
            self._current_team = team_id

        possession_ended = False

        # Score change for this event (signed from home perspective)
        home_delta = self.home_score - self._prev_home_score
        away_delta = self.away_score - self._prev_away_score
        pts_signed = home_delta - away_delta

        if action_type in ("2pt", "3pt") or is_field_goal:
            if shot_result.lower() == "made":
                self._poss_points += pts_signed
                possession_ended = True
                self._current_team = (
                    self.away_team_id if self._current_team == self.home_team_id
                    else self.home_team_id
                )

        elif action_type == "turnover":
            possession_ended = True
            self._current_team = (
                self.away_team_id if self._current_team == self.home_team_id
                else self.home_team_id
            )

        elif action_type == "free throw":
            # Accumulate FT stats on every attempt
            ft_made = shot_result.lower() == "made"
            if is_home_event:
                self.home_fta += 1
                if ft_made:
                    self.home_ftm += 1
            elif is_away_event:
                self.away_fta += 1
                if ft_made:
                    self.away_ftm += 1

            # Detect last FT; skip technical FTs
            is_technical = (
                "technical" in self._prev_sub_type.lower()
                or "technical" in sub_type.lower()
            )
            if _is_last_free_throw(sub_type) and not is_technical:
                self._poss_points += pts_signed
                possession_ended = True
                self._current_team = (
                    self.away_team_id if self._current_team == self.home_team_id
                    else self.home_team_id
                )

        elif action_type == "rebound":
            # Defensive rebound: possession changes to rebounding team
            if team_id != self._current_team and team_id in (self.home_team_id, self.away_team_id):
                possession_ended = True
                self._current_team = team_id

        if possession_ended:
            self._poss_deque.append(self._poss_points)
            self._poss_points = 0

        # Update previous event trackers
        self._prev_sub_type = sub_type
        self._prev_home_score = self.home_score
        self._prev_away_score = self.away_score

        # Log the event
        self.play_log.append(event)

        return True

    def to_feature_vector(self) -> np.ndarray:
        """
        Return the 18-element in-game feature vector (float32).

        Order matches INGAME_FEATURES in model/train_ingame.py exactly.
        Index 16 = quarter (used by StratifiedCalibrator for phase routing).
        """
        score_diff = self.home_score - self.away_score
        seconds_remaining = compute_seconds_remaining(self.period, self.clock_seconds)

        home_fg_pct = self.home_fgm / self.home_fga if self.home_fga > 0 else 0.0
        away_fg_pct = self.away_fgm / self.away_fga if self.away_fga > 0 else 0.0
        home_2pt_pct = self.home_2pm / self.home_2pa if self.home_2pa > 0 else 0.0
        away_2pt_pct = self.away_2pm / self.away_2pa if self.away_2pa > 0 else 0.0
        home_3pt_pct = self.home_3pm / self.home_3pa if self.home_3pa > 0 else 0.0
        away_3pt_pct = self.away_3pm / self.away_3pa if self.away_3pa > 0 else 0.0
        home_ft_pct = self.home_ftm / self.home_fta if self.home_fta > 0 else 0.0
        away_ft_pct = self.away_ftm / self.away_fta if self.away_fta > 0 else 0.0

        turnover_diff = self.home_turnovers - self.away_turnovers
        timeout_remaining_diff = (
            (_TIMEOUTS_PER_TEAM - self.home_timeouts_used)
            - (_TIMEOUTS_PER_TEAM - self.away_timeouts_used)
        )
        last_5_swing = sum(self._poss_deque)
        quarter = self.period
        clutch_flag = int(quarter >= 4 and abs(score_diff) <= 5)

        return np.array([
            score_diff,
            seconds_remaining,
            self.pre_game_prob,
            home_fg_pct,
            away_fg_pct,
            home_2pt_pct,
            away_2pt_pct,
            home_3pt_pct,
            away_3pt_pct,
            home_ft_pct,
            away_ft_pct,
            self.home_fouls,
            self.away_fouls,
            turnover_diff,
            timeout_remaining_diff,
            last_5_swing,
            quarter,
            clutch_flag,
        ], dtype=np.float32)

    @property
    def score_diff(self) -> int:
        return self.home_score - self.away_score

    @property
    def seconds_remaining(self) -> int:
        return compute_seconds_remaining(self.period, self.clock_seconds)

    @property
    def is_clutch(self) -> bool:
        return self.period >= 4 and abs(self.score_diff) <= 5
