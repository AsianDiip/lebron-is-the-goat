"""
ingame.py — In-game feature computation.

Produces one row per play-by-play event, capturing the full game state at that
moment. All features are cumulative up to and including the current event —
no future information is used.

Key design decisions:
  - score_diff: forward-filled from score_home/score_away (NULL on non-scoring plays)
  - seconds_remaining: regulation uses (4-period)*720+clock; OT goes negative
  - last_5_poss_swing: per-game possession state machine (deque of last 5)
  - timeout team: parsed from description field (team_id is 0 for timeouts)
  - pre_game_prob: set to 0.5 placeholder; replaced at training time

Usage:
    from features.ingame import build_ingame_snapshots
    df = build_ingame_snapshots(Path("data/raw/pbp.db"), Path("data/raw/games.db"))
"""

import re
import sqlite3
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

# Regex to detect the last free throw in a sequence (e.g., "Free Throw 2 of 2")
_FT_RE = re.compile(r"Free Throw (\d+) of (\d+)", re.IGNORECASE)

# Regex to extract team name before " Timeout" in description
_TIMEOUT_TEAM_RE = re.compile(r"^(.+?)\s+Timeout", re.IGNORECASE)

# NBA timeout allotment per team per game (post-2017 rules; close enough for the
# 2015-2025 window — small inaccuracies in 2015-16 through 2016-17 are acceptable)
_TIMEOUTS_PER_TEAM = 7


# ---------------------------------------------------------------------------
# Clock encoding
# ---------------------------------------------------------------------------

def compute_seconds_remaining(period: int, clock_seconds: int) -> int:
    """
    Encode time remaining as a single integer.

    Regulation (period 1-4): (4 - period) * 720 + clock_seconds
      → 2880 at tip-off, 0 at end of Q4.

    OT (period > 4): negative values, monotonically decreasing.
      OT1 start = 0, OT1 end = -300, OT2 start = -300, OT2 end = -600, etc.
      Formula: -((period - 5) * 300 + (300 - clock_seconds))

    The model learns that negative values mean overtime. The raw `quarter`
    feature disambiguates Q4-at-0 from OT-at-0.
    """
    if period <= 4:
        return (4 - period) * 720 + clock_seconds
    else:
        return -((period - 5) * 300 + (300 - clock_seconds))


# ---------------------------------------------------------------------------
# Possession state machine (per game)
# ---------------------------------------------------------------------------

def _is_last_free_throw(sub_type: str | None) -> bool:
    """Return True if sub_type describes the final FT in a sequence."""
    if not sub_type:
        return False
    m = _FT_RE.search(sub_type)
    if not m:
        return False
    return m.group(1) == m.group(2)


def _compute_game_features(
    game_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    home_win: int,
    team_abbrev_map: dict[str, int],
) -> pd.DataFrame:
    """
    Compute all in-game features for a single game's PBP events.

    game_df must be sorted by action_number and contain the full game.
    Returns a DataFrame with one row per event plus all feature columns.
    """
    n = len(game_df)
    if n == 0:
        return game_df

    # ---- Score forward-fill ------------------------------------------------
    sh = game_df["score_home"].copy().astype(float)
    sa = game_df["score_away"].copy().astype(float)
    sh.iloc[0] = sh.iloc[0] if pd.notna(sh.iloc[0]) else 0.0
    sa.iloc[0] = sa.iloc[0] if pd.notna(sa.iloc[0]) else 0.0
    sh = sh.ffill().fillna(0.0)
    sa = sa.ffill().fillna(0.0)
    score_diff = (sh - sa).astype(int).tolist()

    # ---- Pre-allocate output arrays ----------------------------------------
    seconds_remaining = np.empty(n, dtype=np.int32)
    home_fg_made = np.zeros(n, dtype=np.int32)
    home_fg_att  = np.zeros(n, dtype=np.int32)
    away_fg_made = np.zeros(n, dtype=np.int32)
    away_fg_att  = np.zeros(n, dtype=np.int32)
    home_fouls   = np.zeros(n, dtype=np.int32)
    away_fouls   = np.zeros(n, dtype=np.int32)
    home_tov     = np.zeros(n, dtype=np.int32)
    away_tov     = np.zeros(n, dtype=np.int32)
    home_to_used = np.zeros(n, dtype=np.int32)
    away_to_used = np.zeros(n, dtype=np.int32)
    last_5_swing = np.zeros(n, dtype=np.int32)
    home_2pm_arr = np.zeros(n, dtype=np.int32)
    home_2pa_arr = np.zeros(n, dtype=np.int32)
    away_2pm_arr = np.zeros(n, dtype=np.int32)
    away_2pa_arr = np.zeros(n, dtype=np.int32)
    home_3pm_arr = np.zeros(n, dtype=np.int32)
    home_3pa_arr = np.zeros(n, dtype=np.int32)
    away_3pm_arr = np.zeros(n, dtype=np.int32)
    away_3pa_arr = np.zeros(n, dtype=np.int32)
    home_ftm_arr = np.zeros(n, dtype=np.int32)
    home_fta_arr = np.zeros(n, dtype=np.int32)
    away_ftm_arr = np.zeros(n, dtype=np.int32)
    away_fta_arr = np.zeros(n, dtype=np.int32)

    # Running counters
    h_fgm = h_fga = a_fgm = a_fga = 0
    h_2pm = h_2pa = a_2pm = a_2pa = 0
    h_3pm = h_3pa = a_3pm = a_3pa = 0
    h_ftm = h_fta = a_ftm = a_fta = 0
    h_fouls = a_fouls = 0
    h_tov = a_tov = 0
    h_to = a_to = 0  # timeouts used

    # Possession state machine
    # current_team: team_id that currently has the ball (None = unknown/jump ball)
    current_team: int | None = None
    poss_points: int = 0  # points scored in current possession
    poss_deque: deque[int] = deque(maxlen=5)  # signed net pts per completed possession

    # Track previous action_type for technical FT detection
    prev_action_type: str = ""
    prev_sub_type: str = ""

    rows = game_df.to_dict("records")

    for i, row in enumerate(rows):
        period = int(row["period"])
        clock_sec = int(row["clock_seconds"]) if pd.notna(row["clock_seconds"]) else 0
        action_type = str(row["action_type"] or "").lower()
        sub_type = str(row["sub_type"] or "") if pd.notna(row["sub_type"]) else ""
        description = str(row["description"] or "") if pd.notna(row["description"]) else ""
        team_id = row["team_id"]
        team_id = int(team_id) if pd.notna(team_id) and team_id else 0
        is_field_goal = int(row["is_field_goal"] or 0)
        shot_result = str(row["shot_result"] or "") if pd.notna(row["shot_result"]) else ""

        # ---- Clock ----------------------------------------------------------
        seconds_remaining[i] = compute_seconds_remaining(period, clock_sec)

        # ---- Period start: reset possession state ---------------------------
        if action_type == "period" and "start" in sub_type.lower():
            current_team = None
            poss_points = 0

        # ---- Determine if team_id belongs to home or away ------------------
        is_home_event = (team_id == home_team_id)
        is_away_event = (team_id == away_team_id)

        # ---- Field goals ----------------------------------------------------
        if is_field_goal:
            is_three = action_type == "3pt"
            made = shot_result.lower() == "made"
            if is_home_event:
                h_fga += 1
                if made: h_fgm += 1
                if is_three:
                    h_3pa += 1
                    if made: h_3pm += 1
                else:
                    h_2pa += 1
                    if made: h_2pm += 1
            elif is_away_event:
                a_fga += 1
                if made: a_fgm += 1
                if is_three:
                    a_3pa += 1
                    if made: a_3pm += 1
                else:
                    a_2pa += 1
                    if made: a_2pm += 1

        # ---- Fouls ----------------------------------------------------------
        if action_type == "foul":
            if is_home_event:
                h_fouls += 1
            elif is_away_event:
                a_fouls += 1

        # ---- Turnovers ------------------------------------------------------
        if action_type == "turnover":
            if is_home_event:
                h_tov += 1
            elif is_away_event:
                a_tov += 1

        # ---- Timeouts -------------------------------------------------------
        if action_type == "timeout":
            to_team = _parse_timeout_team(description, team_abbrev_map,
                                          home_team_id, away_team_id)
            if to_team == home_team_id:
                h_to += 1
            elif to_team == away_team_id:
                a_to += 1

        # ---- Store running counts -------------------------------------------
        home_fg_made[i] = h_fgm
        home_fg_att[i]  = h_fga
        away_fg_made[i] = a_fgm
        away_fg_att[i]  = a_fga
        home_fouls[i]   = h_fouls
        away_fouls[i]   = a_fouls
        home_tov[i]     = h_tov
        away_tov[i]     = a_tov
        home_to_used[i] = h_to
        away_to_used[i] = a_to
        home_2pm_arr[i] = h_2pm; home_2pa_arr[i] = h_2pa
        away_2pm_arr[i] = a_2pm; away_2pa_arr[i] = a_2pa
        home_3pm_arr[i] = h_3pm; home_3pa_arr[i] = h_3pa
        away_3pm_arr[i] = a_3pm; away_3pa_arr[i] = a_3pa
        home_ftm_arr[i] = h_ftm; home_fta_arr[i] = h_fta
        away_ftm_arr[i] = a_ftm; away_fta_arr[i] = a_fta

        # ---- Possession state machine ---------------------------------------
        # Infer possession from current event
        if current_team is None and team_id in (home_team_id, away_team_id):
            current_team = team_id

        possession_ended = False

        if action_type in ("2pt", "3pt") or (action_type == "field goal" or "shot" in action_type):
            # Made shot ends the possession, switch teams
            if shot_result.lower() == "made":
                pts = int(sh[i] - (sh[i - 1] if i > 0 else 0) + sa[i] - (sa[i - 1] if i > 0 else 0))
                # signed from home's perspective
                pts_signed = int(sh[i] - (sh[i - 1] if i > 0 else 0)) - int(sa[i] - (sa[i - 1] if i > 0 else 0))
                poss_points += pts_signed
                possession_ended = True
                # Switch possession
                current_team = away_team_id if current_team == home_team_id else home_team_id

        elif action_type == "turnover":
            # Turnover: end possession with current points (usually 0), switch
            possession_ended = True
            current_team = away_team_id if current_team == home_team_id else home_team_id

        elif action_type == "free throw":
            # Accumulate FT stats on every attempt (not just last-of-sequence)
            ft_made = shot_result.lower() == "made"
            if is_home_event:
                h_fta += 1
                if ft_made: h_ftm += 1
            elif is_away_event:
                a_fta += 1
                if ft_made: a_ftm += 1

            # Detect last FT; skip technical FTs (no possession change)
            is_technical = "technical" in prev_sub_type.lower() or "technical" in sub_type.lower()
            if _is_last_free_throw(sub_type) and not is_technical:
                pts_signed = int(sh[i] - (sh[i - 1] if i > 0 else 0)) - int(sa[i] - (sa[i - 1] if i > 0 else 0))
                poss_points += pts_signed
                possession_ended = True
                current_team = away_team_id if current_team == home_team_id else home_team_id

        elif action_type == "rebound":
            # Defensive rebound: possession changes to the rebounding team
            if team_id != current_team and team_id in (home_team_id, away_team_id):
                possession_ended = True
                current_team = team_id

        if possession_ended:
            poss_deque.append(poss_points)
            poss_points = 0

        last_5_swing[i] = sum(poss_deque)

        # Update previous event trackers
        prev_action_type = action_type
        prev_sub_type = sub_type

    # ---- Compute derived columns -------------------------------------------
    result = game_df[["game_id", "season", "action_number", "period", "clock_seconds"]].copy()

    result["score_diff"] = score_diff
    result["seconds_remaining"] = seconds_remaining
    result["pre_game_prob"] = 0.5  # placeholder; filled by train_ingame.py

    # np.divide with `where` avoids division-by-zero RuntimeWarnings — numpy's
    # np.where evaluates both branches unconditionally before selecting.
    result["home_fg_pct_live"] = np.divide(
        home_fg_made, home_fg_att, out=np.zeros(n, dtype=np.float64), where=home_fg_att > 0
    )
    result["away_fg_pct_live"] = np.divide(
        away_fg_made, away_fg_att, out=np.zeros(n, dtype=np.float64), where=away_fg_att > 0
    )
    result["home_2pt_pct_live"] = np.divide(
        home_2pm_arr, home_2pa_arr, out=np.zeros(n, dtype=np.float64), where=home_2pa_arr > 0
    )
    result["away_2pt_pct_live"] = np.divide(
        away_2pm_arr, away_2pa_arr, out=np.zeros(n, dtype=np.float64), where=away_2pa_arr > 0
    )
    result["home_3pt_pct_live"] = np.divide(
        home_3pm_arr, home_3pa_arr, out=np.zeros(n, dtype=np.float64), where=home_3pa_arr > 0
    )
    result["away_3pt_pct_live"] = np.divide(
        away_3pm_arr, away_3pa_arr, out=np.zeros(n, dtype=np.float64), where=away_3pa_arr > 0
    )
    result["home_ft_pct_live"] = np.divide(
        home_ftm_arr, home_fta_arr, out=np.zeros(n, dtype=np.float64), where=home_fta_arr > 0
    )
    result["away_ft_pct_live"] = np.divide(
        away_ftm_arr, away_fta_arr, out=np.zeros(n, dtype=np.float64), where=away_fta_arr > 0
    )

    result["home_fouls"] = home_fouls
    result["away_fouls"] = away_fouls
    result["turnover_diff_live"] = home_tov - away_tov

    result["timeout_remaining_diff"] = (
        (_TIMEOUTS_PER_TEAM - home_to_used) - (_TIMEOUTS_PER_TEAM - away_to_used)
    )

    result["last_5_poss_swing"] = last_5_swing
    result["quarter"] = game_df["period"].values

    result["clutch_flag"] = (
        (result["quarter"] >= 4) & (result["score_diff"].abs() <= 5)
    ).astype(int)

    result["home_win"] = home_win

    return result.reset_index(drop=True)


def _parse_timeout_team(
    description: str,
    abbrev_map: dict[str, int],
    home_team_id: int,
    away_team_id: int,
) -> int:
    """
    Parse the team responsible for a timeout from the description string.

    Description format: "TEAMNAME Timeout: Regular (Full N Short M)"
    Extracts the first token(s) before ' Timeout' and matches against known
    team abbreviations or name fragments (case-insensitive).

    Returns home_team_id, away_team_id, or 0 if unrecognised.
    """
    m = _TIMEOUT_TEAM_RE.match(description)
    if not m:
        return 0
    token = m.group(1).strip().upper()
    return abbrev_map.get(token, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ingame_snapshots(
    pbp_db_path: Path,
    games_db_path: Path,
) -> pd.DataFrame:
    """
    Build one row per play-by-play event with all in-game features.

    Processes games one at a time to keep memory bounded and allow the
    possession state machine to run per-game. Prints progress every 500 games.

    Args:
        pbp_db_path:   Path to data/raw/pbp.db
        games_db_path: Path to data/raw/games.db

    Returns:
        DataFrame with columns:
            game_id, season, action_number, score_diff, seconds_remaining,
            pre_game_prob, home_fg_pct_live, away_fg_pct_live,
            home_2pt_pct_live, away_2pt_pct_live, home_3pt_pct_live,
            away_3pt_pct_live, home_ft_pct_live, away_ft_pct_live,
            home_fouls, away_fouls, turnover_diff_live, timeout_remaining_diff,
            last_5_poss_swing, quarter, clutch_flag, home_win
    """
    games_conn = sqlite3.connect(games_db_path)
    pbp_conn   = sqlite3.connect(pbp_db_path)

    # Load game metadata: home/away team IDs, winner, team abbreviations
    game_meta = pd.read_sql_query(
        "SELECT game_id, team_id, team_abbrev, is_home, wl FROM game_logs",
        games_conn,
    )

    # Build timeout team lookup: UPPER(abbrev) -> team_id
    abbrev_df = game_meta[["team_abbrev", "team_id"]].drop_duplicates()
    global_abbrev_map: dict[str, int] = {
        row["team_abbrev"].upper(): int(row["team_id"])
        for _, row in abbrev_df.iterrows()
    }

    # home/away and win label per game
    home_rows = game_meta[game_meta["is_home"] == 1].set_index("game_id")
    away_rows = game_meta[game_meta["is_home"] == 0].set_index("game_id")

    game_ids = sorted(home_rows.index.unique().tolist())
    total = len(game_ids)
    print(f"Building in-game snapshots for {total:,} games...")

    all_frames: list[pd.DataFrame] = []

    # Load all PBP once to avoid repeated DB round-trips
    pbp = pd.read_sql_query(
        """
        SELECT game_id, season, action_number, period, clock_seconds,
               team_id, action_type, sub_type, description,
               score_home, score_away, is_field_goal, shot_result
        FROM play_by_play
        ORDER BY game_id, action_number
        """,
        pbp_conn,
    )

    games_conn.close()
    pbp_conn.close()

    pbp_grouped = pbp.groupby("game_id", sort=False)

    for i, game_id in enumerate(game_ids):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i + 1:,}/{total:,}] processing game {game_id}")

        if game_id not in home_rows.index or game_id not in away_rows.index:
            continue  # missing home or away row — skip

        home_team_id = int(home_rows.loc[game_id, "team_id"])
        away_team_id = int(away_rows.loc[game_id, "team_id"])
        home_wl = home_rows.loc[game_id, "wl"]
        home_win = 1 if home_wl == "W" else 0

        if game_id not in pbp_grouped.groups:
            continue

        game_df = pbp_grouped.get_group(game_id).sort_values("action_number").reset_index(drop=True)

        frame = _compute_game_features(
            game_df,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_win=home_win,
            team_abbrev_map=global_abbrev_map,
        )
        all_frames.append(frame)

    if not all_frames:
        raise RuntimeError("No in-game frames produced. Check that pbp.db and games.db are populated.")

    result = pd.concat(all_frames, ignore_index=True)

    # Canonical column order
    result = result[[
        "game_id", "season", "action_number",
        "score_diff", "seconds_remaining", "pre_game_prob",
        "home_fg_pct_live", "away_fg_pct_live",
        "home_2pt_pct_live", "away_2pt_pct_live",
        "home_3pt_pct_live", "away_3pt_pct_live",
        "home_ft_pct_live", "away_ft_pct_live",
        "home_fouls", "away_fouls",
        "turnover_diff_live", "timeout_remaining_diff",
        "last_5_poss_swing", "quarter", "clutch_flag", "home_win",
    ]]

    print(f"Done. {len(result):,} rows across {total:,} games.")
    return result
