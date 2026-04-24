# Phase 2: Feature Engineering — Implementation Plan

## Context

Phase 1 (data collection) is complete. Three SQLite databases exist in `data/raw/` with ~13K games of game logs, play-by-play, and player box scores across 2015-2025. Phase 2 builds the feature matrices that feed both the pre-game LR model and the in-game XGBoost model.

**Deliverables:** `data/processed/pregame_features.parquet`, `data/processed/ingame_snapshots.parquet`, leakage tests passing, updated README and CLAUDE.md.

---

## Implementation Order

```
1. features/elo.py          (standalone)
2. features/pregame.py      (depends on elo.py)
3. features/ingame.py       (standalone)
4. features/pipeline.py     (orchestrates 1-3, writes parquet)
5. tests/test_no_leakage.py (validates outputs)
6. Update README.md + CLAUDE.md
```

---

## File 1: `features/elo.py`

Walk-forward ELO ratings. One public function:

```python
def compute_elo_ratings(games_db_path: Path) -> dict[tuple[int, str], float]:
    """Returns {(team_id, game_id): elo_before_game}"""
```

- Constants: `INITIAL_ELO=1500`, `K=100`, `SEASON_REGRESSION=0.75`
- Query `game_logs ORDER BY game_date, game_id`, group by `game_id` to get both teams
- Maintain `dict[int, float]` of current ELO per team_id
- Record ELO *before* updating (no leakage)
- At each season boundary: `elo = 1500 + 0.75 * (elo - 1500)` for all teams
- New teams default to 1500

---

## File 2: `features/pregame.py`

One row per game with home-minus-away differentials. One public function:

```python
def build_pregame_features(games_db_path, players_db_path, elo_ratings) -> pd.DataFrame
```

**Columns:** `game_id, season, game_date, home_team_id, away_team_id, elo_diff, efg_pct_diff, ortg_diff, drtg_diff, prev_season_win_pct_diff, rest_days_diff, home_flag, ast_pct_diff, tov_pct_diff, home_win`

### Leakage-safe rolling stats approach

| Feature | Source | Method |
|---------|--------|--------|
| `efg_pct_diff` | `player_box_scores` | Season-to-date cumsum with `shift(1)` per (team, season). `eFG = (FGM + 0.5*FG3M) / FGA` |
| `ast_pct_diff` | `player_box_scores` | Same cumsum approach. `AST_rate = AST / (FGA + 0.44*FTA + AST + TOV)` |
| `tov_pct_diff` | `player_box_scores` | Same. `TOV_rate = TOV / (FGA + 0.44*FTA + TOV)` |
| `ortg_diff`, `drtg_diff` | `team_efficiency` | **Previous season** values (true per-100-poss ratings can't be computed from box scores alone) |
| `prev_season_win_pct_diff` | `team_efficiency` | Previous season `w_pct` |
| `rest_days_diff` | `game_logs` | Date diff to each team's prior game. First game of season defaults to 7 days |

For the first game of a season (no prior games to cumsum), fall back to previous-season full values. For 2015-16 game 1, use league averages.

**Performance:** Vectorized via pandas `expanding().sum().shift(1)` grouped by `(team_id, season)` — avoids N^2 queries.

---

## File 3: `features/ingame.py`

One row per PBP event. One public function:

```python
def build_ingame_snapshots(pbp_db_path, games_db_path) -> pd.DataFrame
```

**Columns:** `game_id, season, action_number, score_diff, seconds_remaining, pre_game_prob, home_fg_pct_live, away_fg_pct_live, home_fouls, away_fouls, turnover_diff_live, timeout_remaining_diff, last_5_poss_swing, quarter, clutch_flag, home_win`

### Feature details

**Score forward-fill:** Sort by `action_number`, `ffill()` from initial (0, 0). `score_diff = home - away`.

**Clock encoding:**
- Regulation (period 1-4): `(4 - period) * 720 + clock_seconds`
- OT (period > 4): `-((period - 5) * 300 + (300 - clock_seconds))` — goes negative, monotonically decreasing. Combined with `quarter` feature for disambiguation.

**Live FG%:** Cumulative `fgm/fga` per team from `is_field_goal=1` rows. 0.0 before first attempt.

**Foul counts:** Cumulative count of `action_type='Foul'` per team.

**Turnover diff:** Cumulative `action_type='Turnover'` count, `home - away`.

**Timeout remaining diff:** Each team starts at 7. Decremented on timeout events. Team parsed from `description` field (team_id is 0 for timeouts). Regex: first word before ` Timeout`, matched case-insensitively against team abbreviations from `game_logs`.

**last_5_poss_swing — Possession state machine (per game):**
- Possession ends on: Made Shot, Turnover, last Free Throw in sequence, defensive Rebound
- Track points scored per possession, signed by team (home positive, away negative)
- Maintain `deque(maxlen=5)`, compute swing at every event
- Detect last FT via regex on `sub_type`: `r"Free Throw (\d+) of (\d+)"` where both match
- Skip technical FTs as possession-enders (check preceding foul sub_type)
- And-1 edge case: document as known approximation, don't over-engineer

**clutch_flag:** `1 if period >= 4 and abs(score_diff) <= 5 else 0`

**pre_game_prob:** Set to `0.5` placeholder. Filled at training time by `train_ingame.py`.

**home_win:** Joined from `game_logs` where `is_home=1`. Applied to every row for that game.

### Performance note
~5.5M PBP rows. Vectorize everything except `last_5_poss_swing` (requires per-game state machine via `groupby.apply()`). Expect 15-30 min for full dataset. Print progress every 500 games.

---

## File 4: `features/pipeline.py`

Orchestration script. Calls `elo.py`, `pregame.py`, `ingame.py` in sequence. Saves both parquet files. Prints summary stats (row counts, column counts, home win rate, season coverage).

```bash
python features/pipeline.py
```

---

## File 5: `tests/test_no_leakage.py`

Pytest-based. Sample 10-20 games for each test (not exhaustive, for speed).

| Test | What it checks |
|------|---------------|
| `test_elo_uses_only_prior_games` | ELO for a game equals ELO computed from all games before that date |
| `test_rolling_stats_exclude_current_game` | Season-to-date eFG% excludes current game's box score |
| `test_ingame_features_no_future_events` | score_diff, fouls, FG% at row N match manual computation from events 1..N |
| `test_pregame_uses_previous_season` | ortg/drtg come from prior season, not current |
| `test_no_nan_in_critical_columns` | No NaN in score_diff, seconds_remaining, home_win, elo_diff |
| `test_home_win_is_binary` | All values 0 or 1 |
| `test_seconds_remaining_range` | Regulation: 0-2880; OT: negative, bounded |

---

## File 6: Updates to README.md

- Change Phase 2 status to **Complete**
- Replace the "Not yet implemented" placeholder under Phase 2 with actual usage docs
- Document the rolling stats approach and possession heuristic

## File 7: Updates to CLAUDE.md

Add to "Resolved design decisions":
- Rolling stats method (cumsum+shift from player_box_scores; previous-season for ORtg/DRtg)
- Timeout team parsing from description field
- Possession identification heuristic for last_5_poss_swing
- Score forward-fill strategy
- OT clock encoding (negative seconds_remaining)
- pre_game_prob placeholder (0.5, filled at training time)

---

## Verification

1. `python features/pipeline.py` — runs end-to-end, produces both parquet files
2. `pytest tests/test_no_leakage.py -v` — all leakage tests pass
3. Spot-check: load parquet files, verify row counts (~13K pregame rows, ~5.5M ingame rows), no NaN in critical columns, reasonable value ranges
