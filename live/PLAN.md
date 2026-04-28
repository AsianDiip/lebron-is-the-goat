# Phase 4 — Real-Time Inference Pipeline

## Context

Phases 1-3 (data collection, feature engineering, model training) are complete. Both models (`model/pregame.pkl` and `model/ingame.pkl`) are trained and saved. The `live/` directory is empty — nothing has been started for Phase 4.

Phase 4 builds the real-time inference pipeline: a `GameState` class that tracks mutable in-game state, a `Poller` that hits `PlayByPlayV3` every 30 seconds during a live game, and a FastAPI server for programmatic access.

## Files to Create

1. **`live/game_state.py`** — GameState class
2. **`live/poller.py`** — Polling loop + CLI entry point
3. **`live/api.py`** �� FastAPI server
4. **`tests/test_game_state.py`** — Unit tests for GameState

## Key Files to Reuse

- `features/ingame.py` — `compute_seconds_remaining()`, `_is_last_free_throw()`, `_parse_timeout_team()`, `_TIMEOUT_TEAM_RE`, `_FT_RE`, `_TIMEOUTS_PER_TEAM` (reuse directly, don't duplicate)
- `model/train_ingame.py` — `StratifiedCalibrator` class (must import before `joblib.load`), `INGAME_FEATURES` list
- `model/train_pregame.py` — `PREGAME_FEATURES` list
- `data/fetch_pbp.py` — `parse_clock()`, `CLOCK_RE` for ISO 8601 clock parsing from live API

## Implementation Plan

### Step 1: `live/game_state.py` — GameState Class

The core mutable state object. Mirrors the logic in `features/ingame.py:_compute_game_features()` but operates incrementally (one event at a time) instead of batch.

```python
class GameState:
    game_id: str
    home_team_id: int
    away_team_id: int
    home_score: int = 0
    away_score: int = 0
    period: int = 1
    clock_seconds: int = 720
    pre_game_prob: float  # frozen at tip-off

    # Cumulative counters (mirrors ingame.py running vars)
    home_fouls, away_fouls: int
    home_turnovers, away_turnovers: int
    home_timeouts_used, away_timeouts_used: int
    home_fgm, home_fga, away_fgm, away_fga: int
    home_2pm, home_2pa, away_2pm, away_2pa: int
    home_3pm, home_3pa, away_3pm, away_3pa: int
    home_ftm, home_fta, away_ftm, away_fta: int

    # Possession state machine (deque, maxlen=5)
    _poss_deque: deque[int]
    _current_team: int | None
    _poss_points: int

    # Event tracking
    _seen_event_ids: set[int]
    play_log: list[dict]
    team_abbrev_map: dict[str, int]

    def update(self, event: dict) -> None: ...
    def to_feature_vector(self) -> np.ndarray: ...  # returns 18-element float32 array
```

**Key design decisions:**
- Reuse `compute_seconds_remaining`, `_is_last_free_throw`, `_parse_timeout_team` from `features/ingame.py` directly (import them)
- The possession state machine logic from `_compute_game_features` lines 238-288 is reimplemented as incremental `update()` calls
- `to_feature_vector()` returns features in exact `INGAME_FEATURES` order (index 16 = quarter for StratifiedCalibrator)
- `update()` parses live PlayByPlayV3 fields (camelCase: `actionNumber`, `actionType`, `subType`, `clock`, `scoreHome`, `scoreAway`, `isFieldGoal`, `shotResult`, `teamId`, `actionId`, `period`, `description`) and converts to the same field names used in ingame.py
- Deduplication: `update()` checks `action_id` against `_seen_event_ids`, returns early if duplicate

### Step 2: `live/poller.py` — Polling Loop + Historical Replay

Two modes:
1. **Live mode**: polls `PlayByPlayV3` every 30 seconds for a running game
2. **Replay mode** (`--replay`): reads PBP from `pbp.db` for a completed game and replays events sequentially (for testing/validation)

```
CLI:
  python live/poller.py --game_id 0022301234           # live polling
  python live/poller.py --game_id 0022301234 --replay  # replay from DB
```

**Flow:**
1. Load both models (`pregame.pkl`, `ingame.pkl` — importing `StratifiedCalibrator` first)
2. Look up pregame features for the game from `pregame_features.parquet`
3. Compute `pre_game_prob` by running pregame model on the feature vector
4. Initialize `GameState` with pregame data
5. Loop:
   - Fetch new PBP events (API or DB)
   - Parse ISO 8601 clock → seconds
   - Feed each new event to `GameState.update()`
   - Call `ingame_model.predict_proba(state.to_feature_vector().reshape(1, -1))[:, 1]`
   - Print and log: `[timestamp, period, clock, event_description, home_win_prob]`
   - Write to CSV: `live/{game_id}_probability.csv` (append on new events only)
   - Sleep 30s (live mode) or continue immediately (replay mode)
6. Stop when game ends (period start/end events indicate final, or no new events after repeated polls)

**Pregame feature lookup for live games (2024-25):**
- If game_id exists in `pregame_features.parquet`, use those features
- If not (truly live, brand-new game): compute from DB using `features/pregame.py` helpers + `features/elo.py`
- This keeps live mode working for current-season games

### Step 3: `live/api.py` — FastAPI Server

Lightweight FastAPI app with two endpoints:

- `GET /pregame/{game_id}` — returns `{ game_id, home_team, away_team, pre_game_prob }`
- `GET /live/{game_id}` — returns `{ game_id, period, clock, score_home, score_away, home_win_prob, last_event }`

Internally manages a dict of active `GameState` objects keyed by `game_id`. Spawns a background polling task per game on first request. Uses `asyncio` for non-blocking polling.

### Step 4: `tests/test_game_state.py` — Unit Tests

- **State machine correctness**: Feed a known sequence of events → verify score, fouls, turnovers, timeouts, possession swing
- **Feature vector ordering**: Verify `to_feature_vector()` produces 18 elements in `INGAME_FEATURES` order
- **Deduplication**: Feed same event_id twice → state only updates once
- **Clock encoding**: Regulation Q1-Q4, OT periods produce correct `seconds_remaining`
- **Clutch flag**: Test Q4 close-game detection
- **End-to-end replay**: Replay a known historical game, compare final probability to model evaluation output

## Verification

1. **Replay test**: `python live/poller.py --game_id <known_game> --replay` — verify the probability curve matches training data expectations
2. **Unit tests**: `pytest tests/test_game_state.py -v`
3. **API smoke test**: Start `uvicorn live.api:app`, hit `/pregame/{game_id}` and `/live/{game_id}`
4. **Live test** (if game in progress): `python live/poller.py --game_id <live_game_id>` — verify polling, deduplication, CSV output

## Order of Implementation

1. `live/game_state.py` (no external dependencies beyond features/ingame.py imports)
2. `tests/test_game_state.py` (validate GameState in isolation)
3. `live/poller.py` (integrates GameState + models + API/DB)
4. `live/api.py` (thin wrapper around poller logic)
