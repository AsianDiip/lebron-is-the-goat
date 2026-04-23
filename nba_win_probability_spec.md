# NBA Win Probability Model — Agent Specification

**Sport:** NBA Basketball  
**Stack:** Python, XGBoost, nba_api  
**Goal:** Real-time win probability that updates after every play

---

## 1. Project Overview

Build an end-to-end ML system that ingests historical NBA player, team, and matchup stats to train a win probability model, then updates that probability live throughout a game as play-by-play events arrive.

The output is a single metric — win probability for the home team — updated after every play. The system operates in two modes:

- **Pre-game:** prior probability based on team strength, rest days, home/away, and head-to-head context
- **In-game:** live probability updated after every scored basket, foul, turnover, or timeout

---

## 2. System Architecture

Five layers run in sequence during training, then loop continuously during live inference.

### 2.1 Data layer

All historical data comes from the NBA Stats API via the `nba_api` Python package. No API key required. Rate-limit all calls (~1 req/sec).

**Target seasons:** 2021-22 through 2023-24 for training, 2024-25 for validation — recent enough to reflect current pace and style of play.

| Data type | nba_api endpoint | Key fields |
|---|---|---|
| Game logs | `LeagueGameLog` | game_id, date, home/away, W/L, scores |
| Team efficiency | `LeagueDashTeamStats` | OffRtg, DefRtg, Pace, eFG%, TOV%, OREB% |
| Player box scores | `BoxScoreTraditionalV2` | PTS, AST, REB, STL, BLK, TOV, FG%, TS% |
| Play-by-play | `PlayByPlayV2` | event_type, score, clock, period, team |

**Storage:** Use **SQLite** for raw fetched data (game logs, box scores, play-by-play) with indexes on `game_id`, `date`, and `team_id` — this enables fast filtered queries during feature engineering without scanning entire files. Use **parquet** only for the final processed feature matrices (`train.parquet`, `val.parquet`).

### 2.2 Feature engineering layer

Two categories of features are concatenated into a single vector at inference time.

**Pre-game features** (computed once before tip-off, re-fetched at T-10 min to capture late injury/lineup news):
- Rolling 10-game averages for both teams: OffRtg, DefRtg, Pace, eFG%, TOV%, OREB%
- ELO rating for each team (updated after every game result)
- Home/away flag
- Days of rest for each team
- Travel distance (back-to-back flag as proxy)
- Head-to-head win % over last 2 seasons
- Starting lineup quality score (sum of top-5 player TS% rolling averages)

**In-game features** (updated after every play event):
- Score differential (home minus away)
- Time remaining in game (seconds) — see clock encoding note in Section 7
- Current quarter/period
- Possession indicator
- Home team foul count, away team foul count
- Foul trouble indicator: flag for any top-3 player (by TS%) with 4+ fouls
- Momentum: points scored by each team in last 3 minutes and last 5 minutes (game clock, not wall clock)
- Run indicator: current scoring run length (e.g. home team on a 10-0 run)
- Quarters behind: number of quarters in which the team trailed at the buzzer (captures "dug a hole early" vs "just fell behind")
- Clutch time flag: Q4 (or OT) with ≤5 minutes remaining and score differential ≤5

**Training row structure:** Each play-by-play event becomes one training row. Label = 1 if the home team won the game, 0 otherwise. A single season produces ~500,000+ rows.

### 2.3 Model layer

**Primary model:** XGBoost classifier (`XGBClassifier`)
- Objective: `binary:logistic`
- Eval metric: `logloss`
- Train on 2021-22 and 2022-23, calibrate on first half of 2023-24, validate on second half of 2023-24
- Apply isotonic regression calibration (`CalibratedClassifierCV(cv='prefit', method='isotonic')`) fitted on the calibration split — **not** the validation split, to avoid contaminating the held-out evaluation
- Apply `scale_pos_weight` or stratified sampling to account for class skew in late-game blowout rows

**Validation checks:**
- Reliability diagram: when model says 70%, home team should win ~70% of the time; flag any bucket with fewer than 500 samples as unreliable
- Plot win probability curves for 5–10 historical games and verify they tell the right story (big leads → high probability, comebacks → visible swings)
- Brier score as secondary calibration metric

**Optional upgrade (after baseline works):** LSTM or Transformer that treats each game as a sequence of events — better at capturing momentum and runs than XGBoost on tabular snapshots. `GameState.play_log` already stores the full event sequence; ensure event encoding is consistent from day one to avoid reprocessing.

### 2.4 Real-time inference pipeline

Polling loop that runs during a live game:

```
every ~15 seconds:
  1. Poll PlayByPlayV2 for new events since last check
  2. Deduplicate new events by event_id (not index — API may reorder buffer)
  3. For each new event:
     a. Update in-game state object (score, clock, fouls, possession)
     b. Recompute in-game feature vector
     c. Concatenate with pre-game feature vector (frozen at tip-off)
     d. Run model.predict_proba()
     e. Emit updated probability
  4. Log (timestamp, event_description, home_win_prob) to output store on new event only — not on every poll
```

Keep the full game state as an in-memory object updated incrementally — do not recompute from scratch on each poll.

**Pre-game lineup lock:** At T-10 minutes before tip-off, re-fetch injury reports and starting lineup data, recompute `pregame_features`, then freeze them for the duration of the game.

### 2.5 Output layer

- **Live probability value:** float 0.0–1.0, updated after every play
- **Probability log:** CSV with columns `[timestamp, period, clock, event, home_win_prob]` — written on new events only, not every poll cycle
- **Dashboard (Phase 5):** Streamlit app showing the probability curve over time with quarter markers and key play annotations

---

## 3. File Structure

```
nba-win-prob/
├── data/
│   ├── raw/                  # Fetched API data (SQLite)
│   ├── processed/            # Feature matrices (parquet)
│   ├── fetch_games.py        # LeagueGameLog + LeagueDashTeamStats fetching
│   ├── fetch_pbp.py          # PlayByPlayV2 fetching
│   └── fetch_players.py      # BoxScoreTraditionalV2 fetching
├── features/
│   ├── pregame.py            # Pre-game feature computation
│   ├── ingame.py             # In-game state + feature updates
│   └── pipeline.py           # Combines both into a single vector
├── model/
│   ├── train.py              # XGBoost training + calibration
│   ├── evaluate.py           # Reliability diagram, Brier score, curve plots
│   └── model.joblib          # Saved trained model (joblib format)
├── live/
│   ├── game_state.py         # GameState class (in-memory state object)
│   └── poller.py             # Polling loop + probability logging / output
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── requirements.txt
└── README.md
```

---

## 4. Key Classes & Interfaces

### `GameState` (live/game_state.py)
Holds all mutable in-game state. Updated after each new play event.

```python
class GameState:
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    clock_seconds: int        # seconds remaining in current quarter
    possession: str           # "home" | "away"
    home_fouls: int
    away_fouls: int
    play_log: list[dict]      # all events so far this game (used for sequence models)
    pregame_features: dict    # static, frozen after lineup lock at T-10 min

    def update(self, event: dict) -> None: ...
    def to_feature_vector(self) -> np.ndarray: ...
```

### `WinProbModel` (model/train.py)
Thin wrapper around the trained XGBoost + calibrator.

```python
class WinProbModel:
    def predict(self, feature_vector: np.ndarray) -> float:
        # Returns home team win probability as float 0.0–1.0
```

### `Poller` (live/poller.py)

```python
class Poller:
    def __init__(self, game_id: str, model: WinProbModel, interval_sec: int = 15): ...
    def run(self) -> None:
        # Polls API, deduplicates by event_id, updates GameState,
        # emits and logs probability on each new event
```

---

## 5. Development Plan

### Phase 1 — Data collection & environment setup (~1 week)

**Goal:** Clean, queryable dataset of historical games and play-by-play events.

- Install dependencies: `nba_api`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `requests`, `sqlite3`, `jupyter`, `streamlit`, `joblib`
- Pull 3 seasons of game logs (2021-22 through 2023-24) via `LeagueGameLog`
- Pull team efficiency stats per season via `LeagueDashTeamStats`
- Pull player box scores via `BoxScoreTraditionalV2` for all games
- Pull play-by-play for all games via `PlayByPlayV2` (this takes time — batch and cache aggressively)
- Build local SQLite storage layer: write `fetch_games.py`, `fetch_pbp.py`, and `fetch_players.py` modules that check local cache before hitting the API
- **Data validation:** after fetching, assert that `game_id` values are consistent across all three tables; any game present in logs but missing play-by-play must raise an error rather than silently producing corrupted training rows

**Deliverable:** `/data/raw/` populated, fetch modules working, basic EDA notebook

---

### Phase 2 — Feature engineering (1–2 weeks)

**Goal:** Feature matrix with ~500k+ labeled rows ready for model training.

- Implement `pregame.py`: rolling averages, ELO updater, rest days, H2H win %, home/away flag
- Implement `ingame.py`: score diff, time remaining, momentum windows, run tracker, foul counts, foul trouble indicator, quarters-behind counter, clutch time flag
- Build training dataset: iterate all historical games, generate one row per play event, label with final game outcome
- **Leakage test:** for every training row, assert that no feature value could only be derived from plays that occur after the labeled event; use the play clock timestamp as the enforcement boundary
- EDA: correlation heatmap, feature distributions, verify class balance across game states (early game rows should be near 50/50)
- Feature selection: drop zero-importance or highly correlated features

**Deliverable:** `/data/processed/train.parquet` and `val.parquet`

---

### Phase 3 — Model training & evaluation (~2 weeks)

**Goal:** Calibrated XGBoost model with a verified reliability diagram.

- Train baseline `XGBClassifier` on 2021-22 and 2022-23 data, evaluate log-loss on validation split
- Tune hyperparameters: `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`
- Calibrate outputs: fit `CalibratedClassifierCV(cv='prefit', method='isotonic')` on the calibration split (first half of 2023-24); evaluate final reliability on the held-out validation split (second half of 2023-24)
- Plot reliability diagram — flag buckets with fewer than 500 samples; verify calibration holds in clutch situations (Q4, close games)
- Plot win probability curves for 10 historical games — sanity check that curves make intuitive sense
- Save model to `model/model.joblib` using `joblib.dump()`

**Deliverable:** Saved calibrated model, evaluation notebook with reliability diagram

---

### Phase 4 — Real-time inference pipeline (1–2 weeks)

**Goal:** Polling loop that emits updated win probability after every live play.

- Implement `GameState` class with `update()` and `to_feature_vector()` methods
- Implement lineup lock: re-fetch injury/availability data at T-10 min before tip-off, recompute and freeze `pregame_features`
- Implement `Poller` class: poll `PlayByPlayV2` every 15 seconds, deduplicate new events by `event_id`, update `GameState`, call model, log output
- Test against a completed historical game by replaying its play-by-play in sequence and comparing emitted probabilities to expected curve shape
- Handle edge cases: overtime periods (5-minute OT, not 12-minute quarters), game delays, API rate limits, missing clock data
- Add graceful retry logic for API failures

**Deliverable:** `live/` module working end-to-end on a replayed historical game

---

### Phase 5 — Dashboard & polish (~1 week)

**Goal:** Usable prototype with a live probability display.

- Build Streamlit app: game selector, live probability curve (line chart updating in real time via `st.empty()` + rerun), current probability as large number, quarter markers on x-axis, key play annotations on hover
- Note: Streamlit is not a true push model — document the refresh mechanism and its latency limitation. If reliability matters long-term, consider a FastAPI + WebSocket backend.
- Add pre-game view: show model's prior probability at tip-off with contributing factors (ELO diff, rest advantage, home/away)
- Backtest calibration on hold-out season: reliability diagram, average log-loss per quarter
- Write README with setup instructions and architecture summary

**Deliverable:** Running Streamlit dashboard, backtest report

---

## 6. Dependencies

```
nba_api>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
scikit-learn>=1.3.0
requests>=2.31.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 7. Key Implementation Notes

**Rate limiting:** The NBA Stats API is unofficial and will return 429 errors if polled too aggressively. Add `time.sleep(0.6)` between all API calls during data collection. During live polling, 15-second intervals are safe.

**ELO implementation:** Initialize all teams at 1500. After each game, update winner += K * (1 - expected), loser -= K * (1 - expected), where K=20 and expected = 1 / (1 + 10^((loser_elo - winner_elo) / 400)). Reset toward 1500 at the start of each season (regression to mean).

**Momentum features:** Compute points scored by each team in the last 180 seconds (3 min) and last 300 seconds (5 min) of game clock — not wall-clock time. Use the `clock` field from play-by-play events.

**Feature leakage:** Be careful not to include any features derived from future plays when constructing training rows. Each row's feature vector must only use information available at the moment that play occurred. Enforce this with a per-row timestamp assertion during dataset construction.

**Calibration:** Raw XGBoost probabilities tend to be overconfident. Always apply `CalibratedClassifierCV(cv='prefit', method='isotonic')` on a dedicated calibration split and verify with a reliability diagram before using the model in production. Do not fit the calibrator on your validation set.

**Clock encoding:** Express time remaining as total seconds in the game (not just the quarter). A possession with 10 seconds left in Q4 is very different from one with 10 seconds left in Q1. For regulation: `time_remaining_game = (4 - period) * 720 + clock_seconds`. For overtime: each OT period is 300 seconds, so `time_remaining_game = -(ot_period - 1) * 300 - (300 - clock_seconds)` (negative values indicate overtime, distinct from regulation).

**Model persistence:** Use `joblib.dump()` / `joblib.load()` rather than `pickle` for saving the trained model. joblib is significantly faster for numpy-heavy objects and is already in the dependency list.

**Event deduplication:** During live polling, deduplicate on `event_id` from the play-by-play response, not on positional index. The NBA Stats API occasionally re-orders or re-delivers events in the buffer, and index-based deduplication will silently drop or double-count plays.
