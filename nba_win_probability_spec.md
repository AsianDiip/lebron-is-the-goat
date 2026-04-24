# NBA Win Probability Model — Agent Specification

**Sport:** NBA Basketball  
**Stack:** Python, XGBoost, nba_api  
**Goal:** Real-time win probability that updates after every play

---

## 1. Project Overview

Build an end-to-end ML system that ingests historical NBA player, team, and matchup stats to train a win probability model, then updates that probability live throughout a game as play-by-play events arrive.

The output is a single metric — win probability for the home team — updated after every play. The system operates in two modes:

- **Pre-game:** prior probability based on team strength, rest days, and home/away context, output by a dedicated calibrated LR model
- **In-game:** live probability updated after every scored basket, foul, turnover, or timeout, output by an XGBoost model that takes the pre-game probability as an input feature

---

## 2. System Architecture

Five layers run in sequence during training, then loop continuously during live inference.

### 2.1 Data layer

All historical data comes from the NBA Stats API via the `nba_api` Python package. No API key required. Rate-limit all calls with `time.sleep(0.6)` between requests.

**Target seasons:** 2015-16 through 2024-25 (~13K games)

| Data type | nba_api endpoint | Key fields |
|---|---|---|
| Game logs | `LeagueGameLog` | game_id, date, home/away, W/L, scores |
| Team efficiency | `LeagueDashTeamStats` (Advanced) | OffRtg, DefRtg, Pace, eFG%, TOV%, OREB%, AST% |
| Player box scores | `BoxScoreTraditionalV3` | PTS, AST, REB, STL, BLK, TOV, FG%, TS% |
| Play-by-play | `PlayByPlayV3` | event_type, score, clock, period, team |

**Storage:** SQLite for raw fetched data (`data/raw/*.db`) with indexes on `game_id`, `season`, and `team_id`. Parquet for final processed feature matrices (`data/processed/*.parquet`).

### 2.2 Feature engineering layer

Two categories of features are concatenated into a single vector at inference time.

**Pre-game features** (computed once before tip-off, frozen at tip-off — no lineup-lock re-fetch):
- `elo_diff` — rolling ELO (K=100, walk-forward per game, regress to mean each season)
- `efg_pct_diff` — season-to-date eFG%
- `ortg_diff`, `drtg_diff` — per-100-possession offensive and defensive ratings
- `prev_season_win_pct_diff` — prior season win %, stabilizes early-season predictions
- `rest_days_diff` — days of rest differential
- `home_flag` — always 1 for home team in this model
- `ast_pct_diff`, `tov_pct_diff` — ball movement and turnover rate differential

No H2H (head-to-head) feature. No lineup-lock re-fetch.

**In-game features** (updated after every play event):
- `score_diff` — home minus away score
- `seconds_remaining` — total seconds remaining in game (see clock encoding in Section 7)
- `pre_game_prob` — output of the pre-game LR model; anchors predictions in Q1 when score is near 0
- `home_fg_pct_live`, `away_fg_pct_live` — live field goal percentage for each team
- `home_fouls`, `away_fouls` — raw cumulative foul counts (not FT rate)
- `turnover_diff_live` — cumulative turnover differential
- `timeout_remaining_diff` — timeouts remaining, home minus away
- `last_5_poss_swing` — momentum proxy; net points over last 5 possessions
- `quarter` — current period (1–4, OT)
- `clutch_flag` — 1 if Q4 (or OT) and `abs(score_diff) <= 5`

**Training row structure:** Each play-by-play event becomes one training row. Label = 1 if the home team won the game, 0 otherwise.

### 2.3 Model layer

**Two-stage architecture:**

**Stage 1 — Pre-game model** (`model/train_pregame.py`):
- scikit-learn `LogisticRegression` + `CalibratedClassifierCV` (Platt scaling)
- Input: pre-game features listed above
- Output: `pre_game_prob` (float 0.0–1.0)
- Target: ~66% accuracy, ECE <4%
- Saved to `model/pregame.pkl`

**Stage 2 — In-game model** (`model/train_ingame.py`):
- XGBoost (`binary:logistic`, 500 trees, depth 6)
- Input: in-game features listed above, including `pre_game_prob` from Stage 1
- Post-hoc isotonic calibration on a dedicated calibration split
- Target: Brier score <0.18, ECE <5%
- Saved to `model/ingame.pkl`

**Train/val/test split (by full season — no temporal leakage):**
- Train: 2015–2022 (~8K games)
- Calibration: separate split carved from training data (never the val set)
- Val: 2022–2023 (~1.2K games)
- Test: 2023–2024 (~1.2K games)
- Live: 2024–2025 (held out)

**Calibration:** `CalibratedClassifierCV(cv='prefit', method='isotonic')` fitted on a dedicated calibration split — never on the validation split. Three-way split: train → calibrate → validate.

**Validation checks:**
- Reliability diagram: when model says 70%, home team should win ~70% of the time
- Brier score as primary calibration metric (target <0.18)
- ECE <5% (10-bin)
- AUC-ROC >0.80
- Evaluate separately on the uncertainty region: Q2/Q3, score differential within ±10 pts

**Optional upgrade (after baseline works):** LSTM or Transformer that treats each game as a sequence of events. `GameState.play_log` stores the full event sequence — ensure event encoding is consistent from day one.

### 2.4 Real-time inference pipeline

Polling loop that runs during a live game:

```
every 30 seconds:
  1. Poll PlayByPlayV3 for new events since last check
  2. Deduplicate new events by event_id (not index — API may reorder buffer)
  3. For each new event:
     a. Update in-game state object (score, clock, fouls, timeouts, turnovers)
     b. Recompute in-game feature vector
     c. Concatenate with pre-game feature vector (frozen at tip-off)
     d. Run ingame_model.predict_proba()
     e. Emit updated probability
  4. Log (timestamp, event_description, home_win_prob) on new events only — not every poll
```

Keep the full game state as an in-memory object updated incrementally — do not recompute from scratch on each poll.

### 2.5 Output layer

- **Live probability value:** float 0.0–1.0, updated after every play
- **Probability log:** CSV with columns `[timestamp, period, clock, event, home_win_prob]` — written on new events only, not every poll cycle
- **Dashboard (Phase 5):** Streamlit app showing the probability curve over time with quarter markers and key play annotations

---

## 3. File Structure

```
nba-win-prob/
├── data/
│   ├── raw/                    # SQLite DBs (games.db, pbp.db, players.db)
│   ├── processed/              # Parquet feature matrices
│   ├── fetch_games.py
│   ├── fetch_pbp.py
│   └── fetch_players.py
├── features/
│   ├── elo.py                  # Walk-forward ELO ratings
│   ├── pregame.py              # Pre-game feature computation
│   ├── ingame.py               # In-game snapshot feature computation
│   └── pipeline.py             # Orchestration
├── model/
│   ├── train_pregame.py        # LR + Platt scaling
│   ├── train_ingame.py         # XGBoost + isotonic calibration
│   ├── evaluate.py             # Brier, ECE, reliability diagrams
│   ├── pregame.pkl
│   └── ingame.pkl
├── live/
│   ├── game_state.py           # GameState class (in-memory)
│   ├── poller.py               # PlayByPlayV3 polling loop
│   └── api.py                  # FastAPI server
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   ├── eda.ipynb
│   ├── model_analysis.ipynb
│   └── replay.ipynb
├── tests/
│   └── test_no_leakage.py
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
    home_fouls: int
    away_fouls: int
    home_timeouts: int
    away_timeouts: int
    home_turnovers: int
    away_turnovers: int
    play_log: list[dict]      # all events so far this game
    pregame_features: dict    # static, frozen at tip-off

    def update(self, event: dict) -> None: ...
    def to_feature_vector(self) -> np.ndarray: ...
```

### `Poller` (live/poller.py)

```python
class Poller:
    def __init__(self, game_id: str, pregame_model, ingame_model, interval_sec: int = 30): ...
    def run(self) -> None:
        # Polls PlayByPlayV3, deduplicates by event_id, updates GameState,
        # emits and logs probability on each new event
```

---

## 5. Development Plan

### Phase 1 — Data collection & environment setup

**Goal:** Clean, queryable dataset of historical games and play-by-play events.

- Pull 10 seasons of game logs (2015-16 through 2024-25) via `LeagueGameLog`
- Pull team efficiency stats (Advanced) per season via `LeagueDashTeamStats`
- Pull player box scores via `BoxScoreTraditionalV3` for all games
- Pull play-by-play for all games via `PlayByPlayV3` (batch and cache aggressively — resumable)
- SQLite storage layer: `fetch_games.py`, `fetch_pbp.py`, `fetch_players.py` — all check cache before hitting API
- **Data validation:** every game in `game_logs` must have corresponding PBP and box score rows; missing rows raise an error

**Deliverable:** `data/raw/` populated, fetch modules working, basic EDA notebook

---

### Phase 2 — Feature engineering

**Goal:** Feature matrices ready for two-stage model training.

- `features/elo.py`: walk-forward ELO (K=100, regress to mean each season start)
- `features/pregame.py`: ELO diff, rolling team stats (eFG%, ORtg, DRtg, AST%, TOV%), rest days, prev season win pct
- `features/ingame.py`: score diff, seconds remaining, live FG%, foul counts, timeout diff, turnover diff, last_5_poss_swing, clutch_flag
- Build training dataset: one row per play event, labeled with final game outcome
- **Leakage test:** per-row assertion that no feature uses data from plays after the event timestamp
- Output: `pregame_features.parquet`, `ingame_snapshots.parquet`

**Deliverable:** `data/processed/` populated, leakage tests passing

---

### Phase 3 — Model training & evaluation

**Goal:** Two calibrated models with verified reliability diagrams.

- Train pre-game LR model; calibrate with Platt scaling; evaluate ECE <4%
- Train in-game XGBoost baseline (score_diff + seconds_remaining only)
- Add all in-game features including `pre_game_prob`; hyperparameter sweep
- Post-hoc isotonic calibration on dedicated calibration split (never val set)
- SHAP feature importance; reliability diagrams; win probability curves on 10 historical games
- Save `model/pregame.pkl` and `model/ingame.pkl` with `joblib`

**Deliverable:** Both models saved, evaluation notebook with reliability diagrams, Brier <0.18

---

### Phase 4 — Real-time inference pipeline

**Goal:** Polling loop that emits updated win probability after every live play.

- Implement `GameState` class with `update()` and `to_feature_vector()`
- Implement `Poller`: polls `PlayByPlayV3` every 30 seconds, deduplicates by `event_id`, updates `GameState`, calls both models in sequence, logs output
- Test against a completed historical game by replaying play-by-play in sequence
- Handle edge cases: OT (5-minute periods, not 12), API failures, missing clock data
- Build FastAPI server with `/pregame` and `/live` endpoints

**Deliverable:** `live/` module working end-to-end on a replayed historical game

---

### Phase 5 — Dashboard & polish

**Goal:** Usable prototype with a live probability display.

- Streamlit app: game selector, live probability curve, current probability, quarter markers, key play annotations
- Note: Streamlit is not a true push model — document refresh latency limitation
- Historical replay mode: pick any past game, watch the curve unfold
- Backtest report: per-quarter calibration on test set
- README with setup instructions and architecture summary

**Deliverable:** Running Streamlit dashboard, backtest report

---

### Phase 6 — Sequence Model Upgrade (Optional)

- PyTorch LSTM over play-by-play sequences; target: beat XGBoost Brier by >0.01
- Train with Brier loss; ECE <3%
- Compare XGBoost vs LSTM on uncertainty region specifically

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
fastapi>=0.104.0
uvicorn>=0.24.0
```

---

## 7. Key Implementation Notes

**Rate limiting:** `time.sleep(0.6)` between all API calls during data collection. Live polling uses 30-second intervals.

**ELO implementation:** Initialize all teams at 1500. After each game: winner += K * (1 - expected), loser -= K * (expected), where K=100 and expected = 1 / (1 + 10^((loser_elo - winner_elo) / 400)). Regress toward 1500 at the start of each new season.

**Momentum features:** `last_5_poss_swing` = net points over the last 5 possessions, derived from play-by-play clock timestamps (not wall-clock time).

**Feature leakage:** Each training row's feature vector must only use information available at the moment that play occurred. Enforce with a per-row timestamp assertion during dataset construction in Phase 2.

**Calibration:** Always apply `CalibratedClassifierCV(cv='prefit', method='isotonic')` on a dedicated calibration split. Never fit the calibrator on the validation set. Verify with a reliability diagram before using either model.

**Clock encoding:** Express time remaining as total seconds in the game. For regulation: `time_remaining_game = (4 - period) * 720 + clock_seconds`. OT periods are 300 seconds each and must be handled separately — the formula above produces garbage for OT.

**V3 API:** Use `PlayByPlayV3` and `BoxScoreTraditionalV3` throughout. V2 endpoints are deprecated. V3 clock fields use ISO 8601 format (`PT11M42.00S`); column names are camelCase.

**Model persistence:** `joblib.dump()` / `joblib.load()` — not `pickle`. Artifacts are `model/pregame.pkl` and `model/ingame.pkl`.

**Event deduplication:** During live polling, deduplicate on `event_id` from `PlayByPlayV3`, not on positional index. The API can reorder or re-deliver events in the buffer.
