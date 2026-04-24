# NBA Win Probability Model

Real-time win probability for NBA games, updated after every play. Built with a two-stage model (calibrated Logistic Regression pre-game → XGBoost in-game) and the NBA Stats API.

The model outputs a single float — home team win probability (0.0–1.0) — updated live as play-by-play events arrive during a game.

---

## Architecture

Five layers run in sequence during training, then loop during live inference:

```
Data → Features → Model → Live Inference → Dashboard
```

| Layer | Location | Description |
|---|---|---|
| Data | `data/` | Fetch 2015–2025 game logs, team stats, and play-by-play from the NBA Stats API into SQLite |
| Features | `features/` | Pre-game features (ELO, rolling stats, rest days) + in-game features (score diff, momentum, fouls, clutch flag) |
| Model | `model/` | Two-stage: calibrated LR pre-game model → XGBoost in-game model with isotonic calibration |
| Live inference | `live/` | Polling loop that updates win probability after every play event |
| Dashboard | `dashboard/` | Streamlit probability curve display |

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Data collection | **Complete** | Fetch modules and SQLite storage implemented |
| Phase 2 — Feature engineering | **Complete** | Pre-game and in-game feature computation |
| Phase 3 — Model training | Not started | Two-stage LR + XGBoost training and calibration |
| Phase 4 — Live inference | Not started | Polling loop and GameState class |
| Phase 5 — Dashboard | Not started | Streamlit probability curve display |

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ required. No NBA API key needed — the `nba_api` package uses the public NBA Stats endpoint.

---

## Running Each Phase

### Phase 1 — Data Collection

Fetches 10 seasons (2015-16 through 2024-25) of game logs, team efficiency stats, player box scores, and play-by-play events. Data is stored in SQLite under `data/raw/`. All modules check local cache before hitting the API, so reruns skip already-fetched data.

**Step 1: Game logs and team efficiency stats** (~20 API calls, completes in seconds)

```bash
python data/fetch_games.py
```

This must run first — the other two modules read game IDs from `games.db`.

**Step 2: Play-by-play and player box scores** (~13K API calls each, takes several hours cold)

These can run in parallel in two terminals:

```bash
# Terminal 1
python data/fetch_pbp.py

# Terminal 2
python data/fetch_players.py
```

Both modules print progress as `[i/total] game_id — CACHED/FETCHED`. On interruption, just rerun — cached games are skipped instantly.

To run in the background:

```bash
nohup python data/fetch_pbp.py > data/raw/pbp_fetch.log 2>&1 &
nohup python data/fetch_players.py > data/raw/players_fetch.log 2>&1 &
```

**Handling failures:**

Any game that fails after one retry is logged to `data/raw/failed_games.txt` (or `failed_players.txt`). After the full run completes, retry only those games:

```bash
python data/fetch_pbp.py --resume
python data/fetch_players.py --resume
```

**Fetching a single season:**

```bash
python data/fetch_games.py --season 2023-24
python data/fetch_pbp.py --season 2023-24
python data/fetch_players.py --season 2023-24
```

**Data validation** runs automatically at the end of `fetch_pbp.py`. It asserts that every game in `game_logs` has corresponding play-by-play and box score rows, and prints a coverage summary. Any inconsistency raises an error rather than silently producing bad training data.

**EDA:**

```bash
jupyter notebook notebooks/eda.ipynb
```

---

### Phase 2 — Feature Engineering

```bash
python features/pipeline.py
```

Outputs `data/processed/pregame_features.parquet` and `data/processed/ingame_snapshots.parquet`. Each in-game row is one play-by-play event with a full feature vector and a binary label (1 = home team won).

**Implementation notes:**

- `features/elo.py`: Walk-forward ELO ratings (K=100). Each team's ELO is recorded *before* the game it is used for. Ratings regress 25% toward 1500 at each season boundary.
- `features/pregame.py`: Season-to-date eFG%, AST rate, and TOV rate are computed from `player_box_scores` using a cumulative sum shifted by one game (excludes the current game). Offensive and defensive ratings use previous-season values from `team_efficiency` since true per-possession ratings can't be derived from box scores alone. First season (2015-16) falls back to league averages.
- `features/ingame.py`: `last_5_poss_swing` is computed by a per-game possession state machine that identifies possession-ending events (made shots, turnovers, last free throw in sequence, defensive rebounds) and tracks a rolling deque of the last 5 possessions. Timeout team ownership is parsed from the PBP `description` field (the `team_id` column is 0 for timeout events). OT clock encodes as negative `seconds_remaining` (OT1 runs 0 to -300, OT2 runs -300 to -600, etc.).
- `tests/test_no_leakage.py`: Sample-based leakage assertions. Run with `pytest tests/test_no_leakage.py -v`.

---

### Phase 3 — Model Training

> Not yet implemented.

```bash
# Train pre-game model first — ingame model depends on its output
python model/train_pregame.py
python model/train_ingame.py
python model/evaluate.py
```

Train/val/test split by full season: Train 2015–2022, Val 2022–2023, Test 2023–2024, Live 2024–2025 held out. Calibration uses a dedicated split carved from training data — never the val set. Saves `model/pregame.pkl` and `model/ingame.pkl`.

---

### Phase 4 — Live Inference

> Not yet implemented.

```bash
python live/poller.py --game_id <GAME_ID>
```

Polls `PlayByPlayV3` every 30 seconds, deduplicates events by `event_id`, updates in-memory game state, and logs `[timestamp, period, clock, event, home_win_prob]` to a CSV after each new play.

---

### Phase 5 — Dashboard

> Not yet implemented.

```bash
streamlit run dashboard/app.py
```

Shows the live win probability curve with quarter markers and key play annotations. Uses `st.empty()` + rerun for updates (not a true push model — latency is ~30 seconds).

---

## File Structure

```
nba-win-prob/
├── data/
│   ├── raw/                    # SQLite databases (games.db, pbp.db, players.db)
│   ├── processed/              # Feature matrices (pregame_features.parquet, ingame_snapshots.parquet)
│   ├── fetch_games.py          # LeagueGameLog + LeagueDashTeamStats
│   ├── fetch_pbp.py            # PlayByPlayV3 (includes cross-table validation)
│   └── fetch_players.py        # BoxScoreTraditionalV3
├── features/
│   ├── elo.py                  # Walk-forward ELO ratings (K=100)
│   ├── pregame.py              # ELO diff, rolling team stats, rest days, prev season win pct
│   ├── ingame.py               # Score diff, momentum, foul counts, timeout diff, clutch flag
│   └── pipeline.py             # Combines both into a single feature vector
├── model/
│   ├── train_pregame.py        # Calibrated LR (Platt scaling) → pregame.pkl
│   ├── train_ingame.py         # XGBoost + isotonic calibration → ingame.pkl
│   ├── evaluate.py             # Brier, ECE, reliability diagrams, curve plots
│   ├── pregame.pkl             # Saved pre-game model
│   └── ingame.pkl              # Saved in-game model
├── live/
│   ├── game_state.py           # GameState class (incremental in-memory state)
│   ├── poller.py               # PlayByPlayV3 polling loop + probability logging
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

## Data Storage

| Database | Tables | Contents |
|---|---|---|
| `data/raw/games.db` | `game_logs`, `team_efficiency` | One row per team per game; season-level Advanced efficiency stats |
| `data/raw/pbp.db` | `play_by_play` | One row per play event; clock parsed to seconds |
| `data/raw/players.db` | `player_box_scores` | One row per player per game |

All three databases have indexes on `game_id`, `season`, and `team_id`. SQLite WAL mode is enabled for safe concurrent reads during development.

---

## Key Implementation Notes

**Two-stage model:** The pre-game LR model outputs `pre_game_prob`, which is passed as an explicit input feature to the in-game XGBoost model. This anchors predictions in Q1 when the score is still near 0-0. Train `train_pregame.py` before `train_ingame.py`.

**Season range:** 10 seasons (2015-16 through 2024-25). Train on 2015–2022, calibrate on a dedicated split, validate on 2022–2023, test on 2023–2024. 2024–2025 is held out for live demo.

**API version:** Use `PlayByPlayV3` and `BoxScoreTraditionalV3` throughout. V2 endpoints are deprecated. V3 clock fields use ISO 8601 format (`PT11M42.00S`); column names are camelCase.

**Rate limiting:** All API calls during data collection use `time.sleep(0.6)` between requests. Live polling uses 30-second intervals.

**Clock encoding:** Time remaining is expressed as total seconds in the game, not just the current quarter. For regulation: `time_remaining_game = (4 - period) * 720 + clock_seconds`. OT periods are 300 seconds each and must be handled separately — the formula above is wrong for OT.

**Feature leakage:** Every training row must only contain features derivable from plays at or before that event's timestamp. Enforced with a per-row assertion during dataset construction in Phase 2.

**Calibration:** Three-way split — train → calibrate → validate. `CalibratedClassifierCV(cv='prefit', method='isotonic')` fit on the calibration split only. Never touch the val set during calibration.

**In-game features:** Use raw foul counts (`home_fouls`, `away_fouls`), not FT rate. No H2H feature.

**Event deduplication:** During live polling, deduplicate on `event_id` from `PlayByPlayV3`, not positional index. The API can reorder events in the buffer.

**Model persistence:** `joblib.dump()` / `joblib.load()` — not `pickle`. Artifacts: `model/pregame.pkl` and `model/ingame.pkl`.
