# NBA Win Probability Model

Real-time win probability for NBA games, updated after every play. Built with XGBoost and the NBA Stats API.

The model outputs a single float — home team win probability (0.0–1.0) — updated live as play-by-play events arrive during a game.

---

## Architecture

Five layers run in sequence during training, then loop during live inference:

```
Data → Features → Model → Live Inference → Dashboard
```

| Layer | Location | Description |
|---|---|---|
| Data | `data/` | Fetch historical game logs, team stats, and play-by-play from the NBA Stats API into SQLite |
| Features | `features/` | Pre-game features (ELO, rolling stats, rest days) + in-game features (score diff, momentum, fouls) |
| Model | `model/` | XGBoost classifier with isotonic calibration |
| Live inference | `live/` | Polling loop that updates win probability after every play event |
| Dashboard | `dashboard/` | Streamlit probability curve display |

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Data collection | **Complete** | Fetch modules and SQLite storage implemented |
| Phase 2 — Feature engineering | Not started | Pre-game and in-game feature computation |
| Phase 3 — Model training | Not started | XGBoost training, calibration, evaluation |
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

Fetches 4 seasons (2021-22 through 2024-25) of game logs, team efficiency stats, player box scores, and play-by-play events. Data is stored in SQLite under `data/raw/`. All modules check local cache before hitting the API, so reruns skip already-fetched data.

**Step 1: Game logs and team efficiency stats** (~8 API calls, completes in seconds)

```bash
python data/fetch_games.py
```

This must run first — the other two modules read game IDs from `games.db`.

**Step 2: Play-by-play and player box scores** (~4,920 API calls each, ~50 min cold)

These can run in parallel in two terminals:

```bash
# Terminal 1
python data/fetch_pbp.py

# Terminal 2
python data/fetch_players.py
```

Both modules print progress as `[i/total] game_id — CACHED/FETCHED`. On interruption, just rerun — cached games are skipped instantly.

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

> Not yet implemented.

```bash
python features/pipeline.py
```

Outputs `data/processed/train.parquet` and `data/processed/val.parquet`. Each row is one play-by-play event with a full feature vector (pre-game + in-game features concatenated) and a binary label (1 = home team won).

---

### Phase 3 — Model Training

> Not yet implemented.

```bash
python model/train.py
python model/evaluate.py
```

Trains on 2021-22 and 2022-23, calibrates on first half of 2023-24, validates on second half. Saves the calibrated model to `model/model.joblib`. Evaluation produces a reliability diagram and win probability curves for historical games.

---

### Phase 4 — Live Inference

> Not yet implemented.

```bash
python live/poller.py --game_id <GAME_ID>
```

Polls `PlayByPlayV3` every 15 seconds, deduplicates events by `action_number`, updates in-memory game state, and logs `[timestamp, period, clock, event, home_win_prob]` to a CSV after each new play.

---

### Phase 5 — Dashboard

> Not yet implemented.

```bash
streamlit run dashboard/app.py
```

Shows the live win probability curve with quarter markers and key play annotations. Uses `st.empty()` + rerun for updates (not a true push model — latency is ~15 seconds).

---

## File Structure

```
nba-win-prob/
├── data/
│   ├── raw/                  # SQLite databases (games.db, pbp.db, players.db)
│   ├── processed/            # Feature matrices (train.parquet, val.parquet)
│   ├── fetch_games.py        # LeagueGameLog + LeagueDashTeamStats
│   ├── fetch_pbp.py          # PlayByPlayV3 (includes cross-table validation)
│   └── fetch_players.py      # BoxScoreTraditionalV3
├── features/
│   ├── pregame.py            # ELO, rolling team stats, rest days, H2H win %
│   ├── ingame.py             # Score diff, momentum, fouls, clutch flag
│   └── pipeline.py           # Combines both into a single feature vector
├── model/
│   ├── train.py              # XGBoost training + isotonic calibration
│   ├── evaluate.py           # Reliability diagram, Brier score, curve plots
│   └── model.joblib          # Saved trained model
├── live/
│   ├── game_state.py         # GameState class (incremental in-memory state)
│   └── poller.py             # Polling loop + probability logging
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
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

All three databases have indexes on `game_id`, `date`/`season`, and `team_id`. SQLite WAL mode is enabled for safe concurrent reads during development.

---

## Key Implementation Notes

**API version:** The spec references `PlayByPlayV2` and `BoxScoreTraditionalV2`, but both are deprecated in `nba_api` v1.11.4+. The implementation uses `PlayByPlayV3` and `BoxScoreTraditionalV3`. V3 clock fields use ISO 8601 format (`PT11M42.00S`) and column names are camelCase.

**Rate limiting:** All API calls during data collection use `time.sleep(0.6)` between requests. Live polling uses 15-second intervals, which is safe for the NBA Stats endpoint.

**Clock encoding:** Time remaining is expressed as total seconds in the game, not just the current quarter. For regulation: `time_remaining_game = (4 - period) * 720 + clock_seconds`. Overtime periods are 300 seconds each and use negative values to distinguish from regulation.

**Feature leakage:** Every training row must only contain features derivable from plays at or before that event's timestamp. Enforced with a per-row assertion during dataset construction in Phase 2.

**Calibration:** XGBoost probabilities are calibrated with `CalibratedClassifierCV(cv='prefit', method='isotonic')` fit on a dedicated calibration split (first half of 2023-24). The validation split (second half of 2023-24) is never seen during training or calibration.

**Event deduplication:** During live polling, deduplication uses `action_number` from `PlayByPlayV3`, not positional index. The NBA Stats API occasionally reorders events in the buffer.

**Model persistence:** `joblib.dump()` / `joblib.load()` — not `pickle`.
