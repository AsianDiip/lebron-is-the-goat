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
| Data | `data/` | Fetch 2015–2026 game logs, team stats, and play-by-play from the NBA Stats API into SQLite |
| Features | `features/` | Pre-game features (ELO, rolling stats, rest days) + in-game features (score diff, momentum, live FG%/2PT%/3PT%/FT%, fouls, clutch flag) |
| Model | `model/` | Two-stage: calibrated LR pre-game model → XGBoost in-game model with isotonic calibration |
| Live inference | `live/` | FastAPI server with REST + WebSocket endpoints, background polling per game |
| Dashboard | `web/` | Next.js + TypeScript + Tailwind CSS + Recharts |

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Data collection | **Complete** | Fetch modules and SQLite storage implemented |
| Phase 2 — Feature engineering | **Complete** | Pre-game and in-game feature computation |
| Phase 3 — Model training | **Complete** | Two-stage LR + XGBoost, stratified calibration, evaluation |
| Phase 4 — Live inference | **Complete** | GameState, Poller, FastAPI server with WebSocket |
| Phase 5 — Dashboard | **Complete** | React dashboard with real-time updates, replay, and backtest report |

---

## Setup

```bash
# Python backend
pip install -r requirements.txt

# Frontend
cd web && npm install
```

Python 3.11+ required. Node.js 18+ required for the frontend. No API keys needed — the `nba_api` package uses the public NBA Stats endpoint.

---

## Quick Start (Live Game)

```bash
# Terminal 1: Start the API server
uvicorn live.api:app --reload

# Terminal 2: Start the dashboard
cd web && npm run dev

# Open http://localhost:3000
# Enter a game ID to track live, or browse historical games for replay
```

To find today's game IDs:

```bash
python3 -c "
from nba_api.live.nba.endpoints import scoreboard
sb = scoreboard.ScoreBoard()
for g in sb.get_dict()['scoreboard']['games']:
    print(g['gameId'], g['awayTeam']['teamTricode'], '@', g['homeTeam']['teamTricode'], g['gameStatusText'])
"
```

---

## Running Each Phase

### Phase 1 — Data Collection

Fetches 11 seasons (2015-16 through 2025-26) of game logs, team efficiency stats, player box scores, and play-by-play events. Data is stored in SQLite under `data/raw/`. All modules check local cache before hitting the API, so reruns skip already-fetched data.

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

```bash
# Train pre-game model first — ingame model depends on its output
python model/train_pregame.py      # outputs model/pregame.pkl + data/processed/pregame_probs.parquet
python model/train_ingame.py       # outputs model/ingame.pkl
python model/train_ingame.py --sweep  # optional: 2-stage hyperparameter search (~2 hrs)
python model/evaluate.py           # saves figures to model/eval_figures/
```

Train/val/test split by full season: Train 2015–2022, Val 2022–2023, Test 2023–2024, Live 2024–2025 and 2025–2026 held out.

**Two-stage architecture:**
- Pre-game: calibrated Logistic Regression (Platt scaling) outputs `pre_game_prob`
- In-game: XGBoost (depth 4, lr 0.05, 1000 trees) takes `pre_game_prob` as an input feature. Post-hoc `StratifiedCalibrator` with phase-specific isotonic calibrators for Q1-Q2, Q3-Q4, and OT.

**Current test set performance (2023-24):** Brier 0.151, ECE 1.4%, AUC-ROC 0.862

---

### Phase 4 — Live Inference

All commands must be run from the project root directory.

```bash
# Start the FastAPI server (required for the dashboard)
uvicorn live.api:app --reload

# CLI: poll a live game (prints to stdout + logs to live/{game_id}_probability.csv)
python live/poller.py --game_id 0042500115

# CLI: replay a completed historical game from pbp.db
python live/poller.py --game_id 0022301234 --replay
```

**Components:**
- `live/game_state.py` — `GameState` class: tracks all mutable in-game state incrementally, produces the 18-element feature vector. Deduplicates by `actionNumber` (mapped to `actionId`).
- `live/poller.py` — Two polling modes: live (uses `nba_api.live.nba.endpoints.playbyplay.PlayByPlay`) and replay (reads from SQLite `pbp.db`). Logs `[timestamp, period, clock, event, home_win_prob]` to `live/{game_id}_probability.csv`.
- `live/api.py` — FastAPI server with REST and WebSocket endpoints.

**API Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/pregame/{game_id}` | GET | Pre-game win probability |
| `/pregame/{game_id}/breakdown` | GET | Pre-game feature values + human-readable labels |
| `/live/{game_id}` | GET | Current live state, probability history, play log, and box score |
| `/games?season=2024-25` | GET | Game list for replay selector |
| `/replay/{game_id}` | GET | Full server-side replay from pbp.db |
| `/ws/{game_id}` | WebSocket | Real-time push updates on each poll cycle |
| `/figures/{filename}` | Static | Evaluation figures from model/eval_figures/ |

The WebSocket endpoint pushes the full game state payload to all connected clients whenever the polling loop processes new events. Poll interval is 15 seconds when WebSocket clients are connected (30 seconds otherwise).

**Note:** The stats API `PlayByPlayV3` returns 0 rows for in-progress games. Live polling uses the NBA live endpoint instead. `PlayByPlayV3` is only used for historical replay.

---

### Phase 5 — Dashboard

```bash
# Start the API server first
uvicorn live.api:app --reload

# Start the Next.js dashboard
cd web && npm run dev
# Open http://localhost:3000
```

**Pages:**

- **Game Selector** (`/`) — Browse historical games by season or enter a live game ID. Click any game to view its replay, or enter a game ID for live tracking.
- **Game Dashboard** (`/game/[gameId]`) — Full game view with:
  - **Scoreboard** — Team logos, names in team colors, large score, clock, animated win probability bar
  - **Win Probability Chart** — ESPN-style Recharts area chart with inverted y-axis, team-colored filled regions, quarter boundaries, toss-up band, and tooltips
  - **Play-by-Play** (tab) — Scrollable table with Qtr, Clock, Team, Play, Score, Win%
  - **Box Score** (tab) — Two-column shooting splits (FG/2PT/3PT/FT) with leading-team highlighting
  - **Key Moments** (tab) — Top 5 momentum swings, lead changes count, largest scoring run
  - **Pre-Game** (tab) — Diverging bar chart of the 9 pre-game feature contributions
- **Backtest Report** (`/backtest`) — Metrics table (Brier, ECE, AUC-ROC), pass/fail badges, and evaluation figure grid

**Real-time architecture:** The dashboard connects via WebSocket (`/ws/{game_id}`) for live games. The `useGameWebSocket` hook handles auto-reconnect with exponential backoff and falls back to REST polling (`GET /live/{game_id}` every 5 seconds) if WebSocket fails. For replays, it fetches `GET /replay/{game_id}` once.

**Tech stack:** Next.js 16, TypeScript, Tailwind CSS v4, Recharts. Config: `web/.env.local` sets `NEXT_PUBLIC_API_URL`.

**Legacy:** The Streamlit dashboard (`dashboard/app.py`) still exists but is deprecated. It has a rendering bug where the scoreboard HTML renders as raw text.

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
│   ├── eval_figures/           # Generated evaluation plots
│   ├── pregame.pkl             # Saved pre-game model
│   └── ingame.pkl              # Saved in-game model
├── live/
│   ├── game_state.py           # GameState class (incremental in-memory state)
│   ├── poller.py               # PlayByPlay polling loop + probability logging
│   └── api.py                  # FastAPI server (REST + WebSocket)
├── web/                        # Next.js dashboard
│   ├── src/
│   │   ├── app/                # Pages: /, /game/[gameId], /backtest
│   │   ├── components/         # Scoreboard, WinProbChart, PlayByPlayFeed, BoxScore, etc.
│   │   ├── hooks/              # useGameWebSocket (WebSocket + REST fallback)
│   │   ├── lib/                # teamMeta, api client, formatters, momentum detection
│   │   └── types/              # TypeScript interfaces for API responses
│   ├── .env.local              # NEXT_PUBLIC_API_URL=http://localhost:8000
│   └── package.json
├── dashboard/
│   └── app.py                  # Legacy Streamlit dashboard (deprecated)
├── notebooks/
│   ├── eda.ipynb
│   ├── model_analysis.ipynb
│   └── replay.ipynb
├── tests/
│   ├── test_game_state.py
│   └── test_no_leakage.py
├── requirements.txt
└── README.md
```

---

## Data Storage

| Database | Tables | Contents |
|---|---|---|
| `data/raw/games.db` | `game_logs`, `team_efficiency` | One row per team per game; season-level advanced efficiency stats |
| `data/raw/pbp.db` | `play_by_play` | One row per play event; clock parsed to seconds |
| `data/raw/players.db` | `player_box_scores` | One row per player per game |

All three databases have indexes on `game_id`, `season`, and `team_id`. SQLite WAL mode is enabled for safe concurrent reads during development.

---

## Key Implementation Notes

**Two-stage model:** The pre-game LR model outputs `pre_game_prob`, which is passed as an explicit input feature to the in-game XGBoost model. This anchors predictions in Q1 when the score is still near 0-0. Train `train_pregame.py` before `train_ingame.py`.

**Season range:** 11 seasons (2015-16 through 2025-26). Train on 2015–2022, calibrate on a dedicated split, validate on 2022–2023, test on 2023–2024. 2024–2025 and 2025–2026 are held out.

**API version:** Use `BoxScoreTraditionalV3` for box scores. For live games use `nba_api.live.nba.endpoints.playbyplay.PlayByPlay` — the stats API `PlayByPlayV3` returns 0 rows for in-progress games. V3 clock fields use ISO 8601 format (`PT11M42.00S`); column names are camelCase.

**Rate limiting:** All API calls during data collection use `time.sleep(0.6)` between requests. Live polling uses 30-second intervals (15 seconds when WebSocket clients are connected).

**Clock encoding:** Time remaining is expressed as total seconds in the game, not just the current quarter. For regulation: `time_remaining_game = (4 - period) * 720 + clock_seconds`. OT periods are 300 seconds each and must be handled separately — the formula above is wrong for OT.

**Feature leakage:** Every training row must only contain features derivable from plays at or before that event's timestamp. Enforced with a per-row assertion during dataset construction in Phase 2.

**Calibration:** Three-way split — train → calibrate → validate. `StratifiedCalibrator` uses phase-specific isotonic calibrators. OT calibrator uses val OT rows only (2022-23) — cal OT rows excluded due to era shift in OT home-win rates (~44% in 2015-2022 vs ~64% in 2023-24). Q1-Q4 calibrators never touch val data.

**In-game features:** 18 features including live overall FG%, 2PT%, 3PT%, and FT% for each team (computed from the PBP stream, zero-filled until first attempt). Raw foul counts (`home_fouls`, `away_fouls`), not FT rate. No H2H feature.

**Event deduplication:** During live polling, deduplicate on `actionNumber` (mapped to `actionId` in `GameState`). The live API has no `actionId` field. The stats API can reorder events in the buffer.

**Model persistence:** `joblib.dump()` / `joblib.load()` — not `pickle`. Artifacts: `model/pregame.pkl` and `model/ingame.pkl`.

**StratifiedCalibrator import:** Must be imported before `joblib.load("ingame.pkl")` in any module that loads the in-game model. Both `api.py` and `poller.py` handle this. `api.py` also registers the class on `__main__` for deserialization safety.
