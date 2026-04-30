# NBA Win Probability Model — Technical Specification

## 1. Project overview

An end-to-end ML system that ingests historical NBA data (11 seasons, ~13K games, ~5.5M play-by-play events), trains a two-stage win probability model, and serves real-time predictions during live games via a FastAPI backend and Next.js dashboard.

**Output**: A single float — home team win probability (0.0-1.0) — updated after every play-by-play event.

**Two operating modes**:
- **Pre-game**: Prior probability based on team strength, efficiency, rest days, and home-court advantage. Output by a calibrated Logistic Regression model.
- **In-game**: Live probability updated after every scored basket, foul, turnover, or timeout. Output by an XGBoost model that takes the pre-game probability as an input feature.

**What this project is NOT**:
- Not a betting tool — no point spread, moneyline, or over/under predictions
- Not a player-level model — no individual player impact features (lineup data is not used)
- Not a real-time data provider — depends on the NBA Stats API, which has 15-30s latency
- Does not cover playoffs (only regular season data in the training pipeline)

---

## 2. Full directory map

```
lebron-is-the-goat/
├── data/
│   ├── fetch_games.py            # LeagueGameLog + LeagueDashTeamStats → games.db
│   │                              # Two tables: game_logs (one row per team per game),
│   │                              # team_efficiency (season-level advanced stats per team)
│   ├── fetch_pbp.py              # PlayByPlayV3 → pbp.db / play_by_play table
│   │                              # Includes cross-table validation at end of run
│   │                              # Parses ISO 8601 clock strings to integer seconds
│   ├── fetch_players.py          # BoxScoreTraditionalV3 → players.db / player_box_scores
│   │                              # camelCase V3 column names mapped to snake_case
│   ├── raw/                       # [gitignored] SQLite databases
│   │   ├── games.db              # game_logs + team_efficiency tables
│   │   ├── pbp.db                # play_by_play table (~5.5M rows)
│   │   ├── players.db            # player_box_scores table
│   │   ├── failed_games.txt      # Game IDs that failed during PBP fetch (for --resume)
│   │   ├── failed_players.txt    # Game IDs that failed during player fetch
│   │   ├── pbp_fetch.log         # Background fetch log
│   │   ├── players_fetch.log     # Background fetch log
│   │   └── pipeline.log          # Feature pipeline log
│   └── processed/                 # [gitignored] Parquet feature matrices
│       ├── pregame_features.parquet    # One row per game, 9 features + metadata + label
│       ├── ingame_snapshots.parquet    # One row per PBP event, 18 features + metadata + label
│       └── pregame_probs.parquet       # [game_id, pre_game_prob] for all games (leakage-safe)
│
├── features/
│   ├── elo.py                    # Walk-forward ELO ratings
│   │                              # K=100, season regression 25% toward 1500
│   │                              # Returns dict[(team_id, game_id), float] of pre-game ELO
│   ├── pregame.py                # 9 pre-game features as home-minus-away differentials
│   │                              # Rolling stats from player_box_scores via cumsum().shift(1)
│   │                              # ORtg/DRtg from previous season's team_efficiency
│   │                              # League-average fallbacks for first season / missing data
│   ├── ingame.py                 # 18 in-game features per PBP event
│   │                              # Possession state machine for last_5_poss_swing
│   │                              # Score forward-fill, OT clock encoding (negative values)
│   │                              # Timeout team parsed from description regex
│   ├── pipeline.py               # Orchestration: elo → pregame → ingame → parquet output
│   └── PHASE2_PLAN.md            # Historical planning doc (not maintained)
│
├── model/
│   ├── train_pregame.py          # Stage 1: LogisticRegression + CalibratedClassifierCV
│   │                              # Platt scaling, StandardScaler pipeline
│   │                              # Outputs pregame.pkl + pregame_probs.parquet
│   │                              # Uses cross_val_predict for training game probs (no leakage)
│   ├── train_ingame.py           # Stage 2: XGBoost + StratifiedCalibrator
│   │                              # StratifiedCalibrator class defined here (3 isotonic calibrators)
│   │                              # Reads pregame_probs.parquet to replace placeholder pre_game_prob
│   │                              # --sweep flag for 2-stage hyperparameter search
│   ├── evaluate.py               # Test set evaluation: reliability diagrams, SHAP, curves
│   │                              # Imports StratifiedCalibrator for joblib deserialization
│   ├── train.sh                  # Shell script: runs train_pregame then train_ingame
│   │                              # Supports --sweep flag, logs to model/logs/
│   ├── pregame.pkl               # [gitignored] Saved calibrated LR model
│   ├── ingame.pkl                # [gitignored] Saved StratifiedCalibrator wrapping XGBoost
│   ├── eval_figures/             # Generated evaluation PNGs
│   │   ├── pregame_reliability.png
│   │   ├── ingame_reliability.png
│   │   ├── shap_beeswarm.png
│   │   ├── shap_bar.png
│   │   ├── win_prob_curves.png
│   │   └── per_quarter_calibration.png
│   ├── logs/                     # Training logs from train.sh
│   └── phase3_plan.md            # Historical planning doc
│
├── live/
│   ├── game_state.py             # GameState class: mutable in-game state
│   │                              # Updated incrementally, never recomputed from scratch
│   │                              # to_feature_vector() produces 18-element array matching
│   │                              # INGAME_FEATURES order exactly
│   │                              # Deduplicates events by actionId
│   │                              # Accepts both camelCase (live API) and snake_case (SQLite)
│   ├── poller.py                 # Two modes: live polling + historical replay
│   │                              # lookup_pregame() with fallback to on-the-fly computation
│   │                              # compute_live_pregame() for games not in parquet
│   │                              # CSV probability logger
│   ├── api.py                    # FastAPI server with REST + WebSocket endpoints
│   │                              # Background _poll_loop per game
│   │                              # Registers StratifiedCalibrator on __main__ for joblib
│   │                              # CORS configured for localhost:3000
│   │                              # Serves eval_figures/ as static files at /figures/
│   └── PLAN.md                   # Historical planning doc
│
├── web/                          # Next.js 16 dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # Landing page: live game ID input + historical game browser
│   │   │   ├── game/[gameId]/page.tsx  # Game dashboard: scoreboard, chart, tabbed panels
│   │   │   ├── backtest/page.tsx       # Model evaluation report: metrics + figure grid
│   │   │   ├── layout.tsx        # Root layout: dark theme, nav bar, Geist font
│   │   │   ├── globals.css       # Tailwind v4 imports
│   │   │   └── favicon.ico
│   │   ├── components/
│   │   │   ├── Scoreboard.tsx    # Team logos, names in team colors, score, clock, win prob bar
│   │   │   ├── WinProbChart.tsx  # Recharts area chart, ESPN-style inverted y-axis, team colors
│   │   │   ├── PlayByPlayFeed.tsx  # Scrollable table: Qtr, Clock, Team, Play, Score, Win%
│   │   │   ├── BoxScore.tsx      # Two-column shooting splits (FG/2PT/3PT/FT) with highlighting
│   │   │   ├── MomentumPanel.tsx # Top 5 swings, lead changes, largest scoring run
│   │   │   ├── PregameBreakdown.tsx  # Diverging bar chart of 9 pre-game feature contributions
│   │   │   └── GameSelector.tsx  # Season dropdown + game list for replay selector
│   │   ├── hooks/
│   │   │   └── useGameWebSocket.ts  # WebSocket with auto-reconnect (exp backoff) + REST fallback
│   │   ├── lib/
│   │   │   ├── api.ts            # REST client: fetchLiveGame, fetchReplayGame, fetchGameList, etc.
│   │   │   ├── teamMeta.ts       # 30-team colors, names, IDs, logo URL helper
│   │   │   ├── formatters.ts     # Clock, period, percentage formatting
│   │   │   └── momentum.ts       # topSwings, countLeadChanges, largestRun (client-side)
│   │   └── types/
│   │       └── game.ts           # TypeScript interfaces: GameState, ProbPoint, PlayEvent, etc.
│   ├── .env.local                # NEXT_PUBLIC_API_URL=http://localhost:8000
│   ├── package.json              # next 16.2.4, react 19.2.4, recharts 3.8.1, lucide-react
│   ├── tsconfig.json             # @/* path alias for ./src/*
│   ├── next.config.ts            # Empty config (no custom settings)
│   ├── postcss.config.mjs        # @tailwindcss/postcss plugin
│   ├── AGENTS.md                 # Next.js 16 warning: read docs before editing
│   └── CLAUDE.md                 # Points to AGENTS.md
│
├── dashboard/
│   └── app.py                    # [DEPRECATED] Streamlit dashboard
│                                  # Has rendering bug: HTML/CSS in scoreboard shows as raw text
│                                  # Still runnable but being replaced by web/
│
├── tests/
│   ├── __init__.py
│   ├── test_game_state.py        # 25 unit tests for GameState
│   │                              # Score tracking, clock encoding, shooting stats, fouls,
│   │                              # turnovers, timeouts, possession state machine, dedup,
│   │                              # feature vector shape/order, clutch flag, camelCase input
│   ├── test_no_leakage.py        # Feature leakage + data integrity assertions
│   │                              # ELO leakage, rolling stats leakage, in-game future data,
│   │                              # seconds_remaining ranges, clutch_flag definition,
│   │                              # NaN checks, binary label checks, season coverage
│   └── test_live_pregame.py      # Live pregame fallback tests
│                                  # Verifies playoff games absent from parquet,
│                                  # compute_live_pregame returns valid features,
│                                  # unknown team IDs use league-average fallbacks
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
│
├── requirements.txt              # Python dependencies (15 packages)
├── LICENSE                       # MIT (Copyright 2026 Chen)
├── .gitignore                    # Ignores .db, .pkl, .parquet, .env, CLAUDE.md, node_modules
├── README.md                     # Human-facing project documentation
├── CLAUDE.md                     # AI assistant instructions (gitignored)
└── project_spec.md               # This file
```

---

## 3. Architecture overview

### Data flow

```
NBA Stats API
    ↓ (nba_api, rate-limited 0.6s)
SQLite (data/raw/)
    ↓ (features/pipeline.py)
Parquet (data/processed/)
    ↓ (model/train_pregame.py, then train_ingame.py)
Model artifacts (model/*.pkl)
    ↓ (live/api.py loads at startup)
FastAPI server (:8000)
    ↓ (REST + WebSocket)
Next.js dashboard (:3000)
```

### How a live game request flows

1. User opens `/game/[gameId]` in the browser
2. `useGameWebSocket` hook opens a WebSocket to `/ws/{gameId}`
3. Server calls `_ensure_game_state(gameId)`:
   - Looks up pregame features (parquet first, then NBA live scoreboard fallback)
   - Computes `pre_game_prob` via the pregame model
   - Creates a `GameState` instance
4. Server starts `_poll_loop` as a background `asyncio.Task`:
   - Polls `nba_api.live.nba.endpoints.playbyplay.PlayByPlay` every 15s
   - For each new action: updates `GameState`, runs `ingame_model.predict_proba()`, appends to `prob_history` and `play_log`
   - Broadcasts full payload to all WebSocket clients
5. Frontend receives JSON, renders Scoreboard, WinProbChart, PlayByPlayFeed, BoxScore

### How a replay request flows

1. Frontend calls `GET /replay/{gameId}`
2. Server creates a fresh `GameState`, reads all PBP rows from `pbp.db`
3. Iterates every event, updating state and computing probability
4. Returns the complete `prob_history` and `play_log` in one response
5. Frontend renders identically to a live game (same components, same data shape)

### Where state lives

| State | Location | Lifetime |
|---|---|---|
| Historical data | SQLite (`data/raw/`) | Permanent, gitignored |
| Feature matrices | Parquet (`data/processed/`) | Regenerated by `pipeline.py` |
| Model weights | joblib `.pkl` files | Regenerated by training scripts |
| Active game state | In-memory `_active_games` dict in `api.py` | Per-process, lost on restart |
| WebSocket connections | In-memory `_ws_connections` dict | Per-process |
| Frontend state | React state via `useGameWebSocket` hook | Per-browser-tab |

### Notable design patterns

- **Incremental state machine**: `GameState` is never recomputed from scratch — it processes events one at a time, maintaining running counters for all 18 features.
- **Dual-format input**: `GameState.update()` accepts both camelCase (live API) and snake_case (SQLite replay) event dicts, normalizing internally.
- **Leakage-safe probability injection**: Training games get out-of-fold `cross_val_predict` probabilities for `pre_game_prob`, preventing the in-game model from training on information it wouldn't have at inference time.
- **Graceful degradation**: If a game isn't in the pregame parquet (e.g., playoffs, current season), `lookup_pregame()` falls back to computing features on-the-fly from the NBA live scoreboard API + local databases.

---

## 4. Key files

### `model/train_ingame.py`
**What it does**: Trains the in-game XGBoost model with stratified isotonic calibration. Defines the `StratifiedCalibrator` class.

**Why it exists**: The in-game model needs phase-specific calibration because prediction difficulty varies dramatically between early-game (high uncertainty) and late-game (score is nearly deterministic). OT has different base rates than regulation.

**What breaks if you change it carelessly**:
- Reordering `INGAME_FEATURES` breaks `GameState.to_feature_vector()` and `StratifiedCalibrator.QUARTER_IDX`
- Changing `StratifiedCalibrator` class definition breaks `joblib.load("ingame.pkl")` in all consumers
- Changing the cal/val split logic for OT rows can silently degrade OT calibration

**Depends on**: `pregame_probs.parquet` (from `train_pregame.py`)
**Depended on by**: `evaluate.py`, `poller.py`, `api.py`, `dashboard/app.py` (all import `StratifiedCalibrator`)

### `live/game_state.py`
**What it does**: Mutable in-game state tracker. Processes events one at a time and produces the 18-element feature vector.

**Why it exists**: Live inference needs an incremental state object rather than recomputing features from the full PBP history on each poll cycle.

**What breaks if you change it carelessly**:
- `to_feature_vector()` array order MUST match `INGAME_FEATURES` in `train_ingame.py` — index mismatch produces silently wrong predictions
- Changing possession logic changes `last_5_poss_swing` values, creating train/serve skew
- The FT accumulation logic counts on every attempt (not just last-of-sequence), matching `ingame.py`

**Depends on**: `features.ingame` (reuses `compute_seconds_remaining`, `_is_last_free_throw`, `_parse_timeout_team`)
**Depended on by**: `poller.py`, `api.py`, `dashboard/app.py`

### `live/api.py`
**What it does**: FastAPI server managing game state, background polling, REST endpoints, and WebSocket connections.

**Why it exists**: Central coordination point between the polling loop, model inference, and frontend clients.

**What breaks if you change it carelessly**:
- Must import `StratifiedCalibrator` AND register it on `__main__` before loading `ingame.pkl`
- `_build_game_payload()` response shape is consumed by the frontend — changing keys breaks the TypeScript types
- `_poll_loop` manages `_active_games` dict without locks — not safe for multi-worker uvicorn (single-worker only)

### `features/ingame.py`
**What it does**: Computes all 18 in-game features for every PBP event across all games, producing the training dataset.

**Why it exists**: Separates feature engineering from model training. The possession state machine, score forward-fill, and OT clock encoding are complex enough to warrant isolation.

**What breaks if you change it carelessly**:
- Any change to feature computation creates train/serve skew with `GameState` (which must match exactly)
- The column order in the output DataFrame defines `INGAME_FEATURES` order
- Score forward-fill logic must match `GameState.update()` score handling

### `live/poller.py`
**What it does**: CLI polling loop (live + replay modes), pregame feature lookup with on-the-fly fallback, CSV logging.

**Why it exists**: Provides both a standalone CLI tool and reusable functions (`lookup_pregame`, `compute_live_pregame`, `get_team_abbrev_map`) consumed by `api.py`.

**What breaks if you change it carelessly**:
- `compute_live_pregame()` hardcodes `"2025-26"` as the current season — must be updated each year
- `_PREV_SEASON` mapping must be kept in sync with the fetch scripts' `SEASONS` list
- `_fetch_live_actions()` maps `actionNumber` to `actionId` because the live API lacks `actionId`

---

## 5. Data models

### `game_logs` (games.db)
One row per team per game. A single game produces two rows (home + away). `is_home` flag distinguishes them. `wl` is "W" or "L". `matchup` contains "vs." for home and "@" for away. Indexed on `game_id`, `game_date`, `team_id`, `season`. Unique on `(game_id, team_id)`.

### `team_efficiency` (games.db)
One row per team per season. Advanced stats from `LeagueDashTeamStats`: ORtg, DRtg, net rating, pace, eFG%, TS%, TOV%, OREB%. Used for previous-season lookups in pregame features. Unique on `(season, team_id)`.

### `play_by_play` (pbp.db)
One row per PBP event. `clock_str` is the raw ISO 8601 string (e.g., "PT11M42.00S"), `clock_seconds` is the parsed integer. `score_home`/`score_away` are NULL on ~60% of rows (non-scoring events). `team_id` is 0 for timeout events. `action_type` values include "2pt", "3pt", "free throw", "foul", "turnover", "timeout", "rebound", "period". Unique on `(game_id, action_number)`.

### `player_box_scores` (players.db)
One row per player per game. Traditional box score stats (FGM/FGA/FG3M/AST/TOV/FTA/etc.). Used to compute season-to-date rolling eFG%, assist rate, and turnover rate for pregame features. Unique on `(game_id, player_id)`.

### Parquet schemas

**pregame_features.parquet**: `game_id`, `season`, `game_date`, `home_team_id`, `away_team_id`, 9 feature columns (all floats, home-minus-away differentials), `home_win` (binary label).

**ingame_snapshots.parquet**: `game_id`, `season`, `action_number`, 18 feature columns, `home_win` (binary label). `pre_game_prob` is 0.5 placeholder — replaced at training time.

**pregame_probs.parquet**: `game_id`, `pre_game_prob`. Training games have out-of-fold predictions; val/test/holdout have calibrated model predictions.

---

## 6. External dependencies and integrations

### NBA Stats API (via `nba_api` package)

**Used for**: All historical and live game data.

**Where it's called from**:
- `data/fetch_games.py`: `LeagueGameLog`, `LeagueDashTeamStats`
- `data/fetch_pbp.py`: `PlayByPlayV3`
- `data/fetch_players.py`: `BoxScoreTraditionalV3`
- `live/poller.py`: `nba_api.live.nba.endpoints.playbyplay.PlayByPlay` (live games), `nba_api.live.nba.endpoints.scoreboard.ScoreBoard` (team lookup)

**Env vars**: None. Uses public NBA endpoints, no API key required.

**What happens if unavailable**: Data fetch scripts fail with network errors (retry once, then log to `failed_games.txt`). Live polling retries every 30s. Historical replay is unaffected (reads from local SQLite). On-the-fly pregame computation falls back to league averages if databases are missing.

**Rate limiting**: `time.sleep(0.6)` between all calls during data collection. Live polling uses 15-30s intervals.

### NBA CDN (team logos)

**Used for**: Team logo images in the dashboard.

**Where it's called from**: `web/src/lib/teamMeta.ts` (`teamLogoUrl()` function), `dashboard/app.py` (`team_logo_url()`).

**What happens if unavailable**: Logos don't render; no functional impact.

---

## 7. Environment variables reference

| Variable | Required | Default | Description | Where used |
|---|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8000` | FastAPI server URL | `web/.env.local`, consumed by `web/src/lib/api.ts` |

No other environment variables exist. No API keys, no database connection strings, no secrets.

---

## 8. Known complexity / landmines

### `StratifiedCalibrator` deserialization
The `StratifiedCalibrator` class is defined in `model/train_ingame.py` and serialized inside `ingame.pkl`. Any module that calls `joblib.load("ingame.pkl")` must import the class first, or joblib will fail with an `AttributeError`. Additionally, `api.py` patches it onto `__main__` because uvicorn's import context differs from the training script. If you add a new consumer of `ingame.pkl`, you will hit this.

### Feature vector index coupling
`GameState.to_feature_vector()` returns a raw numpy array with no column names. The order is defined implicitly by the array construction in the method body. It must exactly match `INGAME_FEATURES` in `train_ingame.py`. There is no runtime validation of this — a mismatch produces silently wrong predictions. Index 16 (`quarter`) is especially critical because `StratifiedCalibrator` hardcodes `QUARTER_IDX = 16` for phase routing.

### OT calibrator era shift
OT home-win rates changed from ~44% (2015-2022 training era) to ~64% (2023-24 test era). The OT isotonic calibrator deliberately uses only val-set OT rows (2022-23) and excludes cal-split OT rows to avoid dragging calibration toward the wrong base rate. This is the one place where validation data touches calibration. If you retrain with a different val season, verify that OT calibration still works.

### Score forward-fill correctness
PBP `score_home`/`score_away` are NULL on most events. Both `ingame.py` (training) and `GameState` (inference) implement forward-fill, but via different mechanisms (pandas `ffill()` vs. imperative score tracking). If these ever diverge, the model sees different score_diff values at training vs. inference time.

### Timeout team_id is always 0
The NBA API sets `team_id = 0` for timeout events. The team must be parsed from the `description` string (e.g., "LAL Timeout: Regular"). Both `ingame.py` and `game_state.py` use `_parse_timeout_team()` for this. If the NBA changes the description format, timeout tracking silently breaks.

### Single-worker constraint
`api.py` stores active game state in module-level dicts (`_active_games`, `_ws_connections`, etc.). This is not safe for multi-worker uvicorn. Running with `--workers > 1` will cause each worker to maintain independent state, leading to duplicate polling and inconsistent WebSocket connections.

### Live API vs Stats API
`nba_api.live.nba.endpoints.playbyplay.PlayByPlay` is used for in-progress games. `PlayByPlayV3` (stats API) returns 0 rows for live games and is only valid for completed games. The live API returns different field names (camelCase, no `actionId`). `GameState.update()` handles both formats, but `_fetch_live_actions()` in `poller.py` maps `actionNumber` to `actionId` because the live endpoint lacks `actionId`.

### Hardcoded current season
`poller.py`'s `compute_live_pregame()` function hardcodes `"2025-26"` as the current season for box stat queries and `"2024-25"` as the previous season for efficiency lookups. This must be manually updated each NBA season.

### Playoffs not covered
All three fetch scripts use `season_type_all_star="Regular Season"` (or equivalent filtering). Playoff game IDs (prefix `004`) are never fetched. Live playoff games work via the on-the-fly pregame fallback, but with potentially less accurate features since there's no playoff training data.

---

## 9. Decisions log

### Two-stage model (LR pregame + XGBoost ingame)
**Rationale**: The pre-game probability anchors in-game predictions during Q1 when the score is near 0-0 and in-game features are noisy. Without it, the model over-reacts to early scoring runs. A single unified model would need to handle both the "no game data yet" and "game in progress" regimes.
**Trade-offs**: Adds pipeline complexity (pregame must train first), requires the leakage-safe `pregame_probs.parquet` intermediate artifact.

### Isotonic calibration stratified by game phase
**Rationale**: A single isotonic calibrator averages across Q1 (high uncertainty, many toss-up predictions) and Q4 (low uncertainty, many near-0/1 predictions), producing poor calibration in both regimes. Phase-specific calibrators (Q1-Q2, Q3-Q4, OT) match the model's behavior within each regime.
**Trade-offs**: Three calibrators means three times the calibration data requirement. OT calibrator has especially few samples, requiring the val-data exception.

### SQLite over PostgreSQL
**Rationale**: Zero setup, no daemon, file-based, portable. The data is write-once-read-many (fetched once, queried during feature engineering). WAL mode allows concurrent reads during development.
**Trade-offs**: No concurrent writes (not an issue for this workload). No built-in replication or remote access.

### Parquet for feature matrices
**Rationale**: Columnar storage is efficient for the 5.5M-row ingame dataset. Fast reads in pandas. Schema enforcement. Smaller than CSV.
**Trade-offs**: Not human-readable. Requires pyarrow or fastparquet.

### joblib over pickle
**Rationale**: Better handling of numpy arrays and sklearn estimators. The `StratifiedCalibrator` class contains numpy arrays and sklearn `IsotonicRegression` objects that joblib serializes more efficiently.
**Trade-offs**: Requires importing the custom class before deserialization (the `StratifiedCalibrator` issue).

### Next.js over Streamlit for dashboard
**Rationale**: Streamlit's rerun-based model adds up to 40s latency for live games (30s API poll + 10s dashboard poll). WebSocket support in Next.js enables true push updates with ~15s latency. The Streamlit dashboard also has a rendering bug with HTML in `st.markdown`.
**Trade-offs**: Much more code to build and maintain. Requires Node.js in addition to Python.

### No player-level features
**Rationale**: Lineup data is noisy (injuries, rest, in-game rotations) and requires a separate data pipeline. Team-level features capture most of the signal for win probability. Adding player features would increase model complexity without proportional accuracy gains for this use case.
**Trade-offs**: Cannot capture the impact of specific player matchups or in-game injuries.

### Regular season only (no playoffs)
**Rationale**: The fetch scripts use `LeagueGameLog` with `season_type_all_star="Regular Season"`. Playoff games have different dynamics (7-game series, higher intensity, rest patterns) that would need separate modeling. There are far fewer playoff games, making them hard to train on.
**Trade-offs**: Live playoff games rely on the on-the-fly pregame fallback with less accurate features.

---

## 10. Glossary

| Term | Definition |
|---|---|
| **ELO** | Rating system where teams gain/lose points based on game outcomes relative to expectations. K=100 means a maximum of 100 points transferred per game. |
| **eFG%** | Effective Field Goal Percentage: `(FGM + 0.5 * FG3M) / FGA`. Weights three-pointers at 1.5x to reflect their extra point value. |
| **ORtg / DRtg** | Offensive/Defensive Rating: points scored/allowed per 100 possessions. From `LeagueDashTeamStats` Advanced. |
| **Brier score** | Mean squared error of probability predictions: `mean((predicted - actual)^2)`. Lower is better. Perfect = 0, coin flip = 0.25. |
| **ECE** | Expected Calibration Error: weighted average of |predicted probability - actual frequency| across bins. Measures whether "70% predictions" actually happen 70% of the time. |
| **Platt scaling** | Post-hoc calibration using a sigmoid function fitted on held-out data. Used for the pre-game LR model. |
| **Isotonic regression** | Non-parametric monotonic calibration. Fits a step function mapping raw probabilities to calibrated ones. Used for the in-game XGBoost model. |
| **StratifiedCalibrator** | Custom class in `train_ingame.py` holding the base XGBoost model and three phase-specific isotonic calibrators (Q1-Q2, Q3-Q4, OT). |
| **`pre_game_prob`** | Output of the pregame LR model, passed as input feature to the in-game XGBoost model. Anchors predictions in early quarters. |
| **`last_5_poss_swing`** | Net points from the home team's perspective over the last 5 completed possessions. Momentum proxy. |
| **Clutch** | Q4 or OT with `abs(score_diff) <= 5`. Binary flag in the feature set. |
| **`seconds_remaining`** | Total seconds left in the game. Regulation: `(4 - period) * 720 + clock_seconds` (0-2880). OT: negative values (-300 per OT period). |
| **Walk-forward** | Computing features in chronological order, using only data available up to each point in time. Prevents future information from leaking into features. |
| **Out-of-fold** | Predictions made by a model that was trained on a subset that excludes the data being predicted. Used for `pre_game_prob` on training games via `cross_val_predict`. |
| **PBP** | Play-by-play. One event per action in a game (shots, fouls, turnovers, timeouts, etc.). |
| **Action number** | Sequential integer identifying each PBP event within a game. Used for ordering and deduplication. |
| **WAL mode** | SQLite Write-Ahead Logging. Allows concurrent readers with a single writer. Enabled via `PRAGMA journal_mode=WAL`. |
