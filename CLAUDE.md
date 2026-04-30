# CLAUDE.md

Instructions for AI coding assistants working in this repository.

## Project

NBA real-time win probability model. Outputs a single float (home team win probability, 0.0-1.0) updated after every play-by-play event. Two-stage architecture: calibrated Logistic Regression pre-game model feeds into an XGBoost in-game model. Data sourced from the NBA Stats API via `nba_api`, stored in SQLite, trained on 11 seasons (2015-2026). Live inference via FastAPI + WebSocket, displayed in a Next.js dashboard.

## Tech stack

| Layer | Technology | Version |
|---|---|---|
| ML / data | Python 3.11+, pandas, numpy, XGBoost, scikit-learn, SHAP | see `requirements.txt` |
| Data storage | SQLite (WAL mode), Parquet | |
| API server | FastAPI, uvicorn | |
| Frontend | Next.js 16, React 19, TypeScript 5, Tailwind CSS v4, Recharts 3 | see `web/package.json` |
| Icons | lucide-react | |
| Serialization | joblib (not pickle) | |
| NBA data | nba_api (public, no API key) | |

## Repository structure

```
data/
  fetch_games.py          # LeagueGameLog + LeagueDashTeamStats -> games.db
  fetch_pbp.py            # PlayByPlayV3 -> pbp.db (includes cross-table validation)
  fetch_players.py        # BoxScoreTraditionalV3 -> players.db
  raw/                    # SQLite DBs (games.db, pbp.db, players.db) — gitignored
  processed/              # Parquet feature matrices — gitignored
features/
  elo.py                  # Walk-forward ELO ratings (K=100, season regression 25%)
  pregame.py              # 9 pre-game features as home-minus-away differentials
  ingame.py               # 18 in-game features per PBP event + possession state machine
  pipeline.py             # Orchestrates elo -> pregame -> ingame, writes parquet
model/
  train_pregame.py        # Calibrated LR (Platt scaling) -> pregame.pkl + pregame_probs.parquet
  train_ingame.py         # XGBoost + StratifiedCalibrator -> ingame.pkl
  evaluate.py             # Reliability diagrams, SHAP, per-quarter calibration
  train.sh                # Shell script to run both training stages in sequence
  eval_figures/           # Generated PNG plots (gitignored pkl files)
live/
  game_state.py           # GameState class — incremental in-game state, produces feature vector
  poller.py               # Live polling (NBA live endpoint) + replay (SQLite) + pregame lookup
  api.py                  # FastAPI server: REST + WebSocket + background poll tasks
web/                      # Next.js 16 dashboard (app router, TypeScript, Tailwind v4)
  src/app/                # Pages: / (game selector), /game/[gameId], /backtest
  src/components/         # Scoreboard, WinProbChart, PlayByPlayFeed, BoxScore, etc.
  src/hooks/              # useGameWebSocket (WebSocket + REST fallback)
  src/lib/                # teamMeta, api client, formatters, momentum detection
  src/types/              # TypeScript interfaces for API responses
  .env.local              # NEXT_PUBLIC_API_URL=http://localhost:8000
dashboard/
  app.py                  # Legacy Streamlit dashboard (deprecated, has rendering bugs)
tests/
  test_game_state.py      # Unit tests for GameState (score, clock, shooting, possessions, dedup)
  test_no_leakage.py      # Feature leakage + data integrity assertions (needs populated DBs)
  test_live_pregame.py    # Live pregame fallback computation tests
notebooks/
  eda.ipynb               # Exploratory data analysis
```

## Commands

```bash
# All commands run from project root unless noted otherwise.

# --- Data collection (run once, takes hours cold — caches aggressively) ---
python data/fetch_games.py                    # ~20 API calls, seconds
python data/fetch_pbp.py                      # ~13K API calls, hours cold
python data/fetch_players.py                  # ~13K API calls, hours cold
python data/fetch_pbp.py --resume             # retry only failed games
python data/fetch_games.py --season 2023-24   # single season

# --- Feature engineering ---
python features/pipeline.py                   # outputs pregame_features.parquet + ingame_snapshots.parquet

# --- Training (order matters: pregame first) ---
python model/train_pregame.py                 # -> pregame.pkl + pregame_probs.parquet
python model/train_ingame.py                  # -> ingame.pkl (requires pregame_probs.parquet)
python model/train_ingame.py --sweep          # hyperparameter search (~2 hrs)
bash model/train.sh                           # runs both stages, logs to model/logs/
bash model/train.sh --sweep                   # with hyperparam sweep

# --- Evaluation ---
python model/evaluate.py                      # saves figures to model/eval_figures/

# --- Tests ---
pytest tests/test_game_state.py -v            # fast, no DB needed
pytest tests/test_no_leakage.py -v            # needs populated DBs + parquet
pytest tests/test_live_pregame.py -v          # needs games.db

# --- Live inference ---
uvicorn live.api:app --reload                 # FastAPI server on :8000
python live/poller.py --game_id <ID>          # CLI live polling
python live/poller.py --game_id <ID> --replay # replay from pbp.db

# --- Frontend ---
cd web && npm install && npm run dev          # Next.js on :3000

# --- Find today's live game IDs ---
python3 -c "
from nba_api.live.nba.endpoints import scoreboard
sb = scoreboard.ScoreBoard()
for g in sb.get_dict()['scoreboard']['games']:
    print(g['gameId'], g['awayTeam']['teamTricode'], '@', g['homeTeam']['teamTricode'], g['gameStatusText'])
"
```

## Architecture

Five layers: **data -> features -> model -> live inference -> dashboard**.

Training flows through all five in sequence. Live inference loops through layers 3-5.

### Data layer (`data/`)
Three fetch scripts hit the NBA Stats API, cache to SQLite (`data/raw/`). Always check cache before API. Rate-limit: `time.sleep(0.6)` between calls. Three databases: `games.db` (game_logs + team_efficiency), `pbp.db` (play_by_play), `players.db` (player_box_scores). All use WAL mode with indexes on game_id, season, team_id.

### Feature layer (`features/`)
- `pregame.py`: 9 features as home-minus-away differentials. Frozen at tip-off. Uses `cumsum().shift(1)` for season-to-date stats (excludes current game). ORtg/DRtg from previous season's `team_efficiency`.
- `ingame.py`: 18 features per PBP event. Possession state machine for `last_5_poss_swing`. Timeout team parsed from description (team_id is 0). Score forward-filled with `ffill()`. OT clock goes negative.
- `pipeline.py`: Orchestrates both, writes parquet.

### Model layer (`model/`)
Two-stage:
1. **Pre-game**: `LogisticRegression` + `CalibratedClassifierCV` (Platt scaling). Outputs `pre_game_prob`.
2. **In-game**: XGBoost (depth 4, lr 0.05, 1000 trees, early stopping) + `StratifiedCalibrator` with 3 phase-specific isotonic calibrators (Q1-Q2, Q3-Q4, OT).

Split by season: Train 2015-2022, Val 2022-23, Test 2023-24. Holdout 2024-25 and 2025-26. Calibration carved from training data (15% by game_id). OT calibrator exception: uses val OT rows only (era shift in OT home-win rates).

Intermediate artifact: `pregame_probs.parquet` — training games use out-of-fold predictions (`cross_val_predict`) to prevent leakage.

### Live inference layer (`live/`)
- `GameState`: incremental in-memory state, deduplicates by `actionId`/`actionNumber`, produces 18-element feature vector.
- `poller.py`: live mode uses `nba_api.live.nba.endpoints.playbyplay.PlayByPlay` (stats API `PlayByPlayV3` returns 0 rows for live games). Replay mode reads from `pbp.db`. Falls back to on-the-fly pregame computation for games not in parquet (e.g., playoffs).
- `api.py`: FastAPI with REST + WebSocket. Background `_poll_loop` per game. Poll interval 15s with WS clients, 30s without.

### Dashboard (`web/`)
Next.js 16 app router. WebSocket connection via `useGameWebSocket` hook with auto-reconnect (exponential backoff) and REST polling fallback. `/live/` and `/replay/` return identical response shapes.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/pregame/{game_id}` | GET | Pre-game probability |
| `/pregame/{game_id}/breakdown` | GET | Pre-game features + labels |
| `/live/{game_id}` | GET | Current state + prob_history + play_log + box_score |
| `/games?season=2024-25` | GET | Game list for replay selector |
| `/replay/{game_id}` | GET | Full server-side replay from pbp.db |
| `/ws/{game_id}` | WebSocket | Real-time push on each poll cycle |
| `/figures/{filename}` | Static | Evaluation figures |

## Feature lists (exact order matters)

**PREGAME_FEATURES** (9): `elo_diff`, `efg_pct_diff`, `ortg_diff`, `drtg_diff`, `prev_season_win_pct_diff`, `rest_days_diff`, `home_flag`, `ast_pct_diff`, `tov_pct_diff`

**INGAME_FEATURES** (18): `score_diff`, `seconds_remaining`, `pre_game_prob`, `home_fg_pct_live`, `away_fg_pct_live`, `home_2pt_pct_live`, `away_2pt_pct_live`, `home_3pt_pct_live`, `away_3pt_pct_live`, `home_ft_pct_live`, `away_ft_pct_live`, `home_fouls`, `away_fouls`, `turnover_diff_live`, `timeout_remaining_diff`, `last_5_poss_swing`, `quarter` (index 16), `clutch_flag`

## Critical constraints / gotchas

1. **`StratifiedCalibrator` import order**: Must be imported before `joblib.load("ingame.pkl")`. Both `api.py` and `poller.py` handle this. `api.py` also registers it on `__main__` for deserialization safety. If you add a new module that loads ingame.pkl, you must import `StratifiedCalibrator` first.

2. **Training order**: `train_pregame.py` MUST run before `train_ingame.py` — the in-game model requires `pregame_probs.parquet`.

3. **Clock encoding**: Regulation: `(4 - period) * 720 + clock_seconds`. OT: `-((period - 5) * 300 + (300 - clock_seconds))`. OT produces negative values. Using the regulation formula for OT produces garbage. The `quarter` feature at index 16 disambiguates.

4. **Feature leakage**: Every training row must only use data available at that event's timestamp. Rolling stats use `cumsum().shift(1)` to exclude the current game. ORtg/DRtg come from the *previous* season. `pre_game_prob` uses out-of-fold predictions for training games.

5. **Score forward-fill**: PBP `score_home`/`score_away` are NULL on ~60% of events. Must `ffill()` after sorting by `action_number`, with initial value (0, 0).

6. **Live vs stats API**: `nba_api.live.nba.endpoints.playbyplay.PlayByPlay` for live games. `PlayByPlayV3` for historical replay only. Live API has no `actionId` field — deduplicate on `actionNumber` mapped to `actionId`.

7. **OT calibrator exception**: Uses val OT rows only (2022-23), NOT cal OT rows. This is deliberate — OT home-win rates shifted from ~44% (2015-2022) to ~64% (2023-24). Q1-Q4 calibrators never touch val data.

8. **Timeout team_id is 0**: Parse the team from the `description` field regex, not `team_id`.

9. **`SEASONS` list**: Must cover 2015-16 through 2025-26 in all three fetch scripts. If you add a season, update all three.

10. **Model artifacts are gitignored**: `model/*.pkl` and `data/processed/*.parquet` are not committed. Must regenerate locally.

11. **`GameState.to_feature_vector()` must match `INGAME_FEATURES` order exactly**: If you add/remove/reorder features, update both `train_ingame.py` and `game_state.py` together, and retrain.

12. **Next.js 16 in `web/`**: This is a newer version than typical. Read `web/AGENTS.md` before modifying — APIs may differ from training data.

## Code conventions

- Python: no type stubs, uses `from __future__ import annotations`. `Path` for file paths. `snake_case` everywhere.
- All Python scripts use `sys.path.insert(0, str(ROOT))` for imports from project root.
- Constants defined at module top (UPPER_CASE). No config files — constants live in the modules that use them.
- SQLite connections: `PRAGMA journal_mode=WAL`, explicit `CREATE INDEX IF NOT EXISTS`.
- Frontend: `@/*` path alias for `./src/*`. Components are single-file. No CSS modules — Tailwind only.
- No `.env.example` — the only env var is `NEXT_PUBLIC_API_URL=http://localhost:8000` in `web/.env.local`.

## Environment variables

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | `web/.env.local` | `http://localhost:8000` | FastAPI server URL for frontend |

No other env vars. No API keys needed — `nba_api` uses public NBA endpoints.
