# NBA Win Probability Model

Real-time win probability for NBA games, updated after every play. Built with a two-stage ML model (calibrated Logistic Regression pre-game + XGBoost in-game) and the public NBA Stats API.

## What it does

Predicts the home team's win probability (0.0-1.0) live throughout a game, using play-by-play events as they happen. The model combines pre-game team strength signals (ELO, efficiency ratings, rest days) with in-game context (score, shooting percentages, momentum, fouls, clutch situations).

Includes a broadcast-style React dashboard with an ESPN-like win probability chart, live scoreboard, play-by-play feed, and box score.

## Demo

Start the API server and dashboard, then enter any live or historical game ID:

```
http://localhost:3000/game/0022301234
```

The dashboard connects via WebSocket for real-time updates during live games, or replays historical games from the local database.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- ~3GB disk space for SQLite databases (full data collection)

No API keys needed — the `nba_api` package uses the public NBA Stats endpoint.

## Getting started

```bash
# Clone
git clone <repo-url>
cd lebron-is-the-goat

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd web && npm install && cd ..
```

### Quick start (replay a historical game)

If you already have the trained models and data (or someone shared the `.pkl` and `.db` files with you):

```bash
# Terminal 1: API server
uvicorn live.api:app --reload

# Terminal 2: Frontend
cd web && npm run dev

# Open http://localhost:3000
```

### Full pipeline from scratch

```bash
# 1. Fetch data (takes hours cold — caches aggressively, safe to interrupt and resume)
python data/fetch_games.py
python data/fetch_pbp.py        # can run in parallel with fetch_players
python data/fetch_players.py

# 2. Build features
python features/pipeline.py

# 3. Train models (pregame MUST run first)
python model/train_pregame.py
python model/train_ingame.py

# 4. Evaluate
python model/evaluate.py

# 5. Run the stack
uvicorn live.api:app --reload   # Terminal 1
cd web && npm run dev           # Terminal 2
```

### Track a live game

Find today's game IDs:

```bash
python3 -c "
from nba_api.live.nba.endpoints import scoreboard
sb = scoreboard.ScoreBoard()
for g in sb.get_dict()['scoreboard']['games']:
    print(g['gameId'], g['awayTeam']['teamTricode'], '@', g['homeTeam']['teamTricode'], g['gameStatusText'])
"
```

Then open `http://localhost:3000` and enter the game ID.

## Environment setup

The only environment variable is in `web/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

This is created automatically when you clone the repo. No `.env` files are needed for the Python backend.

## Running tests

```bash
# Fast unit tests (no database needed)
pytest tests/test_game_state.py -v

# Feature integrity tests (requires populated databases + parquet files)
pytest tests/test_no_leakage.py -v

# Live pregame computation tests (requires games.db)
pytest tests/test_live_pregame.py -v

# All tests
pytest -v
```

## Project structure

```
data/           Fetch scripts + SQLite raw storage + Parquet processed features
features/       ELO ratings, pre-game features, in-game features, pipeline
model/          Training scripts, evaluation, saved model artifacts
live/           GameState, polling loop, FastAPI server
web/            Next.js dashboard (React + TypeScript + Tailwind + Recharts)
dashboard/      Legacy Streamlit dashboard (deprecated)
tests/          Pytest test suite
notebooks/      EDA notebook
```

## Architecture

```
Data (NBA API → SQLite) → Features (Parquet) → Model (pregame LR → ingame XGBoost)
                                                         ↓
                                          Live (FastAPI + WebSocket polling)
                                                         ↓
                                          Dashboard (Next.js + Recharts)
```

- **Pre-game model**: Calibrated Logistic Regression on 9 team-strength features
- **In-game model**: XGBoost on 18 features (including pre-game probability as anchor) with phase-specific isotonic calibration
- **Test set performance** (2023-24 season): Brier 0.151, ECE 1.4%, AUC-ROC 0.862

## How to contribute

1. Fork the repo
2. Create a feature branch
3. Run `pytest -v` to ensure tests pass
4. Submit a PR with a clear description

Key things to know before contributing:
- Training order matters: pregame model before ingame model
- Feature vector order in `GameState.to_feature_vector()` must exactly match `INGAME_FEATURES` in `train_ingame.py`
- `StratifiedCalibrator` must be imported before loading `ingame.pkl` with joblib
- See `CLAUDE.md` for detailed design decisions and constraints

## License

MIT
