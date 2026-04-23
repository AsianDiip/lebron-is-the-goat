# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NBA real-time win probability model. Full architecture and implementation roadmap are in `nba_win_probability_spec.md` — read it before making any structural decisions.

## Planned commands (not yet implemented)

Once the codebase is built out per the spec, the expected commands are:

```bash
# Data collection (run once, takes hours — batches and caches aggressively)
python data/fetch_games.py
python data/fetch_pbp.py
python data/fetch_players.py

# Feature engineering
python features/pipeline.py

# Training
python model/train.py

# Evaluation
python model/evaluate.py

# Live inference (provide a game_id)
python live/poller.py --game_id <GAME_ID>

# Dashboard
streamlit run dashboard/app.py
```

## Architecture

The system has five layers: data → features → model → live inference → dashboard. Training flows through them in sequence; live inference loops through layers 3–5.

**Data layer** (`data/`) — Three fetch modules hit the NBA Stats API and cache to SQLite (`data/raw/`). Always check local cache before hitting the API. Rate-limit all calls with `time.sleep(0.6)`.

**Feature layer** (`features/`) — Two feature classes concatenated at inference time:
- `pregame.py` computes static features once before tip-off (ELO, rolling team stats, rest, H2H). These are frozen after a lineup-lock re-fetch at T-10 minutes.
- `ingame.py` updates after every play event (score diff, time remaining, momentum windows, foul trouble, clutch flag).
- `pipeline.py` concatenates both into a single vector for the model.

**Model layer** (`model/`) — XGBoost (`binary:logistic`) trained on 2021-22 and 2022-23, calibrated (isotonic) on first half of 2023-24, validated on second half. Saved with `joblib`. The calibrator is fit with `cv='prefit'` on a separate calibration split — never on the validation split.

**Live inference layer** (`live/`) — `GameState` holds all mutable in-game state and is updated incrementally (never recomputed from scratch). `Poller` hits `PlayByPlayV2` every 15 seconds and deduplicates by `event_id` (not positional index — the API can reorder the buffer).

**Dashboard** (`dashboard/app.py`) — Streamlit. Uses `st.empty()` + rerun for live updates. Not a true push model; document this limitation if latency matters.

## Key constraints

- **Clock encoding:** `time_remaining_game = (4 - period) * 720 + clock_seconds` for regulation. OT periods are 300 seconds (not 720) and must be handled separately or the feature is garbage in OT games.
- **Calibration split:** train → calibrate → validate are three distinct data splits. Never fit the calibrator on validation data.
- **Leakage:** every training row must only contain features derivable from plays at or before that event's clock timestamp. Enforce with a per-row assertion during dataset construction.
- **Logging:** write to the probability CSV on new events only, not on every poll cycle.
- **Model persistence:** use `joblib`, not `pickle`.
