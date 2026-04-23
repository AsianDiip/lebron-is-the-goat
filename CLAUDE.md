# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NBA real-time win probability model. Full architecture and implementation roadmap are in `nba_win_prob_architecture_v2.md` — read it before making any structural decisions.

## Resolved design decisions

These supersede any conflicts between `nba_win_prob_architecture_v2.md` and older notes:

- **Season range:** 2015–2025 (~13K games). `SEASONS` in all fetch scripts must cover 2015-16 through 2024-25.
- **Train/val/test split:** Train 2015–2022, Val 2022–2023 (full season), Test 2023–2024 (full season), Live 2024–2025 held out.
- **Model architecture:** Two-stage. Separate calibrated LR pre-game model outputs `pre_game_prob`, which is passed as a feature into the XGBoost in-game model. Two training scripts: `model/train_pregame.py` and `model/train_ingame.py`.
- **Calibration split:** Three-way: train → separate calibration split → validate. Never fit the calibrator on the validation split. Use `cv='prefit'` with isotonic regression.
- **Pre-game features:** `elo_diff`, `efg_pct_diff`, `ortg_diff`, `drtg_diff`, `prev_season_win_pct_diff`, `rest_days_diff`, `home_flag`, `ast_pct_diff`, `tov_pct_diff`. No H2H feature.
- **In-game features:** `score_diff`, `seconds_remaining`, `pre_game_prob`, `home_fg_pct_live`, `away_fg_pct_live`, `home_fouls`, `away_fouls`, `turnover_diff_live`, `timeout_remaining_diff`, `last_5_poss_swing`, `quarter`, `clutch_flag` (Q4 and `abs(score_diff) <= 5`). Use raw foul counts, not FT rate.
- **Live polling:** `PlayByPlayV3` (not V2) every 30 seconds. Deduplicate by `event_id`, not positional index.
- **Model artifacts:** `model/pregame.pkl` and `model/ingame.pkl`, serialized with `joblib`.

## Planned commands (not yet implemented)

Once the codebase is built out per the spec, the expected commands are:

```bash
# Data collection (run once, takes hours — batches and caches aggressively)
python data/fetch_games.py
python data/fetch_pbp.py
python data/fetch_players.py

# Feature engineering
python features/pipeline.py

# Training (two-stage: pre-game first, then in-game)
python model/train_pregame.py
python model/train_ingame.py

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
- `pregame.py` computes static features once before tip-off (ELO, rolling team stats, rest days, `ast_pct_diff`, `tov_pct_diff`, `prev_season_win_pct_diff`). Frozen at tip-off — no lineup-lock re-fetch.
- `ingame.py` updates after every play event (score diff, time remaining, `last_5_poss_swing`, foul counts, timeout diff, live FG%, `clutch_flag`).
- `pipeline.py` concatenates both into a single vector for the in-game model.

**Model layer** (`model/`) — Two-stage:
1. `train_pregame.py`: scikit-learn `LogisticRegression` + `CalibratedClassifierCV` (Platt scaling). Outputs `pre_game_prob`.
2. `train_ingame.py`: XGBoost (`binary:logistic`, 500 trees, depth 6) takes `pre_game_prob` as an input feature. Post-hoc isotonic calibration on a separate calibration split (never on val). Both saved with `joblib`.

**Live inference layer** (`live/`) — `GameState` holds all mutable in-game state and is updated incrementally (never recomputed from scratch). `Poller` hits `PlayByPlayV3` every 30 seconds and deduplicates by `event_id` (not positional index — the API can reorder the buffer).

**Dashboard** (`dashboard/app.py`) — Streamlit. Uses `st.empty()` + rerun for live updates. Not a true push model; document this limitation if latency matters.

## Key constraints

- **Clock encoding:** `time_remaining_game = (4 - period) * 720 + clock_seconds` for regulation. OT periods are 300 seconds (not 720) and must be handled separately or the feature is garbage in OT games.
- **Calibration split:** train → calibrate → validate are three distinct data splits. Never fit the calibrator on validation data.
- **Leakage:** every training row must only contain features derivable from plays at or before that event's clock timestamp. Enforce with a per-row assertion during dataset construction.
- **Logging:** write to the probability CSV on new events only, not on every poll cycle.
- **Model persistence:** use `joblib`, not `pickle`.
