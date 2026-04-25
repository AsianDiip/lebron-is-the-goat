# Phase 3: Model Training & Evaluation — Implementation Plan

## Context

Phases 1–2 are complete. Two feature matrices are ready:
- `data/processed/pregame_features.parquet` — 11,974 rows (one per game), 15 columns
- `data/processed/ingame_snapshots.parquet` — 5,567,812 rows (one per PBP event), 16 columns

Phase 3 creates three new files and updates one existing file:
1. `model/train_pregame.py` — calibrated LR pre-game model
2. `model/train_ingame.py` — calibrated XGBoost in-game model
3. `model/evaluate.py` — metrics, plots, SHAP
4. `requirements.txt` — add `shap>=0.43.0`

---

## Data Flow

```
pregame_features.parquet
        │
        ▼
  train_pregame.py
        │
        ├──▶ model/pregame.pkl
        ├──▶ data/processed/pregame_probs.parquet  [game_id, pre_game_prob]
        │
ingame_snapshots.parquet ──merge──▶ train_ingame.py
                                         │
                                         ├──▶ model/ingame.pkl
                                         ▼
                                    evaluate.py
                                         │
                                         ├──▶ model/eval_figures/*.png
                                         └──▶ metrics printed to stdout
```

---

## Execution Order

```bash
python model/train_pregame.py           # Step 1 (must run first)
python model/train_ingame.py            # Step 2 (depends on pregame_probs.parquet)
python model/train_ingame.py --sweep    # Step 2 alt (with hyperparam search, ~2hrs)
python model/evaluate.py                # Step 3 (depends on both .pkl files)
```

---

## Season-Based Splits (shared across all files)

| Split | Seasons | Purpose |
|-------|---------|---------|
| Train | 2015-16 through 2021-22 | Base model fitting |
| Calibration | 15% of train game_ids | Post-hoc calibration (carved from train, NEVER val) |
| Val | 2022-23 | Early stopping + metric reporting |
| Test | 2023-24 | Final evaluation |
| Holdout | 2024-25 | Live inference only |

**Critical**: Calibration split is carved by **game_id**, not by row. All ~465 PBP rows from a given game go to the same split. This prevents leaking the game outcome label across splits.

---

## File 1: `model/train_pregame.py`

### Features & Target
- **Features** (9): `elo_diff, efg_pct_diff, ortg_diff, drtg_diff, prev_season_win_pct_diff, rest_days_diff, home_flag, ast_pct_diff, tov_pct_diff`
- **Target**: `home_win`

### Pipeline
1. Load `pregame_features.parquet`
2. Split by season → train / val / test / holdout
3. Carve 15% of train game_ids → calibration split (~1,200 games)
4. `Pipeline([StandardScaler(), LogisticRegression(max_iter=1000)])` on train_proper
5. `CalibratedClassifierCV(cv='prefit', method='sigmoid')` on calibration split (Platt scaling per spec)
6. Print val metrics: accuracy, Brier, ECE, AUC-ROC
7. Generate `pre_game_prob` for ALL games:
   - **Training games**: `cross_val_predict(cv=5, method='predict_proba')` on full train set (avoids leakage — model doesn't predict on data it saw)
   - **Val/test/holdout**: regular `predict_proba` from calibrated model
8. Save `model/pregame.pkl` via `joblib.dump()`
9. Save `data/processed/pregame_probs.parquet` with columns `[game_id, pre_game_prob]`

### Target Metrics
- Accuracy ~66%, ECE <4%, AUC-ROC >0.78

---

## File 2: `model/train_ingame.py`

### Features & Target
- **Features** (12): `score_diff, seconds_remaining, pre_game_prob, home_fg_pct_live, away_fg_pct_live, home_fouls, away_fouls, turnover_diff_live, timeout_remaining_diff, last_5_poss_swing, quarter, clutch_flag`
- **Target**: `home_win`

### Pipeline
1. Load `ingame_snapshots.parquet` (5.5M rows)
2. Load `pregame_probs.parquet` and merge on `game_id` — replaces placeholder 0.5 values
3. Split by season → train / val / test / holdout
4. Carve 15% of train game_ids → calibration split (by game_id, not row)
5. Train `XGBClassifier`:
   - Default params: `objective='binary:logistic', n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10`
   - `eval_set=[(X_val, y_val)]`, `early_stopping_rounds=50`
   - `tree_method='hist'` (default in xgboost>=2.0, handles 5.5M rows in minutes)
6. `CalibratedClassifierCV(cv='prefit', method='isotonic')` on calibration split
7. Print val metrics: Brier, ECE, AUC-ROC
8. Save `model/ingame.pkl` via `joblib.dump()`

### Hyperparameter Sweep (`--sweep` flag)
Two-stage search using val set (temporal split, not CV):
- **Stage 1** (27 combos): `max_depth` × `learning_rate` × `min_child_weight`
- **Stage 2** (27 combos): `subsample` × `colsample_bytree` × `reg_alpha` × `reg_lambda` (or RandomizedSearch n_iter=30 over these 4 params)

Gated behind `argparse --sweep` so default run is fast.

### Target Metrics
- Brier <0.18, ECE <5%, AUC-ROC >0.80

### Memory Notes
- Peak ~3-4 GB (5.5M × 12 features × 8 bytes + XGBoost hist buffers). If tight, cast to float32.

---

## File 3: `model/evaluate.py`

### Metrics Computed
| Metric | Pre-game | In-game |
|--------|----------|---------|
| Accuracy | ✓ | — |
| Brier score | ✓ | ✓ (target <0.18) |
| ECE (10-bin) | ✓ (target <4%) | ✓ (target <5%) |
| AUC-ROC | ✓ | ✓ (target >0.80) |

### Plots (saved to `model/eval_figures/`)
1. **Reliability diagrams** — both models, predicted prob vs actual win rate (10 bins), with prediction histogram
2. **SHAP feature importance** — beeswarm + bar plot from `shap.TreeExplainer` on 50K subsampled rows (unwrap XGBoost from `CalibratedClassifierCV` via `.calibrated_classifiers_[0].estimator`)
3. **Win probability curves** — 10 test games (diverse selection: 3 home wins, 3 away wins, 2 close, 2 blowouts), plotting `seconds_remaining` vs `home_win_prob` with quarter markers
4. **Per-quarter calibration** — table of Brier/ECE/AUC-ROC grouped by quarter (1–4, 5+)
5. **Uncertainty region** — metrics on Q2/Q3 events with `abs(score_diff) <= 10`

### Custom ECE Implementation
Equal-width bins on [0, 1]. ECE = Σ (bin_weight × |avg_predicted − avg_actual|). Report bin counts alongside the diagram.

---

## File 4: `requirements.txt` Update

Add `shap>=0.43.0` (needed for SHAP TreeExplainer in evaluate.py).

---

## Key Design Decisions

1. **Pre-game prob leakage prevention**: `cross_val_predict` for training games ensures the in-game model never sees overfit pregame probabilities on its training data.

2. **Calibration by game_id**: All rows from a game go to the same split. Prevents Q1 events in train and Q4 events in calibration for the same game.

3. **Platt (sigmoid) for pregame, isotonic for ingame**: Per spec. Sigmoid has 2 parameters (good for the smaller pregame dataset). Isotonic is more flexible (good for the 5.5M-row ingame dataset).

4. **Early stopping on val set**: Standard practice. Mild val leakage is acceptable given data volume.

5. **SHAP on uncalibrated model**: `TreeExplainer` needs the raw XGBoost model. Isotonic calibration is monotonic so feature importance ordering is preserved.

---

## Verification

1. **Run training**: `python model/train_pregame.py && python model/train_ingame.py`
2. **Check artifacts exist**: `model/pregame.pkl`, `model/ingame.pkl`, `data/processed/pregame_probs.parquet`
3. **Run evaluation**: `python model/evaluate.py` — verify target metrics are met
4. **Inspect figures**: Check `model/eval_figures/` for reliability diagrams, SHAP plots, win probability curves
5. **Sanity checks**:
   - Pre-game prob distribution centered ~0.55–0.60 (home team advantage)
   - In-game Brier should be much lower in Q4 than Q1
   - Win prob curves should converge toward 0 or 1 as games end
   - SHAP should show `score_diff` and `seconds_remaining` as top features
6. **Run existing tests**: `python -m pytest tests/test_no_leakage.py` — should still pass
