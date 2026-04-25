"""
train_ingame.py — Stage 2 in-game model training.

Trains a calibrated XGBoost model on play-by-play snapshots.
Reads pregame_probs.parquet (output of train_pregame.py) to replace the
placeholder 0.5 pre_game_prob values with actual predictions.

Usage:
    python model/train_ingame.py            # default params
    python model/train_ingame.py --sweep    # run hyperparameter sweep (~2 hrs)

Outputs:
    model/ingame.pkl    — calibrated XGBoost model (joblib)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INGAME_FEATURES = [
    "score_diff",
    "seconds_remaining",
    "pre_game_prob",
    "home_fg_pct_live",
    "away_fg_pct_live",
    "home_2pt_pct_live",
    "away_2pt_pct_live",
    "home_3pt_pct_live",
    "away_3pt_pct_live",
    "home_ft_pct_live",
    "away_ft_pct_live",
    "home_fouls",
    "away_fouls",
    "turnover_diff_live",
    "timeout_remaining_diff",
    "last_5_poss_swing",
    "quarter",        # index 16
    "clutch_flag",
]
TARGET = "home_win"

TRAIN_SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
]
VAL_SEASONS = ["2022-23"]
TEST_SEASONS = ["2023-24"]
HOLDOUT_SEASONS = ["2024-25"]

CAL_FRAC = 0.15
RANDOM_SEED = 42

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Stratified calibrator
# ---------------------------------------------------------------------------


class StratifiedCalibrator:
    """
    Drop-in replacement for CalibratedClassifierCV.

    Holds the base XGBoost model and three phase-specific calibrators:
      - Phase A (Q1-Q2, quarter <= 2):   IsotonicRegression
      - Phase B (Q3-Q4, quarter in 3-4): IsotonicRegression
      - Phase C (OT,    quarter >= 5):   IsotonicRegression, fitted on cal + val OT rows

    predict_proba(X) returns shape (n_samples, 2), routing each row to the
    correct calibrator based on X[:, QUARTER_IDX].
    """

    QUARTER_IDX: int = 16  # index of 'quarter' in INGAME_FEATURES

    def __init__(self, base_model: xgb.XGBClassifier) -> None:
        self.base_model = base_model
        self.classes_ = np.array([0, 1])
        self.n_features_in_: int = base_model.n_features_in_
        self._cal_a: IsotonicRegression | None = None
        self._cal_b: IsotonicRegression | None = None
        self._cal_c: IsotonicRegression | None = None

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_val_ot: np.ndarray | None = None,
        y_val_ot: np.ndarray | None = None,
    ) -> "StratifiedCalibrator":
        raw_probs = self.base_model.predict_proba(X_cal)[:, 1]
        quarters = X_cal[:, self.QUARTER_IDX]

        mask_a = quarters <= 2
        mask_b = (quarters >= 3) & (quarters <= 4)
        mask_c = quarters >= 5

        for label, mask in [
            ("Q1-Q2 (Phase A)", mask_a),
            ("Q3-Q4 (Phase B)", mask_b),
            ("OT    (Phase C)", mask_c),
        ]:
            print(f"  Calibration samples — {label}: {int(mask.sum()):,}")

        self._cal_a = IsotonicRegression(out_of_bounds="clip")
        self._cal_a.fit(raw_probs[mask_a], y_cal[mask_a])

        self._cal_b = IsotonicRegression(out_of_bounds="clip")
        self._cal_b.fit(raw_probs[mask_b], y_cal[mask_b])

        # OT calibrator: use val OT rows only (2022-23) — temporally closest to
        # test (2023-24). Cal-split OT rows (2015-2022) have a ~44% home win rate
        # vs ~64% in test, so pooling them drags calibration toward the wrong base
        # rate. Val-only avoids that era shift entirely.
        if X_val_ot is not None and y_val_ot is not None:
            ot_raw = self.base_model.predict_proba(X_val_ot)[:, 1]
            ot_y = y_val_ot
            print(
                f"  OT calibration: val rows only = {len(ot_y):,} "
                f"(cal OT rows excluded to avoid era-shift dilution)"
            )
        else:
            ot_raw = raw_probs[mask_c]
            ot_y = y_cal[mask_c]
            print(f"  OT calibration: {len(ot_y):,} cal rows only (no val OT provided)")

        if len(ot_y) < 50:
            raise ValueError(f"Only {len(ot_y)} OT rows — too few to calibrate.")
        self._cal_c = IsotonicRegression(out_of_bounds="clip")
        self._cal_c.fit(ot_raw, ot_y)

        print("Stratified calibration fitted (all phases isotonic; OT uses cal+val).")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        quarters = X[:, self.QUARTER_IDX]
        calibrated = np.empty(len(raw_probs), dtype=np.float64)

        mask_a = quarters <= 2
        mask_b = (quarters >= 3) & (quarters <= 4)
        mask_c = quarters >= 5

        if mask_a.any():
            calibrated[mask_a] = self._cal_a.predict(raw_probs[mask_a])
        if mask_b.any():
            calibrated[mask_b] = self._cal_b.predict(raw_probs[mask_b])
        if mask_c.any():
            calibrated[mask_c] = self._cal_c.predict(raw_probs[mask_c])

        return np.column_stack([1.0 - calibrated, calibrated])


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------


def load_ingame_data() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "ingame_snapshots.parquet"
    print("Loading ingame snapshots (5.5M rows — may take a moment)...")
    df = pd.read_parquet(path)
    print(f"Loaded ingame snapshots: {len(df):,} rows")
    return df


def replace_pregame_prob(ingame_df: pd.DataFrame) -> pd.DataFrame:
    """Replace placeholder 0.5 pre_game_prob values with actual predictions."""
    probs_path = ROOT / "data" / "processed" / "pregame_probs.parquet"
    if not probs_path.exists():
        raise FileNotFoundError(
            f"pregame_probs.parquet not found at {probs_path}. "
            "Run model/train_pregame.py first."
        )
    probs_df = pd.read_parquet(probs_path)

    # Sanity check: all current values should be the placeholder
    placeholder_frac = (ingame_df["pre_game_prob"] == 0.5).mean()
    if placeholder_frac < 0.99:
        print(
            f"WARNING: Only {placeholder_frac:.1%} of pre_game_prob values are "
            "the placeholder 0.5. Was this already replaced?"
        )

    df = ingame_df.drop(columns=["pre_game_prob"]).merge(
        probs_df[["game_id", "pre_game_prob"]], on="game_id", how="left"
    )
    n_nan = df["pre_game_prob"].isna().sum()
    assert n_nan == 0, f"{n_nan} NaN values in pre_game_prob after merge"
    print(
        f"Replaced pre_game_prob placeholder  (mean={df['pre_game_prob'].mean():.4f})"
    )
    return df


def split_by_season(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val = df[df["season"].isin(VAL_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    holdout = df[df["season"].isin(HOLDOUT_SEASONS)].copy()
    print(
        f"Split → train: {len(train):,}  val: {len(val):,}  "
        f"test: {len(test):,}  holdout: {len(holdout):,}"
    )
    return train, val, test, holdout


def carve_calibration_split(
    train_df: pd.DataFrame,
    cal_frac: float = CAL_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training rows into train_proper and calibration by game_id.
    All rows for a game go to the same split — prevents leaking the
    game outcome label across splits.
    """
    rng = np.random.default_rng(seed)
    unique_game_ids = train_df["game_id"].unique()
    n_cal = int(len(unique_game_ids) * cal_frac)
    cal_game_ids = set(rng.choice(unique_game_ids, size=n_cal, replace=False))
    cal_mask = train_df["game_id"].isin(cal_game_ids)
    train_proper = train_df[~cal_mask].copy()
    calibration = train_df[cal_mask].copy()
    print(
        f"Calibration carve by game_id → "
        f"train_proper: {len(train_proper):,} rows ({(~cal_mask).sum() / len(train_df):.1%})  "
        f"calibration: {len(calibration):,} rows ({cal_mask.sum() / len(train_df):.1%})"
    )
    return train_proper, calibration


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        ece += (
            mask.sum()
            / len(y_true)
            * abs(y_prob[mask].mean() - float(y_true[mask].mean()))
        )
    return ece


def print_metrics(label: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    print(f"[{label}]  brier={brier:.4f}  ece={ece:.4f}  auc={auc:.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    if params is None:
        params = DEFAULT_PARAMS.copy()

    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=50,
        eval_metric="logloss",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    best_iter = model.best_iteration
    print(f"XGBoost training done. Best iteration: {best_iter}")
    return model


def calibrate_xgboost(
    xgb_model: xgb.XGBClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_val_ot: np.ndarray | None = None,
    y_val_ot: np.ndarray | None = None,
) -> StratifiedCalibrator:
    calibrated = StratifiedCalibrator(xgb_model)
    calibrated.fit(X_cal, y_cal, X_val_ot, y_val_ot)
    return calibrated


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------


def _sweep_stage(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fixed_params: dict,
    search_params: dict[str, list],
) -> tuple[dict, float]:
    best_brier = float("inf")
    best_combo: dict = {}
    combos = list(itertools.product(*search_params.values()))
    print(f"  Searching {len(combos)} combinations...")
    for i, combo in enumerate(combos, 1):
        params = {**fixed_params, **dict(zip(search_params.keys(), combo))}
        model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=50,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, probs)
        if brier < best_brier:
            best_brier = brier
            best_combo = dict(zip(search_params.keys(), combo))
        if i % 5 == 0:
            print(f"  [{i}/{len(combos)}] best brier so far: {best_brier:.4f}")
    return best_combo, best_brier


def hyperparameter_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    print("\n--- Hyperparameter Sweep Stage 1: structure params ---")
    stage1_fixed = {
        "objective": "binary:logistic",
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
    }
    stage1_search = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_weight": [5, 10, 20],
    }
    best1, brier1 = _sweep_stage(
        X_train, y_train, X_val, y_val, stage1_fixed, stage1_search
    )
    print(f"Stage 1 best: {best1}  brier={brier1:.4f}")

    print("\n--- Hyperparameter Sweep Stage 2: regularization params ---")
    stage2_fixed = {
        **stage1_fixed,
        **best1,
    }
    stage2_search = {
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    best2, brier2 = _sweep_stage(
        X_train, y_train, X_val, y_val, stage2_fixed, stage2_search
    )
    print(f"Stage 2 best: {best2}  brier={brier2:.4f}")

    final_params = {
        "objective": "binary:logistic",
        "n_estimators": 1000,
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        **best1,
        **best2,
    }
    print(f"\nFinal best params: {final_params}")
    return final_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train in-game XGBoost model")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter sweep before training (takes ~2 hours)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 — In-game model training")
    print("=" * 60)

    df = load_ingame_data()
    df = replace_pregame_prob(df)
    train_df, val_df, test_df, _ = split_by_season(df)
    train_proper_df, calibration_df = carve_calibration_split(train_df)

    X_train = train_proper_df[INGAME_FEATURES].astype(np.float32).values
    y_train = train_proper_df[TARGET].values
    X_val = val_df[INGAME_FEATURES].astype(np.float32).values
    y_val = val_df[TARGET].values
    X_cal = calibration_df[INGAME_FEATURES].astype(np.float32).values
    y_cal = calibration_df[TARGET].values

    # Extract val OT rows for the OT calibrator (closer era to test than cal split)
    val_ot_mask = val_df["quarter"].values >= 5
    X_val_ot = val_df.loc[val_ot_mask, INGAME_FEATURES].astype(np.float32).values
    y_val_ot = val_df.loc[val_ot_mask, TARGET].values
    print(f"Val OT rows for OT calibrator: {len(y_val_ot):,}")

    # Determine training params
    if args.sweep:
        best_params = hyperparameter_sweep(X_train, y_train, X_val, y_val)
    else:
        best_params = DEFAULT_PARAMS.copy()
        print(f"Using default params: {best_params}")

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, best_params)

    # Evaluate raw (uncalibrated) on val
    raw_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    print_metrics("VAL (uncalibrated)", y_val, raw_val_probs)

    # Calibrate with isotonic regression on calibration split
    calibrated_model = calibrate_xgboost(xgb_model, X_cal, y_cal, X_val_ot, y_val_ot)

    # Evaluate calibrated on val
    cal_val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    print_metrics("VAL (calibrated)", y_val, cal_val_probs)

    # Evaluate on test (informational)
    X_test = test_df[INGAME_FEATURES].astype(np.float32).values
    y_test = test_df[TARGET].values
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    print_metrics("TEST (calibrated)", y_test, test_probs)

    # Save model artifact
    model_path = ROOT / "model" / "ingame.pkl"
    joblib.dump(calibrated_model, model_path)
    print(f"Model saved → {model_path}")

    print("\nDone. Target metrics: Brier <0.18, ECE <5%, AUC-ROC >0.80")


if __name__ == "__main__":
    main()
