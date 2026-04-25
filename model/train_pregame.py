"""
train_pregame.py — Stage 1 pre-game model training.

Trains a calibrated Logistic Regression model on pre-game features and
produces pre_game_prob predictions for all games (used as input to the
in-game model).

Usage:
    python model/train_pregame.py

Outputs:
    model/pregame.pkl                       — calibrated LR model (joblib)
    data/processed/pregame_probs.parquet    — [game_id, pre_game_prob] for all games
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PREGAME_FEATURES = [
    "elo_diff",
    "efg_pct_diff",
    "ortg_diff",
    "drtg_diff",
    "prev_season_win_pct_diff",
    "rest_days_diff",
    "home_flag",
    "ast_pct_diff",
    "tov_pct_diff",
]
TARGET = "home_win"

TRAIN_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22",
]
VAL_SEASONS = ["2022-23"]
TEST_SEASONS = ["2023-24"]
HOLDOUT_SEASONS = ["2024-25"]

CAL_FRAC = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_pregame_data() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "pregame_features.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded pregame features: {len(df):,} rows")
    return df


def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    """Split training data into train_proper and calibration by game_id."""
    rng = np.random.default_rng(seed)
    unique_game_ids = train_df["game_id"].unique()
    n_cal = int(len(unique_game_ids) * cal_frac)
    cal_game_ids = set(rng.choice(unique_game_ids, size=n_cal, replace=False))
    cal_mask = train_df["game_id"].isin(cal_game_ids)
    train_proper = train_df[~cal_mask].copy()
    calibration = train_df[cal_mask].copy()
    print(
        f"Calibration carve → train_proper: {len(train_proper):,} games  "
        f"calibration: {len(calibration):,} games"
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
        avg_pred = y_prob[mask].mean()
        avg_true = float(y_true[mask].mean())
        ece += mask.sum() / len(y_true) * abs(avg_pred - avg_true)
    return ece


def print_metrics(label: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    accuracy = ((y_prob >= 0.5).astype(int) == y_true).mean()
    print(
        f"[{label}]  accuracy={accuracy:.4f}  brier={brier:.4f}  "
        f"ece={ece:.4f}  auc={auc:.4f}"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_base_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)),
    ])


def train_and_calibrate(
    train_proper_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> CalibratedClassifierCV:
    X_tr = train_proper_df[PREGAME_FEATURES].values
    y_tr = train_proper_df[TARGET].values
    X_cal = calibration_df[PREGAME_FEATURES].values
    y_cal = calibration_df[TARGET].values

    base_model = build_base_pipeline()
    base_model.fit(X_tr, y_tr)
    print("Base LR trained.")

    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    print("Platt calibration fitted on calibration split.")
    return calibrated


# ---------------------------------------------------------------------------
# Pre-game probability generation (leakage-safe)
# ---------------------------------------------------------------------------

def generate_pregame_probs(
    calibrated_model: CalibratedClassifierCV,
    pregame_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate pre_game_prob for every game, leakage-safe:
    - Training games: cross_val_predict (out-of-fold) so the in-game model
      never trains on probabilities from a model that has already seen those games.
    - Val/test/holdout: predict_proba from the final calibrated model.
    """
    train_game_ids = set(train_df["game_id"].values)
    result_parts: list[pd.DataFrame] = []

    # Out-of-fold predictions for training data
    train_subset = pregame_df[pregame_df["game_id"].isin(train_game_ids)].copy()
    X_train_all = train_subset[PREGAME_FEATURES].values
    y_train_all = train_subset[TARGET].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    base_pipeline = build_base_pipeline()
    oof_probs = cross_val_predict(
        base_pipeline, X_train_all, y_train_all,
        cv=cv, method="predict_proba",
    )[:, 1]
    print(f"Cross-val predictions for {len(train_subset):,} training games done.")

    result_parts.append(pd.DataFrame({
        "game_id": train_subset["game_id"].values,
        "pre_game_prob": oof_probs,
    }))

    # Calibrated model predictions for non-training games
    non_train = pregame_df[~pregame_df["game_id"].isin(train_game_ids)].copy()
    if len(non_train) > 0:
        X_non_train = non_train[PREGAME_FEATURES].values
        non_train_probs = calibrated_model.predict_proba(X_non_train)[:, 1]
        result_parts.append(pd.DataFrame({
            "game_id": non_train["game_id"].values,
            "pre_game_prob": non_train_probs,
        }))

    probs_df = pd.concat(result_parts, ignore_index=True)
    assert len(probs_df) == len(pregame_df), "Probability count mismatch"
    assert probs_df["pre_game_prob"].isna().sum() == 0, "NaN in pre_game_prob"
    return probs_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 3 — Pre-game model training")
    print("=" * 60)

    df = load_pregame_data()
    train_df, val_df, test_df, holdout_df = split_by_season(df)
    train_proper_df, calibration_df = carve_calibration_split(train_df)

    calibrated_model = train_and_calibrate(train_proper_df, calibration_df)

    # Evaluate on validation set
    X_val = val_df[PREGAME_FEATURES].values
    y_val = val_df[TARGET].values
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    print_metrics("VAL", y_val, val_probs)

    # Evaluate on test set (informational only — don't use for model selection)
    X_test = test_df[PREGAME_FEATURES].values
    y_test = test_df[TARGET].values
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    print_metrics("TEST", y_test, test_probs)

    # Generate pre_game_prob for all games (leakage-safe)
    probs_df = generate_pregame_probs(calibrated_model, df, train_df)
    print(f"pre_game_prob generated for {len(probs_df):,} games  "
          f"(mean={probs_df['pre_game_prob'].mean():.4f})")

    # Save model artifact
    model_path = ROOT / "model" / "pregame.pkl"
    joblib.dump(calibrated_model, model_path)
    print(f"Model saved → {model_path}")

    # Save pre_game_prob artifact
    probs_path = ROOT / "data" / "processed" / "pregame_probs.parquet"
    probs_df.to_parquet(probs_path, index=False)
    print(f"Probabilities saved → {probs_path}")

    print("\nDone. Target metrics: accuracy ~66%, ECE <4%, AUC-ROC >0.78")


if __name__ == "__main__":
    main()
