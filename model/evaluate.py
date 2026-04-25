"""
evaluate.py — Comprehensive evaluation of both pre-game and in-game models.

Loads model/pregame.pkl and model/ingame.pkl, evaluates on the test set
(2023-24 season), and saves figures to model/eval_figures/.

Usage:
    python model/evaluate.py

Outputs:
    model/eval_figures/pregame_reliability.png
    model/eval_figures/ingame_reliability.png
    model/eval_figures/shap_beeswarm.png
    model/eval_figures/shap_bar.png
    model/eval_figures/win_prob_curves.png
    model/eval_figures/per_quarter_calibration.png
    Metrics printed to stdout
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.train_ingame import StratifiedCalibrator  # noqa: F401 — needed for joblib deserialization
FIGURES_DIR = ROOT / "model" / "eval_figures"

PREGAME_FEATURES = [
    "elo_diff", "efg_pct_diff", "ortg_diff", "drtg_diff",
    "prev_season_win_pct_diff", "rest_days_diff", "home_flag",
    "ast_pct_diff", "tov_pct_diff",
]
INGAME_FEATURES = [
    "score_diff", "seconds_remaining", "pre_game_prob",
    "home_fg_pct_live", "away_fg_pct_live",
    "home_2pt_pct_live", "away_2pt_pct_live",
    "home_3pt_pct_live", "away_3pt_pct_live",
    "home_ft_pct_live", "away_ft_pct_live",
    "home_fouls", "away_fouls",
    "turnover_diff_live", "timeout_remaining_diff", "last_5_poss_swing",
    "quarter", "clutch_flag",
]
TARGET = "home_win"
TEST_SEASONS = ["2023-24"]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_models() -> tuple:
    pregame_model = joblib.load(ROOT / "model" / "pregame.pkl")
    ingame_model = joblib.load(ROOT / "model" / "ingame.pkl")
    print("Models loaded.")
    return pregame_model, ingame_model


def load_test_data(ingame_model) -> tuple[pd.DataFrame, pd.DataFrame]:
    pregame_df = pd.read_parquet(ROOT / "data" / "processed" / "pregame_features.parquet")
    ingame_df = pd.read_parquet(ROOT / "data" / "processed" / "ingame_snapshots.parquet")
    probs_df = pd.read_parquet(ROOT / "data" / "processed" / "pregame_probs.parquet")

    pregame_test = pregame_df[pregame_df["season"].isin(TEST_SEASONS)].copy()
    ingame_test = ingame_df[ingame_df["season"].isin(TEST_SEASONS)].copy()

    # Replace placeholder pre_game_prob with actual predictions
    ingame_test = ingame_test.drop(columns=["pre_game_prob"]).merge(
        probs_df[["game_id", "pre_game_prob"]], on="game_id", how="left"
    )
    assert ingame_test["pre_game_prob"].isna().sum() == 0

    print(
        f"Test data loaded → pregame: {len(pregame_test):,} games  "
        f"ingame: {len(ingame_test):,} events"
    )
    return pregame_test, ingame_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    counts = []
    for b in range(n_bins):
        mask = bin_indices == b
        counts.append(mask.sum())
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_prob[mask].mean() - float(y_true[mask].mean()))
    return ece


def print_metrics_table(results: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    header = f"{'Model':<25} {'Brier':>8} {'ECE':>8} {'AUC-ROC':>8} {'Accuracy':>10}"
    print(header)
    print("-" * 60)
    for name, metrics in results.items():
        brier = metrics.get("brier", float("nan"))
        ece = metrics.get("ece", float("nan"))
        auc = metrics.get("auc", float("nan"))
        acc = metrics.get("accuracy", float("nan"))
        print(f"{name:<25} {brier:>8.4f} {ece:>8.4f} {auc:>8.4f} {acc:>10.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, "o-", label="Model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title or "Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins)
    ax1.text(0.05, 0.92, f"Brier={brier:.4f}  ECE={ece:.4f}", transform=ax1.transAxes,
             fontsize=9, va="top", bbox=dict(boxstyle="round", alpha=0.2))

    ax2.hist(y_prob, bins=50, edgecolor="k", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def plot_shap_importance(
    calibrated_ingame_model,
    X_sample: np.ndarray,
    feature_names: list[str],
    save_dir: Path,
) -> None:
    base_xgb: xgb.XGBClassifier = calibrated_ingame_model.base_model

    print(f"Computing SHAP values on {len(X_sample):,} samples...")
    explainer = shap.TreeExplainer(base_xgb)
    shap_values = explainer.shap_values(X_sample)

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Beeswarm)")
    plt.tight_layout()
    beeswarm_path = save_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {beeswarm_path}")

    # Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Mean |SHAP| Feature Importance")
    plt.tight_layout()
    bar_path = save_dir / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {bar_path}")


# ---------------------------------------------------------------------------
# Win probability curves
# ---------------------------------------------------------------------------

def plot_win_probability_curves(
    ingame_model,
    ingame_test_df: pd.DataFrame,
    n_games: int = 10,
    seed: int = RANDOM_SEED,
    save_path: Path | None = None,
) -> None:
    rng = random.Random(seed)

    # Get final outcome per game
    game_outcomes = (
        ingame_test_df.groupby("game_id")["home_win"].first().reset_index()
    )
    home_won_ids = game_outcomes[game_outcomes["home_win"] == 1]["game_id"].tolist()
    away_won_ids = game_outcomes[game_outcomes["home_win"] == 0]["game_id"].tolist()

    # Score margin per game (use last event's score_diff as proxy for final margin)
    last_event = ingame_test_df.sort_values("action_number").groupby("game_id").last()
    close_game_ids = last_event[last_event["score_diff"].abs() <= 5].index.tolist()
    blowout_game_ids = last_event[last_event["score_diff"].abs() >= 15].index.tolist()

    selected: list[str] = []
    # 3 home wins, 3 away wins, 2 close, 2 blowouts (with overlap allowed)
    selected += rng.sample(home_won_ids, min(3, len(home_won_ids)))
    selected += rng.sample(away_won_ids, min(3, len(away_won_ids)))
    close_remaining = [g for g in close_game_ids if g not in selected]
    blowout_remaining = [g for g in blowout_game_ids if g not in selected]
    selected += rng.sample(close_remaining, min(2, len(close_remaining)))
    selected += rng.sample(blowout_remaining, min(2, len(blowout_remaining)))
    selected = list(dict.fromkeys(selected))[:n_games]  # deduplicate, cap at n_games

    n_cols = 5
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    quarter_boundaries_sec = {
        "Q1/Q2": 2160,
        "Q2/Q3": 1440,
        "Q3/Q4": 720,
        "End": 0,
    }

    for i, game_id in enumerate(selected):
        ax = axes_flat[i]
        game_df = ingame_test_df[ingame_test_df["game_id"] == game_id].sort_values("action_number")
        X_game = game_df[INGAME_FEATURES].astype(np.float32).values
        probs = ingame_model.predict_proba(X_game)[:, 1]
        secs = game_df["seconds_remaining"].values
        home_win = game_df["home_win"].iloc[0]

        ax.plot(secs, probs, lw=1.5, color="steelblue")
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
        for name, sec in quarter_boundaries_sec.items():
            if sec > secs.min():
                ax.axvline(sec, color="lightgray", lw=0.8, alpha=0.7)

        outcome_str = "Home W" if home_win else "Home L"
        ax.set_title(f"{game_id[:8]}… ({outcome_str})", fontsize=8)
        ax.set_xlabel("Seconds remaining", fontsize=7)
        ax.set_ylabel("Home win prob", fontsize=7)
        ax.set_ylim(0, 1)
        ax.invert_xaxis()
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Win Probability Curves — Test Set (2023-24)", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-quarter calibration
# ---------------------------------------------------------------------------

def evaluate_per_quarter(
    ingame_model,
    ingame_test_df: pd.DataFrame,
) -> pd.DataFrame:
    X_test = ingame_test_df[INGAME_FEATURES].astype(np.float32).values
    y_test = ingame_test_df[TARGET].values
    probs = ingame_model.predict_proba(X_test)[:, 1]
    ingame_test_df = ingame_test_df.copy()
    ingame_test_df["_prob"] = probs

    rows = []
    for q in sorted(ingame_test_df["quarter"].unique()):
        mask = ingame_test_df["quarter"] == q
        y_q = y_test[mask.values]
        p_q = probs[mask.values]
        if len(y_q) < 10:
            continue
        rows.append({
            "quarter": int(q),
            "n_events": len(y_q),
            "brier": brier_score_loss(y_q, p_q),
            "ece": compute_ece(y_q, p_q),
            "auc": roc_auc_score(y_q, p_q) if len(np.unique(y_q)) > 1 else float("nan"),
        })

    result_df = pd.DataFrame(rows)
    print("\nPer-quarter calibration on test set:")
    print(result_df.to_string(index=False))
    return result_df


def plot_per_quarter_calibration(quarter_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["brier", "ece", "auc"]
    titles = ["Brier Score", "ECE", "AUC-ROC"]
    targets = [0.18, 0.05, 0.80]
    target_dir = ["below", "below", "above"]

    for ax, metric, title, target, direction in zip(axes, metrics, titles, targets, target_dir):
        ax.bar(quarter_df["quarter"].astype(str), quarter_df[metric], color="steelblue", alpha=0.8)
        ax.axhline(target, color="red", linestyle="--", lw=1.2,
                   label=f"Target ({'<' if direction == 'below' else '>'}{target})")
        ax.set_title(title)
        ax.set_xlabel("Quarter")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Quarter Model Performance — Test Set (2023-24)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Uncertainty region
# ---------------------------------------------------------------------------

def evaluate_uncertainty_region(
    ingame_model,
    ingame_test_df: pd.DataFrame,
) -> dict:
    mask = (
        ingame_test_df["quarter"].isin([2, 3]) &
        (ingame_test_df["score_diff"].abs() <= 10)
    )
    subset = ingame_test_df[mask].copy()
    X = subset[INGAME_FEATURES].astype(np.float32).values
    y = subset[TARGET].values
    probs = ingame_model.predict_proba(X)[:, 1]

    metrics = {
        "n_events": len(y),
        "brier": brier_score_loss(y, probs),
        "ece": compute_ece(y, probs),
        "auc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
    }
    print(
        f"\nUncertainty region (Q2/Q3, |score_diff| ≤ 10)  "
        f"n={metrics['n_events']:,}  "
        f"brier={metrics['brier']:.4f}  "
        f"ece={metrics['ece']:.4f}  "
        f"auc={metrics['auc']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3 — Model Evaluation")
    print("=" * 60)

    pregame_model, ingame_model = load_models()
    pregame_test_df, ingame_test_df = load_test_data(ingame_model)

    results: dict[str, dict] = {}

    # --- Pre-game model ---
    print("\n--- Pre-game model (test set) ---")
    X_pg = pregame_test_df[PREGAME_FEATURES].values
    y_pg = pregame_test_df[TARGET].values
    pg_probs = pregame_model.predict_proba(X_pg)[:, 1]

    pg_metrics = {
        "brier": brier_score_loss(y_pg, pg_probs),
        "ece": compute_ece(y_pg, pg_probs),
        "auc": roc_auc_score(y_pg, pg_probs),
        "accuracy": ((pg_probs >= 0.5).astype(int) == y_pg).mean(),
    }
    results["Pre-game LR"] = pg_metrics

    plot_reliability_diagram(
        y_pg, pg_probs,
        title="Pre-game Model — Reliability Diagram (Test 2023-24)",
        save_path=FIGURES_DIR / "pregame_reliability.png",
    )

    # --- In-game model ---
    print("\n--- In-game model (test set) ---")
    X_ig = ingame_test_df[INGAME_FEATURES].astype(np.float32).values
    y_ig = ingame_test_df[TARGET].values
    ig_probs = ingame_model.predict_proba(X_ig)[:, 1]

    ig_metrics = {
        "brier": brier_score_loss(y_ig, ig_probs),
        "ece": compute_ece(y_ig, ig_probs),
        "auc": roc_auc_score(y_ig, ig_probs),
        "accuracy": ((ig_probs >= 0.5).astype(int) == y_ig).mean(),
    }
    results["In-game XGBoost"] = ig_metrics

    plot_reliability_diagram(
        y_ig, ig_probs,
        title="In-game Model — Reliability Diagram (Test 2023-24)",
        save_path=FIGURES_DIR / "ingame_reliability.png",
    )

    # --- SHAP feature importance ---
    print("\n--- SHAP feature importance ---")
    rng = np.random.default_rng(RANDOM_SEED)
    n_shap_sample = min(50_000, len(ingame_test_df))
    shap_idx = rng.choice(len(ingame_test_df), size=n_shap_sample, replace=False)
    X_shap = X_ig[shap_idx]
    plot_shap_importance(ingame_model, X_shap, INGAME_FEATURES, FIGURES_DIR)

    # --- Win probability curves ---
    print("\n--- Win probability curves ---")
    plot_win_probability_curves(
        ingame_model, ingame_test_df,
        save_path=FIGURES_DIR / "win_prob_curves.png",
    )

    # --- Per-quarter calibration ---
    print("\n--- Per-quarter calibration ---")
    quarter_df = evaluate_per_quarter(ingame_model, ingame_test_df)
    plot_per_quarter_calibration(quarter_df, FIGURES_DIR / "per_quarter_calibration.png")

    # --- Uncertainty region ---
    uncertainty_metrics = evaluate_uncertainty_region(ingame_model, ingame_test_df)
    results["In-game (Q2/Q3, |diff|≤10)"] = {
        **uncertainty_metrics,
        "accuracy": float("nan"),
    }

    # --- Summary table ---
    print_metrics_table(results)

    # Target checks
    print("\nTarget checks:")
    print(f"  Pre-game  ECE < 4%:     {'PASS' if pg_metrics['ece'] < 0.04 else 'FAIL'}  ({pg_metrics['ece']:.4f})")
    print(f"  In-game   Brier < 0.18: {'PASS' if ig_metrics['brier'] < 0.18 else 'FAIL'}  ({ig_metrics['brier']:.4f})")
    print(f"  In-game   ECE < 5%:     {'PASS' if ig_metrics['ece'] < 0.05 else 'FAIL'}  ({ig_metrics['ece']:.4f})")
    print(f"  In-game   AUC > 0.80:   {'PASS' if ig_metrics['auc'] > 0.80 else 'FAIL'}  ({ig_metrics['auc']:.4f})")
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
