"""
pipeline.py — Feature engineering orchestration for Phase 2.

Runs all three feature modules in sequence and writes the output parquet files:
  - data/processed/pregame_features.parquet  (one row per game)
  - data/processed/ingame_snapshots.parquet  (one row per PBP event)

Run:
    python features/pipeline.py

Both output files must exist before running model/train_pregame.py or
model/train_ingame.py. The train/val/test split is applied at training time,
not here — this script builds features for all seasons.
"""

import sys
from pathlib import Path

# Allow running as `python features/pipeline.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.elo import compute_elo_ratings
from features.ingame import build_ingame_snapshots
from features.pregame import build_pregame_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = _REPO_ROOT / "data" / "raw"
OUTPUT_DIR = _REPO_ROOT / "data" / "processed"

GAMES_DB = DB_DIR / "games.db"
PBP_DB = DB_DIR / "pbp.db"
PLAYERS_DB = DB_DIR / "players.db"

PREGAME_OUT = OUTPUT_DIR / "pregame_features.parquet"
INGAME_OUT = OUTPUT_DIR / "ingame_snapshots.parquet"


def _check_inputs() -> None:
    """Fail fast if any required database is missing."""
    for path in (GAMES_DB, PBP_DB, PLAYERS_DB):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run the Phase 1 fetch scripts first:\n"
                "  python data/fetch_games.py\n"
                "  python data/fetch_pbp.py\n"
                "  python data/fetch_players.py"
            )


def run() -> None:
    _check_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Walk-forward ELO ratings
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/3 — Computing walk-forward ELO ratings")
    print("=" * 60)
    elo_ratings = compute_elo_ratings(GAMES_DB)
    print(f"  ELO ratings computed for {len(elo_ratings):,} (team, game) pairs.\n")

    # ------------------------------------------------------------------
    # Step 2: Pre-game features
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2/3 — Building pre-game features")
    print("=" * 60)
    pregame_df = build_pregame_features(GAMES_DB, PLAYERS_DB, elo_ratings)
    pregame_df.to_parquet(PREGAME_OUT, index=False)
    print(f"\n  Saved {len(pregame_df):,} rows to {PREGAME_OUT}")
    _print_pregame_summary(pregame_df)

    # ------------------------------------------------------------------
    # Step 3: In-game snapshots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3/3 — Building in-game snapshots")
    print("=" * 60)
    ingame_df = build_ingame_snapshots(PBP_DB, GAMES_DB)
    ingame_df.to_parquet(INGAME_OUT, index=False)
    print(f"\n  Saved {len(ingame_df):,} rows to {INGAME_OUT}")
    _print_ingame_summary(ingame_df)

    print("\n" + "=" * 60)
    print("Phase 2 complete. Feature files written:")
    print(f"  {PREGAME_OUT}")
    print(f"  {INGAME_OUT}")
    print("=" * 60)


def _print_pregame_summary(df) -> None:
    print(f"\n  Pre-game features summary:")
    print(f"    Rows (games):     {len(df):,}")
    print(f"    Columns:          {len(df.columns)}")
    print(f"    Seasons:          {sorted(df['season'].unique())}")
    print(f"    Home win rate:    {df['home_win'].mean():.3f}")
    print(f"    NaN counts:")
    nan_counts = df.isnull().sum()
    for col in nan_counts[nan_counts > 0].index:
        print(f"      {col}: {nan_counts[col]}")
    if nan_counts.sum() == 0:
        print("      (none)")
    print(
        f"    ELO diff range:   [{df['elo_diff'].min():.1f}, {df['elo_diff'].max():.1f}]"
    )


def _print_ingame_summary(df) -> None:
    print(f"\n  In-game snapshots summary:")
    print(f"    Rows (events):    {len(df):,}")
    print(f"    Columns:          {len(df.columns)}")
    print(f"    Games covered:    {df['game_id'].nunique():,}")
    print(f"    Home win rate:    {df['home_win'].mean():.3f}")
    print(f"    Score diff range: [{df['score_diff'].min()}, {df['score_diff'].max()}]")
    print(
        f"    Seconds rem range:[{df['seconds_remaining'].min()}, {df['seconds_remaining'].max()}]"
    )
    clutch_pct = df["clutch_flag"].mean() * 100
    print(f"    Clutch rows:      {clutch_pct:.1f}%")
    nan_critical = (
        df[["score_diff", "seconds_remaining", "home_win"]].isnull().sum().sum()
    )
    print(f"    NaN in critical cols: {nan_critical}")


if __name__ == "__main__":
    run()
