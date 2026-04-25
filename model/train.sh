#!/usr/bin/env bash
# Train both models in sequence. Logs written to model/logs/.
# Usage: bash model/train.sh [--sweep]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/model/logs"
mkdir -p "$LOGS"

SWEEP=""
if [[ "${1:-}" == "--sweep" ]]; then
    SWEEP="--sweep"
fi

echo "=== NBA Win Probability — Model Training ==="
echo "Logs: $LOGS"
echo

# Stage 1: pre-game model
echo "[$(date '+%H:%M:%S')] Starting train_pregame.py..."
python "$ROOT/model/train_pregame.py" 2>&1 | tee "$LOGS/train_pregame.log"
echo "[$(date '+%H:%M:%S')] train_pregame.py complete."
echo

# Stage 2: in-game model
echo "[$(date '+%H:%M:%S')] Starting train_ingame.py${SWEEP:+ $SWEEP}..."
python "$ROOT/model/train_ingame.py" $SWEEP 2>&1 | tee "$LOGS/train_ingame.log"
echo "[$(date '+%H:%M:%S')] train_ingame.py complete."
echo

echo "=== Training complete. Artifacts: ==="
ls -lh "$ROOT/model/"*.pkl 2>/dev/null || echo "  (no .pkl files found)"
ls -lh "$ROOT/data/processed/pregame_probs.parquet" 2>/dev/null || echo "  (pregame_probs.parquet not found)"
