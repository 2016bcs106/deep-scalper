#!/bin/bash
# Train DeepScalper for multiple stocks sequentially.
# Designed for low-memory instances (t4g.micro, 1GB RAM).
#
# Usage:
#   nohup bash scripts/train-all.sh > logs/all.log 2>&1 &
#
# Each stock: download 5 years of data -> train 3 epochs -> evaluate + email.
# Safe to disconnect after launching with nohup.

set -e
cd "$(dirname "$0")/.."

STOCKS=(RELIANCE ADANIENT ADANIGREEN HDFCBANK ICICIBANK)
EPOCHS=3
FROM="2021-05-31"
TO="2026-05-31"

mkdir -p logs

echo "=== Starting training pipeline: ${STOCKS[*]} ==="
echo "=== $(date) ==="
echo ""

for STOCK in "${STOCKS[@]}"; do
  echo "--- [$STOCK] Downloading data ($FROM to $TO) ---"
  npx tsx --env-file=.env scripts/download-data.ts "$STOCK" --from "$FROM" --to "$TO" 2>&1 | tee -a "logs/$STOCK.log"

  echo "--- [$STOCK] Training ($EPOCHS epochs) ---"
  npx tsx --env-file=.env scripts/train.ts "$STOCK" --epochs "$EPOCHS" 2>&1 | tee -a "logs/$STOCK.log"

  echo "--- [$STOCK] Complete at $(date) ---"
  echo ""
done

echo "=== ALL COMPLETE: $(date) ==="
