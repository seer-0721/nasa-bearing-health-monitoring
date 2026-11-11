#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-120}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/xjtu_snn_time_best}"
LOG_FILE="${LOG_FILE:-logs/xjtu_time_best.log}"
NUM_WORKERS="${NUM_WORKERS:-0}"

python -u train.py \
  --data-root data/XJTU/converted_time \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --health-label-mode anomaly \
  --sample-layout row \
  --no-include-bearing-id \
  --limit-good 0 \
  --limit-fault 0 \
  --epochs "${EPOCHS}" \
  --batch-size 256 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --time-steps 16 \
  --health-regression-loss l1 \
  --health-regression-weight 4.0 \
  --health-regression-low-score-weight 1.5 \
  --health-regression-high-score-weight 1.0 \
  --fault-loss-weight 0.4 \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance both \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --dropout 0.03 \
  --selection-metric val_health_score_balanced_acc_at_tolerance \
  --log-format pretty \
  --log-header-interval 20 \
  2>&1 | tee "${LOG_FILE}"
