#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-80}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/paderborn_snn_fault_best}"
LOG_FILE="${LOG_FILE:-logs/paderborn_fault_best.log}"
NUM_WORKERS="${NUM_WORKERS:-0}"

python -u train.py \
  --data-root data/Paderborn/converted \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --health-label-mode source \
  --sample-layout row \
  --limit-good 0 \
  --limit-fault 0 \
  --epochs "${EPOCHS}" \
  --batch-size 256 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --time-steps 16 \
  --health-regression-loss l1 \
  --health-regression-weight 0.2 \
  --health-regression-low-score-weight 0.0 \
  --fault-loss-weight 2.0 \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance fault \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --dropout 0.05 \
  --selection-metric val_fault_macro_acc \
  --log-format pretty \
  --log-header-interval 20 \
  2>&1 | tee "${LOG_FILE}"
