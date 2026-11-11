#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-120}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/cwru_snn_full_fault_best}"
LOG_FILE="${LOG_FILE:-logs/cwru_full_train.log}"
NUM_WORKERS="${NUM_WORKERS:-0}"

python -u train.py \
  --data-root data/CWRU/converted_full \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --health-label-mode source \
  --sample-layout row \
  --limit-good 0 \
  --limit-fault 0 \
  --epochs "${EPOCHS}" \
  --batch-size 256 \
  --time-steps 20 \
  --health-regression-loss l1 \
  --health-regression-weight 0.05 \
  --health-regression-low-score-weight 0.0 \
  --fault-loss-weight 3.0 \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance fault \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --dropout 0.03 \
  --selection-metric val_fault_macro_acc \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --log-format pretty \
  --log-header-interval 20 \
  2>&1 | tee "${LOG_FILE}"
