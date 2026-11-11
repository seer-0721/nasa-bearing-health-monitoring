#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/nasa_bearing_snn_health_reg_best}"
LOG_FILE="${LOG_FILE:-logs/nasa_bearing_best.log}"
NUM_WORKERS="${NUM_WORKERS:-0}"

python -u train.py \
  --data-root data/NasaBearingDataset \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --limit-good 0 \
  --limit-fault 0 \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs "${EPOCHS}" \
  --batch-size 256 \
  --num-workers "${NUM_WORKERS}" \
  --time-steps 16 \
  --health-label-mode anomaly \
  --health-regression-loss l1 \
  --health-regression-weight 2.0 \
  --health-regression-low-score-weight 1.0 \
  --health-regression-high-score-weight 2.0 \
  --fault-loss-weight 0.2 \
  --fault-map B3=inner_ring,B4=rolling_element \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance both \
  --hidden-size1 384 \
  --hidden-size2 192 \
  --dropout 0.05 \
  --selection-metric composite \
  --device "${DEVICE}" \
  --log-format pretty \
  --log-header-interval 20 \
  2>&1 | tee "${LOG_FILE}"
