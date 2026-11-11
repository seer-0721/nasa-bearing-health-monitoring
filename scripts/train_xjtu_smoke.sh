#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data-root data/XJTU/converted \
  --checkpoint-dir checkpoints/xjtu_snn_health_reg \
  --health-label-mode anomaly \
  --limit-good 0 \
  --limit-fault 0 \
  --epochs 80 \
  --batch-size 256 \
  --time-steps 16 \
  --health-regression-loss l1 \
  --health-regression-weight 2.0 \
  --health-regression-low-score-weight 1.0 \
  --fault-loss-weight 0.2 \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance both \
  --hidden-size1 384 \
  --hidden-size2 192 \
  --dropout 0.05 \
  --device cuda
