#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data-root data/XJTU/converted_time \
  --checkpoint-dir checkpoints/xjtu_snn_row_time_best_cpu \
  --health-label-mode anomaly \
  --sample-layout row \
  --no-include-bearing-id \
  --limit-good 0 \
  --limit-fault 0 \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 0 \
  --device cpu \
  --time-steps 12 \
  --health-regression-loss l1 \
  --health-regression-weight 2.0 \
  --health-regression-low-score-weight 1.0 \
  --health-regression-high-score-weight 0.5 \
  --fault-loss-weight 0.5 \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance both \
  --hidden-size1 384 \
  --hidden-size2 192 \
  --dropout 0.05 \
  --log-header-interval 20
