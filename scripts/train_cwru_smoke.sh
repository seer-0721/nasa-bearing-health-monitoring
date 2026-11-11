#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data-root data/CWRU/converted \
  --checkpoint-dir checkpoints/cwru_snn_fault_part \
  --health-label-mode anomaly \
  --limit-good 0 \
  --limit-fault 0 \
  --epochs 80 \
  --batch-size 256 \
  --time-steps 16 \
  --health-regression-loss l1 \
  --health-regression-weight 0.5 \
  --health-regression-low-score-weight 0.0 \
  --fault-loss-weight 1.0 \
  --class-weights balanced \
  --balanced-sampler \
  --sampler-balance fault \
  --hidden-size1 384 \
  --hidden-size2 192 \
  --dropout 0.05 \
  --device cuda
