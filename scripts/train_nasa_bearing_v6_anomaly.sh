#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data-root data/NasaBearingDataset \
  --checkpoint-dir checkpoints/nasa_bearing_snn_health_reg_v6_anomaly \
  --limit-good 0 \
  --limit-fault 0 \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs 100 \
  --batch-size 256 \
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
  --device cuda
