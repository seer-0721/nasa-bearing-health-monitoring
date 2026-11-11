#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python} -m stdp_snn.eval --model results/stdp_snn/nasa/best.pkl --batch-size 512 --num-workers 0
${PYTHON:-python} -m stdp_snn.eval --model results/stdp_snn/cwru/best.pkl --batch-size 512 --num-workers 0
${PYTHON:-python} -m stdp_snn.eval --model results/stdp_snn/paderborn/best.pkl --batch-size 512 --num-workers 0
