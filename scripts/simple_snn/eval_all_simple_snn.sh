#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python} -m simple_snn.eval --model results/simple_snn/nasa/best.pkl --batch-size 512 --num-workers 0
${PYTHON:-python} -m simple_snn.eval --model results/simple_snn/cwru/best.pkl --batch-size 512 --num-workers 0
${PYTHON:-python} -m simple_snn.eval --model results/simple_snn/paderborn/best.pkl --batch-size 512 --num-workers 0
