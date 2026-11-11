#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python} -m rc_esn.eval --model results/rc_esn/nasa/best.pkl --batch-size 512 --num-workers 0
${PYTHON:-python} -m rc_esn.eval --model results/rc_esn/cwru/best.pkl --batch-size 512 --num-workers 0
${PYTHON:-python} -m rc_esn.eval --model results/rc_esn/paderborn/best.pkl --batch-size 512 --num-workers 0
