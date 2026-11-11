#!/usr/bin/env bash
set -euo pipefail

: "${DEVICE:=auto}"
: "${NUM_WORKERS:=0}"

DEVICE="${DEVICE}" NUM_WORKERS="${NUM_WORKERS}" bash scripts/train_nasa_best.sh
DEVICE="${DEVICE}" NUM_WORKERS="${NUM_WORKERS}" bash scripts/train_xjtu_best.sh
DEVICE="${DEVICE}" NUM_WORKERS="${NUM_WORKERS}" bash scripts/train_cwru_full.sh
DEVICE="${DEVICE}" NUM_WORKERS="${NUM_WORKERS}" bash scripts/train_paderborn_best.sh
