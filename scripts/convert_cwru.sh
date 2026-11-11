#!/usr/bin/env bash
set -euo pipefail

python tools/convert_bearing_datasets.py cwru \
  --input-root data/CWRU \
  --output-root data/CWRU/converted
