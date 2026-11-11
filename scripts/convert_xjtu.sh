#!/usr/bin/env bash
set -euo pipefail

python tools/convert_bearing_datasets.py xjtu \
  --input-root data/XJTU/extracted/XJTU-SY_Bearing_Datasets \
  --output-root data/XJTU/converted
