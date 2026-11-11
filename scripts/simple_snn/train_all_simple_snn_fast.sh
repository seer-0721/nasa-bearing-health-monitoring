#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python} -m simple_snn.run_all --datasets nasa,cwru,paderborn --fast
