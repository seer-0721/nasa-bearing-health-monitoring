#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python} -m rc_esn.run_all --datasets nasa,cwru,paderborn
